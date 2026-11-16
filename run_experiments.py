"""Reproducible development calibration and held-out evaluation.

Only the evaluation layer sees clean references and synthesis metadata.
Run `python run_experiments.py --help` for the staged workflow.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('starter_demo', ROOT / 'original/periodic_notch_starter/demo.py')
starter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = starter
spec.loader.exec_module(starter)

KINDS = ('global', 'local', 'chirp', 'orientation')
AMPLITUDES = (0.06, 0.12, 0.18)


def json_dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def metric_row(clean, noisy, output, masks):
    output = np.asarray(output)
    if output.shape != clean.shape or not np.isfinite(output).all():
        raise ValueError('Invalid algorithm output shape or nonfinite values')
    if output.min() < -1e-8 or output.max() > 1 + 1e-8:
        raise ValueError('Algorithm output outside [0,1]')
    err = (output-clean)**2
    mse = float(err.mean())
    noise_mse = float(np.mean((noisy-clean)**2))
    def masked_mean(a, mask):
        return float(a[mask].mean()) if np.any(mask) else ''
    contaminated = np.asarray(masks['contaminated'], bool)
    untouched = np.asarray(masks['clean'], bool)
    return dict(mse=mse, psnr_db=-10*np.log10(mse) if mse else 'inf',
                ssim=float(structural_similarity(clean, output, data_range=1)),
                mae=float(np.abs(output-clean).mean()),
                contaminated_mse=masked_mean(err, contaminated),
                clean_mse=masked_mean(err, untouched),
                clean_modification_mae=masked_mean(np.abs(output-noisy), untouched),
                input_mse=noise_mse,
                degraded=int(mse > noise_mse + 1e-8))


def dataset_records(manifest, split):
    """Accept the versioned manifest provided by data_pipeline."""
    records = manifest.get('images', manifest.get('records', []))
    return [r for r in records if r['split'] == split]


def load_clean(record, data_dir):
    path = record.get('npy_path', record.get('clean_npy', record.get('path')))
    p = Path(path)
    if not p.is_absolute():
        p = Path(data_dir) / p
    return np.load(p)


def sample_seed(image_id, kind, amplitude):
    # Paired amplitude sweep: geometry, phase and frequency stay identical.
    token = f'20261116:{image_id}:{kind}'.encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:4], 'little')


def make_sample(clean, image_id, kind, amplitude):
    from data_pipeline import synthesize
    if kind == 'clean':
        masks = dict(contaminated=np.zeros(clean.shape, bool), clean=np.ones(clean.shape, bool))
        return clean.copy(), {'kind': 'clean', 'seed': 0, 'clipping_fraction': 0.0}, masks
    return synthesize(clean, kind, amplitude, sample_seed(image_id, kind, amplitude))


def time_call(fn):
    t = time.perf_counter()
    result = fn()
    return result, (time.perf_counter()-t)*1000


def evaluate_methods(noisy, config):
    """Receives ONLY the observed image and development-frozen configuration."""
    from local_spectral import LocalConfig, prepare, restore
    from published_baseline import denoise as published_denoise
    cfg = starter.Config(fixed_sigma=config['fixed_sigma'])
    (spectrum, peaks, _), detection_ms = time_call(lambda: starter.detect_peaks(noisy, cfg))
    yield 'input', noisy, 0.0, None
    for method, adaptive in [('global_fixed', False), ('global_adaptive', True)]:
        (result, _, _, _), ms = time_call(lambda: starter.apply_notches(noisy, spectrum, peaks, cfg, adaptive))
        yield method, result, detection_ms+ms, None
    try:
        result, ms = time_call(lambda: published_denoise(noisy))
        if isinstance(result, tuple):
            result = result[0]
        yield 'published_arpenos', result, ms, None
    except Exception as exc:
        yield 'published_arpenos', None, 0., {'error': repr(exc)}
    base = LocalConfig(**config.get('local_config', {}))
    try:
        prepared, prep_ms = time_call(lambda: prepare(noisy, base))
        for method, conservative in [('local_adaptive', True), ('without_conservative', False)]:
            try:
                (result, diag), ms = time_call(lambda: restore(prepared,
                    strength=config['local_strength'], conservative=conservative, return_diagnostics=True))
                yield method, result, prep_ms+ms, diag
            except Exception as exc:
                yield method, None, 0., {'error': repr(exc)}
    except Exception as exc:
        for method in ('local_adaptive','without_conservative'):
            yield method, None, 0., {'error': repr(exc)}
    fixed = replace(base, fixed_scale=128)
    try:
        prepared, prep_ms = time_call(lambda: prepare(noisy, fixed))
        (result, diag), ms = time_call(lambda: restore(prepared, strength=config.get('fixed_local_strength', config['local_strength']),
                                                    conservative=True, return_diagnostics=True))
        yield 'local_fixed128', result, prep_ms+ms, diag
    except Exception as exc:
        yield 'local_fixed128', None, 0., {'error': repr(exc)}
    try:
        # Pure scale-selection ablation: keep all cross-scale evidence, force
        # only the selected analysis scale. This differs from fixed-scale baseline.
        selected128 = replace(base, selection_scale=128)
        prepared, prep_ms = time_call(lambda: prepare(noisy, selected128))
        (result, diag), ms = time_call(lambda: restore(prepared, strength=config['local_strength'],
                                                    conservative=True, return_diagnostics=True))
        yield 'without_scale_selection', result, prep_ms+ms, diag
    except Exception as exc:
        yield 'without_scale_selection', None, 0., {'error': repr(exc)}


def calibrate(data_dir, out, max_images=30):
    """Tune simple operating points on development images, never test images."""
    from data_pipeline import load_manifest
    from local_spectral import LocalConfig, prepare, restore
    manifest = load_manifest(Path(data_dir) / 'manifest.json')
    records = dataset_records(manifest, 'dev')[:max_images]
    sigmas, strengths = (1.0, 2.0, 3.0), (0.5, 0.75, 1.0)
    losses = {f'sigma_{s}': [] for s in sigmas}
    losses.update({f'{mode}_{s}': [] for mode in ('adaptive', 'fixed') for s in strengths})
    for index, record in enumerate(records):
        image_id = str(record.get('id', record.get('image_id')))
        clean = load_clean(record, data_dir)
        for kind in (*KINDS, 'clean'):
            noisy, _, _ = make_sample(clean, image_id, kind, 0.12)
            cfg = starter.Config()
            spectrum, peaks, _ = starter.detect_peaks(noisy, cfg)
            for sigma in sigmas:
                result, *_ = starter.apply_notches(noisy, spectrum, peaks, replace(cfg, fixed_sigma=sigma), False)
                losses[f'sigma_{sigma}'].append(float(np.mean((result-clean)**2)))
            for mode in ('adaptive', 'fixed'):
                prepared = prepare(noisy, LocalConfig(fixed_scale=128 if mode == 'fixed' else None))
                for strength in strengths:
                    result = restore(prepared, strength=strength, conservative=True)
                    losses[f'{mode}_{strength}'].append(float(np.mean((result-clean)**2)))
        print(f'calibration {index+1}/{len(records)} {image_id}', flush=True)
    means = {k: float(np.mean(v)) for k, v in losses.items()}
    config = dict(schema_version=1, fixed_sigma=min(sigmas, key=lambda s: means[f'sigma_{s}']),
                  local_strength=min(strengths, key=lambda s: means[f'adaptive_{s}']),
                  fixed_local_strength=min(strengths, key=lambda s: means[f'fixed_{s}']),
                  local_config=asdict(LocalConfig()),
                  calibration=dict(split='dev', ids=[str(r.get('id', r.get('image_id'))) for r in records],
                                   amplitude=0.12, cases=[*KINDS, 'clean'],
                                   objective='mean image MSE, equal weight per case and image', losses=means))
    json_dump(out, config)
    return config


def run(data_dir, config_path, output_dir, limit=0):
    from data_pipeline import load_manifest
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    manifest = load_manifest(data_dir / 'manifest.json')
    records = dataset_records(manifest, 'test')
    if limit:
        records = records[:limit]
    test_ids = {str(r.get('id',r.get('image_id'))) for r in records}
    dev_ids = set(config['calibration']['ids'])
    if test_ids & dev_ids:
        raise ValueError('Calibration/test image overlap')
    json_dump(output_dir/'source_hashes.json', {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('data_pipeline.py','local_spectral.py','published_baseline.py','run_experiments.py')})
    json_dump(output_dir/'run_config.json', dict(config=config, test_ids=[r.get('id', r.get('image_id')) for r in records],
             python=sys.version, platform=platform.platform(),
             amplitudes=AMPLITUDES, kinds=KINDS, numpy=np.__version__))
    from importlib.metadata import version
    json_dump(output_dir/'environment.json', {n:version(n) for n in ('numpy','scipy','scikit-image','Pillow','matplotlib')})
    fields = ['image_id','kind','amplitude','method','runtime_ms','clipping_fraction','mse','psnr_db','ssim','mae',
              'contaminated_mse','clean_mse','clean_modification_mae','input_mse','degraded','output_projection_fraction','error']
    failed = []
    meta_file = (output_dir/'synthesis.jsonl').open('w',encoding='utf-8')
    with (output_dir/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fields)
        writer.writeheader()
        for index, record in enumerate(records):
            image_id = str(record.get('id',record.get('image_id')))
            clean = load_clean(record, data_dir)
            cases = [(k,a) for k in KINDS for a in AMPLITUDES]+[('clean',0.0)]
            for kind, amplitude in cases:
                noisy, metadata, masks = make_sample(clean, image_id, kind, amplitude)
                meta_file.write(json.dumps(dict(image_id=image_id, **metadata), ensure_ascii=False)+'\n')
                try:
                    for method, result, ms, diag in evaluate_methods(noisy, config):
                        row = dict(image_id=image_id, kind=kind, amplitude=amplitude, method=method,
                                   runtime_ms=ms, clipping_fraction=metadata['clipping_fraction'], error='')
                        if result is None:
                            row['error'] = diag['error']
                            failed.append(dict(image_id=image_id,kind=kind,amplitude=amplitude,method=method,error=row['error']))
                            writer.writerow(row)
                            continue
                        # Common feasible intensity projection for every method.
                        # Count affected pixels; do not contrast-stretch outputs.
                        row['output_projection_fraction'] = float(np.mean((result<0)|(result>1)))
                        result = np.clip(result,0,1)
                        row.update(metric_row(clean, noisy, result, masks))
                        writer.writerow(row)
                        # Predetermined first three IDs: no selection by visual quality.
                        if index < 3 and amplitude in (0.12, 0.0):
                            folder = output_dir/'examples'/f'{image_id}_{kind}'
                            folder.mkdir(parents=True,exist_ok=True)
                            np.save(folder/f'{method}.npy',result.astype('float32'))
                            if method == 'input':
                                np.save(folder/'reference.npy',clean.astype('float32'))
                                for key, mask in masks.items():
                                    if isinstance(mask,np.ndarray): np.save(folder/f'mask_{key}.npy',mask)
                            if method == 'local_adaptive' and diag:
                                np.savez_compressed(folder/'diagnostics.npz', **{k:v for k,v in diag.items() if isinstance(v,np.ndarray)})
                                json_dump(folder/'tiles.json',diag.get('tiles',[]))
                except Exception as exc:
                    failed.append(dict(image_id=image_id,kind=kind,amplitude=amplitude,error=repr(exc)))
                    print('ERROR',failed[-1],flush=True)
                file.flush()
                meta_file.flush()
            print(f'evaluation {index+1}/{len(records)} {image_id}',flush=True)
    meta_file.close()
    json_dump(output_dir/'exceptions.json',failed)
    if failed:
        raise RuntimeError(f'{len(failed)} failed samples; inspect exceptions.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('calibrate','test'))
    parser.add_argument('--data',default='data')
    parser.add_argument('--config',default='configs/frozen.json')
    parser.add_argument('--out',default='results/main')
    parser.add_argument('--limit',type=int,default=0,help='Optional explicit image limit for smoke runs')
    args = parser.parse_args()
    if args.stage == 'calibrate':
        calibrate(args.data,args.config,args.limit or 30)
    else:
        run(args.data,args.config,args.out,args.limit)


if __name__ == '__main__':
    main()
