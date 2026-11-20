"""Paired regression of the unified project against its two component branches."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

import denoiser_v3
import narrow_denoiser
from context_controls import make_controls
from image_io_v3 import preprocess_image, sha256_file
from project_pipeline import denoise, load_config
from run_experiments import metric_row
from support_data import make_sample as narrow_sample
from validation_data_v3 import make_sample as general_sample


ROOT = Path(__file__).resolve().parent
AMP = .12
DEV_IDS = ('140055', '145053')
REGRESSION_ID = '69022'
CONTROLS = ('compact_legitimate_stripe_clean', 'broad_phase_banded_clean')
EXAMPLES = {('69022', 'edge8'), ('140055', 'global'), ('140055', 'clean')}
SOURCES = ('project_pipeline.py', 'project_comparison.py', 'denoiser_v3.py', 'local_estimation_v3.py',
           'texture_guard_v3.py', 'narrow_denoiser.py', 'narrow_profiles.py', 'narrow_support.py',
           'narrow_context.py', 'run_experiments.py', 'data_pipeline.py', 'support_data.py',
           'validation_data_v3.py', 'image_io_v3.py', 'context_controls.py', 'configs/v3_p0_coupled.json',
           'configs/context_context_full.json', 'rejection_protocol.json', 'data/v3_manifest.json')


def array_hash(image):
    return hashlib.sha256(np.asarray(image, dtype='<f8').tobytes(order='C')).hexdigest()


def cases():
    data = ROOT / 'data'
    protocol = json.loads((ROOT / 'rejection_protocol.json').read_text(encoding='utf-8'))
    manifest = json.loads((data / 'v3_manifest.json').read_text(encoding='utf-8'))
    files = {r['id']: r for r in manifest['images'] + manifest['textures']}

    def reference(image_id):
        record = files[image_id]
        path = data / record['npy_path']
        if sha256_file(path) != record['clean_sha256']:
            raise ValueError(f'Reference file changed: {image_id}')
        return np.load(path, allow_pickle=False)

    def mixed_case(clean, image_id, variant):
        broad, _, bm = general_sample(clean, image_id, 'local', AMP)
        observed, _, nm = narrow_sample(broad, image_id, 'edge16', AMP, variant)
        masks = dict(contaminated=bm['contaminated'] | nm['contaminated'],
                     clean=bm['clean'] & nm['clean'],
                     broad_only=bm['contaminated'] & nm['clean'],
                     narrow_only=nm['contaminated'] & bm['clean'])
        return 'mixed', image_id, 'local+edge16', variant, clean, observed, masks

    for image_id in DEV_IDS:
        clean = reference(image_id)
        for kind in ('clean', 'global', 'local', 'chirp', 'orientation'):
            observed, _, masks = general_sample(clean, image_id, kind, 0 if kind == 'clean' else AMP)
            yield 'development', image_id, kind, '', clean, observed, masks
        if image_id == DEV_IDS[0]:
            yield mixed_case(clean, image_id, 0)

    source = next(r for r in protocol['regression_images'] if r['id'] == REGRESSION_ID)
    path = data / source['source_jpg']
    if sha256_file(path) != source['source_sha256']:
        raise ValueError(f'Regression JPEG changed: {REGRESSION_ID}')
    clean, _ = preprocess_image(path, size=256)
    for kind in ('clean', 'edge8', 'edge16', 'edge32'):
        observed, _, masks = narrow_sample(clean, REGRESSION_ID, kind, 0 if kind == 'clean' else AMP, 3)
        yield 'regression', REGRESSION_ID, kind, 3, clean, observed, masks
    yield mixed_case(clean, REGRESSION_ID, 3)

    for image_id in protocol['regression_textures']:
        clean = reference(image_id)
        observed, _, masks = general_sample(clean, image_id, 'clean', 0.)
        yield 'DTD', image_id, 'clean', '', clean, observed, masks
    for name, clean, observed, masks, _ in make_controls():
        if name in CONTROLS:
            yield 'analytic', name, 'clean', '', clean, observed, masks


def scores(clean, observed, output, masks):
    metrics = metric_row(clean, observed, output, masks)
    row = {name: metrics[name] for name in ('mse', 'psnr_db', 'ssim', 'mae', 'contaminated_mse', 'clean_mse')}
    row['changed_pixels'] = int(np.count_nonzero(output != observed))
    clean_mask = masks['clean']
    adjacent = (clean_mask & (distance_transform_edt(clean_mask) <= 16)
                if masks['contaminated'].any() else np.zeros_like(clean_mask))
    error = (output - clean) ** 2
    for key, mask in (('adjacent_clean', adjacent), ('broad_only', masks.get('broad_only')),
                      ('narrow_only', masks.get('narrow_only'))):
        row[key + '_mse'] = float(error[mask].mean()) if mask is not None and mask.any() else ''
    return row


def compare(destination):
    if any((destination / f).exists() for f in ('metrics.csv', 'run.json')):
        raise FileExistsError('Choose a fresh output directory to preserve previous measurements')
    source_hashes = {name: sha256_file(ROOT / name) for name in SOURCES}
    config = load_config()
    rows, inputs = [], []
    for cohort, image_id, kind, variant, clean, observed, masks in cases():
        # Only observed intensities and shared configs cross the inference boundary.
        narrow = narrow_denoiser.denoise(observed, config.narrow)
        general = denoiser_v3.denoise(observed, config.general)
        restored, diag = denoise(observed, config, return_diagnostics=True)
        identity = dict(cohort=cohort, image_id=image_id, kind=kind, variant=variant)
        expected = {'narrow': narrow, 'unchanged': observed}
        if diag['route'] in expected and not np.array_equal(expected[diag['route']], restored):
            raise AssertionError(f'{identity}: unified output disagrees with selected route')
        if np.any(restored[~diag['support_mask']] != observed[~diag['support_mask']]):
            raise AssertionError(f'{identity}: unified output changed outside declared support')
        for method, output in (('input', observed), ('general', general), ('narrow', narrow), ('project', restored)):
            rows.append(identity | dict(method=method, route=diag['route'] if method == 'project' else '')
                        | scores(clean, observed, output, masks))
        inputs.append(identity | dict(reference_sha256=array_hash(clean), observed_sha256=array_hash(observed),
                                      route=diag['route'], reason=diag['reason'], evidence=diag['evidence']))
        if (image_id, kind) in EXAMPLES:
            examples = destination / 'examples'
            examples.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(examples / f'{image_id}_{kind}.npz', clean=clean, observed=observed,
                                restored=restored, support_mask=diag['support_mask'],
                                difference=observed-restored, route=diag['route'])
        print(f'{cohort} / {image_id} / {kind}: {diag["route"]}', flush=True)

    if any(sha256_file(ROOT / name) != digest for name, digest in source_hashes.items()):
        raise RuntimeError('Comparison sources changed during inference; rerun with the final sources')
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(inputs=len(inputs), metric_rows=len(rows), route_counts=dict(Counter(r['route'] for r in inputs)),
                  reproduce=['python prepare_v3_data.py', 'python project_comparison.py --out results/project_reproduction'],
                  source_sha256=source_hashes,
                  amplitude=AMP, variants='edge side 0=left, 3=bottom; widths in pixels',
                  comparison='Fixed regression materials; each of four methods sees the same observed image.',
                  mixed_recipe='legacy local then edge16, amplitude .12 each, clip at both steps; '
                               'separate metrics use contamination of one and clean support of the other',
                  masks='contaminated and clean use the synthesis masks, adjacent = true clean within '
                        '16 pixels of non-clean support; empty regions have blank metrics',
                  array_hash='SHA256 of row-major little-endian float64 bytes',
                  configs=dict(general=denoiser_v3.config_dict(config.general),
                               narrow=narrow_denoiser.config_dict(config.narrow)),
                  cases=inputs)
    (destination / 'run.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                                          encoding='utf-8')
    print(f'{len(inputs)} inputs, {len(rows)} paired rows: {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/project_reproduction')
    compare(parser.parse_args().out)
