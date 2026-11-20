"""Paired regression and balanced inference timing for the local project refinement."""
import argparse
import csv
import json
import math
import os
import platform
from collections import defaultdict
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from context_controls import make_controls
from project_comparison import AMP, ROOT, SOURCES, array_hash, cases, general_sample, scores
from image_io_v3 import sha256_file
from project_pipeline import denoise as baseline, load_config
from project_refined import denoise as refined, routing_parameters


EXTRA_DEV = ('147021', '151087', '153093', '156079')
DESIGN_TEXTURES = {'banded', 'braided', 'chequered', 'crosshatched', 'grid', 'grooved'}
ARCHIVE = ROOT / 'results/project'
NEW_SOURCES = ('evaluate_project_refinement.py', 'project_refined.py', 'project_texture.py',
               'project_directional.py', 'local_spectral.py', 'requirements-lock.txt')
TIE = 1e-12


def extended_cases():
    """Rebuild the previous 30 inputs, then reuse known DTD and BSDS development sources."""
    yield from cases()
    for name, clean, observed, masks, meta in make_controls():
        if name == 'compact_legitimate_stripe_clean':
            flat = np.full_like(clean, .5)
            band = np.zeros(clean.shape, dtype=bool)
            band[:meta['support_width'], :] = True
            yield 'analytic_twin', 'flat_with_true_edge8', 'edge8', '', flat, observed, dict(contaminated=band, clean=~band)
        elif name == 'smooth_background_edge16':
            yield 'analytic_positive', name, 'edge16', '', clean, observed, masks
    manifest = json.loads((ROOT / 'data/v3_manifest.json').read_text(encoding='utf8'))
    records = {r['id']: r for r in manifest['images'] + manifest['textures']}
    texture_ids = [r['id'] for r in manifest['textures']]
    for image_id in [*texture_ids, *EXTRA_DEV]:
        r = records[image_id]
        is_dev = image_id in EXTRA_DEV
        if is_dev and r['split'] != 'dev':
            raise ValueError(f'Unexpected data split for {image_id}')
        path = ROOT / 'data' / r['npy_path']
        if sha256_file(path) != r['clean_sha256']:
            raise ValueError(f'Unexpected reference for {image_id}')
        clean = np.load(path, allow_pickle=False)
        kinds = ('clean', 'global', 'local', 'chirp', 'orientation') if is_dev else ('global', 'chirp')
        for kind in kinds:
            observed, _, masks = general_sample(clean, image_id, kind, 0. if kind == 'clean' else AMP)
            yield 'development_extension' if is_dev else 'DTD_paired', image_id, kind, '', clean, observed, masks


def identity(row):
    return (row['cohort'], row['image_id'], row['kind'], str(row['variant']))


def texture_subset(image_id):
    return 'design' if image_id.split('_')[1] in DESIGN_TEXTURES else 'remaining'


def archive_rows():
    with (ARCHIVE / 'metrics.csv').open(newline='', encoding='utf8') as f:
        metrics = {identity(row): row for row in csv.DictReader(f) if row['method'] == 'project'}
    run = json.loads((ARCHIVE / 'run.json').read_text(encoding='utf8'))
    for name, digest in run['source_sha256'].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f'Historical comparison source changed: {name}')
    inputs = {identity(row): row for row in run['cases']}
    if len(inputs) != 30 or inputs.keys() != metrics.keys():
        raise ValueError('Historical 30-input baseline is incomplete')
    return inputs, metrics


def assert_archive_match(source, baseline_row, original, old_metrics):
    for field in ('observed_sha256', 'reference_sha256'):
        if source[field] != original[field]:
            raise AssertionError(f'{identity(source)} mismatched {field}')
    if baseline_row['route'] != old_metrics['route']:
        raise AssertionError(f'{identity(source)} mismatched route')
    for field in ('mse', 'psnr_db', 'ssim', 'mae', 'contaminated_mse', 'clean_mse',
                  'changed_pixels', 'adjacent_clean_mse', 'broad_only_mse', 'narrow_only_mse'):
        new, old = baseline_row[field], old_metrics[field]
        if new == '' or old == '':
            same = new == old
        else:
            same = np.isclose(float(new), float(old), atol=1e-14, rtol=1e-12)
        if not same:
            raise AssertionError(f'{identity(source)} mismatched {field}: {new} versus {old}')


def timed_pair(index, observed, cfg):
    """Two runs per method, once first and once second on the very same input."""
    order = ('baseline', 'refined') if index % 2 == 0 else ('refined', 'baseline')
    methods = dict(baseline=baseline, refined=refined)
    times = defaultdict(list)
    outputs = {}
    for method in (*order, *reversed(order)):
        start = perf_counter()
        output, diag = methods[method](observed, cfg, return_diagnostics=True)
        times[method].append((perf_counter() - start) * 1000.)
        support = diag['support_mask']
        if support.shape != observed.shape or support.dtype != np.bool_:
            raise AssertionError(f'{method}: invalid authorized support')
        if not np.array_equal(output[~support], observed[~support]):
            raise AssertionError(f'{method}: changed pixels outside authorized region')
        if not np.array_equal(diag['residual'], observed-output):
            raise AssertionError(f'{method}: saved residual disagrees with output')
        if method in outputs:
            first, first_diag = outputs[method]
            if (not np.array_equal(first, output) or first_diag['route'] != diag['route']
                    or not np.array_equal(first_diag['support_mask'], support)):
                raise AssertionError(f'{method}: repeated inference disagrees')
        else:
            outputs[method] = (output, diag)
    return outputs, times, order[0]


def aggregate(rows):
    """Per-image statistics. Absent regions have no value and never enter means."""
    def mean(field):
        vals = [row[field] for row in rows if row[field] != '']
        return float(np.mean(vals)) if vals else None

    return dict(inputs=len(rows), modified=sum(r['changed_pixels'] > 0 for r in rows),
                mse=mean('mse'), mae=mean('mae'), contaminated_mse=mean('contaminated_mse'),
                clean_mse=mean('clean_mse'), adjacent_clean_mse=mean('adjacent_clean_mse'),
                broad_only_mse=mean('broad_only_mse'), narrow_only_mse=mean('narrow_only_mse'),
                worst_fifth_mae=float(np.mean(sorted((r['mae'] for r in rows), reverse=True)
                                              [:math.ceil(len(rows)/5)])))


def evaluate(destination):
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'Choose an empty output directory: {destination}')
    sources = {n: sha256_file(ROOT / n) for n in (*SOURCES, *NEW_SOURCES,
               'results/project/run.json', 'results/project/metrics.csv')}
    originals, old_metrics = archive_rows()
    cfg = load_config()
    yy, xx = np.indices((144, 144))
    warmup = .45 + .08*np.sin(.7*xx + .4*yy) + .02*np.random.default_rng(32).normal(size=xx.shape)
    for method in (baseline, refined):
        method(warmup, cfg, return_diagnostics=True)
    rows, sources_by_case, checked = [], [], set()
    totals = defaultdict(list)
    for index, (cohort, image_id, kind, variant, clean, observed, masks) in enumerate(extended_cases()):
        key = dict(cohort=cohort, image_id=image_id, kind=kind, variant=variant)
        source = key | dict(reference_sha256=array_hash(clean), observed_sha256=array_hash(observed))
        if cohort in ('DTD', 'DTD_paired'):
            source['texture_subset'] = texture_subset(image_id)
        outputs, times, first = timed_pair(index, observed, cfg)
        row_scores = {}
        for method in ('input', 'baseline', 'refined'):
            output, diag = (observed, {}) if method == 'input' else outputs[method]
            row = key | dict(method=method, route=diag.get('route', ''), reason=diag.get('reason', ''))
            row.update(scores(clean, observed, output, masks))
            ev = diag.get('evidence', {})
            screen = ev.get('texture_screen', {})
            row.update(narrow_supported_pixels=ev.get('narrow_supported_pixels'),
                       spatially_eligible_tiles=ev.get('general_corroborated_tiles'),
                       texture_vetoed_tiles=screen.get('texture_vetoed_tiles'),
                       component_decisions='|'.join(c['reason'] for c in screen.get('components', [])))
            row.update(changed_clean_pixels=int(np.count_nonzero((output != observed) & masks['clean'])),
                       output_sha256=array_hash(output), first_method=first,
                       run1_ms='', run2_ms='', mean_ms='')
            if method != 'input':
                row['run1_ms'], row['run2_ms'] = times[method]
                row['mean_ms'] = float(np.mean(times[method]))
                totals[method].append(row['mean_ms'])
            rows.append(row)
            row_scores[method] = row
        if identity(key) in originals:
            assert_archive_match(source, row_scores['baseline'], originals[identity(key)], old_metrics[identity(key)])
            checked.add(identity(key))
        sources_by_case.append(source)
        print(index+1, image_id, kind, row_scores['baseline']['route'], row_scores['refined']['route'], flush=True)
    if checked != originals.keys() or len(rows) != 228 or len({identity(r) for r in sources_by_case}) != 76:
        raise AssertionError('Incomplete pairing or duplicate cases')
    twins = {'compact_legitimate_stripe_clean', 'flat_with_true_edge8'}
    for method in ('input','baseline','refined'):
        outputs = [r['output_sha256'] for r in rows if r['image_id'] in twins and r['method'] == method]
        if len(outputs) != 2 or outputs[0] != outputs[1]:
            raise AssertionError(f'{method}: identical observations produced different outputs')
    if any(sha256_file(ROOT / name) != digest for name, digest in sources.items()):
        raise RuntimeError('A comparison source changed during inference')
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups = [r['cohort'] + '/' + r['kind']]
        if r['cohort'] in ('DTD', 'DTD_paired'):
            subset = texture_subset(r['image_id'])
            groups.append('DTD_' + subset + '/' + r['kind'])
        if r['cohort'] in ('development', 'development_extension'):
            groups.append('all_development/' + r['kind'])
        for group in groups:
            grouped[group][r['method']].append(r)
    summary = {}
    for group, methods in grouped.items():
        summary[group] = {method: aggregate(records) for method, records in methods.items()}
        input_mse = np.array([r['mse'] for r in methods['input']])
        for method in ('baseline', 'refined'):
            mse = np.array([r['mse'] for r in methods[method]])
            summary[group][method].update(improved_vs_input=int(np.sum(mse < input_mse-TIE)),
                                          harmed_vs_input=int(np.sum(mse > input_mse+TIE)))
        comparison = {}
        for field in ('mse','contaminated_mse','clean_mse','adjacent_clean_mse'):
            delta = np.array([new[field] - old[field]
                              for old, new in zip(methods['baseline'], methods['refined'])
                              if old[field] != '' and new[field] != ''])
            comparison[field] = dict(available=len(delta), better=int(np.sum(delta < -TIE)),
                                     tied=int(np.sum(np.abs(delta) <= TIE)), worse=int(np.sum(delta > TIE)))
        summary[group]['paired_refined_vs_baseline'] = comparison
    elapsed = {method: dict(total_mean_seconds=sum(values)/1000, median_mean_ms=float(np.median(values)),
                            mean_ms=float(np.mean(values))) for method, values in totals.items()}
    report = dict(inputs=76, paired_rows=228, original_inputs_verified=len(checked), tie_mse=TIE,
                  design_dtd_classes=sorted(DESIGN_TEXTURES),
                  original_baseline_check='Routes, input/reference hashes and all prior numerical metric fields checked; '
                  'atol 1e-14, rtol 1e-12. No archived output arrays exist for all previous inputs.',
                  extended_data='Existing BSDS development IDs; same 12 DTD photos, plus deterministic .12 global/chirp.',
                  ambiguity_twin='The compact legitimate texture and flat-field true edge8 have identical observed '
                                 'pixels; each method must return identical output for the two interpretations.',
                  timing='Single process, synthetic warmup for both methods, ABBA/BAAB order alternates per image. '
                         'Per-image mean of two diagnostics-enabled calls; I/O, configuration and scoring excluded.',
                  timing_ms=elapsed, timing_total_speedup=sum(totals['baseline'])/sum(totals['refined']),
                  environment=dict(os=platform.platform(), python=platform.python_version(),
                                   cpu=platform.processor(), libraries={n: version(n) for n in ('numpy','scipy','scikit-image')},
                                   thread_environment={k: os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')}),
                  configuration=asdict(cfg), routing_parameters=routing_parameters(), source_sha256=sources, cases=sources_by_case,
                  grouped_summary='summary.json, means per image; worst fifth averages ceil(n/5) largest MAEs',
                  reproduce='python evaluate_project_refinement.py --out results/project_refinement_reproduction')
    destination.mkdir(parents=True, exist_ok=True)
    with (destination/'metrics.csv').open('w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for filename, data in (('summary.json',summary), ('run.json',report)):
        (destination/filename).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf8')
    print(f'{len(rows)} rows, {len(checked)} prior baseline cases verified; {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/project_refinement_reproduction')
    evaluate(parser.parse_args().out)
