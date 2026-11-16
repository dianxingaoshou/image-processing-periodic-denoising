"""Verify the completed frozen benchmark and supplementary artifacts.

This checks consistency and reproducibility of stored results. It does not tune
parameters, change algorithms or certify scientific superiority. Run only after
run_experiments.py, validate_tradeoff.py and audit_failures.py have completed.
Exit status is nonzero if any check fails; details go to results/verification.json.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
METHODS = {
    'input', 'global_fixed', 'global_adaptive', 'published_arpenos',
    'local_adaptive', 'local_fixed128', 'without_scale_selection',
    'without_conservative',
}
KINDS = ('global', 'local', 'chirp', 'orientation')
AMPLITUDES = (.06, .12, .18)
STRENGTHS = (0., .125, .25, .375, .5, .625, .75, .875, 1., 1.25, 1.5, 2.)
MODES = ('conservative', 'without_conservative')
SOURCES = ('data_pipeline.py', 'local_spectral.py', 'published_baseline.py', 'run_experiments.py')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sha_array(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype='<f8').tobytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, *, atol=1e-12, rtol=1e-9):
    require(math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol),
            f'{message}: got {actual}, expected {expected}')


def quality_relation(mse, psnr, label):
    mse, psnr = float(mse), float(psnr)
    require(math.isfinite(mse) and mse >= 0, f'{label}: invalid MSE {mse}')
    if mse == 0:
        require(psnr == math.inf, f'{label}: zero MSE must have +inf PSNR')
    else:
        close(psnr, -10 * math.log10(mse), f'{label}: PSNR/data_range=1 relation', atol=1e-9)


def sample_key(row):
    return (str(row['image_id']), row['kind'], float(row.get('amplitude', 0.0)))


class Verifier:
    def __init__(self, data, main, supplement, config):
        self.data, self.main, self.supplement, self.config_path = map(Path, (data, main, supplement, config))
        self.checks = []
        self.metrics = []
        self.config = None
        self.run_config = None
        self.manifest = None
        self.synthesis = []
        self.metric_index = {}

    def check(self, name, function):
        try:
            details = function()
            item = {'check': name, 'passed': True, 'details': details}
        except Exception as exc:
            item = {'check': name, 'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
        self.checks.append(item)
        print(('PASS' if item['passed'] else 'FAIL') + ' ' + name, flush=True)

    def load(self):
        self.config = read_json(self.config_path)
        self.run_config = read_json(self.main / 'run_config.json')
        self.manifest = read_json(self.data / 'manifest.json')
        self.metrics = read_csv(self.main / 'metrics.csv')
        with (self.main / 'synthesis.jsonl').open(encoding='utf-8') as stream:
            self.synthesis = [json.loads(line) for line in stream if line.strip()]
        self.metric_index = {(*sample_key(r), r['method']): r for r in self.metrics}
        return {'metrics_rows': len(self.metrics), 'synthesis_rows': len(self.synthesis)}

    def splits(self):
        images = self.manifest['images']
        dev = [str(r.get('id', r.get('image_id'))) for r in images if r['split'] == 'dev']
        test = [str(r.get('id', r.get('image_id'))) for r in images if r['split'] == 'test']
        require(len(dev) == len(set(dev)) == 30, 'Expected 30 unique development images')
        require(len(test) == len(set(test)) == 60, 'Expected 60 unique test images')
        require(not set(dev) & set(test), 'Image-level development/test overlap')
        require(self.run_config['test_ids'] == test, 'Run test IDs/order differ from manifest')
        require(set(self.config['calibration']['ids']) == set(dev), 'Calibration did not use the manifest development split')
        require(self.config['calibration']['split'] == 'dev', 'Calibration split label is not dev')
        for record in images:
            allowed = {'train', 'val'} if record['split'] == 'dev' else {'test'}
            require(record['official_split'] in allowed, f'Official split violation: {record["id"]}')
        return {'development_images': len(dev), 'test_images': len(test), 'overlap': 0}

    def configuration(self):
        require(self.config == self.run_config['config'], 'Frozen configuration differs from embedded run configuration')
        require(self.run_config['kinds'] == list(KINDS), 'Unexpected main-run degradation kinds')
        require(self.run_config['amplitudes'] == list(AMPLITUDES), 'Unexpected main-run amplitudes')
        notes = read_json(self.supplement / 'validation_notes.json')
        actual_hash = sha_file(self.config_path)
        require(notes['configuration_sha256'] == actual_hash, 'Supplement used a different frozen-config file hash')
        require(notes['local_config'] == self.config['local_config'], 'Supplement local settings differ')
        require(notes['no_parameter_tuning'] is True, 'Supplement tuning declaration missing')
        return {'frozen_file_sha256': actual_hash,
                'frozen_canonical_sha256': canonical_hash(self.config),
                'run_embedded_canonical_sha256': canonical_hash(self.run_config['config']),
                'supplement_frozen_file_sha256': notes['configuration_sha256']}

    def source_hashes(self):
        recorded = read_json(self.main / 'source_hashes.json')
        actual = {name: sha_file(ROOT / name) for name in SOURCES}
        require(set(recorded) == set(SOURCES), 'Source hash inventory differs from required four files')
        require(recorded == actual, f'Current source differs from frozen run: {[n for n in actual if recorded.get(n) != actual[n]]}')
        return actual

    def data_hashes(self):
        images = self.manifest['images']
        for record in images:
            clean = self.data / record.get('npy_path', record['clean_npy'])
            require(sha_file(clean) == record['clean_sha256'], f'Clean reference hash mismatch: {record["id"]}')
            source = self.data / record['source_jpg']
            require(sha_file(source) == record['source_sha256'], f'Source JPEG hash mismatch: {record["id"]}')
        archive = self.data / 'source' / 'BSR_bsds500.tgz'
        require(sha_file(archive) == self.manifest['archive_sha256'], 'BSDS500 archive hash mismatch')
        return {'clean_references_checked': len(images), 'source_jpegs_checked': len(images),
                'archive_sha256': self.manifest['archive_sha256']}

    def main_inventory(self):
        test_ids = self.run_config['test_ids']
        cases = [(kind, amplitude) for kind in KINDS for amplitude in AMPLITUDES] + [('clean', 0.)]
        expected = {(image_id, kind, amplitude, method) for image_id in test_ids
                    for kind, amplitude in cases for method in METHODS}
        counts = Counter((*sample_key(r), r['method']) for r in self.metrics)
        require(len(self.metrics) == 6240, f'Expected 6240 rows, found {len(self.metrics)}; run may still be incomplete')
        require(set(counts) == expected, f'Unexpected/missing metric keys: missing={len(expected-set(counts))}, extra={len(set(counts)-expected)}')
        require(all(n == 1 for n in counts.values()), 'Duplicate metric keys')
        per_method = Counter(r['method'] for r in self.metrics)
        require(set(per_method) == METHODS and set(per_method.values()) == {780}, 'Expected eight methods with 780 rows each')
        return {'rows': len(self.metrics), 'unique_keys': len(counts), 'method_counts': dict(sorted(per_method.items()))}

    def main_numeric(self):
        for index, row in enumerate(self.metrics):
            label = f'metrics row {index + 2} {sample_key(row)} {row["method"]}'
            require(not row['error'], f'{label}: recorded method error')
            quality_relation(row['mse'], row['psnr_db'], label)
            ssim = float(row['ssim'])
            require(math.isfinite(ssim) and -1 - 1e-12 <= ssim <= 1 + 1e-12, f'{label}: SSIM outside [-1,1]')
            for name in ('mse', 'mae', 'input_mse', 'runtime_ms', 'contaminated_mse', 'clean_mse', 'clean_modification_mae'):
                if row[name] != '':
                    value = float(row[name])
                    require(math.isfinite(value) and value >= 0, f'{label}: invalid {name}')
            for name in ('clipping_fraction', 'output_projection_fraction'):
                value = float(row[name])
                require(math.isfinite(value) and 0 <= value <= 1, f'{label}: invalid {name}')
            require(int(row['degraded']) == int(float(row['mse']) > float(row['input_mse']) + 1e-8), f'{label}: degraded flag inconsistent')
            source = self.metric_index[(*sample_key(row), 'input')]
            close(row['input_mse'], source['mse'], f'{label}: input MSE does not match input row')
            if row['method'] == 'input':
                close(row['mse'], row['input_mse'], f'{label}: input MSE mismatch')
                require(float(row['runtime_ms']) == 0, f'{label}: input runtime must be zero')
                if row['kind'] == 'clean':
                    for name in ('mse', 'mae', 'input_mse', 'clean_mse', 'clean_modification_mae'):
                        require(float(row[name]) == 0, f'{label}: clean-input {name} must be zero')
                    require(float(row['ssim']) == 1, f'{label}: clean-input SSIM must be one')
            if row['kind'] in ('global', 'chirp', 'orientation'):
                require(row['clean_mse'] == '' and row['clean_modification_mae'] == '', f'{label}: no clean region expected')
        errors = read_json(self.main / 'exceptions.json')
        require(errors == [], f'Expected zero exceptions, found {len(errors)}')
        return {'numeric_rows_checked': len(self.metrics), 'ssim_range': '[-1,1]',
                'psnr_data_range': 1, 'exception_count': 0}

    def paired_synthesis(self):
        expected = {(image_id, kind, amp) for image_id in self.run_config['test_ids']
                    for kind in KINDS for amp in AMPLITUDES}
        expected |= {(image_id, 'clean', 0.) for image_id in self.run_config['test_ids']}
        counts = Counter(sample_key(row) for row in self.synthesis)
        require(len(self.synthesis) == 780 and set(counts) == expected, 'Expected exactly 780 synthesis records')
        require(all(n == 1 for n in counts.values()), 'Duplicate synthesis records')
        grouped = defaultdict(list)
        for row in self.synthesis:
            if row['kind'] == 'clean':
                require(row['seed'] == 0 and row['clipping_fraction'] == 0., 'Clean synthesis metadata must have seed/clipping zero')
            else:
                grouped[(row['image_id'], row['kind'])].append(row)
        for (image_id, kind), rows in grouped.items():
            require(sorted(float(r['amplitude']) for r in rows) == list(AMPLITUDES), f'Unpaired amplitudes: {image_id} {kind}')
            token = f'20261116:{image_id}:{kind}'.encode()
            seed = int.from_bytes(hashlib.sha256(token).digest()[:4], 'little')
            require({r['seed'] for r in rows} == {seed}, f'Seed mismatch across amplitudes: {image_id} {kind}')
            require(len({canonical_hash(r['parameters']) for r in rows}) == 1, f'Geometry/frequency/phase changed across amplitudes: {image_id} {kind}')
        # Recreate synthesis metadata with the recorded references and current
        # frozen data pipeline. Algorithms are not rerun in this check.
        from run_experiments import load_clean, make_sample
        by_id = {str(r['id']): r for r in self.manifest['images']}
        references = {}
        for row in self.synthesis:
            image_id = row['image_id']
            if image_id not in references:
                references[image_id] = load_clean(by_id[image_id], self.data)
            _, metadata, _ = make_sample(references[image_id], image_id, row['kind'], float(row.get('amplitude', 0.)))
            require(dict(image_id=image_id, **metadata) == row, f'Synthesis metadata reproduction mismatch: {sample_key(row)}')
        return {'synthesis_records': len(self.synthesis), 'paired_amplitude_groups': len(grouped),
                'metadata_records_reproduced': len(self.synthesis)}

    def supplementary_sweep(self):
        rows = read_csv(self.supplement / 'strength_curve.csv')
        notes = read_json(self.supplement / 'validation_notes.json')
        ids = self.run_config['test_ids'][:10]
        require(notes['test_ids'] == ids, 'Supplement subset is not the first ten manifest test IDs')
        require(notes['kinds'] == ['local', 'chirp', 'orientation'] and notes['strengths'] == list(STRENGTHS), 'Unexpected supplementary sweep design')
        require(notes['sweep_rows'] == len(rows) == 720, 'Expected 720 supplementary sweep rows')
        key = lambda r: (r['image_id'], r['kind'], r['mode'], float(r['strength']))
        counts = Counter(key(row) for row in rows)
        expected = {(image_id, kind, mode, strength) for image_id in ids
                    for kind in ('local', 'chirp', 'orientation') for mode in MODES for strength in STRENGTHS}
        require(set(counts) == expected and all(n == 1 for n in counts.values()), 'Supplement sweep keys incomplete or duplicated')
        metadata = {sample_key(r): r for r in self.synthesis}
        for row in rows:
            sk = (row['image_id'], row['kind'], .12)
            require(float(row['amplitude']) == .12, 'Supplement uses another amplitude')
            require(int(row['seed']) == metadata[sk]['seed'], f'Supplement seed differs from main: {sk}')
            source = self.metric_index[(*sk, 'input')]
            close(row['input_contaminated_mse'], source['contaminated_mse'], f'Supplement/main input region MSE {sk}')
            for name in ('contaminated_mse', 'clean_mse', 'input_contaminated_mse', 'overall_mse'):
                if row[name] != '':
                    value = float(row[name])
                    require(math.isfinite(value) and value >= 0, f'Supplement invalid {name}')
            if row['kind'] != 'local':
                require(row['clean_mse'] == '', 'Changing full-image interference must not have a clean-region metric')
            if float(row['strength']) == 0:
                close(row['overall_mse'], source['mse'], f'Zero-strength identity {sk}')
                if row['clean_mse'] != '':
                    require(float(row['clean_mse']) == 0, 'Zero strength modified a clean region')
            if float(row['strength']) == float(self.config['local_strength']):
                method = 'local_adaptive' if row['mode'] == 'conservative' else 'without_conservative'
                main = self.metric_index[(*sk, method)]
                close(row['overall_mse'], main['mse'], f'Supplement frozen-point/main MSE {sk} {method}')
        return {'sweep_rows': len(rows), 'unique_keys': len(counts), 'test_images': len(ids),
                'frozen_operating_point_matches_main': True}

    def texture_ambiguity(self):
        rows = read_csv(self.supplement / 'texture_metrics.csv')
        notes = read_json(self.supplement / 'validation_notes.json')
        require(notes['texture_rows'] == len(rows), 'Texture row count differs from notes')
        pairs = defaultdict(list)
        with np.load(self.supplement / 'texture_arrays.npz') as arrays:
            for row in rows:
                name, mode = row['case'], row['mode']
                observed = arrays[f'{name}_input']
                output = arrays[f'{name}_{mode}']
                reference = observed if row['interpretation'] == 'legitimate_texture' else arrays[f'{name}_alternative_reference']
                require(sha_array(observed) == row['input_sha256'], f'Texture input hash mismatch {name} {mode}')
                require(sha_array(output) == row['output_sha256'], f'Texture output hash mismatch {name} {mode}')
                close(row['mse'], np.mean((output-reference)**2), f'Texture recomputed MSE {name} {mode}')
                close(row['input_mse'], np.mean((observed-reference)**2), f'Texture input MSE {name} {mode}')
                quality_relation(row['mse'], row['psnr_db'], f'Texture {name} {mode}')
                require(float(row['strength']) == float(self.config['local_strength']), 'Texture uses another operating strength')
                pairs[(name, mode)].append(row)
        dual = 0
        for (name, mode), group in pairs.items():
            if name == 'brick_packaged_skimage':
                require(len(group) == 1 and group[0]['interpretation'] == 'legitimate_texture', 'Unexpected natural-texture interpretation')
                continue
            require(len(group) == 2 and {r['interpretation'] for r in group} == {'legitimate_texture', 'noise_on_flat_field'}, f'Missing dual interpretation {name} {mode}')
            require(len({r['input_sha256'] for r in group}) == 1, f'Dual interpretation changed input pixels {name} {mode}')
            require(len({r['output_sha256'] for r in group}) == 1, f'Dual interpretation changed output pixels {name} {mode}')
            dual += 1
        require(dual == 8, f'Expected four synthetic cases x two modes, found {dual} dual-interpretation groups')
        return {'texture_rows': len(rows), 'identical_pixel_and_output_interpretation_pairs': dual}

    def failure_audit(self):
        folder = self.main / 'failure_audit'
        selected = read_json(folder / 'selection.json')
        require(len(selected) == 5 and {r['kind'] for r in selected} == {*KINDS, 'clean'}, 'Expected five audited condition worst cases')
        from local_spectral import LocalConfig, denoise
        from run_experiments import load_clean, make_sample
        cfg = replace(LocalConfig(**self.config['local_config']), strength=self.config['local_strength'])
        by_id = {str(r['id']): r for r in self.manifest['images']}
        confirmed = []
        for audit in selected:
            kind = audit['kind']
            group = [r for r in self.metrics if r['method'] == 'local_adaptive' and r['kind'] == kind]
            worst = max(group, key=lambda r: float(r['mse']) - float(r['input_mse']))
            require((*sample_key(audit), audit['method']) == (*sample_key(worst), worst['method']), f'Audit did not select actual worst regression for {kind}')
            clean = load_clean(by_id[audit['image_id']], self.data)
            observed, meta, _ = make_sample(clean, audit['image_id'], kind, float(audit['amplitude']))
            require(meta == audit['synthesis'], f'Audit synthesis metadata differs for {kind}')
            require(hashlib.sha256(observed.tobytes()).hexdigest() == audit['input_sha256'], f'Audit regenerated input hash differs for {kind}')
            restored = denoise(observed, cfg)
            mse = float(np.mean((restored-clean)**2))
            close(mse, worst['mse'], f'Audit rerun/main MSE {kind}')
            close(mse, audit['rerun_mse'], f'Audit rerun/selection MSE {kind}')
            with np.load(folder / f'{kind}_worst.npz') as arrays:
                require(np.array_equal(arrays['reference'], clean), f'Audit reference array differs {kind}')
                require(np.array_equal(arrays['input'], observed), f'Audit input array differs {kind}')
                require(np.array_equal(arrays['output'], restored), f'Audit rerun output is not byte-identical {kind}')
                require(np.array_equal(arrays['residual'], observed-restored), f'Audit residual differs {kind}')
            confirmed.append({'kind': kind, 'image_id': audit['image_id'],
                              'amplitude': float(audit['amplitude']), 'rerun_mse': mse,
                              'output_sha256': sha_array(restored)})
        return {'rerun_cases': confirmed, 'configuration_changed': False}

    def run(self):
        for name, fn in (
            ('load_artifacts', self.load),
            ('image_level_splits', self.splits),
            ('frozen_configuration', self.configuration),
            ('recorded_source_hashes', self.source_hashes),
            ('reference_and_source_data_hashes', self.data_hashes),
            ('main_complete_unique_inventory', self.main_inventory),
            ('main_numeric_metrics_and_exceptions', self.main_numeric),
            ('paired_amplitudes_and_synthesis_reproduction', self.paired_synthesis),
            ('supplementary_sweep_and_main_consistency', self.supplementary_sweep),
            ('identical_pixels_dual_interpretation', self.texture_ambiguity),
            ('failure_audit_frozen_rerun', self.failure_audit),
        ):
            self.check(name, fn)
        passed = sum(c['passed'] for c in self.checks)
        return {'schema_version': 1, 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
                'all_passed': passed == len(self.checks), 'checks_passed': passed,
                'checks_total': len(self.checks), 'paths': {'data': str(self.data.resolve()),
                'main': str(self.main.resolve()), 'supplement': str(self.supplement.resolve()),
                'frozen_config': str(self.config_path.resolve())},
                'verifier_sha256': sha_file(__file__),
                'scope': 'Artifact consistency and five deterministic failure-case reruns; no parameter tuning or claims of superiority.',
                'checks': self.checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=ROOT / 'data', type=Path)
    parser.add_argument('--main', default=ROOT / 'results/main', type=Path)
    parser.add_argument('--supplement', default=ROOT / 'results/supplement', type=Path)
    parser.add_argument('--config', default=ROOT / 'configs/frozen.json', type=Path)
    parser.add_argument('--out', default=ROOT / 'results/verification.json', type=Path)
    args = parser.parse_args()
    report = Verifier(args.data, args.main, args.supplement, args.config).run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'{report["checks_passed"]}/{report["checks_total"]} checks passed; saved {args.out}', flush=True)
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
