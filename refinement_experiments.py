"""Focused risk-score study. References enter evaluation only, never inference.

Stages: plan -> development -> freeze -> regression -> check -> report.
Historical v3 tables/configurations are not overwritten. The independent check
uses four previously unused BSDS test photographs selected without viewing them.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from denoiser_v3 import config_from_dict, config_dict, Prepared, restore
from evaluation_v3 import estimation_metrics, tail_mean
from gain_diagnostics import candidate_gain_rows, summarize_gain_rows
from image_io_v3 import preprocess_image, sha256_file
from local_estimation_v3 import prepare as estimate
from reproducibility_v3 import capture_snapshot, verify_snapshot
from run_experiments import metric_row
from texture_guard_v3 import guard_weights
from validation_data_v3 import make_sample

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'results/refinement'
PLAN = ROOT / 'refinement_protocol.json'
FREEZE = OUT / 'freeze.json'
STRENGTHS = (.5, 1., 1.5, 2.)
METHODS = ('coupled_baseline', 'decoupled_control', 'structure_risk')
KINDS = ('clean', 'global', 'local', 'chirp', 'orientation', 'edge_strip', 'irregular')
SOURCES = ('refinement_experiments.py', 'gain_diagnostics.py', 'denoiser_v3.py',
    'local_estimation_v3.py', 'texture_guard_v3.py', 'image_io_v3.py',
    'evaluation_v3.py', 'run_experiments.py', 'validation_data_v3.py',
    'data_pipeline.py', 'local_spectral.py', 'reproducibility_v3.py',
    'original/periodic_notch_starter/demo.py', 'requirements-lock.txt',
    'configs/v3_p0_coupled.json', 'configs/refined_risk.json',
    'refinement_protocol.json', 'data/v3_manifest.json')


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
                          if isinstance(value, (dict, list)) else value for key, value in row.items()}
                         for row in rows)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def plan():
    if PLAN.exists():
        raise FileExistsError('Protocol already exists; preserve its selection rather than silently reselecting.')
    manifest = read(DATA / 'v3_manifest.json')
    used = {row['id'] for row in manifest['images']}
    pool = sorted(path for path in (DATA / 'source/images/test').glob('*.jpg') if path.stem not in used)
    if len(pool) != 20:
        raise ValueError('Restore the official BSDS images first; expected 20 unused test images.')
    chosen = sorted(np.random.default_rng(2026111604).choice(len(pool), 4, replace=False))
    check = [dict(id=pool[i].stem, source_jpg=pool[i].relative_to(DATA).as_posix(),
                  source_sha256=sha256_file(pool[i])) for i in chosen]
    dev = [row for row in manifest['images'] if row['split'] == 'dev'][:8]
    audit = [row for row in manifest['images'] if row['split'] == 'audit_v3']
    dump(PLAN, dict(schema_version=1, seed=2026111604, development_ids=[r['id'] for r in dev],
        independent_check=check, independent_check_selection='4 of 20 unused official test IDs; seeded filenames only, before pixel loading',
        regression_texture_ids=[r['id'] for r in manifest['textures']],
        regression_edge_ids=[r['id'] for r in audit[:4]], diagnostic_global_id='10081',
        kinds=list(KINDS), amplitude=.12, strengths=list(STRENGTHS), methods=list(METHODS),
        baseline='P0 frequency-grid fix, legacy risk, coupled strength=2',
        main_change='input-side periodic ambiguity and removed-structure risk; candidate residuals unchanged',
        support_extension='deferred: this study selects risk scoring as the one main algorithm change',
        selection_rule='fixed coarse feature settings; no audit/DTD tuning; retain coupled baseline CLI default unless development and independent evidence support replacement',
        paired_quality='same image/condition/strength grid; same residual for all risk modes; no extrapolation in matched-quality comparisons',
        historical_data_status='old audit60 and DTD12 are regression/diagnostic material after review; not an untouched confirmation set',
        oracle_note='local preclip/pre-overlap weighted oracle gain is evaluation-only; not the globally optimal restoration',
        preprocessing='image_io_v3.preprocess_image: size=256, short-side Lanczos resize, center crop, shared floating luminance, no contrast normalization',
        source_manifest_sha256=sha256_file(DATA / 'v3_manifest.json')))
    print(PLAN)


def configs():
    baseline = config_from_dict(read(ROOT / 'configs/v3_p0_coupled.json')['configuration'])
    experiment = config_from_dict(read(ROOT / 'configs/refined_risk.json')['configuration'])
    if replace(experiment, guard=baseline.guard) != baseline:
        raise ValueError('The risk-only experiment must not change the estimator, strength or combination.')
    return dict(coupled_baseline=baseline, decoupled_control=replace(baseline, combination='decoupled'),
                structure_risk=experiment)


def verify_frozen():
    frozen = read(FREEZE)
    protocol = read(PLAN)
    verify_snapshot(frozen['snapshot'], paths=SOURCES,
        audit_ids=[row['id'] for row in protocol['independent_check']], base_dir=ROOT)
    if sha256_file(OUT / 'development/metrics.csv') != frozen['development_metrics_sha256']:
        raise ValueError('Development metrics changed after freezing')
    if frozen['configurations'] != frozen['snapshot']['metadata']['configurations']:
        raise ValueError('Focused configurations differ from sealed snapshot metadata')
    for method, cfg in configs().items():
        if config_dict(cfg) != frozen['configurations'][method]:
            raise ValueError('Configuration differs from the frozen focused study')
    return frozen


def freeze():
    if FREEZE.exists():
        raise FileExistsError('Focused study already frozen; use another output directory for a new study.')
    protocol = read(PLAN)
    configurations = {name: config_dict(cfg) for name, cfg in configs().items()}
    if not (OUT / 'development/metrics.csv').is_file():
        raise FileNotFoundError('Run development diagnostics before freezing.')
    dump(FREEZE, dict(configurations=configurations, recommendation='retain coupled baseline default; structure risk remains opt-in',
        development_metrics_sha256=sha256_file(OUT / 'development/metrics.csv'),
        snapshot=capture_snapshot(SOURCES, [r['id'] for r in protocol['independent_check']], base_dir=ROOT,
            metadata=dict(configurations=configurations, study='focused input-side risk scoring'))))
    print(FREEZE)


def samples(stage):
    protocol = read(PLAN); manifest = read(DATA / 'v3_manifest.json')
    if sha256_file(DATA / 'v3_manifest.json') != protocol['source_manifest_sha256']:
        raise ValueError('Input manifest differs from the focused study protocol')
    images = {r['id']: r for r in manifest['images'] + manifest['textures']}

    def load_reference(record):
        path = DATA / record['npy_path']
        if sha256_file(path) != record['clean_sha256']:
            raise ValueError(f"Reference array changed: {record['id']}")
        return np.load(path)
    if stage == 'development':
        records = [images[i] for i in protocol['development_ids']]
    elif stage == 'check':
        records = []
        for record in protocol['independent_check']:
            source = DATA / record['source_jpg']
            if sha256_file(source) != record['source_sha256']:
                raise ValueError('Independent-check source changed')
            records.append(record)
    else:
        records = [images[i] for i in protocol['regression_texture_ids']]
    for index, record in enumerate(records):
        if stage == 'check':
            clean, preprocessing = preprocess_image(DATA / record['source_jpg'], size=256)
        else:
            clean = load_reference(record)
            preprocessing = dict(reference_npy=record['npy_path'], reference_sha256=record['clean_sha256'])
        kinds = ('clean', 'global', 'chirp') if stage == 'regression' else KINDS
        for kind in kinds:
            variant = index % 4 if kind == 'edge_strip' else None
            yield record['id'], clean, kind, variant, preprocessing
    if stage == 'regression':
        for index, image_id in enumerate(protocol['regression_edge_ids']):
            record = images[image_id]
            yield image_id, load_reference(record), 'edge_strip', index % 4, dict(reference_npy=record['npy_path'], reference_sha256=record['clean_sha256'])
        image_id = protocol['diagnostic_global_id']
        record = images[image_id]
        yield image_id, load_reference(record), 'global', None, dict(reference_npy=record['npy_path'], reference_sha256=record['clean_sha256'])


def run(stage):
    if stage != 'development':
        verify_frozen()
    destination = OUT / stage
    if (destination / 'metrics.csv').exists():
        raise FileExistsError(f'{stage} results already exist; no silent overwrite/repeated check.')
    source_hashes = {path: sha256_file(ROOT / path) for path in SOURCES}
    protocol = read(PLAN); cfgs = configs(); rows = []; gains = []; inputs = []
    for index, (image_id, clean, kind, variant, preprocessing) in enumerate(samples(stage)):
        amplitude = 0. if kind == 'clean' else protocol['amplitude']
        observed, meta, masks = make_sample(clean, image_id, kind, amplitude, variant=variant)
        identity = dict(stage=stage, image_id=image_id, kind=kind, amplitude=amplitude)
        input_hash = hashlib.sha256(observed.tobytes()).hexdigest()
        inputs.append(identity | dict(shape=list(clean.shape), float_input_sha256=input_hash,
                                      preprocessing=preprocessing, synthesis=meta))
        rows.append(identity | dict(method='input', strength=0., runtime_ms=0.) | metric_row(clean, observed, observed, masks))
        start = time.perf_counter()
        estimated = estimate(observed, cfgs['coupled_baseline'].estimation)
        estimate_ms = (time.perf_counter() - start) * 1000
        # Reuse identical residuals: risk changes cannot improve candidate recall.
        for method, cfg in cfgs.items():
            start = time.perf_counter()
            q, s, global_diag, decisions = guard_weights(observed, estimated.tiles, cfg.guard)
            prepared = Prepared(cfg, estimated, q, s, global_diag, decisions)
            guard_ms = (time.perf_counter() - start) * 1000
            for strength in STRENGTHS:
                start = time.perf_counter()
                output, diag = restore(prepared, strength=strength, return_diagnostics=True)
                elapsed = (time.perf_counter() - start) * 1000 + estimate_ms + guard_ms
                rows.append(identity | dict(method=method, strength=strength, runtime_ms=elapsed) |
                    metric_row(clean, observed, output, masks) | estimation_metrics(diag, masks))
            gains.extend(identity | dict(method=method) | row for row in
                candidate_gain_rows(observed, clean, prepared, strength=2.))
        print(f'{stage}: {index+1} {image_id} {kind}', flush=True)
    if source_hashes != {path: sha256_file(ROOT / path) for path in SOURCES}:
        raise RuntimeError('Study sources changed during execution; rerun this stage with stable sources.')
    write_csv(destination / 'metrics.csv', rows)
    write_csv(destination / 'candidate_gains.csv', gains)
    dump(destination / 'gain_calibration.json', {method: summarize_gain_rows(
        [row for row in gains if row['method'] == method]) for method in METHODS})
    dump(destination / 'gain_by_condition.json', {f'{method}/{kind}': summarize_gain_rows(
        [row for row in gains if row['method'] == method and row['kind'] == kind])
        for method in METHODS for kind in sorted({row['kind'] for row in gains})})
    dump(destination / 'inputs.json', inputs)
    dump(destination / 'run.json', dict(stage=stage, n_inputs=len(inputs), n_metric_rows=len(rows),
        source_hashes=source_hashes,
        note='single-machine timings; estimation reused across risk ablations; reference-free inference, postclip/OLA final metrics'))


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row['method'] != 'input' and float(row['strength']) != 2.:
            continue
        grouped[(row['kind'], row['method'])].append(row)
    result = []
    for (kind, method), values in sorted(grouped.items()):
        record = dict(kind=kind, method=method, n=len(values))
        for field in ('mse', 'input_mse', 'mae', 'psnr_db', 'ssim', 'contaminated_mse', 'clean_mse',
                'clean_modification_mae', 'correct_candidate_coverage', 'processed_block_coverage', 'runtime_ms'):
            numbers = [float(row[field]) for row in values if row.get(field, '') != '']
            record[field] = float(np.mean(numbers)) if numbers else ''
        record['degraded_fraction'] = float(np.mean([int(row['degraded']) for row in values]))
        record['tail20_mae'] = tail_mean([float(row['mae']) for row in values])
        record['mean_mse_change'] = record['mse'] - record['input_mse']
        result.append(record)
    return result


def matched_quality(rows):
    """Dirty-region recovery versus clean-region damage, same-image interpolation.

    Only common sampled recovery ranges are used. The point is diagnostic,
    not another sample and not a guarantee of monotonic response.
    """
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row['kind'] not in ('local', 'edge_strip', 'irregular') or row['method'] == 'input':
            continue
        groups[(row['image_id'], row['kind'])][row['method']].append(row)
    records = []
    for (image_id, kind), methods in groups.items():
        curves = []
        invalid_reason = None
        for method in ('coupled_baseline', 'structure_risk'):
            points = [(float(r['contaminated_mse']), float(r['clean_mse'])) for r in sorted(methods[method], key=lambda r: float(r['strength']))
                            if r.get('contaminated_mse', '') != '' and r.get('clean_mse', '') != '']
            difference = np.diff([p[0] for p in points])
            if np.any(difference > 1e-12) and np.any(difference < -1e-12):
                invalid_reason = 'nonmonotonic recovery curve; ambiguous strength branch'
            if np.any(np.abs(difference) <= 1e-12):
                invalid_reason = 'duplicate recovery level; no unique interpolated damage'
            curves.append(sorted(points))
        a, b = curves
        if invalid_reason:
            records.append(dict(image_id=image_id, kind=kind, matched=False, reason=invalid_reason))
            continue
        if not a or not b:
            records.append(dict(image_id=image_id, kind=kind, matched=False, reason='no dirty/clean pixels'))
            continue
        low, high = max(a[0][0], b[0][0]), min(a[-1][0], b[-1][0])
        if high - low <= 1e-12:
            records.append(dict(image_id=image_id, kind=kind, matched=False, reason='no nondegenerate common quality interval'))
            continue
        target = (low + high) / 2
        da = float(np.interp(target, [p[0] for p in a], [p[1] for p in a]))
        db = float(np.interp(target, [p[0] for p in b], [p[1] for p in b]))
        records.append(dict(image_id=image_id, kind=kind, matched=True, target_dirty_mse=target,
            baseline_clean_mse=da, structure_clean_mse=db, clean_mse_difference=db-da))
    return records


def report():
    lines = ['# 风险评分聚焦验证', '',
        '本轮先修离散像素频率范围、默认配置和数据测试边界，以风险评分为唯一主要算法改动。未增加高阶相位、前2候选重排或空间包络自由度。', '',
        '三种决策复用完全相同的候选残差：P0修复后耦合基线、解耦机制对照、输入周期歧义与移除结构风险。所有表为强度2；曲线另存每图原始记录。', '',
        '局部参考增益只在离线评价计算，忽略裁剪与邻块融合，不能当作全图最优解。得分分箱仅诊断排序，非概率校准。', '']
    for stage in ('development', 'regression', 'check'):
        path = OUT / stage / 'metrics.csv'
        if not path.exists():
            continue
        rows = read_csv(path); summary = aggregate(rows)
        write_csv(OUT / stage / 'summary.csv', summary)
        matched = matched_quality(rows)
        if matched:
            write_csv(OUT / stage / 'matched_quality.csv', matched)
        lines += [f'## {stage}', '', '| 条件 | 方法 | 图数 | 平均PSNR dB | MSE | 比输入更差比例 | 最差20% MAE |',
            '|---|---|---:|---:|---:|---:|---:|']
        for row in summary:
            lines.append(f"| {row['kind']} | {row['method']} | {row['n']} | {row['psnr_db']:.3f} | {row['mse']:.7f} | {row['degraded_fraction']:.1%} | {row['tail20_mae']:.6f} |")
        valid = [r for r in matched if r['matched']]
        lines += ['', f'局部条件相近恢复质量：{len(valid)}/{len(matched)} 对具有非退化共同区间；' +
            (f"输入结构风险相对基线干净区MSE差均值 {np.mean([r['clean_mse_difference'] for r in valid]):.8g}。" if valid else '无可匹配对。'), '']
    lines += ['## 取舍与边界', '',
        '保留P0耦合版为CLI默认，结构风险通过显式配置启用。算法拟合残差不变，因此候选召回及窄条支撑不足不会由风险评分修复。', '',
        '开发8图用于诊断，旧审计/DTD用于回归；独立检查只有4张未使用BSDS图，不能推广为全面性能结论。未根据回归或独立检查再调参数。', '',
        '全量旧v3实验未重跑。旧3900条主审计保持原样，精确源码位于v3-validated-baseline标签；当前修订不得沿用旧源码快照冒充同一实验。', '']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print(OUT / 'RESULTS.md')


def main():
    global OUT, FREEZE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'development', 'freeze', 'regression', 'check', 'report'))
    parser.add_argument('--out', type=Path, default=OUT, help='Use a fresh directory for another reproducibility run')
    args = parser.parse_args()
    OUT = args.out.resolve(); FREEZE = OUT / 'freeze.json'
    if args.stage in ('development', 'regression', 'check'):
        run(args.stage)
    else:
        globals()[args.stage]()


if __name__ == '__main__':
    main()
