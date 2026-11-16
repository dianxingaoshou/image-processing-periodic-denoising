"""Generate a Chinese v3 code-validation note from completed frozen artifacts.

This script does not run denoisers or choose parameters. It refuses incomplete
main/stress/texture inventories and requires the final verification to pass.
Unprovided metrics are shown as missing, never converted into zero damage.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

METHODS = ('input', 'v2_frozen', 'v3_coupled_P0', 'v3_decoupled', 'v3_unprotected')
LABELS = dict(input='含噪输入', v2_frozen='冻结v2', v3_coupled_P0='P0修复后耦合权重',
              v3_decoupled='P0修复后解耦权重', v3_unprotected='P0修复后关闭保护')
KIND_NAMES = dict(global_='全局定频', local='局部定频', chirp='频率渐变', orientation='方向变化',
                  clean='未另加干扰', cubic='三次相位', random_phase='平滑随机相位',
                  corner='角落局部干扰', edge_strip='边缘窄条干扰', irregular='不规则区域干扰')
KIND_NAMES['global'] = KIND_NAMES.pop('global_')
PRIMARY_KINDS = ('global', 'local', 'chirp', 'orientation')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path, required=True):
    path = Path(path)
    if not path.exists() and not required:
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def numeric(row, *keys, default=None):
    for key in keys:
        value = row.get(key)
        if value is None or value == '':
            continue
        try:
            return float(value)
        except (ValueError, TypeError):
            continue
    return default


def fmt(value, digits=3, scientific=False):
    if value is None:
        return '未提供'
    if math.isinf(value):
        return '∞'
    if not math.isfinite(value):
        return '未提供'
    return f'{value:.{digits}e}' if scientific else f'{value:.{digits}f}'


def values(rows, key):
    result = [numeric(row, key) for row in rows]
    return np.asarray([x for x in result if x is not None and not math.isnan(x)], dtype=float)


def mean(rows, key):
    a = values(rows, key)
    return float(a.mean()) if len(a) else None


def tail20(rows, key='mae'):
    a = values(rows, key)
    return float(np.sort(a)[-max(1, math.ceil(.2 * len(a))):].mean()) if len(a) else None


def metric_quantile(rows, q, key='mae'):
    a = values(rows, key)
    return float(np.quantile(a, q)) if len(a) else None


def group(rows):
    result = defaultdict(list)
    for row in rows:
        result[(row['kind'], row['method'])].append(row)
    return result


def table(lines, headers, rows):
    lines += ['', '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(str(x).replace('|', '\\|').replace('\n', ' ') for x in row) + ' |')
    lines.append('')


def inventory(root, stage, expected):
    rows = read_csv(root / stage / 'metrics.csv')
    keys = {(r['image_id'], r['kind'], r['amplitude'], r['method']) for r in rows}
    if len(rows) != expected or len(keys) != expected:
        raise RuntimeError(f'{stage}: expected {expected} complete unique rows, got {len(rows)}/{len(keys)}')
    if set(r['method'] for r in rows) != set(METHODS) or any(r.get('error') for r in rows):
        raise RuntimeError(f'{stage}: methods incomplete or algorithm exceptions present')
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def paired_comparisons(rows):
    lookup = {(r['image_id'], r['kind'], r['amplitude'], r['method']): r for r in rows}
    rng = np.random.default_rng(2026111603)
    result = []
    for kind in PRIMARY_KINDS:
        for baseline in ('v2_frozen', 'v3_coupled_P0', 'v3_unprotected'):
            differences = defaultdict(list)
            for (image_id, k, amplitude, method), row in lookup.items():
                if k != kind or method != 'v3_decoupled':
                    continue
                reference = lookup.get((image_id, k, amplitude, baseline))
                if reference is not None:
                    differences[image_id].append(float(row['psnr_db']) - float(reference['psnr_db']))
            d = np.asarray([np.mean(v) for v in differences.values()])
            if not len(d):
                continue
            resampled = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
            low, high = np.quantile(resampled, [.025, .975])
            result.append(dict(kind=kind, baseline=baseline, n_images=len(d),
                               mean_delta_psnr=float(d.mean()), image_bootstrap_low=float(low),
                               image_bootstrap_high=float(high)))
    return result


def pair_interpretation(row):
    lo, hi = row['image_bootstrap_low'], row['image_bootstrap_high']
    if lo > 0:
        return '当前样本支持解耦版PSNR更高'
    if hi < 0:
        return '当前样本支持解耦版PSNR更低'
    if lo == hi == 0:
        return '当前样本未观测到差异'
    return '区间含0，方向尚不确定'


def flatten_numbers(value, prefix=''):
    output = []
    if isinstance(value, dict):
        for key, child in value.items():
            output += flatten_numbers(child, f'{prefix}.{key}' if prefix else str(key))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output.append((prefix, float(value)))
    return output


def append_p0(lines):
    lines += ['## 正确性与复现边界修复', '',
              '冻结v2源码和原结果保留。v3中会改变候选估计和输出的数值修复，与只影响调用或复现核验的修复分开记录。']
    table(lines, ['修复', '实际变化', '对旧结果的含义'], [
        ['取消固定隔点拟合', '候选初筛和相位细化使用全部像素；0.25及附近频率、多方向和相位进入边界测试',
         '会改变候选排序、振幅、相位及复原。旧低频实验没有因此自动失效；v2结果保持历史冻结记录'],
        ['按实际分析窗口去重', '候选记录analysis_window_id；有效位置与尺寸相同的请求尺度不再构成额外支撑',
         '会影响小图和矩形图的证据与权重；不同窗口仍有重叠，称为窗口一致性，不称统计独立样本'],
        ['共享浮点灰度', '实验和CLI统一使用0.2125R+0.7154G+0.0721B；RGBA明确背景合成',
         '原实验RGB路径可逐位复现；旧CLI的PIL灰度路径会与新CLI不同，不混称算法收益'],
        ['配置缺失立即报错', '只有显式--use-default-config允许未校准默认值；输出配置来源及SHA256',
         '避免缺文件时静默更换运行点，本身不是去噪算法改进'],
        ['结构种子与幅度分离', '新入口结构种子不含幅度，三档逐项核对频率、相位、包络',
         '已核实现行v2 make_sample本来就配对；未使用的历史sample_seed接口差异不证明v2结果有误'],
        ['扩大冻结范围', '纳入生成器、指标、基线、底层帮助函数、v3代码、协议、清单与依赖分发文件；核对审计ID准确顺序',
         '防止源代码、数值环境、数据划分或审计顺序静默改变，不改变合法的既有像素输入'],
    ])


def append_configuration(lines, frozen, raw):
    cfg = frozen.get('configuration', frozen.get('config', {}))
    calibration = frozen.get('calibration', {})
    strength = numeric(cfg, 'strength', default=numeric(calibration, 'selected_strength'))
    est = cfg.get('estimation', {})
    lines += ['## 权重解耦与开发集校准', '',
              '拟合因子q包含可靠性、幅度与外推约束；安全因子s包含纹理和歧义保护；候选不具减除资格时增益为0。'
              '在相同候选残差和评分下比较以下组合：', '',
              '- 耦合：`g = min(alpha * q * s, 1)`。',
              '- 解耦：`g = min(alpha * q, 1) * s`。', '',
              '解耦使最终增益不超过s，增大拟合强度不能取消安全上限。这是机制保证，'
              '并不保证s正确识别了所有真实纹理；去噪收益仍需按相近去噪量比较。', '',
              f'冻结配置的alpha为{fmt(strength, 2)}。参数只使用原30张开发图；风险阈值来自冻结v2在开发集的基准，'
              '不是看过本轮审计或DTD结果后放宽。尾部指标为误改最严重20%图像的平均MAE；30张开发图对应最差6张，'
              '比只依赖一个P95点更直接反映尾部，但仍不是总体风险保证。']
    if strength is not None and strength == 0:
        lines += ['', '**本次风险约束只允许选择零强度。该运行点等价于不处理，不能据此声称实现了有效去噪。**', '']
    criterion = calibration.get('criterion')
    if criterion:
        lines += ['', '配置记录的校准准则：' + str(criterion), '']
    bounds = flatten_numbers(calibration.get('bounds', {}))
    if bounds:
        table(lines, ['开发集约束字段', '冻结阈值'], [[key, fmt(value, scientific=True)] for key, value in bounds])
    modes = []
    method_strengths = {}
    for method in METHODS:
        selected = [r for r in raw if r['method'] == method]
        alphas = sorted(set(float(r['strength']) for r in selected if r.get('strength') not in (None, '')))
        method_strengths[method] = alphas
        combinations = sorted(set(r['combination'] for r in selected if r.get('combination')))
        modes.append([LABELS[method], '、'.join(fmt(a, 2) for a in alphas) or '未记录', '、'.join(combinations) or '不适用或未记录'])
    table(lines, ['方法', '实际alpha', '记录的组合'], modes)
    coupled = method_strengths['v3_coupled_P0']; decoupled = method_strengths['v3_decoupled']
    if coupled and decoupled and coupled == decoupled:
        lines += ['耦合P0与解耦版使用相同alpha，主表可更直接检验组合顺序；关闭保护也保留相同估计器残差。'
                  '与冻结v2的差异还包含数值修复，若alpha不同也包含运行点差异。', '']
    else:
        lines += ['耦合P0与解耦版的alpha记录不同或不完整，不能将主表差异全部归因于权重组合顺序；'
                  '应优先看相近去噪量曲线。', '']
    lines += [f'候选精细拟合数量为{est.get("refine_candidates", "未提供")}，重叠残差重排序开关为'
              f'{est.get("use_overlap_reranking", "未提供")}。P2探索与默认版本分开解释。', '']


def append_primary(lines, raw, pairs):
    lookup = group(raw)
    lines += ['## 新60图审计结果', '',
              '主审计每图包含四类退化、三档幅度和一项未加干扰对照，共780个输入；五种方法共3900条记录。'
              '冻结v2在同一批输入重新运行，旧审计图只作回归资料。', '',
              'PSNR为逐样本dB均值；MSE、MAE、SSIM分别平均，不能由平均PSNR反推出平均MSE。'
              '不同条件总干扰能量不同，应在同一条件内比较。']
    table(lines, ['条件', '方法', '平均PSNR dB', '平均MSE', '平均MAE', '平均SSIM'], [
        [KIND_NAMES[kind], LABELS[method], fmt(mean(lookup[(kind, method)], 'psnr_db'), 2),
         fmt(mean(lookup[(kind, method)], 'mse'), scientific=True), fmt(mean(lookup[(kind, method)], 'mae'), scientific=True),
         fmt(mean(lookup[(kind, method)], 'ssim'), 4)]
        for kind in PRIMARY_KINDS for method in METHODS])
    lines += ['### 解耦版的配对变化', '',
              '以下差值为解耦版减对照。三档强度先在同一原图内平均，再以原图为单位进行2000次固定种子自举，'
              '避免把同图的退化版本当作独立受试者。区间只描述本次样本证据。']
    table(lines, ['条件', '对照', '配对PSNR差 dB', '按原图自举95%区间', '当前解释'], [
        [KIND_NAMES[r['kind']], LABELS[r['baseline']], fmt(r['mean_delta_psnr']),
         f'[{fmt(r["image_bootstrap_low"])}, {fmt(r["image_bootstrap_high"])}]', pair_interpretation(r)] for r in pairs])
    lines += ['### 干净对照尾部与局部干净区域', '',
              '本表中的MAE直接量化额外修改。尾部20%在60张对照中为最差12张；P95和最大值一起保留，'
              '不以平均PSNR中的∞掩盖少量严重误改。']
    table(lines, ['方法', '对照MAE均值', '最差20%平均MAE', 'MAE P95', 'MAE最大值', '修改图像数'], [
        [LABELS[m], fmt(mean(lookup[('clean', m)], 'mae'), scientific=True), fmt(tail20(lookup[('clean', m)]), scientific=True),
         fmt(metric_quantile(lookup[('clean', m)], .95), scientific=True), fmt(metric_quantile(lookup[('clean', m)], 1.), scientific=True),
         f'{int(np.sum(values(lookup[("clean", m)], "mae") > 1e-12))}/{len(lookup[("clean", m)])}'] for m in METHODS])
    table(lines, ['方法', '局部干净区MSE', '局部干净区修改MAE'], [
        [LABELS[m], fmt(mean(lookup[('local', m)], 'clean_mse'), scientific=True),
         fmt(mean(lookup[('local', m)], 'clean_modification_mae'), scientific=True)] for m in METHODS])
    old = mean(lookup[('clean', 'v3_coupled_P0')], 'mae'); new = mean(lookup[('clean', 'v3_decoupled')], 'mae')
    old_tail = tail20(lookup[('clean', 'v3_coupled_P0')]); new_tail = tail20(lookup[('clean', 'v3_decoupled')])
    if all(x is not None for x in (old, new, old_tail, new_tail)):
        direction = '下降' if new_tail < old_tail else '上升' if new_tail > old_tail else '不变'
        lines += [f'相对P0耦合版，解耦版对照平均MAE为{fmt(old,scientific=True)}→{fmt(new,scientific=True)}，'
                  f'尾部20%均值为{fmt(old_tail,scientific=True)}→{fmt(new_tail,scientific=True)}（{direction}）。'
                  '这个观察必须与去噪量一起解释，不能仅凭改动较少判断保护成功。', '']
    risky_rows = []
    for method in METHODS[1:]:
        selected = [r for r in raw if r['method'] == method]
        risky = values(selected, 'risky_full_blocks'); eligible = values(selected, 'eligible_blocks')
        totals = values(selected, 'total_blocks')
        risky_rows.append([LABELS[method], len(risky), int(risky.sum()) if len(risky) else '未记录',
                           int(eligible.sum()) if len(eligible) else '未记录', int(totals.sum()) if len(totals) else '未记录'])
    table(lines, ['方法', '有诊断的样本数', '有风险但满额减除的块次', '有资格减除的块次', '总块次'], risky_rows)
    lines += ['块次对所有退化样本累计，重复出现的空间块不是独立样本。风险定义来自模型安全权重小于1，'
              '不等于已知真实误删标签；解耦式在数值容差外应消除“有安全限制仍满额减除”的机制。', '']
    table(lines, ['条件', '方法', '质量退化比例', '平均耗时 ms'], [
        [KIND_NAMES[k], LABELS[m], fmt(100 * mean(lookup[(k, m)], 'degraded'), 1) + '%' if mean(lookup[(k, m)], 'degraded') is not None else '未提供',
         fmt(mean(lookup[(k, m)], 'runtime_ms'), 1)] for k in PRIMARY_KINDS for m in METHODS[1:]])
    lines += ['质量退化沿用输出MSE高于输入MSE加1e-8的定义，与程序异常分开。时间为本机单次墙钟时间，'
              '不同阶段可能共享机器资源，不作严格速度排名。', '']


def append_frequency(lines, raw):
    lookup = group(raw)
    lines += ['## 估计阶段的频率与覆盖评价', '',
              '在实际合成块中心，按共轭等价误差 `min(||fhat-f||, ||fhat+f||)` 比较频率，单位为周期/像素。'
              '重叠平均的频率显示图只是块中心频率的可视化，不是完整二次相位频率场预测。', '',
              '候选覆盖率检查是否有候选池，正确候选覆盖率检查池内是否至少一个频率误差不超过0.01周期/像素。'
              '有效选定覆盖率排除被估计器拒绝或幅度为零的模型；这些覆盖率和处理覆盖率均以所有含噪块中心为分母，'
              '包括无候选或被拒绝的块。频率误差均值只在有效选定模型时定义，所以同时报告其误差块数。'
              '实际处理还要求最终增益至少0.05；'
              '干净区误处理率以所有干净块中心为分母，不将空区域视为0误处理。']
    table(lines, ['条件', '方法', '选定频率误差均值', '频率误差块数', '候选覆盖率', '正确候选覆盖率', '有效选定覆盖率', '实际处理覆盖率', '干净区误处理率'], [
        [KIND_NAMES[k], LABELS[m], fmt(mean(lookup[(k, m)], 'frequency_error_mean'), scientific=True),
         int(values(lookup[(k, m)], 'frequency_error_count').sum()) if len(values(lookup[(k, m)], 'frequency_error_count')) else '未提供',
         fmt(mean(lookup[(k, m)], 'candidate_block_coverage'), 3), fmt(mean(lookup[(k, m)], 'correct_candidate_coverage'), 3),
         fmt(mean(lookup[(k, m)], 'selected_block_coverage'), 3),
         fmt(mean(lookup[(k, m)], 'processed_block_coverage'), 3), fmt(mean(lookup[(k, m)], 'clean_block_processing_rate'), 3)]
        for k in PRIMARY_KINDS for m in METHODS[1:]])
    lines += ['表中比例先在各样本计算再平均，频率误差也按存在有效选定模型的样本均值平均；不将覆盖率不足的结果'
              '解释为全图估计准确。接受候选仍不能证明它必然是噪声。', '']


def append_stress(lines, rows):
    lookup = group(rows)
    order = ('cubic', 'random_phase', 'corner', 'edge_strip', 'irregular')
    kinds = [k for k in order if any(r['kind'] == k for r in rows)]
    kinds += sorted(set(r['kind'] for r in rows) - set(kinds))
    lines += ['## 模型之外与偏置位置验证 仅20图', '',
              '该组使用事先固定的20张新审计图、五类中档干扰，共100个输入和500条方法记录。'
              '三次相位、平滑随机相位不与估计器的二次相位模型完全一致；角落、边缘窄条和不规则区域'
              '检验位置偏置、外推与重叠融合。它们不是完整60图三档强度结论。']
    table(lines, ['条件', '方法', '平均PSNR dB', '平均MSE', '平均MAE', '干净区MSE', '频率误差均值'], [
        [KIND_NAMES.get(k, k), LABELS[m], fmt(mean(lookup[(k, m)], 'psnr_db'), 2), fmt(mean(lookup[(k, m)], 'mse'), scientific=True),
         fmt(mean(lookup[(k, m)], 'mae'), scientific=True), fmt(mean(lookup[(k, m)], 'clean_mse'), scientific=True),
         fmt(mean(lookup[(k, m)], 'frequency_error_mean'), scientific=True)] for k in kinds for m in METHODS])
    lines += ['位置变化时不同掩膜的干扰能量可能不同，应在相同输入上比较方法。非二次相位上的残留或失败'
              '用于界定模型适用范围，不通过逐图调参隐藏。', '']


def append_textures(lines, rows):
    lookup = group(rows)
    lines += ['## 自然纹理保真与叠加干扰配对', '',
              'DTD使用固定12张自然纹理照片，每张包含原始未另加干扰、叠加全局定频、叠加频率渐变三种输入，'
              '五种方法共180条结果。原始照片不是严格物理意义上的无噪声图；这里评价的是是否额外修改其已有内容。'
              'DTD不参与阈值或强度选择。']
    clean_kind = 'clean' if ('clean', 'input') in lookup else next((k for k, _ in lookup if k in ('original', 'texture_clean')), 'clean')
    table(lines, ['方法', '未加干扰MAE均值', '最差20%平均MAE', 'MAE P95', 'MAE最大值', '修改图像数'], [
        [LABELS[m], fmt(mean(lookup[(clean_kind, m)], 'mae'), scientific=True), fmt(tail20(lookup[(clean_kind, m)]), scientific=True),
         fmt(metric_quantile(lookup[(clean_kind, m)], .95), scientific=True), fmt(metric_quantile(lookup[(clean_kind, m)], 1.), scientific=True),
         f'{int(np.sum(values(lookup[(clean_kind, m)], "mae")>1e-12))}/{len(lookup[(clean_kind, m)])}'] for m in METHODS])
    lines += ['12张照片的尾部20%对应最差3张，样本较少，不能把该量当作稳定的总体上界。'
              '叠加干扰版本使用同一原始照片作参考，下面同时检查是否真正去噪，避免把全部拒绝处理解释为纹理保护。']
    noisy_kinds = [k for k in ('global', 'chirp') if (k, 'input') in lookup]
    table(lines, ['叠加条件', '方法', '平均PSNR dB', '平均MSE', '平均MAE', '平均SSIM'], [
        [KIND_NAMES[k], LABELS[m], fmt(mean(lookup[(k, m)], 'psnr_db'), 2), fmt(mean(lookup[(k, m)], 'mse'), scientific=True),
         fmt(mean(lookup[(k, m)], 'mae'), scientific=True), fmt(mean(lookup[(k, m)], 'ssim'), 4)] for k in noisy_kinds for m in METHODS])
    details = []
    for method in METHODS[1:]:
        candidates = lookup[(clean_kind, method)]
        if candidates:
            worst = max(candidates, key=lambda r: float(r['mae']))
            details.append([LABELS[method], worst['image_id'], fmt(float(worst['mae']), scientific=True)])
    table(lines, ['方法', '纹理误改最大案例ID', '修改MAE'], details)
    coupled = tail20(lookup[(clean_kind, 'v3_coupled_P0')]); decoupled = tail20(lookup[(clean_kind, 'v3_decoupled')])
    if coupled is not None and decoupled is not None:
        direction = '下降' if decoupled < coupled else '上升' if decoupled > coupled else '不变'
        lines += [f'本DTD子集的解耦相对P0耦合尾部修改为{fmt(coupled,scientific=True)}→'
                  f'{fmt(decoupled,scientific=True)}（{direction}）。仍需与叠噪恢复和严重单图失败一起判断，'
                  '不能把合成规则图案的保留推广为通用自然纹理识别。', '']


def append_matching(lines, root):
    folder = root / 'supplement'
    curves = read_csv(folder / 'weight_curves.csv')
    matched = read_csv(folder / 'matched_removal.csv', required=False)
    coverage = read_csv(folder / 'matched_coverage.csv')
    modes = sorted(set(r.get('mode', r.get('combination', '')) for r in curves))
    lines += ['## 相近去噪量下比较安全保护', '',
              '该组仅使用固定前10张新审计图的局部中档退化，比较耦合、解耦和关闭保护，复用相同候选残差。'
              '匹配轴是含噪区域MSE降低比例，不是相同strength；只在采样Pareto曲线共同可达区间插值，'
              '不从曲线重新选择测试运行参数。', '', f'保存的曲线模式为：{"、".join(modes)}；曲线记录{len(curves)}条。']
    cover_headers = ['image_id', 'matched_points', 'reason']
    additional = sorted({k for r in coverage for k in r if k not in cover_headers})
    table(lines, ['图像ID', '匹配点数', '覆盖说明', *additional], [
        [r.get('image_id', '未提供'), r.get('matched_points', '未提供'), r.get('reason', '未提供'),
         *[r.get(k, '') for k in additional]] for r in coverage])
    if not matched:
        lines += ['没有保存可匹配的共同区间。本轮不据此宣称在相同去噪量下改善保护。', '']
        return
    fraction_keys = ('matched_contaminated_mse_reduction_fraction', 'matched_reduction_fraction', 'removal_fraction')
    positive = [r for r in matched if (numeric(r, *fraction_keys, default=0.) or 0.) > 1e-12]
    individual = [r for r in positive if r.get('group', 'individual') in ('individual', 'per_image')]
    ids = {r.get('image_id') for r in individual if r.get('image_id')}
    lines += [f'具有正去噪共同区间并进入个体匹配的图像ID数为{len(ids)}；覆盖表保留全部未匹配案例。'
              '合并均值曲线不能替代个体覆盖，多个插值点也不是独立实验样本。', '']
    pooled = [r for r in positive if r.get('group') in ('subset_mean', 'pooled', 'mean')]
    comparison = []
    for label, selected in (('个体匹配点', individual), ('10图均值曲线匹配点', pooled)):
        if not selected:
            continue
        q = [numeric(r, 'coupled_clean_mse') for r in selected]
        s = [numeric(r, 'decoupled_clean_mse') for r in selected]
        u = [numeric(r, 'unprotected_clean_mse', 'without_conservative_clean_mse') for r in selected]
        for r, a, b, c in zip(selected, q, s, u):
            comparison.append([label, r.get('image_id', '均值曲线'), fmt(numeric(r, *fraction_keys), 4),
                               fmt(a, scientific=True), fmt(b, scientific=True), fmt(c, scientific=True)])
    # Show finite samples rather than an unjustified aggregate over interpolated points.
    if comparison:
        index = sorted(set([0, len(comparison) // 2, len(comparison) - 1]))
        table(lines, ['组别', '案例', '相同含噪MSE降低比例', '耦合干净区MSE', '解耦干净区MSE', '关闭保护干净区MSE'],
              [comparison[i] for i in index])
    lines += ['上述仅展示有限匹配点，完整匹配与未匹配原因见CSV。若新方法只能在接近零去噪量处比较，'
              '不能将较小修改当作有效保护；匹配结果也不能证明对所有真实周期纹理都可辨识。', '']


def _has_numeric_reference_quality(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in ('psnr', 'psnr_db', 'ssim') and isinstance(child, (int, float)):
                return True
            if _has_numeric_reference_quality(child):
                return True
    elif isinstance(value, list):
        return any(_has_numeric_reference_quality(v) for v in value)
    return False


def append_real(lines, root, project_root):
    folder = root / 'real'
    json_files = sorted(folder.rglob('*.json')) if folder.exists() else []
    outputs = []
    for path in json_files:
        payload = read_json(path)
        if _has_numeric_reference_quality(payload):
            raise RuntimeError(f'Unpaired real example contains numeric PSNR/SSIM: {path}')
        outputs.append(path.relative_to(root).as_posix())
    provenance_path = project_root / 'third_party/real_examples/provenance.json'
    provenance = read_json(provenance_path) if provenance_path.exists() else {}
    entries = provenance.get('records', [])
    lines += ['## 两项真实来源案例 仅定性', '',
              '真实来源图像没有配对干净参考，未计算PSNR、SSIM，也不以输出更平滑作为质量证明。'
              '各方法跨案例使用各自同一份冻结配置，不逐图调参。恢复图、残差和频谱只作为定性检查。']
    if entries:
        table(lines, ['案例ID', '来源', '解释边界'], [
            [r.get('id', '未提供'), r.get('source_page', r.get('url', '未提供')),
             r.get('interpretation', r.get('evaluation', '无配对真值，定性展示'))] for r in entries])
    else:
        lines += ['本地来源清单未找到，不能仅凭图像外观确认干扰来源。', '']
    lines += [f'真实案例阶段保存JSON记录{len(outputs)}份，PNG图{len(list(folder.rglob("*.png"))) if folder.exists() else 0}份。'
              '画布纹理案例中的周期成分也可能是物理载体或图像内容，处理目标不同于严格传感器噪声真值。', '',
              '尚未完成作者MATLAB/Octave原版与Python移植版的小批输入数值对照。现有ARPENOS仍是已说明数值替换的源码指导移植，'
              '真实案例展示不构成原实现等价性核验，也不冒称复现原论文指标。', '']


def append_p2(lines, root, cfg):
    est = cfg.get('estimation', {})
    paths = sorted(p for p in root.rglob('*.csv') if 'p2' in p.name.lower() and 'report' not in p.name.lower())
    lines += ['## 可选候选重排序的状态', '',
              '前2候选精细拟合、空间分层留出验证和共享区域残差一致性是估计稳定性检查，'
              '不能替代风险门控；邻块一致也不证明候选是噪声。', '',
              f'当前冻结参数refine_candidates={est.get("refine_candidates", "未提供")}，'
              f'use_overlap_reranking={est.get("use_overlap_reranking", "未提供")}。']
    if paths:
        table(lines, ['开发探索记录', '记录数', '性质'], [[p.relative_to(root).as_posix(), len(read_csv(p)),
              '仅开发探索；默认配置和主审计方法表决定是否启用，不把不稳定探索作为成功项'] for p in paths])
    else:
        lines += ['', '未发现独立P2开发CSV，不能对该可选项声称已有稳定收益。', '']
    if est.get('refine_candidates') == 1 and not est.get('use_overlap_reranking', False):
        lines += ['P2未进入默认算法。当前主结论集中于数值正确性、不可被强度取消的安全权重及模型之外的验证，'
                  '不以增加模型数量补充工作量。', '']


def write_report(root, config_path, project_root):
    verification = read_json(root / 'verification.json')
    if verification.get('all_passed') is not True:
        raise RuntimeError('Complete v3 artifact verification must pass before writing the report')
    audit = inventory(root, 'audit', 3900)
    stress = inventory(root, 'stress', 500)
    textures = inventory(root, 'textures', 180)
    frozen = read_json(config_path)
    cfg = frozen.get('configuration', frozen.get('config', {}))
    pair_rows = read_csv(root / 'audit/paired_comparisons.csv', required=False)
    pairs = [dict(kind=r['kind'], baseline=r['baseline'], n_images=int(float(r['n_images'])),
                  mean_delta_psnr=float(r['mean_delta_psnr']), image_bootstrap_low=float(r['image_bootstrap_low']),
                  image_bootstrap_high=float(r['image_bootstrap_high'])) for r in pair_rows] if pair_rows else paired_comparisons(audit)
    checks = verification.get('checks', {})
    if isinstance(checks, dict) and all(isinstance(v, bool) for v in checks.values()):
        check_text = f'{sum(checks.values())}/{len(checks)}项核验通过'
    else:
        check_text = '最终all_passed核验通过，细项见verification.json'
    lines = ['# 多尺度局部相位建模与保守决策的v3代码验证', '',
             '本轮围绕数值与复现边界、不可被强度取消的安全保护，以及模型匹配之外的证据展开。'
             '这是实现与实验记录，不是通用最优周期去噪的声明。', '',
             f'新60图主审计3900条、20图模型外与位置压力测试500条、12张DTD配对纹理测试180条，库存完整；{check_text}。'
             '参数只在开发集选择，新审计、DTD与真实案例不用于调整阈值。算法接收含噪图与配置，参考和合成真值仅在评价层使用。', '']
    xml_paths = (root / 'tests.xml', root.parent / 'v3_tests.xml', root.parent / 'tests_v3.xml')
    for path in xml_paths:
        if path.exists():
            tests = ET.parse(path).getroot().findall('.//testcase')
            failures = sum(t.find('failure') is not None or t.find('error') is not None for t in tests)
            lines += [f'保存的数值测试记录包含{len(tests)}项，失败或错误{failures}项。', '']
            break
    append_p0(lines)
    append_configuration(lines, frozen, audit)
    append_primary(lines, audit, pairs)
    append_frequency(lines, audit)
    append_stress(lines, stress)
    append_textures(lines, textures)
    append_matching(lines, root)
    append_real(lines, root, project_root)
    append_p2(lines, root, cfg)
    lines += ['## 复现与解释范围', '',
              '浮点图像范围为[0,1]，PSNR和SSIM使用data_range=1，不做逐结果最小最大拉伸。'
              '空区域指标显示为未提供，不解释为零损伤。运行耗时、质量退化和程序异常分开记录。', '',
              '自然周期纹理与附加干扰仍可能有相同像素解释。安全上限可阻止强度取消保护，'
              '但安全评分本身仍是启发式；有限压力测试和真实图展示不能证明已解决单图分离的不可辨识性。', '',
              '- 主审计、模型外与纹理原始指标：`audit/metrics.csv`、`stress/metrics.csv`、`textures/metrics.csv`。',
              '- 尾部与频率/覆盖汇总：各阶段`summary.csv`，本说明另生成`report_paired_comparisons.csv`。',
              '- 权重匹配及覆盖：`supplement/weight_curves.csv`、`matched_removal.csv`、`matched_coverage.csv`。',
              '- 真实案例：`real/`；数据与来源：v3数据清单、DTD及真实案例来源JSON。',
              '- 配置、代码、依赖及审计ID顺序核验：冻结配置中的snapshot与`verification.json`。',
              '- 本说明由`v3_report.py`从已完成产物生成；未重新运行去噪，也未选择参数。', '']
    output = root / 'RESULTS.md'
    output.write_text('\n'.join(lines), encoding='utf-8')
    write_csv(root / 'report_paired_comparisons.csv', pairs)
    print(f'Wrote {output} and report_paired_comparisons.csv')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/v3'))
    p.add_argument('--config', type=Path, default=Path('configs/v3_frozen.json'))
    p.add_argument('--project-root', type=Path, default=Path.cwd())
    args = p.parse_args()
    write_report(args.root, args.config, args.project_root)


if __name__ == '__main__':
    main()
