"""Write a Chinese code-validation note from completed, verified v2 artifacts.

This script does not run denoisers, select parameters, or invent expected gains.
Run after audit, ablations, decisions, supplement and verification have finished.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET


KINDS = ('global', 'local', 'chirp', 'orientation')
KIND_NAMES = dict(global_='全局定频', local='局部定频', chirp='频率渐变',
                  orientation='方向变化', clean='无噪声对照')
KIND_NAMES['global'] = KIND_NAMES.pop('global_')
MAIN_METHODS = ('input', 'v1_full', 'global_adaptive', 'published_arpenos', 'v2_full')
LABELS = {
    'input': '含噪输入', 'v1_full': 'v1完整方法',
    'global_adaptive': '原型全局自适应', 'published_arpenos': 'ARPENOS移植',
    'v2_full': 'v2完整方法', 'v2_without_guard': '关闭全部保守决策',
    'v2_fixed_selection128': '强制选择128尺度', 'v2_without_chirp': '关闭二次相位',
    'v2_constant_amplitude': '关闭变幅模型',
    'v2_without_texture_risk': '仅关闭局部纹理风险',
    'v2_peak_only': '仅频峰局部评分',
    'v2_without_candidate_gate': '仅关闭候选减除资格门槛',
}
ABLATION_METHODS = ('v2_fixed_selection128', 'v2_without_guard',
                    'v2_without_chirp', 'v2_constant_amplitude')
DECISION_METHODS = ('v2_without_texture_risk', 'v2_peak_only', 'v2_without_candidate_gate')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path, required=True):
    path = Path(path)
    if not path.exists() and not required:
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def number(row, key, default=None):
    value = row.get(key, '')
    return default if value == '' or value is None else float(value)


def fmt(value, digits=3, scientific=False):
    if value is None:
        return '—'
    if math.isinf(value):
        return '∞'
    if not math.isfinite(value):
        return '—'
    return f'{value:.{digits}e}' if scientific else f'{value:.{digits}f}'


def table(lines, headers, rows):
    lines.extend(['', '| ' + ' | '.join(headers) + ' |',
                  '| ' + ' | '.join('---' for _ in headers) + ' |'])
    lines.extend('| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows)
    lines.append('')


def condition_lookup(rows):
    return {(r['kind'], r['method']): r for r in rows}


def pair_lookup(rows):
    return {(r['kind'], r['baseline']): r for r in rows}


def signed_evidence(pair):
    if not pair:
        return '未提供按原图自举区间，只能作描述性比较'
    lo = number(pair, 'image_bootstrap_low')
    hi = number(pair, 'image_bootstrap_high')
    mean = number(pair, 'mean_delta_psnr')
    if lo is None or hi is None or mean is None:
        return '区间不完整，不作方向判断'
    if lo > 0:
        return '当前样本支持完整方法PSNR更高'
    if hi < 0:
        return '当前样本支持完整方法PSNR更低'
    if abs(mean) <= 1e-12 and abs(lo) <= 1e-12 and abs(hi) <= 1e-12:
        return '未观测到PSNR差异'
    return '区间跨越或接触0，增益方向仍不确定'


def stage_gate(root, stage, expected):
    path = root / stage / 'validation.json'
    if path.exists():
        result = read_json(path)
        if not result.get('complete'):
            raise RuntimeError(f'{stage}: experiment is not complete')
        actual = result.get('actual', result.get('actual_rows'))
        if actual is not None and int(actual) != expected:
            raise RuntimeError(f'{stage}: expected {expected}, actual {actual}')
    rows = read_csv(root / stage / 'metrics.csv')
    keys = {(r['image_id'], r['kind'], r['amplitude'], r['method']) for r in rows}
    if len(rows) != expected or len(keys) != expected or any(r.get('error') for r in rows):
        raise RuntimeError(f'{stage}: incomplete, duplicated or erroneous metric rows')
    return rows


def append_implementation(lines, config):
    est = config.get('estimation', {})
    guard = config.get('guard', {})
    scales = '、'.join(str(s) for s in est.get('scales', ()))
    chirp = '启用' if est.get('enable_chirp', False) else '关闭'
    amplitude = '启用' if est.get('enable_amplitude_model', False) else '关闭'
    forced = est.get('force_scale')
    lines += ['## 原提案中的优化如何落实', '',
              '下表描述已实现的机制；是否有效由后面的新审计结果与消融回答。频率统一使用周期/像素，可靠性分数不解释为噪声概率。']
    table(lines, ['优化项', '代码机制与当前配置', '本轮检验'], [
        ['局部尺度选择', f'分析尺度{scales}；每尺度最多{est.get("max_candidates", "—")}个候选；'
         f'在同一合成区域验证拟合并选择一个尺度；当前强制尺度={forced}',
         '前20图强制128消融保留跨尺度候选和评分证据，不把多尺度结果直接平均'],
        ['按可靠程度保守抑制', '频峰、邻域和跨尺度证据，加局部结构复杂度风险、候选减除资格门槛及全局周期歧义拒绝；'
         f'纹理风险={guard.get("use_texture_risk", True)}，歧义拒绝={guard.get("use_ambiguity_guard", True)}',
         '分别关闭决策模块，并在前10图比较相近去噪量下的干净区损伤'],
        ['局部候选残差', f'连续频率细化；二次相位{chirp}；非负变幅模型{amplitude}；拟合改善与复杂度约束控制模型选择',
         '前20图分别关闭二次相位与变幅模型；不把拟合误差降低当作噪声判定依据'],
        ['重叠复原与工程验证', f'合成块{est.get("synthesis_size", "—")}像素、步长{est.get("stride", "—")}；'
         '正端点窗口及归一化重叠减除；保留未抑制候选供消融复用',
         '数值测试、唯一记录清点、配置和源码哈希、确定性重跑'],
    ])


def append_main_results(lines, summary, pairs, raw, n_images):
    lookup = condition_lookup(summary)
    plook = pair_lookup(pairs)
    lines += ['## 新60图主实验', '',
              f'主实验使用{n_images}张此前未参与算法选择的官方测试图，每图4类退化、3档强度及1个无噪声对照。'
              'v1也在这同一批图上重新运行；旧60张已看过的测试图只作为回归资料，不混入本表。', '',
              'PSNR是逐样本PSNR的平均值，MSE和MAE分别是逐样本像素误差的平均值，不能相互替代。'
              '不同退化条件总噪声能量不同，应在同一条件内比较方法。']
    rows = []
    overview = []
    for kind in KINDS:
        full = lookup[(kind, 'v2_full')]
        for method in MAIN_METHODS:
            r = lookup[(kind, method)]
            pair = plook.get((kind, method))
            mse, full_mse = number(r, 'mean_mse'), number(full, 'mean_mse')
            delta = number(full, 'mean_psnr_db') - number(r, 'mean_psnr_db')
            rows.append([KIND_NAMES[kind], LABELS[method], fmt(number(r, 'mean_psnr_db'), 2),
                         fmt(mse, scientific=True), fmt(number(r, 'mean_mae'), scientific=True),
                         fmt(number(r, 'mean_ssim'), 4)])
            overview.append(dict(split='audit_test', n_images=n_images, kind=kind, method=method,
                n_samples=int(float(r['n'])), mean_psnr_db=number(r, 'mean_psnr_db'),
                mean_mse=mse, mean_mae=number(r, 'mean_mae'), mean_ssim=number(r, 'mean_ssim'),
                v2_minus_method_mean_psnr_db=delta,
                v2_minus_method_mse=full_mse-mse,
                v2_mse_relative_to_method=full_mse/mse if mse else '',
                paired_image_bootstrap_low=number(pair or {}, 'image_bootstrap_low', ''),
                paired_image_bootstrap_high=number(pair or {}, 'image_bootstrap_high', '')))
    table(lines, ['条件', '方法', '平均PSNR dB', '平均MSE', '平均MAE', '平均SSIM'], rows)
    lines += ['### v2相对同批图上v1的变化', '',
              '下面差值均为v2减v1。MSE变化率为（v2 MSE − v1 MSE）/ v1 MSE，负值表示像素误差降低。'
              '区间按原图自举，三档强度先在同图内平均；这仍只是本次数据的证据，不能证明普遍领先。']
    rows = []
    for kind in KINDS:
        pair = plook.get((kind, 'v1_full'))
        old, full = lookup[(kind, 'v1_full')], lookup[(kind, 'v2_full')]
        change = (number(full, 'mean_mse')/number(old, 'mean_mse')-1)*100
        rows.append([KIND_NAMES[kind], fmt(number(pair or {}, 'mean_delta_psnr'), 3),
                     f'[{fmt(number(pair or {}, "image_bootstrap_low"), 3)}, '
                     f'{fmt(number(pair or {}, "image_bootstrap_high"), 3)}]',
                     fmt(change, 1)+'%', signed_evidence(pair)])
    table(lines, ['条件', '配对PSNR差 dB', '按原图自举95%区间', 'MSE变化率', '当前解释'], rows)
    lines += ['### 无噪声对照和局部干净区域', '',
              '无噪声对照直接衡量算法额外引入的修改。修改图像数使用单图MAE大于1e-12的口径；'
              '不要用平均PSNR中的∞掩盖少量严重改动。']
    clean_rows = []
    for method in MAIN_METHODS:
        r = lookup[('clean', method)]
        clean_rows.append([LABELS[method], fmt(number(r, 'mean_mae'), scientific=True),
            fmt(number(r, 'median_mae'), scientific=True), fmt(number(r, 'p95_mae'), scientific=True),
            fmt(number(r, 'max_mae'), scientific=True), f'{int(float(r["modified_images"]))}/{n_images}'])
    table(lines, ['方法', 'MAE均值', 'MAE中位数', 'MAE P95', '单图MAE最大值', '修改图像数'], clean_rows)
    old, full = lookup[('clean', 'v1_full')], lookup[('clean', 'v2_full')]
    delta = number(full, 'mean_mae')-number(old, 'mean_mae')
    if delta > 1e-12:
        lines += [f'v2无噪声对照平均MAE由v1的{fmt(number(old,"mean_mae"),scientific=True)}增至'
                  f'{fmt(number(full,"mean_mae"),scientific=True)}。因此去噪PSNR提升不能解释为没有额外损伤。', '']
    elif delta < -1e-12:
        lines += [f'v2无噪声对照平均MAE由v1的{fmt(number(old,"mean_mae"),scientific=True)}降至'
                  f'{fmt(number(full,"mean_mae"),scientific=True)}，仍需结合P95、最大值及规则纹理反例判断。', '']
    else:
        lines += ['v2与v1的无噪声平均改变量在上述精度下接近；仍需检查单图尾部与规则纹理反例。', '']
    table(lines, ['方法', '局部条件干净区域MSE', '局部条件干净区域修改MAE'], [
        [LABELS[m], fmt(number(lookup[('local',m)], 'mean_clean_mse'),scientific=True),
         fmt(number(lookup[('local',m)], 'mean_clean_modification_mae'),scientific=True)]
        for m in MAIN_METHODS])
    lines += ['只有局部退化和无噪声对照存在真正干净区域；全局、频率渐变、方向变化的空区域误差不解释为0。', '']
    table(lines, ['条件', '方法', '输出比输入更差比例', '平均时间 ms'], [
        [KIND_NAMES[k], LABELS[m], fmt(100*number(lookup[(k,m)],'mean_degraded'),1)+'%',
         fmt(number(lookup[(k,m)],'mean_runtime_ms'),1)]
        for k in KINDS for m in MAIN_METHODS if m!='input'])
    lines += ['质量退化定义为输出MSE高于输入MSE加1e-8，与程序异常分开统计。时间为本机单次运行，'
              '准备阶段开销计入各方法，不作为严格速度排名。', '']
    return overview


def append_ablation(lines, title, summary, pairs, methods, n_images=20):
    lookup = condition_lookup(summary)
    plook = pair_lookup(pairs)
    lines += [title, '', f'这部分仅使用事先固定的前{n_images}张新审计图，每种方法100个样本'
              '（4类中档退化及无噪声对照）。它不代表全部60图，也不代表全部三档强度。', '',
              '表中的PSNR差为完整v2减去对应消融；正值支持保留该模块，负值说明当前模块降低了该条件的PSNR。'
              '局部干净区与无噪声误差必须同时检查。']
    rows = []
    for method in methods:
        for kind in KINDS:
            pair = plook.get((kind, method))
            full, other = lookup[(kind, 'v2_full')], lookup[(kind, method)]
            delta = number(pair or {}, 'mean_delta_psnr', number(full,'mean_psnr_db')-number(other,'mean_psnr_db'))
            rows.append([LABELS[method], KIND_NAMES[kind], fmt(delta,3),
                f'[{fmt(number(pair or {},"image_bootstrap_low"),3)}, '
                f'{fmt(number(pair or {},"image_bootstrap_high"),3)}]',
                signed_evidence(pair)])
    table(lines, ['消融', '条件', '完整方法PSNR差 dB', '按原图自举95%区间', '当前解释'], rows)
    rows=[]
    for method in ('v2_full', *methods):
        clean=lookup[('clean',method)];local=lookup[('local',method)]
        rows.append([LABELS[method], fmt(number(clean,'mean_mae'),scientific=True),
                     fmt(number(clean,'p95_mae'),scientific=True),fmt(number(clean,'max_mae'),scientific=True),
                     fmt(number(local,'mean_clean_mse'),scientific=True)])
    table(lines, ['方法', '无噪声MAE均值', '无噪声MAE P95', '无噪声MAE最大', '局部干净区MSE'],rows)


def append_matching(lines, root):
    folder=root/'supplement'
    coverage=read_csv(folder/'matched_coverage.csv')
    matched=read_csv(folder/'matched_removal.csv',required=False)
    positive=[r for r in matched if number(r,'matched_contaminated_mse_reduction_fraction',0)>1e-12]
    individual=[r for r in positive if r.get('group')=='individual']
    pooled=[r for r in positive if r.get('group')=='subset_mean']
    eligible_ids={r['image_id'] for r in individual}
    eligible=len(eligible_ids)
    total=len(coverage)
    lines += ['## 相近去噪量下的损伤比较', '',
              f'仅用固定前10张新审计图的局部中档退化。具有可插值共同去噪区间的个体为'
              f'{eligible}/{total}（{fmt(100*eligible/total if total else 0,1)}%）；'
              f'其余{total-eligible}例不进入个体匹配，仍保留在覆盖表中。', '',
              '比较轴是含噪区域MSE降低比例，不是相同strength。先保留采样点的Pareto前沿，再只在'
              '共同区间内线性插值；这是有限强度扫描近似，不选择新的运行参数。']
    table(lines, ['图像ID','匹配点数','覆盖说明'],[
        [r['image_id'],sum(point['image_id']==r['image_id'] for point in individual),
         '存在正去噪共同区间' if r['image_id'] in eligible_ids else
         '没有正去噪共同区间；不计入个体保护效果'] for r in coverage])
    rows=[]
    for label,points in (('个体匹配点',individual),('10图均值曲线匹配点',pooled)):
        if not points:
            rows.append([label,0,'—','—','—','无可比较正去噪区间'])
            continue
        better=sum(number(r,'clean_mse_difference',0)<-1e-18 for r in points)
        worse=sum(number(r,'clean_mse_difference',0)>1e-18 for r in points)
        lo=min(number(r,'matched_contaminated_mse_reduction_fraction') for r in points)
        hi=max(number(r,'matched_contaminated_mse_reduction_fraction') for r in points)
        rows.append([label,len(points),fmt(100*lo,2)+'%至'+fmt(100*hi,2)+'%',better,worse,
                     '计数为同一曲线上的相关插值点，不是独立图像个数'])
    table(lines,['比较','正去噪匹配点数','覆盖去噪范围','保守损伤较低点','保守损伤较高点','解释'],rows)
    if pooled:
        mid=pooled[len(pooled)//2]
        lines += [f'以均值曲线的一个中间匹配点为例，含噪区MSE降低'
                  f'{fmt(100*number(mid,"matched_contaminated_mse_reduction_fraction"),2)}%时，'
                  f'保守决策的干净区MSE为{fmt(number(mid,"conservative_clean_mse"),scientific=True)}，'
                  f'关闭决策为{fmt(number(mid,"without_conservative_clean_mse"),scientific=True)}。'
                  '该点只用于说明权衡，不作为总体优势证明。', '']
    lines += ['只有在共同去噪区间内损伤更低，才支持本子集上的保守决策价值；'
              '没有正去噪区间、仅不处理或只降低强度，都不能单独构成纹理保护证据。', '']


def append_texture(lines, root):
    rows=read_csv(root/'supplement/texture_metrics.csv')
    lookup={(r['case'],r['method'],r['interpretation']):r for r in rows}
    cases=list(dict.fromkeys(r['case'] for r in rows))
    lines += ['## 规则纹理与单图歧义反例', '',
              '这些是固定压力检查，部分自然底图来自开发集，不是新增独立测试样本。'
              '每个合成案例保留两种真实解释：周期成分属于合法纹理，或属于背景上的附加干扰。'
              '自然底图复合案例的替代参考是自然底图；CSV沿用的noise_on_flat_field标签在这些案例中不表示背景为常数。']
    display=[]
    for case in cases:
        records=[]
        for method in ('v1','v2','v2_without_guard'):
            r=lookup.get((case,method,'legitimate_texture'))
            records.append(fmt(number(r or {},'modification_mae'),scientific=True))
        old=number(lookup.get((case,'v1','legitimate_texture'),{}),'modification_mae')
        new=number(lookup.get((case,'v2','legitimate_texture'),{}),'modification_mae')
        if old is None or new is None:decision='数据不全'
        elif new<old-1e-12:decision='v2改动量较小；仍需看绝对误差'
        elif new>old+1e-12:decision='v2改动量增大，保留为失败/风险'
        else:decision='改动量接近；不据此判断是否识别了噪声'
        display.append([case,*records,decision])
    table(lines,['合法纹理案例','v1修改MAE','v2修改MAE','关闭guard修改MAE','当前观察'],display)
    for case in cases:
        if 'brick' not in case.lower():
            continue
        old=number(lookup.get((case,'v1','legitimate_texture'),{}),'modification_mae')
        new=number(lookup.get((case,'v2','legitimate_texture'),{}),'modification_mae')
        if old is not None and new is not None and new>old+1e-12:
            ratio=f'，约为v1的{fmt(new/old,2)}倍' if old>0 else ''
            lines += [f'真实砖纹理出现更大误改：修改MAE从v1的{fmt(old,scientific=True)}增至'
                      f'v2的{fmt(new,scientific=True)}{ratio}。因此人工全图规则纹理受到保留，'
                      '不能解释为通用真实纹理保护已经成功。','']
    lines += ['全部纹理与双解释指标保留如下。替代参考表示把周期部分视为附加干扰时的评价，'
              '不是第二张算法输入；合法纹理参考就是输入自身。']
    table(lines,['案例','评价参考解释','方法','MSE','相对参考MAE','输入修改MAE'],[
        [r['case'],'合法纹理' if r['interpretation']=='legitimate_texture' else '替代背景参考',
         r['method'],fmt(number(r,'mse'),scientific=True),fmt(number(r,'mae'),scientific=True),
         fmt(number(r,'modification_mae'),scientific=True)] for r in rows])
    groups=defaultdict(list)
    for row in rows:groups[(row['case'],row['method'])].append(row)
    dual=[group for group in groups.values() if len(group)==2]
    same=sum(len({r['input_sha256'] for r in group})==1 and
             len({r['output_sha256'] for r in group})==1 for group in dual)
    lines += [f'双解释组共{len(dual)}组，其中输入与输出哈希均相同的有{same}组。'
              '像素和算法决定没有改变，改变的只有评价参考，因此该检查展示不可辨识性，而不是自动判别能力。', '']
    local_cases=[case for case in cases if case.startswith('local_periodic_on_natural_')]
    if local_cases:
        max_mod=max(number(lookup[(c,'v2','legitimate_texture')],'modification_mae',0) for c in local_cases)
        lines += [f'自然图中局部周期内容的v2最大修改MAE为{fmt(max_mod,scientific=True)}。'
                  '这类输入既可表示绘制或印制的真实周期内容，也可表示局部干扰。全图规则纹理被拒绝处理，'
                  '不能证明局部真实纹理一定受到保护。', '']
    lines += ['因此，保守不处理是一种显式策略；它也可能保留平滑背景上的真实干扰。'
              '不能将有限纹理压力检查通过写成彻底解决纹理与噪声的分离。', '']


def write_report(root):
    verification=read_json(root/'verification.json')
    if verification.get('all_passed') is not True:
        raise RuntimeError('The complete optimization verification must pass first')
    audit_raw=stage_gate(root,'audit',3900)
    stage_gate(root,'ablations',500)
    stage_gate(root,'decisions',400)
    run=read_json(root/'audit/run.json')
    frozen=run['config'];config=frozen['configuration']
    ids=run['ids']
    if len(ids)!=60 or len(set(ids))!=60:
        raise RuntimeError('Expected exactly60 unique fresh audit IDs')
    audit_summary=read_csv(root/'audit/summary.csv')
    audit_pairs=read_csv(root/'audit/paired_comparisons.csv')
    checks=verification.get('checks',{})
    lines=['# 优化代码实现与新审计结果','',
           '本说明回答原提案中的优化是否已经落实，以及本轮验证支持哪些改善。它是代码验证记录，不是论文；'
           '实现完成、程序运行成功与研究假设成立分开判断。','',
           f'本轮主实验3900条、尺度与残差消融500条、决策消融400条，均通过完整性核验。'
           f'交付一致性检查{sum(bool(v) for v in checks.values())}/{len(checks)}通过。'
           '算法只接收含噪输入与冻结配置，参考图、合成频率与区域真值仅用于校准和评价。','',
           f'参数只在原30张开发图上选择；本次减除强度为{fmt(number(config,"strength"),2)}。'
           '使用60张新的官方测试图进行审计，旧60张已查看图像改为回归资料。'
           '四类条件使用0.06、0.12、0.18三档幅度，同图同条件的频率、方向、相位和包络保持配对。','']
    junit=root.parent/'optimization_tests.xml'
    if junit.exists():
        tests=ET.parse(junit).getroot().findall('.//testcase')
        failures=sum(t.find('failure') is not None or t.find('error') is not None for t in tests)
        lines += [f'保存的数值测试记录包含{len(tests)}项，失败或错误{failures}项。','']
    append_implementation(lines,config)
    overview=append_main_results(lines,audit_summary,audit_pairs,audit_raw,len(ids))
    append_ablation(lines,'## 尺度与残差模型消融 仅20图',
        read_csv(root/'ablations/summary.csv'),read_csv(root/'ablations/paired_comparisons.csv'),ABLATION_METHODS)
    lines += ['强制128保留了其他尺度的证据计算，检验的是尺度选择。二次相位和变幅模型分别关闭；'
              '本轮没有额外同时关闭二者，因此没有独立识别两者的交互作用。关闭全部保守决策保留相同候选残差。','']
    append_ablation(lines,'## 可靠性与纹理决策消融 仅20图',
        read_csv(root/'decisions/summary.csv'),read_csv(root/'decisions/paired_comparisons.csv',required=False),DECISION_METHODS)
    lines += ['仅频峰评分仍保留全局周期歧义拒绝和候选残差，局部邻域与跨尺度评分、局部纹理风险关闭；'
              '仅关闭纹理风险和仅关闭候选减除资格门槛是更窄的独立消融。资格由频峰评分或跨尺度与邻域支撑判断，'
              '不是能量阈值。不同决策复用同一候选残差，避免将重新拟合误当成评分作用。','']
    append_matching(lines,root)
    append_texture(lines,root)
    lines += ['## 指标与复现范围','',
              'PSNR和SSIM使用data_range=1，不对各方法结果单独拉伸。指标来自浮点数组，PNG只作显示。'
              '本轮是可控合成退化验证，不代表真实传感器干扰上的效果已经成立。ARPENOS为作者源码指导的Python移植，'
              '其robustfit数值替换已记录，不能冒称原论文指标复现。','',
              '- 主指标与区间：`audit/summary.csv`、`audit/paired_comparisons.csv`。',
              '- 尺度/残差与决策消融：`ablations/`、`decisions/`。',
              '- 匹配去噪量、覆盖及纹理：`supplement/`。',
              '- 冻结配置、数据和源码一致性：`verification.json`与各阶段`run.json`。',
              '- 可比较机器表：`overview_compare.csv`。',
              '- 从项目根目录运行`run_optimization.ps1`复现；本说明由`write_optimization_report.py`从完成的结果生成。','']
    (root/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    with (root/'overview_compare.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(overview[0]))
        writer.writeheader();writer.writerows(overview)
    print(f'Wrote {root / "RESULTS.md"} and {root / "overview_compare.csv"}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/optimization_v2'))
    args=parser.parse_args()
    write_report(args.root)


if __name__=='__main__':
    main()
