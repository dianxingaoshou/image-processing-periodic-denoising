"""Frozen 2x2x2 support ablation: analysis32, synthesis32 and grid envelope.

Run plan, development, freeze, regression, check, report in that order.
References/masks exist only at this evaluation layer. All variants retain the
same legacy coupled guard and strength; risk-score changes are not mixed in.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from denoiser_v3 import config_dict, config_from_dict, prepare, restore
from evaluation_v3 import estimation_metrics, tail_mean
from image_io_v3 import preprocess_image, sha256_file
from reproducibility_v3 import capture_snapshot, verify_snapshot
from run_experiments import metric_row
from support_data import KINDS, make_sample

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; OUT=ROOT/'results/support'; PLAN=ROOT/'support_protocol.json'
STRENGTHS=(.5,1.,2.)
FACTORS={('_'.join(name for name,flag in zip(('a32','s32','grid'),flags) if flag) or 'baseline'):flags
         for flags in itertools.product((False,True),repeat=3)}
CONFIG_PATHS=tuple(f'configs/support_{name}.json' for name in FACTORS)
SOURCES=('support_experiments.py','support_data.py','small_window.py','spatial_envelope.py',
    'local_estimation_v3.py','denoiser_v3.py','texture_guard_v3.py','local_spectral.py',
    'evaluation_v3.py','run_experiments.py','validation_data_v3.py','data_pipeline.py','image_io_v3.py',
    'reproducibility_v3.py','original/periodic_notch_starter/demo.py','requirements-lock.txt',
    'configs/v3_p0_coupled.json','data/v3_manifest.json','refinement_protocol.json','support_protocol.json',*CONFIG_PATHS)
TIE_TOLERANCE=1e-12


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def write_csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def factorial_configs():
    baseline=config_from_dict(read(ROOT/'configs/v3_p0_coupled.json')['configuration'])
    configurations={}
    for name,(analysis32,synthesis32,grid) in FACTORS.items():
        estimation=replace(baseline.estimation,scales=([32,64,128,256] if analysis32 else [64,128,256]),
            synthesis_size=32 if synthesis32 else 64,stride=16 if synthesis32 else 32,
            amplitude_model='grid' if grid else 'affine')
        configurations[name]=replace(baseline,estimation=estimation)
    return configurations


def plan():
    if PLAN.exists():raise FileExistsError('Protocol already exists; never silently reselect check images.')
    manifest=read(DATA/'v3_manifest.json');previous=read(ROOT/'refinement_protocol.json')
    used={r['id'] for r in manifest['images']}|{r['id'] for r in previous['independent_check']}
    pool=sorted(p for p in (DATA/'source/images/test').glob('*.jpg') if p.stem not in used)
    if len(pool)!=16:raise ValueError('Restore BSDS raw images first; expected 16 unused test files.')
    indices=sorted(np.random.default_rng(2026111605).choice(len(pool),4,replace=False))
    check=[dict(id=pool[i].stem,source_jpg=pool[i].relative_to(DATA).as_posix(),source_sha256=sha256_file(pool[i])) for i in indices]
    for name,cfg in factorial_configs().items():
        path=ROOT/f'configs/support_{name}.json'
        if path.exists():raise FileExistsError(path)
        dump(path,dict(configuration=config_dict(cfg),purpose='Fixed factorial support ablation, legacy coupled guard; not a calibrated universal replacement.'))
    dump(PLAN,dict(schema_version=1,seed=2026111605,
        development_ids=previous['development_ids'],independent_check=check,
        independent_selection='4 of 16 unused official test filenames, seeded before loading pixels; excludes old210 and priorcheck4',
        regression_texture_ids=[r['id'] for r in manifest['textures']],
        regression_edge_ids=previous['independent_check'],kinds=list(KINDS),
        methods={k:list(v) for k,v in FACTORS.items()},factor_order=['analysis32','synthesis32','grid_envelope'],
        amplitude=.12,strengths=list(STRENGTHS),primary_strength=2.,
        primary_metrics=['contaminated_mse','clean_mse','adjacent_clean_mse'],
        adjacent_clean_definition='uncontaminated pixels within Euclidean distance16 of non-clean support; truth used only in evaluation',
        tie_tolerance=TIE_TOLERANCE,decision='report raw per-image deltas; jointly improve dirty MSE by >1e-12 without raising clean or adjacent MSE by >1e-12; this tolerance is numerical, not proof of universal safety',
        controls='12 DTD clean regression photos and clean versions of every BSDS; original photos are not guaranteed physically noise-free',
        scale_caveat='synthesis32 uses stride16 to maintain50percent overlap; counts/centres change, so block-centre recall is secondary and not directly comparable across geometries',
        envelope='3x3 bilinear nonnegative controls, normalized smoothness .002 and ridge .0002; cap2.5 times constant-fit amplitude; penalized input-fit improvement required; no true support provided',
        small_window='>=3 radial cycles and>=.8 parity-balanced split coefficient stability; otherwise larger analysis windows remain candidates',
        preprocessing='size256; short-side Lanczos resize and center crop; shared float luminance; no contrast rescaling',
        historical_scope='old audit, DTD and priorcheck4 are regression after prior design; independent check is used once after freezing',
        source_manifest_sha256=sha256_file(DATA/'v3_manifest.json')))
    print(PLAN)


def load_configs():
    actual={name:config_from_dict(read(ROOT/f'configs/support_{name}.json')['configuration']) for name in FACTORS}
    expected=factorial_configs()
    if actual!=expected:raise ValueError('Factorial configs changed an unintended setting')
    return actual


def freeze():
    destination=OUT/'freeze.json'
    if destination.exists():raise FileExistsError('Study is already frozen')
    configurations={name:config_dict(cfg) for name,cfg in load_configs().items()}
    protocol=read(PLAN);dev=OUT/'development/metrics.csv'
    if not dev.exists():raise FileNotFoundError('Run development first')
    dump(destination,dict(configurations=configurations,development_metrics_sha256=sha256_file(dev),
        snapshot=capture_snapshot(SOURCES,[r['id'] for r in protocol['independent_check']],base_dir=ROOT,
            metadata=dict(configurations=configurations,selection='fixed factorial variants; no check-set tuning'))))
    print(destination)


def verify_frozen():
    frozen=read(OUT/'freeze.json');protocol=read(PLAN)
    verify_snapshot(frozen['snapshot'],paths=SOURCES,audit_ids=[r['id'] for r in protocol['independent_check']],base_dir=ROOT)
    if frozen['development_metrics_sha256']!=sha256_file(OUT/'development/metrics.csv'):
        raise ValueError('Development metrics changed after freezing')
    if frozen['configurations']!=frozen['snapshot']['metadata']['configurations']:
        raise ValueError('Configurations differ from sealed metadata')
    if frozen['configurations']!={name:config_dict(cfg) for name,cfg in load_configs().items()}:
        raise ValueError('Current configuration differs from freeze')
    return frozen


def samples(stage):
    protocol=read(PLAN);manifest=read(DATA/'v3_manifest.json')
    if sha256_file(DATA/'v3_manifest.json')!=protocol['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    by_id={r['id']:r for r in manifest['images']+manifest['textures']}
    def load_reference(record):
        path=DATA/record['npy_path']
        if sha256_file(path)!=record['clean_sha256']:raise ValueError('Reference array changed')
        return np.load(path),dict(reference=record['npy_path'],sha256=record['clean_sha256'])
    def load_source(record):
        path=DATA/record['source_jpg']
        if sha256_file(path)!=record['source_sha256']:raise ValueError('Source JPEG changed')
        return preprocess_image(path,size=256)
    if stage=='development':
        records=[by_id[i] for i in protocol['development_ids']]
    elif stage=='check':records=protocol['independent_check']
    elif stage=='regression':records=[by_id[i] for i in protocol['regression_texture_ids']]
    else:raise ValueError('Unknown stage')
    for index,record in enumerate(records):
        clean,preprocessing=(load_source if stage=='check' else load_reference)(record)
        kinds=('clean',) if stage=='regression' else KINDS
        for kind in kinds:
            yield record['id'],clean,kind,index%4,preprocessing,('DTD_regression' if stage=='regression' else 'BSDS')
    if stage=='regression':
        for index,record in enumerate(protocol['regression_edge_ids']):
            clean,preprocessing=load_source(record)
            yield record['id'],clean,'edge16',index%4,preprocessing,'BSDS_regression'


def spatial_metrics(clean,output,masks):
    error=(output-clean)**2
    def mean(mask):return float(error[mask].mean()) if np.any(mask) else ''
    return dict(adjacent_clean_mse=mean(masks['adjacent_clean']),
                far_clean_mse=mean(masks['clean']&~masks['adjacent_clean']))


def run(stage):
    if stage!='development':verify_frozen()
    destination=OUT/stage
    if (destination/'metrics.csv').exists():raise FileExistsError('Stage exists; use a fresh --out directory')
    before={p:sha256_file(ROOT/p) for p in SOURCES}
    configurations=load_configs();rows=[];inputs=[]
    for index,(image_id,clean,kind,variant,preprocessing,dataset) in enumerate(samples(stage)):
        amplitude=0. if kind=='clean' else .12
        observed,meta,masks=make_sample(clean,image_id,kind,amplitude,variant)
        identity=dict(stage=stage,dataset=dataset,image_id=image_id,kind=kind,variant=variant,amplitude=amplitude)
        inputs.append(identity|dict(synthesis=meta,preprocessing=preprocessing,
            float_input_sha256=hashlib.sha256(observed.tobytes()).hexdigest()))
        rows.append(identity|dict(method='input',strength=0.,runtime_ms=0.)|
            metric_row(clean,observed,observed,masks)|spatial_metrics(clean,observed,masks))
        for method,cfg in configurations.items():
            start=time.perf_counter();prepared=prepare(observed,cfg);prepare_ms=(time.perf_counter()-start)*1000
            tiles=prepared.estimation.tiles
            selected_small=sum(t['selected'] is not None and t['selected']['scale']==32 for t in tiles)
            selected_grid=sum('grid_amplitude' in str(t.get('model','')) for t in tiles)
            checks=[c for t in tiles for c in t.get('small_window_checks',[])]
            grid=[t['envelope_diagnostics'] for t in tiles if t.get('envelope_diagnostics') is not None]
            details=dict(selected_small_fraction=selected_small/len(tiles),selected_grid_fraction=selected_grid/len(tiles),
                small_candidates_checked=len(checks),small_candidates_eligible=sum(c['eligible'] for c in checks),
                envelope_upper_mean=float(np.mean([g['amplitude_upper'] for g in grid])) if grid else '',
                total_blocks=len(tiles))
            for strength in STRENGTHS:
                start=time.perf_counter();output,diag=restore(prepared,strength=strength,return_diagnostics=True)
                rows.append(identity|dict(method=method,strength=strength,runtime_ms=prepare_ms+(time.perf_counter()-start)*1000)|
                    metric_row(clean,observed,output,masks)|spatial_metrics(clean,output,masks)|estimation_metrics(diag,masks)|details)
        print(f'{stage}: {index+1} {image_id} {kind}',flush=True)
    if before!={p:sha256_file(ROOT/p) for p in SOURCES}:raise RuntimeError('Sources changed during execution')
    write_csv(destination/'metrics.csv',rows);dump(destination/'inputs.json',inputs)
    dump(destination/'run.json',dict(stage=stage,n_inputs=len(inputs),n_metric_rows=len(rows),source_hashes=before,
        note='Final postclip/OLA regional metrics; fixed legacy guard; single-machine runtime; centre counts not directly comparable across synthesis sizes'))


def paired_deltas(rows):
    primary=[r for r in rows if r['method']!='input' and float(r['strength'])==2.]
    baseline={(r['dataset'],r['image_id'],r['kind']):r for r in primary if r['method']=='baseline'}
    result=[]
    for row in primary:
        if row['method']=='baseline':continue
        reference=baseline[row['dataset'],row['image_id'],row['kind']]
        record={k:row[k] for k in ('dataset','image_id','kind','method')}
        for field in ('mse','contaminated_mse','clean_mse','adjacent_clean_mse','mae'):
            record[field+'_delta']=float(row[field])-float(reference[field]) if row.get(field,'')!='' and reference.get(field,'')!='' else ''
        dirty=record['contaminated_mse_delta'];clean=record['clean_mse_delta'];adjacent=record['adjacent_clean_mse_delta']
        record['dirty_improved']=bool(dirty!='' and dirty < -TIE_TOLERANCE)
        record['clean_increased']=bool(clean!='' and clean>TIE_TOLERANCE)
        record['adjacent_increased']=bool(adjacent!='' and adjacent>TIE_TOLERANCE)
        record['joint_success']=bool(dirty!='' and clean!='' and adjacent!='' and record['dirty_improved']
                                     and not record['clean_increased'] and not record['adjacent_increased'])
        result.append(record)
    return result


def summary(rows):
    grouped=defaultdict(list)
    for row in rows:
        if row['method']!='input' and float(row['strength'])!=2.:continue
        grouped[row['dataset'],row['kind'],row['method']].append(row)
    output=[]
    for (dataset,kind,method),values in sorted(grouped.items()):
        record=dict(dataset=dataset,kind=kind,method=method,n=len(values))
        for field in ('mse','psnr_db','mae','contaminated_mse','clean_mse','adjacent_clean_mse','runtime_ms',
                'selected_small_fraction','selected_grid_fraction','correct_candidate_coverage','processed_block_coverage'):
            numbers=[float(r[field]) for r in values if r.get(field,'')!='']
            record[field]=float(np.mean(numbers)) if numbers and not (field=='psnr_db' and kind=='clean') else ''
        record['degraded_fraction']=float(np.mean([int(r['degraded']) for r in values]))
        record['tail20_mae']=tail_mean([float(r['mae']) for r in values])
        for field in ('clean_mse','adjacent_clean_mse'):
            values2=[float(r[field]) for r in values if r.get(field,'')!='']
            record['tail20_'+field]=tail_mean(values2) if values2 else ''
        output.append(record)
    return output


def matched_quality(rows):
    groups=defaultdict(lambda:defaultdict(list))
    for row in rows:
        if row['method']!='input' and row.get('contaminated_mse','')!='' and row.get('adjacent_clean_mse','')!='':
            groups[row['dataset'],row['image_id'],row['kind']][row['method']].append(row)
    output=[]
    for (dataset,image_id,kind),methods in groups.items():
        for method in FACTORS:
            if method=='baseline':continue
            record=dict(dataset=dataset,image_id=image_id,kind=kind,method=method,matched=False)
            curves=[];reason=None
            for name in ('baseline',method):
                curve=[(float(r['contaminated_mse']),float(r['adjacent_clean_mse'])) for r in sorted(methods[name],key=lambda r:float(r['strength']))]
                delta=np.diff([p[0] for p in curve])
                if len(curve)<2:reason='insufficient points'
                elif np.any(np.abs(delta)<=TIE_TOLERANCE):reason='duplicate quality levels'
                elif np.any(delta>0) and np.any(delta<0):reason='nonmonotonic recovery'
                curves.append(sorted(curve))
            if reason:record['reason']=reason
            else:
                a,b=curves;low=max(a[0][0],b[0][0]);high=min(a[-1][0],b[-1][0])
                if high-low<=TIE_TOLERANCE:record['reason']='no nondegenerate common quality interval'
                else:
                    target=(low+high)/2
                    ca=float(np.interp(target,[p[0] for p in a],[p[1] for p in a]))
                    cb=float(np.interp(target,[p[0] for p in b],[p[1] for p in b]))
                    record.update(matched=True,target_dirty_mse=target,baseline_adjacent_mse=ca,method_adjacent_mse=cb,adjacent_mse_delta=cb-ca)
            output.append(record)
    return output


def report():
    lines=['# 32像素窗口与空间包络对照','',
        '2×2×2因子对照：分析窗口增加32、复原块32/步长16、3×3非负空间包络。保护配置固定为旧风险耦合版，主表固定强度2。', '',
        '评价完整重叠融合和裁剪之后的含噪区、全部干净区及相邻16像素干净区。逐图差值以同输入baseline为参照；1e-12只用于数值并列判断，不等于普遍安全保证。', '']
    for stage in ('development','regression','check'):
        path=OUT/stage/'metrics.csv'
        if not path.exists():continue
        rows=read_csv(path);aggregated=summary(rows);paired=paired_deltas(rows);matched=matched_quality(rows)
        write_csv(OUT/stage/'summary.csv',aggregated);write_csv(OUT/stage/'paired_deltas.csv',paired)
        if matched:write_csv(OUT/stage/'matched_quality.csv',matched)
        lines += [f'## {stage}','', '| 数据/条件 | 方法 | 图数 | 含噪区MSE | 干净区MSE | 相邻干净区MSE | 最差20% MAE |', '|---|---|---:|---:|---:|---:|---:|']
        for row in aggregated:
            if row['method']=='input':continue
            def fmt(value):return f'{value:.8g}' if value!='' else '—'
            lines.append(f"| {row['dataset']}/{row['kind']} | {row['method']} | {row['n']} | {fmt(row['contaminated_mse'])} | {fmt(row['clean_mse'])} | {fmt(row['adjacent_clean_mse'])} | {row['tail20_mae']:.7g} |")
        lines+=['','### 逐图配对：含噪区改善且两类干净区均不增加损伤','', '| 方法 | 可评价局部输入 | 同时达标 | 干净区损伤增加 | 相邻区损伤增加 | 相近质量可匹配 |', '|---|---:|---:|---:|---:|---:|']
        for method in FACTORS:
            if method=='baseline':continue
            valid=[r for r in paired if r['method']==method and r['contaminated_mse_delta']!=''
                   and r['clean_mse_delta']!='' and r['adjacent_clean_mse_delta']!='']
            mq=[r for r in matched if r['method']==method]
            lines.append(f"| {method} | {len(valid)} | {sum(r['joint_success'] for r in valid)} | {sum(r['clean_increased'] for r in valid)} | {sum(r['adjacent_increased'] for r in valid)} | {sum(r['matched'] for r in mq)}/{len(mq)} |")
        lines+=['']
    lines+=['## 解释边界','',
        'a32只增加分析窗口；s32改变复原块并保持50%重叠率，因此同时改变块数。grid固定相位后拟合非负网格，并与原幅度模型比较惩罚后收益。', '',
        '32窗口要求至少3周期及稳定分层拟合，否则保留大窗口候选；网格上限仍为常幅估计2.5倍，可能限制极窄干扰。模型自由度、候选召回和最终纹理安全不是同一件事。', '',
        'DTD12图与过往4张检查图只作回归，新4图只在冻结后检查一次，不能当作大规模确认。强度点及重叠块不增加独立图像数。', '',
        '旧结果不覆盖：风险评分完整基线在risk-refinement-baseline标签，旧3900条v3主审计在v3-validated-baseline标签。', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(OUT/'RESULTS.md')


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('plan','development','freeze','regression','check','report'))
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.out.resolve()
    if args.stage in ('development','regression','check'):run(args.stage)
    else:globals()[args.stage]()


if __name__=='__main__':main()
