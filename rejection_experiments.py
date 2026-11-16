"""Qualified output versus amplitude context: rejection-case controlled study."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
import time

import numpy as np

import narrow_experiments as previous
import narrow_denoiser as narrow
from context_controls import make_controls
from image_io_v3 import preprocess_image,sha256_file
from reproducibility_v3 import capture_snapshot,verify_snapshot
from run_experiments import metric_row
from support_data import make_sample

base=previous.base;ROOT=base.ROOT;DATA=base.DATA
OUT=ROOT/'results/context';PLAN=ROOT/'rejection_protocol.json'
METHODS=('baseline','ratio_relaxed','context_no_width','context_full');STRENGTHS=(.5,.75,1.)
SOURCES=tuple(dict.fromkeys((*previous.SOURCES,'rejection_experiments.py','rejection_protocol.json',
    'narrow_context.py','context_controls.py',*(f'configs/context_{name}.json' for name in METHODS))))
TOL=1e-12


def load_configs():
    configs={name:narrow.config_from_dict(base.read(ROOT/f'configs/context_{name}.json')['configuration']) for name in METHODS}
    baseline=configs['baseline']
    if (baseline.support.contrast_context!='qualified' or not baseline.use_gate or not baseline.compact_support
            or baseline.strength!=1. or baseline.support.max_flank_ratio!=.35):
        raise ValueError('The baseline must preserve strict original gates and strength1')
    expected=dict(baseline=baseline,
        ratio_relaxed=replace(baseline,support=replace(baseline.support,max_flank_ratio=.9)),
        context_no_width=replace(baseline,support=replace(baseline.support,contrast_context='amplitude',limit_context_width=False)),
        context_full=replace(baseline,support=replace(baseline.support,contrast_context='amplitude',limit_context_width=True)))
    if configs!=expected:raise ValueError('Context ablations changed an unintended setting')
    return configs


def samples(stage):
    if stage=='controls':
        for image_id,clean,observed,masks,metadata in make_controls():
            yield image_id,clean,observed,masks,metadata,dict(source='fixed analytic controls'),'analytic',0
        return
    protocol=base.read(PLAN);manifest=base.read(DATA/'v3_manifest.json')
    if sha256_file(DATA/'v3_manifest.json')!=protocol['source_manifest_sha256']:raise ValueError('Manifest changed')
    images={r['id']:r for r in manifest['images']+manifest['textures']}
    def reference(record):
        path=DATA/record['npy_path']
        if sha256_file(path)!=record['clean_sha256']:raise ValueError('Reference NPY changed')
        return np.load(path),dict(reference=record['npy_path'],sha256=record['clean_sha256'])
    def source(record):
        path=DATA/record['source_jpg']
        if sha256_file(path)!=record['source_sha256']:raise ValueError('Source JPEG changed')
        return preprocess_image(path,size=256)
    if stage=='development':records=[images[i] for i in protocol['development_ids']]
    elif stage=='check':records=protocol['independent_check']
    elif stage=='regression':records=[images[i] for i in protocol['regression_textures']]
    else:raise ValueError('Unknown stage')
    for index,record in enumerate(records):
        clean,prep=(source if stage=='check' else reference)(record)
        for kind in (('clean',) if stage=='regression' else protocol['kinds']):
            observed,metadata,masks=make_sample(clean,record['id'],kind,0. if kind=='clean' else .12,index%4)
            yield record['id'],clean,observed,masks,metadata,prep,('DTD_regression' if stage=='regression' else 'BSDS'),index%4
    if stage=='regression':
        for index,record in enumerate(protocol['regression_images']):
            clean,prep=source(record)
            for kind in ('clean','edge8','edge16','edge32'):
                observed,metadata,masks=make_sample(clean,record['id'],kind,0. if kind=='clean' else .12,index%4)
                yield record['id'],clean,observed,masks,metadata,prep,'BSDS_regression',index%4


def freeze():
    destination=OUT/'freeze.json'
    if destination.exists():raise FileExistsError('Study already frozen')
    configs={name:narrow.config_dict(cfg) for name,cfg in load_configs().items()}
    dev=OUT/'development/metrics.csv';controls=OUT/'controls/metrics.csv'
    if not dev.exists() or not controls.exists():raise FileNotFoundError('Run development and fixed controls first')
    base.dump(destination,dict(configurations=configs,development_metrics_sha256=sha256_file(dev),
        control_metrics_sha256=sha256_file(controls),snapshot=capture_snapshot(SOURCES,
        [r['id'] for r in base.read(PLAN)['independent_check']],base_dir=ROOT,
        metadata=dict(configurations=configs,decision='fixed context correction; no gate relaxation in full; no test-set tuning'))))
    print(destination)


def verify_frozen():
    frozen=base.read(OUT/'freeze.json')
    verify_snapshot(frozen['snapshot'],paths=SOURCES,audit_ids=[r['id'] for r in base.read(PLAN)['independent_check']],base_dir=ROOT)
    for key,path in (('development_metrics_sha256','development/metrics.csv'),('control_metrics_sha256','controls/metrics.csv')):
        if sha256_file(OUT/path)!=frozen[key]:raise ValueError('Pre-freeze metrics changed')
    if frozen['configurations']!=frozen['snapshot']['metadata']['configurations']:raise ValueError('Sealed metadata changed')
    if frozen['configurations']!={name:narrow.config_dict(cfg) for name,cfg in load_configs().items()}:
        raise ValueError('Current configuration differs from freeze')
    return frozen


def run(stage):
    if stage in ('regression','check'):verify_frozen()
    destination=OUT/stage
    if (destination/'metrics.csv').exists():raise FileExistsError('Stage exists; choose a fresh --out')
    before={p:sha256_file(ROOT/p) for p in SOURCES};configs=load_configs()
    rows=[];inputs=[];intervals=[]
    for index,(image_id,clean,observed,masks,metadata,prep,dataset,variant) in enumerate(samples(stage)):
        kind=metadata.get('kind',image_id)
        identity=dict(stage=stage,dataset=dataset,image_id=image_id,kind=kind,variant=variant)
        inputs.append(identity|dict(preprocessing=prep,synthesis=metadata,float_input_sha256=hashlib.sha256(observed.tobytes()).hexdigest()))
        rows.append(identity|dict(method='input',strength=0.,runtime_ms=0.)|
            metric_row(clean,observed,observed,masks)|base.spatial_metrics(clean,observed,masks))
        start=time.perf_counter();prepared=narrow.prepare(observed,configs['baseline']);prepare_ms=(time.perf_counter()-start)*1000
        baseline_output,baseline_diag=narrow.restore(prepared,return_diagnostics=True)
        baseline_support=baseline_diag['support_mask']
        for method,cfg in configs.items():
            p=narrow.Prepared(prepared.image,cfg,prepared.profiles)
            for strength in STRENGTHS:
                start=time.perf_counter();output,diag=narrow.restore(p,strength=strength,return_diagnostics=True)
                added=diag['support_mask']&~baseline_support
                removed=baseline_support&~diag['support_mask']
                new_clean=added&masks['clean'];new_adjacent=added&masks['adjacent_clean']
                details=dict(supported_pixels=int(diag['support_mask'].sum()),added_support_pixels=int(added.sum()),
                    removed_support_pixels=int(removed.sum()),added_clean_pixels=int(new_clean.sum()),added_adjacent_pixels=int(new_adjacent.sum()),
                    added_clean_mse=float(np.mean((output[new_clean]-clean[new_clean])**2)) if new_clean.any() else '',
                    added_adjacent_mse=float(np.mean((output[new_adjacent]-clean[new_adjacent])**2)) if new_adjacent.any() else '')
                rows.append(identity|dict(method=method,strength=strength,runtime_ms=prepare_ms+(time.perf_counter()-start)*1000)|
                    metric_row(clean,observed,output,masks)|base.spatial_metrics(clean,output,masks)|details)
                if strength==1.:
                    for profile in diag['profiles']:
                        for interval in profile['intervals']:
                            intervals.append(identity|dict(method=method,axis=profile['axis'],frequency=profile['frequency_cycles_per_pixel'])|interval)
        print(f'{stage}: {index+1} {image_id} {kind}',flush=True)
    if before!={p:sha256_file(ROOT/p) for p in SOURCES}:raise RuntimeError('Sources changed during inference')
    base.write_csv(destination/'metrics.csv',rows)
    if intervals:base.write_csv(destination/'intervals.csv',intervals)
    base.dump(destination/'inputs.json',inputs)
    base.dump(destination/'run.json',dict(stage=stage,n_inputs=len(inputs),n_metric_rows=len(rows),source_hashes=before,
        note='Identical profile pool and coefficients for every method. Output remains strict qualified runs; only contrast context / width control / ratio ablation varies.'))


def primary_rows(rows):return [r for r in rows if r['method']=='input' or float(r['strength'])==1.]


def paired_rows(rows):
    selected=primary_rows(rows);lookup={(r['dataset'],r['image_id'],r['kind'],r['method']):r for r in selected}
    pairs=[]
    for row in selected:
        if row['method'] in ('input','baseline'):continue
        identity=(row['dataset'],row['image_id'],row['kind'])
        baseline=lookup[*identity,'baseline'];observed=lookup[*identity,'input']
        result={k:row[k] for k in ('dataset','image_id','kind','method')}
        for field in ('mse','contaminated_mse','clean_mse','adjacent_clean_mse'):
            result[field+'_delta']=float(row[field])-float(baseline[field]) if row.get(field,'')!='' and baseline.get(field,'')!='' else ''
        available=all(row.get(k,'')!='' and baseline.get(k,'')!='' and observed.get(k,'')!='' for k in
                      ('contaminated_mse','clean_mse','adjacent_clean_mse'))
        if available:
            dirty=float(row['contaminated_mse']);old_dirty=float(baseline['contaminated_mse']);input_dirty=float(observed['contaminated_mse'])
            safe=float(row['clean_mse'])<=TOL and float(row['adjacent_clean_mse'])<=TOL
            old_safe=float(baseline['clean_mse'])<=TOL and float(baseline['adjacent_clean_mse'])<=TOL
            old_success=old_dirty<input_dirty-TOL and old_safe
            new_success=dirty<input_dirty-TOL and safe
            joint=dirty<min(old_dirty,input_dirty)-TOL and safe
            rejected=old_dirty>=input_dirty-TOL
            regressed=old_success and (dirty>old_dirty+TOL or float(row['clean_mse'])>float(baseline['clean_mse'])+TOL or
                                      float(row['adjacent_clean_mse'])>float(baseline['adjacent_clean_mse'])+TOL)
        else:old_success=new_success=joint=rejected=regressed=False
        result.update(local_evaluable=available,input_net_joint=bool(joint),new_recovery=bool(joint and rejected),
            old_success=bool(old_success),old_success_lost=bool(old_success and not new_success),old_success_regressed=bool(regressed),
            clean_increased=bool(result['clean_mse_delta']!='' and result['clean_mse_delta']>TOL),
            adjacent_increased=bool(result['adjacent_clean_mse_delta']!='' and result['adjacent_clean_mse_delta']>TOL))
        pairs.append(result)
    return pairs


def report():
    lines=['# 拒绝案例：对比上下文与减除资格分离','',
        '保持候选、系数、0.02幅度／0.8稳定性／0.08收益门槛和至少3个连续资格行不变，仅修改邻带取样上下文。完整方案不填资格孔、不合并输出段、不放宽0.35比值；还约束幅度上下文宽度≤32。', '',
        'ratio_relaxed只将比值放宽至0.9，用作负面对照；context_no_width用于检查缺少宽幅度区域保护的后果。所有主表强度1，方法共用同一候选池与系数。', '']
    for stage in ('development','controls','regression','check'):
        path=OUT/stage/'metrics.csv'
        if not path.exists():continue
        rows=base.read_csv(path);selected=primary_rows(rows)
        summary=base.summary([dict(r,strength=2.) for r in selected]);pairs=paired_rows(rows)
        base.write_csv(OUT/stage/'summary.csv',summary);base.write_csv(OUT/stage/'paired_deltas.csv',pairs)
        lines += [f'## {stage}','', '| 数据/条件 | 方法 | 图数 | 含噪区MSE | 全部干净区MSE | 相邻干净区MSE | 最差20% MAE |', '|---|---|---:|---:|---:|---:|---:|']
        def fmt(value):return f'{value:.8g}' if value!='' else '—'
        for row in summary:
            if row['method']=='input':continue
            lines.append(f"| {row['dataset']}/{row['kind']} | {row['method']} | {row['n']} | {fmt(row['contaminated_mse'])} | {fmt(row['clean_mse'])} | {fmt(row['adjacent_clean_mse'])} | {row['tail20_mae']:.7g} |")
        lines += ['', '| 方法 | 局部输入 | 严格净改善 | 新恢复旧拒绝 | 旧成功退步/丢失 | 新增干净/相邻损伤 |', '|---|---:|---:|---:|---:|---:|']
        for method in METHODS[1:]:
            valid=[p for p in pairs if p['method']==method and p['local_evaluable']]
            lines.append(f"| {method} | {len(valid)} | {sum(p['input_net_joint'] for p in valid)} | {sum(p['new_recovery'] for p in valid)} | {sum(p['old_success_regressed'] for p in valid)}/{sum(p['old_success_lost'] for p in valid)} | {sum(p['clean_increased'] for p in valid)}/{sum(p['adjacent_increased'] for p in valid)} |")
        lines+=['']
    lines+=['## 边界','',
        '69022及上一轮所有图均为已用诊断/回归材料，不作为未参与设计的确认集。新4图在冻结后检查一次；控制例包括合法紧条纹的不可辨识反例，不能从零自然图误改推广为普遍纹理安全。', '',
        '上下文只用于邻带对比，严格资格失败行始终不处理。因此可能仍残留条纹，证据不足时继续拒绝。宽背景或频率偏差造成的半区不一致未通过降低门槛掩盖。', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8');print(OUT/'RESULTS.md')


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('development','controls','freeze','regression','check','report'))
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.out.resolve()
    if args.stage in ('development','controls','regression','check'):run(args.stage)
    else:globals()[args.stage]()


if __name__=='__main__':main()
