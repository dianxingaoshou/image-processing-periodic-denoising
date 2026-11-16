"""Diagnosis-led narrow-band proposal / gate / support ablations.

Old algorithms and their snapshots are retained unchanged. New candidates and
their support are input-only; truth is used only in post-restoration metrics.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
import time

import numpy as np

import support_experiments as base
import narrow_denoiser as narrow
from narrow_profiles import profiles_from_frequencies
from denoiser_v3 import config_from_dict as legacy_config,prepare as legacy_prepare,restore as legacy_restore
from image_io_v3 import preprocess_image,sha256_file
from reproducibility_v3 import capture_snapshot,verify_snapshot
from run_experiments import metric_row
from support_data import make_sample

ROOT=base.ROOT;DATA=base.DATA;OUT=ROOT/'results/narrow';PLAN=ROOT/'narrow_protocol.json'
NEW_STRENGTHS=(.5,.75,1.);OLD_STRENGTHS=(.5,1.,2.)
METHODS=('legacy64','legacy32_grid','old_peaks_new_support','proposal_only','profile_gate','compact_only','full')
VARIANTS=dict(proposal_only='proposal_only',profile_gate='gate_only',compact_only='support_only',full='full')
SOURCES=tuple(dict.fromkeys((*base.SOURCES,'narrow_experiments.py','narrow_profiles.py','narrow_support.py',
    'narrow_denoiser.py','filter_narrow.py','narrow_protocol.json',*(f'configs/narrow_{name}.json' for name in VARIANTS.values()))))
TOL=1e-12


def load_configs():
    configs={name:narrow.config_from_dict(base.read(ROOT/f'configs/narrow_{path}.json')['configuration']) for name,path in VARIANTS.items()}
    full=configs['full']
    if not full.use_gate or not full.compact_support:
        raise ValueError('Full configuration must enable both gate and compact support')
    expected=dict(full=full,proposal_only=replace(full,use_gate=False,compact_support=False),
                  profile_gate=replace(full,compact_support=False),compact_only=replace(full,use_gate=False))
    if configs!=expected:raise ValueError('Narrow ablations must differ only in gate/support switches')
    return configs


def samples(stage):
    protocol=base.read(PLAN);manifest=base.read(DATA/'v3_manifest.json')
    if sha256_file(DATA/'v3_manifest.json')!=protocol['source_manifest_sha256']:raise ValueError('Input manifest changed')
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
        image,preprocessing=(source if stage=='check' else reference)(record)
        for kind in (('clean',) if stage=='regression' else protocol['kinds']):
            yield record['id'],image,kind,index%4,preprocessing,('DTD_regression' if stage=='regression' else 'BSDS')
    if stage=='regression':
        for index,record in enumerate(protocol['regression_images']):
            image,preprocessing=source(record)
            for kind in ('clean','edge8','edge16','edge32'):
                yield record['id'],image,kind,index%4,preprocessing,'BSDS_regression'


def old_peak_profiles(observed,legacy_estimation,config):
    """Input-only detector control: old 2-D pool, same new fit/gate/support."""
    ranked=sorted((candidate for tile in legacy_estimation.tiles for candidate in tile['candidates']),
        key=lambda c:-float(c.get('selection_score',c['amplitude']**2*c['evidence'])))
    profiles=[]
    for axis in (0,1):
        length=observed.shape[1-axis];frequencies=[]
        for candidate in ranked:
            frequency=abs(candidate['fx'] if axis==0 else candidate['fy'])
            if frequency*(length-1)<config.profiles.minimum_cycles:continue
            if frequency>config.profiles.max_frequency:continue
            if any(abs(frequency-prior)<1.5/length for prior in frequencies):continue
            frequencies.append(frequency)
            if len(frequencies)>=config.profiles.max_candidates:break
        profiles.extend(profiles_from_frequencies(observed,axis,frequencies,config.profiles,refine=True))
    return narrow.Prepared(observed.copy(),config,profiles)


def freeze():
    destination=OUT/'freeze.json'
    if destination.exists():raise FileExistsError('Narrow study already frozen')
    dev=OUT/'development/metrics.csv'
    if not dev.exists():raise FileNotFoundError('Run development first')
    configs={name:narrow.config_dict(c) for name,c in load_configs().items()}
    protocol=base.read(PLAN)
    base.dump(destination,dict(configurations=configs,development_metrics_sha256=sha256_file(dev),
        snapshot=capture_snapshot(SOURCES,[r['id'] for r in protocol['independent_check']],base_dir=ROOT,
            metadata=dict(configurations=configs,decision='specialized opt-in; fixed coarse settings, no test/DTD tuning'))))
    print(destination)


def verify_frozen():
    frozen=base.read(OUT/'freeze.json');protocol=base.read(PLAN)
    verify_snapshot(frozen['snapshot'],paths=SOURCES,audit_ids=[r['id'] for r in protocol['independent_check']],base_dir=ROOT)
    if sha256_file(OUT/'development/metrics.csv')!=frozen['development_metrics_sha256']:raise ValueError('Development metrics changed')
    if frozen['configurations']!=frozen['snapshot']['metadata']['configurations']:raise ValueError('Frozen metadata changed')
    if frozen['configurations']!={n:narrow.config_dict(c) for n,c in load_configs().items()}:raise ValueError('Configuration changed')
    return frozen


def run(stage):
    if stage!='development':verify_frozen()
    destination=OUT/stage
    if (destination/'metrics.csv').exists():raise FileExistsError('Stage exists; use a fresh --out')
    before={p:sha256_file(ROOT/p) for p in SOURCES}
    configs=load_configs();legacy_configs={name:legacy_config(base.read(ROOT/path)['configuration']) for name,path in
        [('legacy64','configs/v3_p0_coupled.json'),('legacy32_grid','configs/support_a32_s32_grid.json')]}
    rows=[];inputs=[];decisions=[]
    for index,(image_id,clean,kind,variant,preprocessing,dataset) in enumerate(samples(stage)):
        observed,metadata,masks=make_sample(clean,image_id,kind,0. if kind=='clean' else .12,variant)
        identity=dict(stage=stage,dataset=dataset,image_id=image_id,kind=kind,variant=variant)
        inputs.append(identity|dict(preprocessing=preprocessing,synthesis=metadata,float_input_sha256=hashlib.sha256(observed.tobytes()).hexdigest()))
        rows.append(identity|dict(method='input',strength=0.,runtime_ms=0.)|metric_row(clean,observed,observed,masks)|base.spatial_metrics(clean,observed,masks))
        old_prepared={};old_times={}
        for method,cfg in legacy_configs.items():
            start=time.perf_counter();prepared=legacy_prepare(observed,cfg);old_times[method]=(time.perf_counter()-start)*1000
            old_prepared[method]=prepared
            for strength in OLD_STRENGTHS:
                start=time.perf_counter();output=legacy_restore(prepared,strength=strength)
                rows.append(identity|dict(method=method,strength=strength,runtime_ms=old_times[method]+(time.perf_counter()-start)*1000)|
                    metric_row(clean,observed,output,masks)|base.spatial_metrics(clean,output,masks))
        start=time.perf_counter();new_prepared=narrow.prepare(observed,configs['full']);new_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter();old_detector=old_peak_profiles(observed,old_prepared['legacy32_grid'].estimation,configs['full'])
        old_detector_ms=(time.perf_counter()-start)*1000+old_times['legacy32_grid']
        for method in ('old_peaks_new_support',*VARIANTS):
            prepared=old_detector if method=='old_peaks_new_support' else new_prepared
            cfg=configs['full'] if method=='old_peaks_new_support' else configs[method]
            elapsed=old_detector_ms if method=='old_peaks_new_support' else new_ms
            for strength in NEW_STRENGTHS:
                start=time.perf_counter();output,diag=narrow.restore(prepared,strength=strength,
                    use_gate=cfg.use_gate,compact_support=cfg.compact_support,return_diagnostics=True)
                predicted_support=diag['support_mask']
                dirty=masks['contaminated'];clean_mask=masks['clean']
                details=dict(profiles=len(prepared.profiles),supported_pixels=int(predicted_support.sum()),
                    dirty_support_fraction=float(predicted_support[dirty].mean()) if dirty.any() else '',
                    clean_support_fraction=float(predicted_support[clean_mask].mean()) if clean_mask.any() else '')
                rows.append(identity|dict(method=method,strength=strength,runtime_ms=elapsed+(time.perf_counter()-start)*1000)|
                    metric_row(clean,observed,output,masks)|base.spatial_metrics(clean,output,masks)|details)
                if strength==1.:
                    for profile in diag['profiles']:
                        decisions.append(identity|dict(method=method,axis=profile['axis'],frequency=profile['frequency_cycles_per_pixel'],
                            supported_rows=profile['supported_rows'],qualified_rows=profile['qualified_rows'],
                            accepted_intervals=sum(i['accepted'] for i in profile['intervals']),
                            interval_reasons=';'.join(i['reason'] for i in profile['intervals'])))
        print(f'{stage}: {index+1} {image_id} {kind}',flush=True)
    if before!={p:sha256_file(ROOT/p) for p in SOURCES}:raise RuntimeError('Study source changed during execution')
    base.write_csv(destination/'metrics.csv',rows);base.write_csv(destination/'profile_decisions.csv',decisions)
    base.dump(destination/'inputs.json',inputs)
    base.dump(destination/'run.json',dict(stage=stage,n_inputs=len(inputs),n_metric_rows=len(rows),source_hashes=before,
        note='Final clipped image metrics; old strengths .5/1/2, new .5/.75/1. Independent checks only after freeze; profile proposals shared among gate/support ablations.'))


def primary_rows(rows):
    return [r for r in rows if r['method']=='input' or float(r['strength'])==(2. if r['method'].startswith('legacy') else 1.)]


def paired_rows(rows):
    primary=primary_rows(rows);lookup={(r['dataset'],r['image_id'],r['kind'],r['method']):r for r in primary}
    paired=[]
    for row in primary:
        if row['method'] in ('input','legacy64','legacy32_grid'):continue
        identity=(row['dataset'],row['image_id'],row['kind'])
        reference=lookup[*identity,'legacy32_grid'];observed=lookup[*identity,'input']
        result={k:row[k] for k in ('dataset','image_id','kind','method')}
        for field in ('mse','contaminated_mse','clean_mse','adjacent_clean_mse'):
            result[field+'_delta']=float(row[field])-float(reference[field]) if row.get(field,'')!='' and reference.get(field,'')!='' else ''
        dirty=result['contaminated_mse_delta'];clean=result['clean_mse_delta'];adjacent=result['adjacent_clean_mse_delta']
        result['relative_joint']=bool(dirty!='' and clean!='' and adjacent!='' and dirty < -TOL and clean<=TOL and adjacent<=TOL)
        result['input_net_joint']=bool(dirty!='' and clean!='' and adjacent!='' and dirty < -TOL and
            float(row['contaminated_mse'])<float(observed['contaminated_mse'])-TOL and float(row['clean_mse'])<=TOL and float(row['adjacent_clean_mse'])<=TOL)
        result['new_clean_damage']=bool(row.get('clean_mse','')!='' and float(row['clean_mse'])>TOL)
        result['new_adjacent_damage']=bool(row.get('adjacent_clean_mse','')!='' and float(row['adjacent_clean_mse'])>TOL)
        paired.append(result)
    return paired


def report():
    lines=['# 8/16像素窄条：诊断后定向修改','',
        '新通道沿条带方向提出1D周期候选，按法向逐行拟合系数，以分半一致性和紧支撑约束决定减除。它不在其他区域运行旧频率残差，是显式专用分支，不能冒称通用方法的单参数升级。', '',
        '旧候选＋新支撑控制使用相同的新拟合／门控／支撑，仅将频率来源换回旧2D池前5个不同切向频率；其他新方案复用同一组1D候选。真实参考与掩膜只进入评价。', '']
    for stage in ('development','regression','check'):
        path=OUT/stage/'metrics.csv'
        if not path.exists():continue
        rows=base.read_csv(path);selected=primary_rows(rows)
        # Reuse the regional/tail aggregation with a common reporting strength.
        normalized=[dict(r,strength=2.) for r in selected]
        summary=base.summary(normalized);paired=paired_rows(rows)
        base.write_csv(OUT/stage/'summary.csv',summary);base.write_csv(OUT/stage/'paired_deltas.csv',paired)
        lines += [f'## {stage}','', '| 数据/条件 | 方法 | 图数 | 含噪区MSE | 全部干净区MSE | 相邻干净区MSE | 最差20% MAE |', '|---|---|---:|---:|---:|---:|---:|']
        for row in summary:
            if row['method']=='input':continue
            def fmt(v):return f'{v:.8g}' if v!='' else '—'
            lines.append(f"| {row['dataset']}/{row['kind']} | {row['method']} | {row['n']} | {fmt(row['contaminated_mse'])} | {fmt(row['clean_mse'])} | {fmt(row['adjacent_clean_mse'])} | {row['tail20_mae']:.7g} |")
        lines+=['','| 方法 | 局部输入 | 相对旧32基线联合改善 | 比输入与基线净改善且两类干净区零新增损伤 | 新干净区损伤 | 新相邻区损伤 |', '|---|---:|---:|---:|---:|---:|']
        for method in METHODS[2:]:
            valid=[r for r in paired if r['method']==method and r['contaminated_mse_delta']!='' and r['adjacent_clean_mse_delta']!='']
            lines.append(f"| {method} | {len(valid)} | {sum(r['relative_joint'] for r in valid)} | {sum(r['input_net_joint'] for r in valid)} | {sum(r['new_clean_damage'] for r in valid)} | {sum(r['new_adjacent_damage'] for r in valid)} |")
        lines+=['']
    lines+=['## 解释边界','',
        '主表：旧基线强度2，新通道强度1；另存固定强度曲线，不视为独立样本。阈值为开发前固定粗参数，不根据DTD或新检查调参。', '',
        '新通道假设沿水平或垂直条带方向近似定频、延伸足够长。分半检验只隔离周期系数；候选频率来自完整输入，不是完全独立交叉验证。局部真实规则纹理仍可能满足全部条件，硬支撑并不证明语义上一定为噪声。', '',
        '诊断里的已知相位、真实频率和孤立块参考增益只用于定位，不参与新方法。旧实验保持原样，默认通用入口不变。', '']
    (OUT/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8');print(OUT/'RESULTS.md')


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('development','freeze','regression','check','report'))
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.out.resolve()
    if args.stage in ('development','regression','check'):run(args.stage)
    else:globals()[args.stage]()


if __name__=='__main__':main()
