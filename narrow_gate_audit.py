"""Reference-only gate attribution on already-used development images.

Counterfactual gains reuse the same estimated residuals; no truth enters the
denoiser. Overlap masses are diagnostic tile weights, not independent pixels.
"""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from denoiser_v3 import config_from_dict, prepare, restore, effective_gains
from gain_diagnostics import candidate_gain_rows
from local_estimation_v3 import _geometry, restore as reconstruct
from support_data import make_sample

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/narrow_diagnosis'


def distance(c,fy,fx):
    return min(float(np.hypot(c['fy']-fy,c['fx']-fx)),float(np.hypot(c['fy']+fy,c['fx']+fx)))


def write_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def metrics(image,reference,masks):
    error=(image-reference)**2
    def mean(key):
        mask=masks[key]
        return float(error[mask].mean()) if np.any(mask) else None
    return dict(mse=float(error.mean()),dirty_mse=mean('contaminated'),clean_mse=mean('clean'),adjacent_mse=mean('adjacent_clean'))


def classify(tile,decision,pooled_match,selected_match,gain):
    # Continuous fitting can occasionally recover a frequency just outside
    # the pool-match tolerance. Such a tile proceeds to gate attribution.
    if not pooled_match and not selected_match:return 'no_correct_candidate'
    if not selected_match:return 'wrong_selected_or_fitted_frequency'
    if str(tile.get('model','')).startswith('rejected'):return 'frequency_rejected'
    if not decision.get('candidate_gate',False):return 'candidate_gate_rejected'
    if decision.get('reliability_weight',0)<=1e-12:return 'reliability_zero'
    if decision.get('amplitude_gate',0)<=1e-12:return 'amplitude_gate_zero'
    if decision.get('extrapolation_gate',0)<=1e-12:return 'extrapolation_gate_zero'
    if decision.get('safety_weight',0)<=1e-12:return 'safety_zero'
    if gain<.05:return 'positive_but_gain_below_005'
    return 'processed_correct_candidate'


def save_outputs(tile_rows,image_rows):
    stage_masses=defaultdict(float);kind_masses=defaultdict(lambda:defaultdict(float));case_masses=defaultdict(lambda:defaultdict(float))
    pool_miss=defaultdict(float);kind_total=defaultdict(float)
    for row in tile_rows:
        mass=float(row['dirty_hann_mass']);reason=row['stage'];kind=row['kind']
        stage_masses[reason]+=mass;kind_masses[kind][reason]+=mass;case_masses[row['image_id'],kind][reason]+=mass
        kind_total[kind]+=mass
        if row['correct_pool_candidate'] in (False,'False'):pool_miss[kind]+=mass
    for row in image_rows:
        for key in list(row):
            if key.endswith('_mass_fraction'):row.pop(key)
        masses=case_masses[row['image_id'],row['kind']];total=sum(masses.values())
        row.update({reason+'_mass_fraction':mass/total for reason,mass in masses.items()})
    write_csv(OUT/'gate_tiles.csv',tile_rows);write_csv(OUT/'gate_images.csv',image_rows)
    total=sum(stage_masses.values())
    summary=dict(aggregate_mass_fractions={key:value/total for key,value in stage_masses.items()},
        by_kind={kind:{key:value/sum(masses.values()) for key,value in masses.items()} for kind,masses in kind_masses.items()},
        raw_pool_miss_mass_fraction={kind:pool_miss[kind]/total for kind,total in kind_total.items()},
        interpretation='Already-used dev8, amplitude .12, supportall32/grid. All dirty-intersecting tiles included. Fractions weighted by Hann mass of dirty overlap; correlated and not pixel recall. Counterfactuals reuse existing residuals; local oracle is not a full-image optimum. Frequency recovered by continuous fitting proceeds to gate attribution even if raw pool missed.')
    (OUT/'gate_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# 门控逐层定位（开发集，只读诊断）','',summary['interpretation'],'',
        '| 条件 | 模式 | 平均含噪区MSE | 平均相邻干净区MSE |','|---|---|---:|---:|']
    modes=('input','normal','bypass_candidate_gate','all_gains_one','local_oracle_existing_residuals')
    for kind in ('edge8','edge16','edge32'):
        for mode in modes:
            rows=[r for r in image_rows if r['kind']==kind and r['mode']==mode]
            lines.append(f"| {kind} | {mode} | {np.mean([float(r['dirty_mse']) for r in rows]):.9g} | {np.mean([float(r['adjacent_mse']) for r in rows]):.9g} |")
    lines+=['','## 主要互斥失败阶段（按含噪重叠Hann质量加权）','',
        '| 条件 | 阶段 | 质量占比 |','|---|---|---:|']
    for kind,masses in summary['by_kind'].items():
        for reason,fraction in sorted(masses.items(),key=lambda v:-v[1]):lines.append(f'| {kind} | {reason} | {fraction:.2%} |')
    lines+=['','## 解释','',
        '8像素主要缺少正确候选；放开门控对同一组残差反而同时增加含噪区与邻近损伤。16像素存在正确候选被资格门控拒绝，但直接解除门控会带来明显相邻区损伤；不能据此全局降低阈值。', '',
        '被门控拒绝的16像素正确频率候选，峰值分数中位数约3.89，低于4.5；频谱证据约0.714、邻块一致性约0.953，但跨窗支撑中位数为0，无法进入要求跨窗>=0.8的回退。', '',
        '同残差的孤立块参考最优增益仍不能完全恢复8像素条纹，说明仅改门控不足。参考增益包含真值，仅用于离线诊断；不作为算法输入。']
    (OUT/'gate_findings.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def summarize_saved():
    with (OUT/'gate_tiles.csv').open(encoding='utf-8-sig',newline='') as stream:tile_rows=list(csv.DictReader(stream))
    with (OUT/'gate_images.csv').open(encoding='utf-8-sig',newline='') as stream:image_rows=list(csv.DictReader(stream))
    for row in tile_rows:
        decision={key:(row[key]=='True' if key=='candidate_gate' else float(row[key]) if row[key] else 0.)
                  for key in ('candidate_gate','reliability_weight','amplitude_gate','extrapolation_gate','safety_weight')}
        row['stage']=classify(row,decision,row['correct_pool_candidate']=='True',row['selected_match']=='True',float(row['effective_gain']))
    save_outputs(tile_rows,image_rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=config_from_dict(json.loads((ROOT/'configs/support_a32_s32_grid.json').read_text(encoding='utf-8'))['configuration'])
    protocol=json.loads((ROOT/'support_protocol.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'data/v3_manifest.json').read_text(encoding='utf-8'))
    by_id={r['id']:r for r in manifest['images']}
    tile_rows=[];image_rows=[];stage_masses=defaultdict(float);kind_masses=defaultdict(lambda:defaultdict(float))
    for index,image_id in enumerate(protocol['development_ids']):
        record=by_id[image_id];reference=np.load(ROOT/'data'/record['npy_path'])
        for kind in ('edge8','edge16','edge32'):
            observed,meta,masks=make_sample(reference,image_id,kind,.12,index%4)
            p=prepare(observed,cfg)
            gains=effective_gains(p.fit_weights,p.safety_weights,cfg.strength,cfg.combination)
            fy,fx=meta['fy'],meta['fx']
            masses=defaultdict(float)
            for tile_index,(tile,decision,gain) in enumerate(zip(p.estimation.tiles,p.decisions,gains)):
                y,x,h,w=(tile[key] for key in ('y','x','h','w'));sl=np.s_[y:y+h,x:x+w]
                dirty=masks['contaminated'][sl];n_dirty=int(dirty.sum())
                if not n_dirty:continue
                weight=_geometry((h,w))[4]
                mass=float(weight[dirty].sum())
                pool=tile.get('candidates',[]);selected=tile.get('selected')
                pooled_match=any(distance(c,fy,fx)<=.01 for c in pool)
                selected_error=distance(selected,fy,fx) if selected else None
                selected_match=selected_error is not None and selected_error<=.01
                reason=classify(tile,decision,pooled_match,selected_match,float(gain))
                masses[reason]+=mass;stage_masses[reason]+=mass;kind_masses[kind][reason]+=mass
                row=dict(image_id=image_id,kind=kind,tile_index=tile_index,y=y,x=x,dirty_pixels=n_dirty,dirty_hann_mass=mass,
                    correct_pool_candidate=pooled_match,selected_match=selected_match,selected_error=selected_error,
                    stage=reason,effective_gain=float(gain),fit_weight=float(p.fit_weights[tile_index]),
                    selected_scale=selected['scale'] if selected else None,model=tile.get('model'),
                    selected_peak_score=selected['score'] if selected else None,evidence=selected['evidence'] if selected else None,
                    neighbor=tile['neighbor'],cross_scale=tile['cross_scale'],amplitude=tile['amplitude'])
                for key in ('candidate_gate','support_reason','reliability','reliability_weight','amplitude_gate',
                        'extrapolation_gate','safety_weight','unweighted_improvement','fit_improvement','texture_risk'):
                    row[key]=decision.get(key)
                tile_rows.append(row)
            raw_q=np.array([decision.get('fit_weight',0.) for decision in p.decisions])
            bypass_gate=effective_gains(raw_q,p.safety_weights,cfg.strength,cfg.combination)
            oracle=candidate_gain_rows(observed,reference,p)
            oracle_gains=np.array([r['oracle_gain'] or 0. for r in oracle])
            outputs=dict(input=observed,normal=restore(p),
                bypass_candidate_gate=reconstruct(p.estimation,weights=bypass_gate,strength=1.),
                all_gains_one=restore(p,conservative=False),
                local_oracle_existing_residuals=reconstruct(p.estimation,weights=oracle_gains,strength=1.))
            total=sum(masses.values())
            for mode,output in outputs.items():
                row=dict(image_id=image_id,kind=kind,mode=mode,dirty_tile_mass=total,**metrics(output,reference,masks))
                row.update({reason+'_mass_fraction':mass/total for reason,mass in masses.items()})
                image_rows.append(row)
            print(f'{image_id} {kind}: '+', '.join(f'{k}={v/total:.3f}'for k,v in masses.items()),flush=True)
    save_outputs(tile_rows,image_rows)


if __name__=='__main__':
    import sys
    if '--summarize-only' in sys.argv:summarize_saved()
    else:main()
