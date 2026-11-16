"""Read-only attribution of rejected profile support, using already-seen images.

Reference projections identify the failure mechanism only. The alternative
contrast context below uses observed amplitude and unchanged qualification;
it is a diagnostic counterfactual, not a production implementation.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from image_io_v3 import preprocess_image
from narrow_denoiser import Config,prepare,restore
from narrow_profiles import profiles_from_frequencies
from narrow_support import SupportConfig,build_support,_runs
from run_experiments import metric_row
from support_data import make_sample

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/rejection_diagnosis'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='')as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def context_support(profile,cfg=None):
    """Change only flank context; never enlarge output beyond qualified rows."""
    cfg=cfg or SupportConfig();old=build_support(profile,cfg)
    amplitude=np.asarray(profile['amplitude'])
    amplitude_mask=amplitude>=cfg.min_amplitude
    amplitude_runs=list(_runs(amplitude_mask));mask=np.zeros(len(amplitude),bool);intervals=[]
    for start,stop in _runs(old['qualified']):
        left_bound,right_bound=next((a,b)for a,b in amplitude_runs if a<=start and b>=stop)
        inside=float(np.median(amplitude[start:stop]))
        left=amplitude[max(0,left_bound-cfg.flank_width):left_bound]
        right=amplitude[right_bound:min(len(amplitude),right_bound+cfg.flank_width)]
        flanks=[float(np.median(x))for x in(left,right)if len(x)]
        ratio=max(flanks)/max(inside,1e-15)if flanks else None
        if stop-start<cfg.min_width:reason='too_few_supported_rows'
        elif right_bound-left_bound>cfg.max_width:reason='broad_amplitude_context'
        elif ratio is None:reason='no_observed_flank'
        elif ratio>cfg.max_flank_ratio:reason='insufficient_outside_contrast'
        else:reason='context_supported_region'
        accepted=reason=='context_supported_region'
        if accepted:mask[start:stop]=True
        intervals.append(dict(start=start,stop=stop,context_start=left_bound,context_stop=right_bound,
            context_width=right_bound-left_bound,outside_to_inside_ratio=ratio,
            median_amplitude=inside,accepted=accepted,reason=reason))
    assert not np.any(mask&~old['qualified'])
    return dict(mask=mask,qualified=old['qualified'],intervals=intervals)


def reconstruct_context(prepared):
    raw=np.zeros_like(prepared.image);scores=np.zeros_like(raw);mask_all=np.zeros_like(raw,bool)
    for profile in prepared.profiles:
        support=context_support(profile,prepared.config.support)
        axis=profile['axis'];length=prepared.image.shape[1-axis]
        phase=2*np.pi*profile['frequency']*np.arange(length)
        coefficients=profile['coefficients']
        residual=coefficients[:,0,None]*np.sin(phase)+coefficients[:,1,None]*np.cos(phase)
        score=np.broadcast_to((support['mask']*profile['amplitude']**2)[:,None],residual.shape)
        if axis==1:residual=residual.T;score=score.T
        update=score>scores;raw[update]=residual[update];scores[update]=score[update];mask_all|=score>0
    return np.clip(prepared.image-raw,0,1),mask_all


def records():
    manifest=read(ROOT/'data/v3_manifest.json');protocol=read(ROOT/'narrow_protocol.json')
    images={r['id']:r for r in manifest['images']}
    for index,image_id in enumerate(protocol['development_ids']):
        record=images[image_id];path=ROOT/'data'/record['npy_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record['clean_sha256']
        yield image_id,np.load(path),index%4,'development'
    for index,record in enumerate(protocol['independent_check']):
        source=ROOT/'data'/record['source_jpg']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==record['source_sha256']
        clean,_=preprocess_image(source,size=256)
        yield record['id'],clean,index%4,'previous_check_regression'
        if record['id']=='69022':
            for variant in range(3):yield record['id'],clean,variant,'69022_orientation_diagnostic'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rows=[];cases=[];intervals=[]
    for image_id,clean,variant,split in records():
        for kind in ('edge8','edge16','edge32'):
            observed,metadata,truth=make_sample(clean,image_id,kind,.12,variant)
            prepared=prepare(observed)
            normal_axis=1 if variant<2 else 0
            true_tangent=abs(metadata['fy']if normal_axis==1 else metadata['fx'])
            pool=[p for p in prepared.profiles if p['axis']==normal_axis]
            nearest=min(pool,key=lambda p:abs(p['frequency']-true_tangent))
            frequency=nearest['frequency']
            background=profiles_from_frequencies(clean,normal_axis,[frequency])[0]
            old=build_support(nearest);context=context_support(nearest)
            output,diagnostics=restore(prepared,return_diagnostics=True)
            changed,support_mask=reconstruct_context(prepared)
            identity=dict(image_id=image_id,split=split,kind=kind,variant=variant,axis=normal_axis)
            def masked(z,key):
                mask=truth[key];return float(np.mean((z[mask]-clean[mask])**2))if mask.any()else None
            original_intervals=old['intervals']
            cases.append(identity|dict(true_tangent=true_tangent,estimated_frequency=frequency,
                frequency_error=abs(frequency-true_tangent),clipping_fraction=metadata['clipping_fraction'],
                input_dirty_mse=masked(observed,'contaminated'),old_dirty_mse=masked(output,'contaminated'),
                context_dirty_mse=masked(changed,'contaminated'),old_clean_mse=masked(output,'clean'),
                context_clean_mse=masked(changed,'clean'),old_adjacent_mse=masked(output,'adjacent_clean'),
                context_adjacent_mse=masked(changed,'adjacent_clean'),old_supported_pixels=int(diagnostics['support_mask'].sum()),
                context_supported_pixels=int(support_mask.sum()),qualified_rows=int(old['qualified'].sum()),
                old_supported_rows=int(old['mask'].sum()),context_supported_rows=int(context['mask'].sum()),
                old_reasons=';'.join(i['reason']for i in original_intervals),
                context_reasons=';'.join(i['reason']for i in context['intervals'])))
            for method,result in [('old',old),('amplitude_context',context)]:
                for record in result['intervals']:intervals.append(identity|dict(method=method)|record)
            envelope=truth['envelope']
            normal_envelope=envelope.max(axis=normal_axis^1)
            for index in range(len(nearest['amplitude'])):
                coefficient=nearest['coefficients'][index];bg=background['coefficients'][index]
                added=coefficient-bg
                first,second=nearest['split_coefficients'][:,index]
                # Retain the local tail and any row with strong periodic evidence.
                # This selection is for diagnostic output only, not inference.
                if normal_envelope[index]==0 and nearest['amplitude'][index]<.02:
                    if variant%2==0 and index>=64:continue
                    if variant%2==1 and index<len(nearest['amplitude'])-64:continue
                rows.append(identity|dict(row=index,envelope=float(normal_envelope[index]),
                    coefficient_sin=float(coefficient[0]),coefficient_cos=float(coefficient[1]),
                    first_sin=float(first[0]),first_cos=float(first[1]),second_sin=float(second[0]),second_cos=float(second[1]),
                    amplitude=float(nearest['amplitude'][index]),stability=float(nearest['stability'][index]),
                    validation_gain=float(nearest['validation_gain'][index]),qualified=bool(old['qualified'][index]),
                    old_supported=bool(old['mask'][index]),context_supported=bool(context['mask'][index]),
                    background_amplitude=float(np.linalg.norm(bg)),added_noise_amplitude=float(np.linalg.norm(added)),
                    observed_phase=float(np.arctan2(coefficient[1],coefficient[0])),
                    background_phase=float(np.arctan2(bg[1],bg[0])),added_noise_phase=float(np.arctan2(added[1],added[0]))))
            print(image_id,variant,kind,cases[-1]['old_supported_rows'],cases[-1]['context_supported_rows'],flush=True)
    write_csv(OUT/'case_rows.csv',rows);write_csv(OUT/'case_metrics.csv',cases);write_csv(OUT/'case_intervals.csv',intervals)
    baseline=read(ROOT/'narrow_protocol.json')
    hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()for p in('narrow_profiles.py','narrow_support.py','narrow_denoiser.py','support_data.py','narrow_protocol.json','rejection_69022_audit.py')}
    (OUT/'case_provenance.json').write_text(json.dumps(dict(baseline_commit='79da40c222f590c1d552968627b72b2c5a67a0c0',
        scope='8development +4previouscheck, allalreadyseen; extra69022orientations; no unusedcheckpixels',
        hashes=hashes,truth_use='closestfrequency/backgroundprojection/envelope only for diagnosis; counterfactualsupport uses observed profiles only',
        context_constraint='outputmask is a subset of originalqualifiedrows; context must be one amplitude-connected interval of width<=32'),indent=2),encoding='utf-8')
    lines=['# 69022拒绝处理：支撑与资格掩膜混用','',
        '先保留原版源码与固定参数，只读诊断已使用的开发图和上一轮检查图。参考背景投影仅解释机制，不进入下面输入侧反事实。','',
        '69022/variant3的edge8中，原资格通过区间251:255，稳定性未通过的row255仍有幅度0.14348；原实现把该端点当作右侧外部邻带，得到外/内比1.4448而拒绝。该行参考背景幅度仅0.0308，新增干扰投影约0.1127，主要仍属于条纹，不是外部背景。','',
        'edge16同理，250:255的右邻row255导致比值1.3126；其稳定性0.773低于0.8。资格掩膜的断裂与真实空间包络边界不是同一件事。','',
        '输入侧反事实：仍按原幅度/稳定性/交叉收益形成资格掩膜，输出只允许原资格行；但是邻带从其所在幅度连通区外采样。幅度连通上下文仍受32行宽度上限限制，不填资格孔、不扩张输出、不降低0.8或0.08阈值。','',
        '| 69022/variant3 | 原正确候选处理行 | 上下文反事实处理行 | 输入含噪MSE | 反事实含噪MSE | 反事实干净MSE | 反事实相邻MSE |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for c in cases:
        if c['image_id']=='69022'and c['variant']==3:
            lines.append(f"| {c['kind']} | {c['old_supported_rows']} | {c['context_supported_rows']} | {c['input_dirty_mse']:.8g} | {c['context_dirty_mse']:.8g} | {c['context_clean_mse']:.8g} | {c['context_adjacent_mse']:.8g} |")
    lines+=['','这个输入侧规则只是诊断方案，仍须由主代理在开发/纹理/新检查独立对照；不能因为69022正例就宣称适用于所有纹理。']
    (OUT/'case_FINDINGS.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
