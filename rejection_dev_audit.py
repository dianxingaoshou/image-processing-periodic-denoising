"""Read-only attribution of already-used development narrow-band rejections.

Clean/noise coefficient decompositions are evaluation-only. The production
profile extractor and support gate receive only the observed image.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from narrow_profiles import _fit,propose_profiles
from narrow_support import build_support,_runs
from support_data import make_sample

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/rejection_diagnosis'
IDS={'140055','145053','151087','181079'}


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def fit_parts(rows,frequency):
    length=rows.shape[1];positions=np.arange(length)
    full,_=_fit(rows,frequency,positions,length)
    first,_=_fit(rows,frequency,positions[:length//2],length)
    second,_=_fit(rows,frequency,positions[length//2:],length)
    return full[:,:2],np.stack([first[:,:2],second[:,:2]])


def coefficient_stability(split):
    mass=np.linalg.norm(split[0],axis=1)+np.linalg.norm(split[1],axis=1)
    return np.clip(1-np.linalg.norm(split[0]-split[1],axis=1)/np.maximum(mass,1e-15),0,1)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'data/v3_manifest.json').read_text(encoding='utf-8'))
    dev=[r for r in manifest['images'] if r['split']=='dev'][:8]
    details=[];summary=[];interval_rows=[]
    for index,record in enumerate(dev):
        if record['id'] not in IDS:continue
        source=ROOT/'data'/record['npy_path']
        assert hashlib.sha256(source.read_bytes()).hexdigest()==record['clean_sha256']
        clean=np.load(source);axis=1 if index%4<2 else 0
        for kind in ('edge8','edge16'):
            observed,metadata,masks=make_sample(clean,record['id'],kind,.12,index%4)
            truth_frequency=abs(metadata['fy'] if axis==1 else metadata['fx'])
            candidates=propose_profiles(observed,axis)
            profile=min(candidates,key=lambda c:abs(c['frequency']-truth_frequency))
            gate=build_support(profile)
            normal=lambda a:a if axis==0 else a.T
            observed_rows=normal(observed);clean_rows=normal(clean);noise_rows=observed_rows-clean_rows
            reference_coef,reference_split=fit_parts(clean_rows,profile['frequency'])
            noise_coef,noise_split=fit_parts(noise_rows,profile['frequency'])
            _,noise_true_split=fit_parts(noise_rows,truth_frequency)
            dirty=np.any(normal(masks['contaminated']),axis=1)
            clean_stability=coefficient_stability(reference_split);noise_stability=coefficient_stability(noise_split)
            signal_amplitude=np.linalg.norm(noise_coef,axis=1)
            bg_amplitude=np.linalg.norm(reference_coef,axis=1)
            bg_split_delta=np.linalg.norm(reference_split[0]-reference_split[1],axis=1)
            noise_split_delta=np.linalg.norm(noise_split[0]-noise_split[1],axis=1)
            amplitude_support=profile['amplitude']>=.02
            for row in range(len(dirty)):
                details.append(dict(image_id=record['id'],kind=kind,axis=axis,row=row,
                    true_dirty=bool(dirty[row]),observed_amplitude=float(profile['amplitude'][row]),
                    observed_stability=float(profile['stability'][row]),observed_cross_gain=float(profile['validation_gain'][row]),
                    qualified=bool(gate['qualified'][row]),supported=bool(gate['mask'][row]),
                    reference_amplitude=float(bg_amplitude[row]),actual_added_noise_amplitude=float(signal_amplitude[row]),
                    reference_half_delta=float(bg_split_delta[row]),noise_half_delta=float(noise_split_delta[row]),
                    reference_stability=float(clean_stability[row]),noise_stability=float(noise_stability[row])))
            for interval in gate['intervals']:
                left=np.arange(max(0,interval['start']-16),interval['start'])
                right=np.arange(interval['stop'],min(len(dirty),interval['stop']+16))
                flank=np.r_[left,right]
                interval_rows.append(dict(image_id=record['id'],kind=kind,proposal='qualified_run',
                    start=interval['start'],stop=interval['stop'],width=interval['width'],
                    stability_median=interval['median_stability'],cross_gain_median=interval['median_validation_gain'],
                    flank_ratio=interval['outside_to_inside_ratio'],reason=interval['reason'],
                    true_dirty_inside=int(dirty[interval['start']:interval['stop']].sum()),
                    true_dirty_in_measured_flank=int(dirty[flank].sum()),
                    measured_flank_rows=len(flank)))
            for start,stop in _runs(amplitude_support):
                if not dirty[start:stop].any():continue
                inside=profile['amplitude'][start:stop]
                sides=[profile['amplitude'][max(0,start-16):start],profile['amplitude'][stop:min(len(dirty),stop+16)]]
                sides=[side for side in sides if len(side)]
                contrast=max(float(np.median(side)) for side in sides)/float(np.median(inside)) if sides else None
                interval_rows.append(dict(image_id=record['id'],kind=kind,proposal='amplitude_run_diagnostic',
                    start=start,stop=stop,width=stop-start,
                    stability_median=float(np.median(profile['stability'][start:stop])),
                    cross_gain_median=float(np.median(profile['validation_gain'][start:stop])),
                    flank_ratio=contrast,reason='not_a_production_decision',true_dirty_inside=int(dirty[start:stop].sum()),
                    true_dirty_in_measured_flank=int(dirty[np.r_[np.arange(max(0,start-16),start),np.arange(stop,min(len(dirty),stop+16))]].sum()),
                    measured_flank_rows=sum(map(len,sides))))
            meaningful=dirty&(signal_amplitude>=.02)
            identity=dict(image_id=record['id'],kind=kind,axis=axis,frequency_error=abs(profile['frequency']-truth_frequency))
            summary.append(identity|dict(dirty_rows=int(dirty.sum()),meaningful_noise_rows=int(meaningful.sum()),
                qualified_dirty_rows=int((gate['qualified']&dirty).sum()),supported_rows=gate['supported_rows'],
                median_observed_stability=float(np.median(profile['stability'][meaningful])),
                median_noise_only_stability=float(np.median(noise_stability[meaningful])),
                median_noise_true_frequency_stability=float(np.median(coefficient_stability(noise_true_split)[meaningful])),
                median_observed_cross_gain=float(np.median(profile['validation_gain'][meaningful])),
                median_reference_amplitude=float(np.median(bg_amplitude[meaningful])),
                median_actual_noise_amplitude=float(np.median(signal_amplitude[meaningful])),
                median_reference_half_delta=float(np.median(bg_split_delta[meaningful])),
                median_noise_half_delta=float(np.median(noise_split_delta[meaningful])),
                coefficient_decomposition_max_error=float(np.max(np.abs(profile['coefficients']-reference_coef-noise_coef))),
                synthesis_clipping_fraction=float(metadata.get('clipping_fraction',0.))))
    write_csv(OUT/'dev_rows.csv',details);write_csv(OUT/'dev_intervals.csv',interval_rows);write_csv(OUT/'dev_summary.csv',summary)
    findings=['# 已用开发集拒绝原因：逐行门控与支撑边界耦合','',
        '仅140055、145053、151087、181079原开发图，edge8/16，均保持原参数。真实频率仅选取对应诊断候选；干净图和实际新增噪声的系数分解只用于离线归因。', '',
        '| 图/条件 | 有效新增噪声行 | 通过逐行资格的含噪行 | 输出支撑行 | 观测稳定性中位数 | 噪声在估计频率稳定性 | 噪声在真实频率稳定性 | 背景两半系数差 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in summary:
        findings.append(f"| {row['image_id']}/{row['kind']} | {row['meaningful_noise_rows']} | {row['qualified_dirty_rows']} | {row['supported_rows']} | {row['median_observed_stability']:.3f} | {row['median_noise_only_stability']:.3f} | {row['median_noise_true_frequency_stability']:.3f} | {row['median_reference_half_delta']:.4f} |")
    findings+=['','## 已确认的机制','',
        '所有8个拒绝输入均有接近真实切向频率的候选，但通过稳定性和交叉拟合收益的行呈碎片化。现有程序先逐行门控，再用资格行形成支撑，因此邻带可能仍位于真实条纹内部。', '',
        '140055/edge16资格区间3..8，左邻带0..2本身仍被加噪，导致外侧/内部幅度比约1.01；远离真实干扰的右侧中位幅度约0.016。145053/edge8只通过第254行，右侧第255行仍为真实条纹，邻带比约0.952；edge16同样约0.955。', '',
        '181079的edge8资格行252、254、255，edge16为250、252、254、255，连续长度不足3。151087在正常输入中几乎没有连续稳定行，属于更严重的背景系数干扰/低交叉收益，不能只通过合并资格段恢复。', '',
        '已知新增噪声的分半系数通常远稳定于观测系数；两半差异主要由自然场景背景投影到同一频率造成。因而低稳定性不总表示新增条纹本身不稳定，但输入侧无法直接获得这个分解。', '',
        '151087/edge8估计频率误差约0.00056周期／像素，噪声自身在该频率下稳定性约0.776；在真实频率重拟合后约1，说明切向频率偏差也参与拒绝。真实频率仅用作诊断上限，不能传给新算法。其观测稳定性仍低得多，不能认为仅修频率即可解决。', '',
        '以幅度下限独立提出区间时，140055、145053和181079常能得到完整边缘带；但151087/edge8会得到240..255的16行，其中8行实际干净。因此幅度区间本身也不是安全支撑，必须保留额外验证。', '',
        '幅度区间对照只是定位，不是新算法或放宽门控建议。可考虑把输入侧支撑候选与置信判断分开，并对整个候选区间验证相位/系数结构；不能把碎片空洞直接填充后当成安全真值。', '',
        '本文没有使用新检查图片，也未改动生产代码。逐行原始数值、资格区间中真实干扰混入邻带的计数及系数分解误差均随CSV保存。']
    (OUT/'dev_FINDINGS.md').write_text('\n'.join(findings),encoding='utf-8')
    print('\n'.join(findings))


if __name__=='__main__':main()
