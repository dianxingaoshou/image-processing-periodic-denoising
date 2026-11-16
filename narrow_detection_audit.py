"""Evaluation-only diagnosis of narrow-stripe candidate losses.

Truth identifies dirty-intersecting tiles and measures recall, never proposes
production frequencies. The pure-noise spectrum is an explicitly oracle-only
diagnostic control, not a restoration method.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from denoiser_v3 import config_from_dict, prepare
from local_estimation_v3 import spectral_candidates
from support_data import make_sample

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/narrow_diagnosis'


def distance(candidate, fy, fx):
    return min(float(np.hypot(candidate['fy']-fy,candidate['fx']-fx)),
               float(np.hypot(candidate['fy']+fy,candidate['fx']+fx)))


def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    manifest=json.loads((ROOT/'data/v3_manifest.json').read_text(encoding='utf-8'))
    protocol=json.loads((ROOT/'support_protocol.json').read_text(encoding='utf-8'))
    records={r['id']:r for r in manifest['images']}
    configurations={name:config_from_dict(json.loads((ROOT/f'configs/support_{name}.json').read_text(encoding='utf-8'))['configuration'])
                    for name in ('baseline','a32_s32_grid')}
    rows=[];tiles_out=[]
    for index,image_id in enumerate(protocol['development_ids']):
        clean=np.load(ROOT/'data'/records[image_id]['npy_path'])
        for kind in ('edge8','edge16','edge32'):
            observed,meta,masks=make_sample(clean,image_id,kind,.12,index%4)
            fy,fx=meta['fy'],meta['fx']
            for method,cfg in configurations.items():
                p=prepare(observed,cfg);tiles=p.estimation.tiles
                selected_map=np.zeros(clean.shape,bool);pool_map=np.zeros(clean.shape,bool)
                raw_map=np.zeros(clean.shape,bool)
                dirty_tiles=raw_hits=pool_hits=selected_hits=0
                correct_32_checks=correct_32_rejected=all_32_checks=0
                rejection_reasons={}
                energy_ratios=[];flat_ratios=[];ideal_hits=[];windows=set()
                for tile in tiles:
                    sl=np.s_[tile['y']:tile['y']+tile['h'],tile['x']:tile['x']+tile['w']]
                    dirty=masks['contaminated'][sl]
                    if not dirty.any():continue
                    dirty_tiles+=1
                    pool=any(distance(c,fy,fx)<=.01 for c in tile['candidates'])
                    checks=tile.get('small_window_checks',[])
                    raw=pool or any(distance(c,fy,fx)<=.01 for c in checks)
                    selected=tile['selected'] is not None and distance(tile['selected'],fy,fx)<=.01
                    raw_hits+=raw;pool_hits+=pool;selected_hits+=selected
                    if raw:raw_map[sl]|=dirty
                    if pool:pool_map[sl]|=dirty
                    if selected:selected_map[sl]|=dirty
                    for check in checks:
                        all_32_checks+=1
                        if distance(check,fy,fx)<=.01:
                            correct_32_checks+=1
                            if not check['eligible']:
                                correct_32_rejected+=1
                                rejection_reasons[check['reason']]=rejection_reasons.get(check['reason'],0)+1
                    tiles_out.append(dict(image_id=image_id,kind=kind,method=method,y=tile['y'],x=tile['x'],
                        dirty_pixels=int(dirty.sum()),raw_match=raw,pool_match=pool,selected_match=selected,
                        selected_error=distance(tile['selected'],fy,fx) if tile['selected'] is not None else '',
                        candidate_count=len(tile['candidates']),small_checks=len(checks)))
                    for c in tile['candidates']:
                        window=tuple(c['analysis_window_id'])
                        if window in windows:continue
                        windows.add(window);ay,ax,h,w=window
                        window_slice=np.s_[ay:ay+h,ax:ax+w]
                        noise=masks['true_noise'][window_slice]
                        if not np.any(masks['contaminated'][window_slice]):continue
                        hann=np.outer(np.hanning(h),np.hanning(w))
                        flat=np.broadcast_to(np.hanning(h)[:,None],(h,w)) if index%4<2 else np.broadcast_to(np.hanning(w)[None,:],(h,w))
                        energy=float(np.mean(noise**2))
                        if energy>0:
                            energy_ratios.append(float(np.sum((noise*hann)**2)/np.sum(hann**2)/energy))
                            flat_ratios.append(float(np.sum((noise*flat)**2)/np.sum(flat**2)/energy))
                        ideal=spectral_candidates(.5+noise,cfg.estimation)
                        ideal_hits.append(any(distance(candidate,fy,fx)<=.01 for candidate in ideal))
                dirty_count=int(masks['contaminated'].sum())
                rows.append(dict(image_id=image_id,kind=kind,method=method,fy=fy,fx=fx,
                    dirty_intersecting_tiles=dirty_tiles,raw_match_tiles=raw_hits,pool_match_tiles=pool_hits,
                    selected_match_tiles=selected_hits,raw_tile_recall=raw_hits/max(dirty_tiles,1),
                    pool_tile_recall=pool_hits/max(dirty_tiles,1),selected_tile_recall=selected_hits/max(dirty_tiles,1),
                    dirty_pixel_raw_union=float(raw_map.sum()/dirty_count),dirty_pixel_pool_union=float(pool_map.sum()/dirty_count),
                    dirty_pixel_selected_union=float(selected_map.sum()/dirty_count),all_32_checks=all_32_checks,
                    correct_32_checks=correct_32_checks,correct_32_rejected=correct_32_rejected,
                    correct_32_reject_reasons=json.dumps(rejection_reasons,sort_keys=True),
                    hann_noise_energy_ratio_median=float(np.median(energy_ratios)) if energy_ratios else '',
                    flat_normal_noise_energy_ratio_median=float(np.median(flat_ratios)) if flat_ratios else '',
                    oracle_pure_noise_peak_recall=float(np.mean(ideal_hits)) if ideal_hits else ''))
                print(image_id,kind,method,raw_hits,pool_hits,selected_hits,'/',dirty_tiles,flush=True)
    write_csv(OUT/'detection_per_image.csv',rows)
    write_csv(OUT/'detection_tiles.csv',tiles_out)
    summary=[]
    for kind in ('edge8','edge16','edge32'):
        for method in configurations:
            values=[r for r in rows if r['kind']==kind and r['method']==method]
            record=dict(kind=kind,method=method,n_images=len(values))
            for field in ('dirty_intersecting_tiles','raw_match_tiles','pool_match_tiles','selected_match_tiles',
                          'correct_32_checks','correct_32_rejected'):
                record[field]=sum(r[field] for r in values)
            for field in ('raw_tile_recall','pool_tile_recall','selected_tile_recall','dirty_pixel_raw_union',
                          'dirty_pixel_pool_union','dirty_pixel_selected_union','hann_noise_energy_ratio_median',
                          'flat_normal_noise_energy_ratio_median','oracle_pure_noise_peak_recall'):
                valid=[r[field] for r in values if r[field]!='']
                record[field]=float(np.mean(valid)) if valid else ''
            summary.append(record)
    write_csv(OUT/'detection_summary.csv',summary)
    lines=['# 窄条候选检测诊断','',
        '仅原8张开发图，edge8/16/32；真实频率只用于评价，不传入算法。统计所有与含噪像素相交的复原块，避免块中心遗漏窄条。', '',
        'raw包括32资格筛选之前的峰；pool为筛选后候选池；selected为精拟合后选定频率。频率容差0.01周期/像素，正负共轭等价。像素union为正确候选所在块对含噪区域的覆盖，并不保证最终残差正确。', '',
        '| 条件 | 方法 | raw块召回 | pool块召回 | selected块召回 | pool像素union | selected像素union |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['kind']} | {r['method']} | {r['raw_tile_recall']:.3f} | {r['pool_tile_recall']:.3f} | {r['selected_tile_recall']:.3f} | {r['dirty_pixel_pool_union']:.3f} | {r['dirty_pixel_selected_union']:.3f} |")
    lines+=['','纯噪声峰是oracle隔离诊断，不能作为算法或性能结果；flat-normal能量对照用已知边缘方向，仅说明窗口衰减机制，未提供可部署的检测方案。','']
    (OUT/'detection_FINDINGS.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':main()
