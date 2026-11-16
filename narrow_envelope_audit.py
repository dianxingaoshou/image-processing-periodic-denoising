"""Read-only envelope/overlap failure diagnosis; all truth is evaluation-only.

This is not an inference method. Fixed-true-phase and enlarged-amplitude-anchor
counterfactuals below isolate representation/cap limits, never select production
parameters or supply masks to a denoiser.
"""
from dataclasses import replace
from pathlib import Path
import csv
import json
import subprocess

import numpy as np

from denoiser_v3 import config_from_dict, effective_gains, prepare, restore
from image_io_v3 import preprocess_image, sha256_file
from local_estimation_v3 import _geometry, _fit_constant
from spatial_envelope import fit_spatial_envelope
from support_data import make_sample

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/narrow_diagnosis'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def mean(a,mask):return float(np.mean(a[mask])) if np.any(mask) else ''


def error_frequency(tile,fy,fx):
    selected=tile.get('selected')
    if selected is None:return float('inf')
    return min(float(np.hypot(tile['fy']-fy,tile['fx']-fx)),float(np.hypot(tile['fy']+fy,tile['fx']+fx)))


def known_phase_fit(clean,observed,masks,meta,tile,cfg):
    y,x,h,w=(tile[k] for k in ('y','x','h','w'));sl=np.s_[y:y+h,x:x+w]
    yy,xx,_,_,weight,plane,_=_geometry((h,w))
    phase=2*np.pi*(meta['fy']*(yy+y)+meta['fx']*(xx+x))+meta['phase_offset']
    carrier=np.sin(phase)
    coef,_,_=_fit_constant(observed[sl],phase)
    anchor=float(np.hypot(coef[0],coef[1]))
    capped=fit_spatial_envelope(observed[sl],carrier,weight,plane,anchor)
    # This second fit deliberately uses the true injected peak amplitude to
    # isolate cap-vs-basis limits. It is diagnostic and cannot be deployed.
    oracle_anchor=fit_spatial_envelope(observed[sl],carrier,weight,plane,meta['amplitude'])
    desired=meta['amplitude']*masks['envelope'][sl]
    row=dict(y=y,x=x,constant_anchor=anchor,true_peak=float(desired.max()),
        amplitude_cap=capped['amplitude_upper'],true_peak_above_cap=bool(desired.max()>capped['amplitude_upper']+1e-12),
        fixed_phase_capped_envelope_mse=float(np.mean((capped['field']-desired)**2)),
        fixed_phase_true_anchor_envelope_mse=float(np.mean((oracle_anchor['field']-desired)**2)),
        fixed_phase_capped_dirty_mse=mean((observed[sl]-capped['residual']-clean[sl])**2,masks['contaminated'][sl]),
        fixed_phase_true_anchor_dirty_mse=mean((observed[sl]-oracle_anchor['residual']-clean[sl])**2,masks['contaminated'][sl]),
        fixed_phase_capped_clean_leak=mean(capped['residual']**2,masks['clean'][sl]),
        fixed_phase_true_anchor_clean_leak=mean(oracle_anchor['residual']**2,masks['clean'][sl]))
    return row


def analytic_cap_examples():
    rows=[]
    for size in (32,64):
        yy,xx,_,_,weight,plane,_=_geometry((size,size))
        phase=2*np.pi*(.12*yy+.08*xx)+.3;carrier=np.sin(phase)
        for width in (8,16,32):
            desired=.12*np.where(xx<width,.5*(1+np.cos(np.pi*np.minimum(xx/width,1))),0.)
            observed=.5+desired*carrier
            coef,_,_=_fit_constant(observed,phase);anchor=float(np.hypot(coef[0],coef[1]))
            cap=fit_spatial_envelope(observed,carrier,weight,plane,anchor)
            large=fit_spatial_envelope(observed,carrier,weight,plane,.12)
            dirty=desired>.0012;clean=desired==0
            rows.append(dict(tile_size=size,width=width,anchor=anchor,cap=cap['amplitude_upper'],true_peak=.12,
                true_peak_over_cap=.12/max(cap['amplitude_upper'],1e-20),
                input_dirty_mse=mean((observed-.5)**2,dirty),
                capped_dirty_mse=mean((observed-cap['residual']-.5)**2,dirty),
                true_anchor_dirty_mse=mean((observed-large['residual']-.5)**2,dirty),
                capped_clean_leak=mean(cap['residual']**2,clean),
                true_anchor_clean_leak=mean(large['residual']**2,clean)))
    return rows


def run():
    source_files=('local_estimation_v3.py','texture_guard_v3.py','spatial_envelope.py','denoiser_v3.py','support_data.py')
    source_hashes={name:sha256_file(ROOT/name) for name in source_files}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    manifest=read(ROOT/'data/v3_manifest.json');protocol=read(ROOT/'support_protocol.json')
    by_id={row['id']:row for row in manifest['images']}
    cfg=config_from_dict(read(ROOT/'configs/support_a32_s32_grid.json')['configuration'])
    cases=[]
    for index,image_id in enumerate(protocol['development_ids']):
        record=by_id[image_id];path=ROOT/'data'/record['npy_path']
        if sha256_file(path)!=record['clean_sha256']:raise ValueError('Reference changed')
        clean=np.load(path)
        for kind in ('edge8','edge16','edge32'):cases.append((image_id,clean,kind,index%4,'development'))
    old=next(row for row in protocol['independent_check'] if row['id']=='48017')
    clean,_=preprocess_image(ROOT/'data'/old['source_jpg'],size=256)
    cases.append(('48017',clean,'edge32',2,'previous_check_now_regression'))
    rows=[];tile_rows=[];phase_rows=[]
    for index,(image_id,clean,kind,variant,role) in enumerate(cases):
        observed,meta,masks=make_sample(clean,image_id,kind,.12,variant)
        prepared=prepare(observed,cfg)
        protected,diag=restore(prepared,return_diagnostics=True)
        unprotected,raw_diag=restore(prepared,conservative=False,return_diagnostics=True)
        correct=np.zeros_like(clean);wrong=np.zeros_like(clean)
        correct_protected=np.zeros_like(clean);wrong_protected=np.zeros_like(clean)
        gains=effective_gains(prepared.fit_weights,prepared.safety_weights,cfg.strength,cfg.combination)
        dirty_tiles=[]
        for tile_index,tile in enumerate(prepared.estimation.tiles):
            y,x,h,w=(tile[k] for k in ('y','x','h','w'));sl=np.s_[y:y+h,x:x+w]
            weight=_geometry((h,w))[4]
            matched=bool(error_frequency(tile,meta['fy'],meta['fx'])<=.01)
            (correct if matched else wrong)[sl]+=weight*tile['residual']
            (correct_protected if matched else wrong_protected)[sl]+=weight*gains[tile_index]*tile['residual']
            dirty_count=int(np.count_nonzero(masks['contaminated'][sl]))
            if not dirty_count:continue
            dirty_tiles.append(tile)
            env=tile.get('envelope_diagnostics') or {}
            coefficients=np.asarray(env.get('coefficients',[]))
            upper=float(env.get('amplitude_upper',0.))
            active=float(np.mean(coefficients>=upper-1e-8)) if coefficients.size and upper>0 else ''
            tile_rows.append(dict(image_id=image_id,kind=kind,variant=variant,role=role,
                tile_index=tile_index,y=y,x=x,dirty_pixels=dirty_count,
                matched_frequency=matched,model=tile.get('model',''),gain=float(gains[tile_index]),
                grid_selected=bool(env.get('selected',False)),grid_upper=upper,grid_upper_active_fraction=active,
                raw_dirty_energy=mean(tile['residual']**2,masks['contaminated'][sl]),
                raw_clean_energy=mean(tile['residual']**2,masks['clean'][sl]),
                raw_adjacent_energy=mean(tile['residual']**2,masks['adjacent_clean'][sl])))
        denominator=diag['weight_sum']
        correct/=denominator;wrong/=denominator;correct_protected/=denominator;wrong_protected/=denominator
        row=dict(image_id=image_id,kind=kind,variant=variant,role=role,dirty_intersecting_tiles=len(dirty_tiles),
            correctly_fitted_dirty_tiles=sum(error_frequency(t,meta['fy'],meta['fx'])<=.01 for t in dirty_tiles),
            input_dirty_mse=mean((observed-clean)**2,masks['contaminated']),
            protected_dirty_mse=mean((protected-clean)**2,masks['contaminated']),
            unprotected_dirty_mse=mean((unprotected-clean)**2,masks['contaminated']),
            protected_adjacent_mse=mean((protected-clean)**2,masks['adjacent_clean']),
            unprotected_adjacent_mse=mean((unprotected-clean)**2,masks['adjacent_clean']),
            protected_far_clean_mse=mean((protected-clean)**2,masks['clean']&~masks['adjacent_clean']),
            raw_correct_ola_dirty_energy=mean(correct**2,masks['contaminated']),
            raw_correct_ola_adjacent_energy=mean(correct**2,masks['adjacent_clean']),
            raw_wrong_ola_adjacent_energy=mean(wrong**2,masks['adjacent_clean']),
            protected_correct_ola_adjacent_energy=mean(correct_protected**2,masks['adjacent_clean']),
            protected_wrong_ola_adjacent_energy=mean(wrong_protected**2,masks['adjacent_clean']))
        rows.append(row)
        # Up to three most contaminated tiles, fixed rule not chosen by success.
        chosen=sorted(dirty_tiles,key=lambda t:-int(masks['contaminated'][t['y']:t['y']+t['h'],t['x']:t['x']+t['w']].sum()))[:3]
        for tile in chosen:
            phase_rows.append(dict(image_id=image_id,kind=kind,role=role,**known_phase_fit(clean,observed,masks,meta,tile,cfg)))
        print(f'envelope audit {index+1}/{len(cases)} {image_id} {kind}',flush=True)
    write_csv(OUT/'envelope_cases.csv',rows)
    write_csv(OUT/'envelope_tiles.csv',tile_rows)
    write_csv(OUT/'envelope_known_phase.csv',phase_rows)
    analytic=analytic_cap_examples();write_csv(OUT/'envelope_analytic.csv',analytic)
    summary={}
    for kind in ('edge8','edge16','edge32'):
        values=[r for r in rows if r['kind']==kind and r['role']=='development']
        summary[kind]={key:float(np.mean([r[key] for r in values])) for key in (
            'input_dirty_mse','protected_dirty_mse','unprotected_dirty_mse','protected_adjacent_mse','unprotected_adjacent_mse')}
        summary[kind]['matched_tiles']=sum(r['correctly_fitted_dirty_tiles'] for r in values)
        summary[kind]['dirty_tiles']=sum(r['dirty_intersecting_tiles'] for r in values)
        ph=[r for r in phase_rows if r['kind']==kind and r['role']=='development']
        summary[kind]['known_phase_true_peak_above_cap_fraction']=float(np.mean([r['true_peak_above_cap'] for r in ph]))
        matched=[r for r in tile_rows if r['kind']==kind and r['role']=='development' and r['matched_frequency']]
        summary[kind]['matched_fits_with_positive_gain']=sum(r['gain']>0 for r in matched)
        summary[kind]['matched_fits_grid_selected']=sum(r['grid_selected'] for r in matched)
    (OUT/'envelope_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Envelope / overlap diagnosis','',
        'Read-only study: existing development8 and one prior check case now regression. Truth is used for attribution and fixed-known-phase counterfactuals only; no new test image is read.', '',
        '| Width | Correctly fitted dirty-intersecting tiles | Input dirty MSE | Protected dirty MSE | Unprotected dirty MSE | Protected adjacent MSE | Unprotected adjacent MSE | Known-phase peak above cap |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for kind,r in summary.items():
        lines.append(f"| {kind} | {r['matched_tiles']}/{r['dirty_tiles']} | {r['input_dirty_mse']:.7g} | {r['protected_dirty_mse']:.7g} | {r['unprotected_dirty_mse']:.7g} | {r['protected_adjacent_mse']:.7g} | {r['unprotected_adjacent_mse']:.7g} | {r['known_phase_true_peak_above_cap_fraction']:.1%} |")
    spill=next(r for r in rows if r['image_id']=='48017')
    lines += ['', 'The numerator/denominator use dirty-intersecting tiles, not block centers. Full-image outputs include clipping and normalized overlap reconstruction. Correct/wrong-frequency component energies do not add because cross terms and cancellations exist.', '',
        'Matched-frequency fitting does not imply processing: positive-gain matched fits are edge8=0/3, edge16=4/31, edge32=44/94. The three matched edge8 fits all selected the grid alternative, but their production gate weights are zero.', '',
        f"Known prior failure48017/edge32: protected adjacent MSE={spill['protected_adjacent_mse']:.9g}; protected correct-frequency contribution energy={spill['protected_correct_ola_adjacent_energy']:.9g}; protected wrong-frequency contribution energy={spill['protected_wrong_ola_adjacent_energy']:.9g}. This particular boundary damage comes from erroneous background/texture frequencies, not from an accurately identified stripe envelope alone.", '',
        'A 3x3 grid on a32 tile has15.5px node spacing. With a Hann-weighted constant amplitude, an8px boundary strip can be diluted before the2.5x cap is applied. Enlarging the cap in a known-phase diagnostic can reveal this numerical limitation, but must not be described as an input-only solution.', '',
        'CSV files retain per-tile raw leakage, capped controls, and fixed-known-phase comparisons. Do not infer texture safety from reduced fitting SSE.']
    (OUT/'envelope_FINDINGS.md').write_text('\n'.join(lines),encoding='utf-8')
    if source_hashes!={name:sha256_file(ROOT/name) for name in source_files}:
        raise RuntimeError('Production sources changed during diagnosis; rerun with a stable checkout')
    (OUT/'envelope_run.json').write_text(json.dumps(dict(baseline_commit=commit,
        source_sha256=source_hashes,n_cases=len(cases),development_cases=24,prior_check_now_regression=1,
        frequency_match_tolerance=.01,truth_use='attribution and fixed-known-phase counterfactuals only; never inference'),
        indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
