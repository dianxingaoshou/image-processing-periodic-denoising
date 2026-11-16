"""Equal-removal comparison, texture counterexamples and v2 delivery checks."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from optimization_experiments import DATA,OUT,CONFIG,records,frozen_config,read_csv,write_csv
from run_experiments import load_clean,make_sample
from optimized_denoiser import prepare,restore,denoise
from validate_tradeoff import _match,_plot_tradeoff,_texture_inputs


def decisions():
    """Reuse identical candidate residuals to isolate decision components."""
    from dataclasses import replace
    from run_experiments import metric_row,time_call
    from texture_guard_v2 import guard_weights
    from local_estimation_v2 import restore as reconstruct
    _,cfg=frozen_config();folder=OUT/'decisions';folder.mkdir(parents=True,exist_ok=True)
    settings=[('v2_full',cfg.guard),('v2_without_texture_risk',replace(cfg.guard,use_texture_risk=False)),
              ('v2_peak_only',replace(cfg.guard,use_consistency=False,use_texture_risk=False)),
              ('v2_without_candidate_gate',replace(cfg.guard,use_candidate_gate=False))]
    rows=[]
    for i,record in enumerate(records('audit_test')[:20]):
        clean=load_clean(record,DATA)
        for kind in ('global','local','chirp','orientation','clean'):
            amplitude=0. if kind=='clean' else .12
            observed,_,masks=make_sample(clean,record['id'],kind,amplitude)
            p,prep_ms=time_call(lambda:prepare(observed,cfg))
            for method,guard in settings:
                (weights,_,_),guard_ms=time_call(lambda:guard_weights(observed,p.estimation.tiles,guard))
                result,ms=time_call(lambda:reconstruct(p.estimation,weights,strength=cfg.strength))
                row=dict(image_id=record['id'],kind=kind,amplitude=amplitude,method=method,runtime_ms=prep_ms+guard_ms+ms)
                row.update(metric_row(clean,observed,result,masks));rows.append(row)
        print(f'v2 decision ablation {i+1}/20',flush=True)
    write_csv(folder/'metrics.csv',rows)
    groups=defaultdict(list)
    for r in rows:groups[(r['kind'],r['method'])].append(r)
    summary=[]
    for (kind,method),group in sorted(groups.items()):
        row=dict(kind=kind,method=method,n=len(group))
        for key in ('psnr_db','mse','mae','ssim','contaminated_mse','clean_mse','degraded'):
            values=[float(r[key]) for r in group if r[key]!='']
            row['mean_'+key]=float(np.mean(values)) if values else ''
        summary.append(row)
    write_csv(folder/'summary.csv',summary)


def supplement():
    frozen,cfg=frozen_config();folder=OUT/'supplement';folder.mkdir(parents=True,exist_ok=True)
    strengths=(0.,.125,.25,.375,.5,.75,1.,1.25,1.5,2.)
    rows=[];matched=[];coverage=[]
    for i,record in enumerate(records('audit_test')[:10]):
        clean=load_clean(record,DATA)
        observed,_,masks=make_sample(clean,record['id'],'local',.12)
        p=prepare(observed,cfg);input_mse=float(np.mean((observed[masks['contaminated']]-clean[masks['contaminated']])**2))
        case=[]
        for mode,guard in (('conservative',True),('without_conservative',False)):
            for strength in strengths:
                result=restore(p,strength=strength,conservative=guard);err=(result-clean)**2
                row=dict(image_id=record['id'],kind='local',mode=mode,strength=strength,
                    contaminated_mse=float(err[masks['contaminated']].mean()),clean_mse=float(err[masks['clean']].mean()),
                    input_contaminated_mse=input_mse)
                rows.append(row);case.append(row)
        matches=_match(case,'individual',record['id'],'local');matched+=matches
        coverage.append(dict(image_id=record['id'],matched_points=len(matches),
            reason='matched' if matches else 'no common positive removal interval'))
        print(f'v2 matched sweep {i+1}/10',flush=True)
    grouped=defaultdict(list)
    for r in rows:grouped[(r['mode'],r['strength'])].append(r)
    mean=[]
    for (mode,strength),group in grouped.items():
        mean.append(dict(kind='local',mode=mode,strength=strength,**{k:float(np.mean([g[k] for g in group]))
            for k in ('contaminated_mse','clean_mse','input_contaminated_mse')}))
    matched+=_match(mean,'subset_mean','all10','local')
    write_csv(folder/'strength_curve.csv',rows);write_csv(folder/'strength_curve_mean.csv',mean)
    write_csv(folder/'matched_coverage.csv',coverage)
    if matched:write_csv(folder/'matched_removal.csv',matched)
    _plot_tradeoff(mean,matched,folder)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from local_spectral import denoise as v1
    cases=_texture_inputs()
    # The same composite pixels can represent a painted periodic patch or
    # additive interference. These development-image counterexamples explicitly
    # test the boundary that a whole-image ambiguity guard does not resolve.
    for record in records('dev')[:2]:
        background=load_clean(record,DATA)
        observed,_,_=make_sample(background,record['id'],'local',.12)
        cases.append((f'local_periodic_on_natural_{record["id"]}',observed,background))
    fig,axs=plt.subplots(len(cases),4,figsize=(11,2.7*len(cases)),layout='constrained')
    texture_rows=[];arrays={}
    for i,(name,observed,alternative_clean) in enumerate(cases):
        p=prepare(observed,cfg);protected=restore(p);raw=restore(p,conservative=False)
        old=v1(observed);outputs={'v1':old,'v2':protected,'v2_without_guard':raw}
        for method,result in outputs.items():
            for interpretation,reference in (('legitimate_texture',observed),('noise_on_flat_field',alternative_clean)):
                if reference is None:continue
                mse=float(np.mean((result-reference)**2))
                texture_rows.append(dict(case=name,method=method,interpretation=interpretation,mse=mse,
                    psnr_db=-10*np.log10(mse) if mse else 'inf',mae=float(np.abs(result-reference).mean()),
                    modification_mae=float(np.abs(result-observed).mean()),
                    input_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),output_sha256=hashlib.sha256(result.tobytes()).hexdigest()))
            arrays[f'{name}_{method}']=result
        arrays[f'{name}_input']=observed
        if alternative_clean is not None:arrays[f'{name}_alternative_reference']=alternative_clean
        for j,(a,title) in enumerate(((observed,'Legitimate input'),(old,'v1'),(protected,'v2'),(raw,'v2 without guard'))):
            axs[i,j].imshow(a,cmap='gray',vmin=0,vmax=1);axs[i,j].set_title(title,fontsize=9)
            axs[i,j].set_xticks([]);axs[i,j].set_yticks([])
        axs[i,0].set_ylabel(name,fontsize=8)
    fig.savefig(folder/'texture_comparison.png',dpi=150);plt.close(fig)
    write_csv(folder/'texture_metrics.csv',texture_rows)
    np.savez_compressed(folder/'texture_arrays.npz',**arrays)
    (folder/'scope.json').write_text(json.dumps(dict(ids=[r['id'] for r in records('audit_test')[:10]],
        config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),strengths=strengths,
        matching='Finite sweep Pareto frontier then interpolate clean MSE only in common contaminated-MSE reduction interval.',
        ambiguity='Pixel-identical texture/noise interpretations must produce identical output; abstention is a conservative policy.'),indent=2),encoding='utf-8')


def verify():
    frozen,cfg=frozen_config();checks={}
    splits={name:{r['id'] for r in records(name)} for name in ('dev','regression','audit_test')}
    checks['disjoint_splits']=all(not splits[a]&splits[b] for a,b in (('dev','regression'),('dev','audit_test'),('regression','audit_test')))
    checks['split_sizes']=[len(splits[s]) for s in ('dev','regression','audit_test')]==[30,60,60]
    manifest=json.loads((DATA/'optimization_manifest.json').read_text(encoding='utf-8'))
    checks['all_reference_hashes']=all(hashlib.sha256((DATA/r['npy_path']).read_bytes()).hexdigest()==r['clean_sha256'] and
        hashlib.sha256((DATA/r['source_jpg']).read_bytes()).hexdigest()==r['source_sha256'] for r in manifest['images'])
    for stage,count,per_method in (('audit',3900,780),('ablations',500,100)):
        rows=read_csv(OUT/stage/'metrics.csv')
        keys={(r['image_id'],r['kind'],r['amplitude'],r['method']) for r in rows}
        groups=defaultdict(list)
        for r in rows:groups[r['method']].append(r)
        checks[stage+'_complete']=len(keys)==len(rows)==count and len(groups)==5 and all(len(v)==per_method for v in groups.values())
        checks[stage+'_numeric']=all(np.isfinite(float(r['mse'])) and float(r['mse'])>=0 and
            np.isfinite(float(r['ssim'])) and -1<=float(r['ssim'])<=1 and not r['error'] for r in rows)
        checks[stage+'_psnr_relation']=all((float(r['psnr_db'])==float('inf') if float(r['mse'])==0 else
            np.isclose(float(r['psnr_db']),-10*np.log10(float(r['mse'])),rtol=1e-10,atol=1e-10)) for r in rows)
        run=json.loads((OUT/stage/'run.json').read_text(encoding='utf-8'))
        checks[stage+'_frozen_configuration']=run['config']==frozen
        meta=[json.loads(line) for line in (OUT/stage/'synthesis.jsonl').read_text(encoding='utf-8').splitlines()]
        checks[stage+'_synthesis_inventory']=len(meta)==(100 if stage=='ablations' else 780)
        by_case=defaultdict(list)
        for item in meta:
            if item['kind']!='clean':by_case[(item['image_id'],item['kind'])].append(item)
        checks[stage+'_paired_amplitudes']=all(len({r['seed'] for r in group})==1 and
            len({json.dumps(r['parameters'],sort_keys=True) for r in group})==1 for group in by_case.values())
    texture=read_csv(OUT/'supplement/texture_metrics.csv');pairs=defaultdict(list)
    for row in texture:pairs[(row['case'],row['method'])].append(row)
    checks['identical_interpretations']=all(len({r['input_sha256'] for r in group})==len({r['output_sha256'] for r in group})==1
        for group in pairs.values() if len(group)==2)
    sweep=read_csv(OUT/'supplement/strength_curve.csv')
    checks['matched_sweep_inventory']=len(sweep)==200 and len({(r['image_id'],r['mode'],r['strength']) for r in sweep})==200
    decision_rows=read_csv(OUT/'decisions/metrics.csv')
    checks['decision_ablation_inventory']=len(decision_rows)==400 and len({(r['image_id'],r['kind'],r['method']) for r in decision_rows})==400
    audit_rows=read_csv(OUT/'audit/metrics.csv')
    audit_lookup={(r['image_id'],r['kind'],r['amplitude']):r for r in audit_rows if r['method']=='v2_full'}
    checks['decision_full_matches_audit']=all(np.isclose(float(r['mse']),float(audit_lookup[(r['image_id'],r['kind'],r['amplitude'])]['mse']),rtol=1e-10,atol=1e-12)
        for r in decision_rows if r['method']=='v2_full')
    checks['original_code_unchanged']=all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest for name,digest in
        json.loads(Path('results/main/source_hashes.json').read_text(encoding='utf-8')).items())
    # Deterministic re-evaluation at five predetermined audit samples.
    reruns=[]
    main=read_csv(OUT/'audit/metrics.csv')
    record=records('audit_test')[0];clean=load_clean(record,DATA)
    for kind in ('global','local','chirp','orientation','clean'):
        amplitude=0. if kind=='clean' else .12
        observed,_,_=make_sample(clean,record['id'],kind,amplitude);result=denoise(observed,cfg)
        row=next(r for r in main if r['image_id']==record['id'] and r['kind']==kind and float(r['amplitude'])==amplitude and r['method']=='v2_full')
        reruns.append(bool(np.isclose(np.mean((result-clean)**2),float(row['mse']),rtol=1e-10,atol=1e-12)))
    checks['deterministic_reruns']=all(reruns)
    report=dict(all_passed=all(checks.values()),checks=checks,config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest())
    (OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not report['all_passed']:raise RuntimeError('v2 artifact verification failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('decisions','supplement','verify'))
    args=p.parse_args();{'decisions':decisions,'supplement':supplement,'verify':verify}[args.stage]()
