"""P0 numerical fixes and P1 safety-order study, with untouched audit images."""
from __future__ import annotations
import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from denoiser_v3 import Config,config_dict,config_from_dict,prepare,restore
from evaluation_v3 import metric_row,estimation_metrics,tail_mean
from reproducibility_v3 import capture_snapshot,verify_snapshot
from run_experiments import load_clean,time_call
from validation_data_v3 import make_sample,LEGACY_KINDS,EXTRA_KINDS

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data';OUT=ROOT/'results/v3';CONFIG=ROOT/'configs/v3_frozen.json'
METHODS=('input','v2_frozen','v3_coupled_P0','v3_decoupled','v3_unprotected')
SOURCES=('v3_experiments.py','denoiser_v3.py','texture_guard_v3.py','local_estimation_v3.py',
    'evaluation_v3.py','validation_data_v3.py','image_io_v3.py','reproducibility_v3.py','filter_image_v3.py',
    'v3_evidence.py','v3_verify.py','v3_report.py','data_pipeline.py','run_experiments.py','local_spectral.py',
    'local_estimation_v2.py','texture_guard_v2.py','optimized_denoiser.py','published_baseline.py',
    'original/periodic_notch_starter/demo.py','v3_protocol.json','data/v3_manifest.json','requirements-lock.txt',
    'configs/optimized_v2.json','data/manifest.json','data/optimization_manifest.json')
BASE_FIELDS=['image_id','kind','amplitude','method','strength','combination','runtime_ms',
    'mse','psnr_db','ssim','mae','contaminated_mse','clean_mse','clean_modification_mae','input_mse','degraded',
    'total_blocks','dirty_block_centres','clean_block_centres','candidate_block_coverage','correct_candidate_coverage',
    'selected_block_coverage','processed_block_coverage','clean_block_processing_rate','clean_candidate_rate',
    'frequency_error_count','frequency_error_mean','accepted_frequency_error_mean','risky_full_blocks','eligible_blocks','error']


def manifest():return json.loads((DATA/'v3_manifest.json').read_text(encoding='utf-8'))


def records(split):
    m=manifest()
    return m['textures'] if split=='texture_test' else [r for r in m['images'] if r['split']==split]


def dump(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def legacy_config():
    from optimized_denoiser import config_from_dict as v2config
    return v2config(json.loads((ROOT/'configs/optimized_v2.json').read_text(encoding='utf-8'))['configuration'])


def annotate_v2(diag):
    risky=0;eligible=0
    for t in diag.get('tiles',[]):
        d=t.get('guard_decision',{})
        s=(1-float(d.get('texture_risk',0)))*(1-float(d.get('global_ambiguity_risk',0)))
        gain=float(t.get('suppression',0))
        risky+=int(s<1-1e-8 and gain>=1-1e-8)
        eligible+=int(d.get('candidate_gate',False))
    diag['risky_full_subtraction_blocks']=risky;diag['eligible_blocks']=eligible
    return diag


def calibrate():
    """Only strength varies; all v2 reliability/risk formulas are preserved."""
    from optimized_denoiser import denoise as old
    cfg=Config();alphas=(0.,.5,.75,1.,1.25,1.5,2.);dev=records('dev')
    losses={a:dict(relative=[],clean_mae=[],local_clean_mse=[]) for a in alphas}
    legacy=dict(clean_mae=[],local_clean_mse=[]);rows=[]
    for i,r in enumerate(dev):
        clean=load_clean(r,DATA)
        for kind in (*LEGACY_KINDS,'clean'):
            observed,_,masks=make_sample(clean,r['id'],kind,.12)
            p=prepare(observed,cfg);input_mse=float(np.mean((observed-clean)**2))
            if kind in ('clean','local'):
                b=old(observed,legacy_config())
                if kind=='clean':legacy['clean_mae'].append(float(np.abs(b-clean).mean()))
                else:legacy['local_clean_mse'].append(float(np.mean((b[masks['clean']]-clean[masks['clean']])**2)))
            for a in alphas:
                result=restore(p,strength=a);mse=float(np.mean((result-clean)**2));mae=float(np.abs(result-clean).mean())
                if kind=='clean':losses[a]['clean_mae'].append(mae)
                else:
                    losses[a]['relative'].append(mse/max(input_mse,1e-8))
                    if kind=='local':losses[a]['local_clean_mse'].append(float(np.mean((result[masks['clean']]-clean[masks['clean']])**2)))
                rows.append(dict(image_id=r['id'],kind=kind,strength=a,mse=mse,input_mse=input_mse,mae=mae))
        print(f'v3 calibration {i+1}/{len(dev)} {r["id"]}',flush=True)
    bounds=dict(mean_clean_mae=float(np.mean(legacy['clean_mae'])),tail20_clean_mae=tail_mean(legacy['clean_mae']),
                mean_local_clean_mse=float(np.mean(legacy['local_clean_mse'])))
    scores={str(a):dict(mean_relative_mse=float(np.mean(v['relative'])),mean_clean_mae=float(np.mean(v['clean_mae'])),
                       tail20_clean_mae=tail_mean(v['clean_mae']),mean_local_clean_mse=float(np.mean(v['local_clean_mse'])))
            for a,v in losses.items()}
    feasible=[a for a in alphas if all(scores[str(a)][k]<=v+1e-12 for k,v in bounds.items())]
    chosen=min(feasible,key=lambda a:(scores[str(a)]['mean_relative_mse'],scores[str(a)]['tail20_clean_mae']))
    cfg=replace(cfg,strength=chosen)
    calibration=dict(ids=[r['id'] for r in dev],split='dev',cases=[*LEGACY_KINDS,'clean'],amplitude=.12,
        criterion='minimum mean normalized noisy MSE; clean mean MAE, top20% clean MAE mean and local clean-region MSE must not exceed frozen v2 on the same development inputs',
        tail_definition='arithmetic mean of ceil(0.2*n) largest per-image MAE values; n=30',
        bounds=bounds,scores=scores,selected_strength=chosen,
        method_strengths={m:(2. if m=='v2_frozen' else chosen) for m in METHODS if m!='input'})
    audit_ids=[r['id'] for r in records('audit_v3')]
    snapshot=capture_snapshot(SOURCES,audit_ids,base_dir=ROOT,metadata=dict(configuration=config_dict(cfg),calibration=calibration))
    dump(CONFIG,dict(schema_version=3,configuration=config_dict(cfg),calibration=calibration,snapshot=snapshot))
    write_csv(OUT/'development_calibration.csv',rows)


def load_frozen():
    frozen=json.loads(CONFIG.read_text(encoding='utf-8'))
    ids=[r['id'] for r in records('audit_v3')]
    verify_snapshot(frozen['snapshot'],paths=SOURCES,audit_ids=ids,base_dir=ROOT)
    if frozen['configuration']!=frozen['snapshot']['metadata']['configuration']:
        raise ValueError('Frozen configuration differs from its sealed snapshot')
    if frozen['calibration']!=frozen['snapshot']['metadata']['calibration']:
        raise ValueError('Calibration record differs from sealed snapshot')
    return frozen,config_from_dict(frozen['configuration'])


def outputs(observed,cfg):
    from optimized_denoiser import denoise as old
    yield 'input',observed,0.,None,0.,'input'
    (result,diag),ms=time_call(lambda:old(observed,legacy_config(),return_diagnostics=True))
    yield 'v2_frozen',result,ms,annotate_v2(diag),2.,'coupled_v2'
    p,prep_ms=time_call(lambda:prepare(observed,cfg))
    for method,mode,guard in (('v3_coupled_P0','coupled',True),('v3_decoupled','decoupled',True),('v3_unprotected','decoupled',False)):
        (result,diag),ms=time_call(lambda:restore(p,combination=mode,conservative=guard,return_diagnostics=True))
        yield method,result,prep_ms+ms,diag,cfg.strength,mode if guard else 'unprotected'


def stage_samples(stage):
    if stage=='audit':
        for i,r in enumerate(records('audit_v3')):
            for kind in LEGACY_KINDS:
                for a in (.06,.12,.18):yield i,r,kind,a,None
            yield i,r,'clean',0.,None
    elif stage=='stress':
        for i,r in enumerate(records('audit_v3')[:20]):
            for kind in EXTRA_KINDS:yield i,r,kind,.12,i%4 if kind in ('corner','edge_strip') else None
    elif stage=='textures':
        for i,r in enumerate(records('texture_test')):
            for kind in ('clean','global','chirp'):yield i,r,kind,0. if kind=='clean' else .12,None
    else:raise ValueError(stage)


def run(stage):
    frozen,cfg=load_frozen();folder=OUT/stage;folder.mkdir(parents=True,exist_ok=True)
    counts=dict(audit=3900,stress=500,textures=180);rows=[];exceptions=[];meta=[];last_id=None
    dump(folder/'run.json',dict(configuration=frozen['configuration'],snapshot=frozen['snapshot'],
        ids=[r['id'] for r in (records('texture_test') if stage=='textures' else records('audit_v3')[:20] if stage=='stress' else records('audit_v3'))],stage=stage))
    for i,r,kind,amplitude,variant in stage_samples(stage):
        if last_id is not None and last_id!=r['id']:print(f'v3 {stage} completed {i} images',flush=True)
        last_id=r['id'];clean=load_clean(r,DATA)
        observed,metadata,masks=make_sample(clean,r['id'],kind,amplitude,variant)
        meta.append(dict(image_id=r['id'],**metadata))
        try:
            for method,result,ms,diag,strength,combination in outputs(observed,cfg):
                row=dict(image_id=r['id'],kind=kind,amplitude=amplitude,method=method,strength=strength,
                         combination=combination,runtime_ms=ms,error='')
                row.update(metric_row(clean,observed,result,masks));row.update(estimation_metrics(diag,masks));rows.append(row)
                if (stage=='audit' and i<2 and amplitude in (.12,0.)) or stage=='textures':
                    ex=folder/'examples'/f'{r["id"]}_{kind}';ex.mkdir(parents=True,exist_ok=True)
                    np.save(ex/f'{method}.npy',result)
                    if method=='input':np.save(ex/'reference.npy',clean)
                    if method in ('v3_decoupled','v3_coupled_P0'):
                        dump(ex/f'{method}_decisions.json',dict(global_assessment=diag['global_assessment'],tiles=diag['tiles'],
                            risky_full_blocks=diag['risky_full_subtraction_blocks']))
                        if method=='v3_decoupled':np.savez_compressed(ex/'diagnostics.npz',**{k:v for k,v in diag.items() if isinstance(v,np.ndarray)})
        except Exception as exc:
            exceptions.append(dict(image_id=r['id'],kind=kind,amplitude=amplitude,error=repr(exc)))
            print('ERROR',exceptions[-1],flush=True)
        write_csv(folder/'metrics.csv',rows,BASE_FIELDS)
        (folder/'synthesis.jsonl').write_text(''.join(json.dumps(m,ensure_ascii=False)+'\n' for m in meta),encoding='utf-8')
    dump(folder/'exceptions.json',exceptions)
    keys={(r['image_id'],r['kind'],r['amplitude'],r['method']) for r in rows}
    complete=len(keys)==len(rows)==counts[stage] and not exceptions
    dump(folder/'validation.json',dict(expected=counts[stage],actual=len(rows),unique=len(keys),complete=complete,exceptions=len(exceptions)))
    if not complete:raise RuntimeError('Incomplete stage '+stage)
    print(f'v3 {stage} complete {len(rows)} rows',flush=True)


def real():
    _,cfg=load_frozen();folder=OUT/'real';folder.mkdir(parents=True,exist_ok=True)
    from image_io_v3 import to_float_gray
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cases=[]
    for case in manifest()['real_examples']:
        with Image.open(ROOT/case['path']) as image:
            image=image.convert('RGB');factor=min(512/max(image.size),1.)
            image=image.resize((round(image.width*factor),round(image.height*factor)),Image.Resampling.LANCZOS)
            observed=to_float_gray(np.asarray(image))
        outputs_meta={};fig,axs=plt.subplots(1,5,figsize=(15,3.5),layout='constrained')
        for ax,(method,result,ms,diag,strength,combination) in zip(axs,outputs(observed,cfg)):
            path=folder/f'{case["id"]}_{method}.npy';np.save(path,result)
            Image.fromarray(np.rint(result*255).astype('uint8')).save(path.with_suffix('.png'))
            outputs_meta[method]=dict(runtime_ms=ms,strength=strength,combination=combination,path=str(path.relative_to(ROOT)),
                modification_mae=float(np.abs(result-observed).mean()))
            ax.imshow(result,cmap='gray',vmin=0,vmax=1);ax.set_title(method,fontsize=9);ax.axis('off')
        fig.suptitle(case['id']+' | no paired reference; shared fixed settings')
        fig.savefig(folder/f'{case["id"]}_comparison.png',dpi=160);plt.close(fig)
        cases.append(dict(id=case['id'],reference_available=False,source=case,shape=list(observed.shape),outputs=outputs_meta))
    dump(folder/'metadata.json',cases)


def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for stage in ('audit','stress','textures'):
        folder=OUT/stage;rows=read_csv(folder/'metrics.csv');groups=defaultdict(list)
        for r in rows:groups[(r['kind'],r['method'])].append(r)
        summary=[]
        for (kind,method),group in sorted(groups.items()):
            row=dict(kind=kind,method=method,n=len(group))
            for field in BASE_FIELDS:
                if field in ('image_id','kind','method','combination','error'):continue
                values=[float(r[field]) for r in group if r.get(field,'')!='']
                row['mean_'+field]=float(np.mean(values)) if values else ''
            maes=[float(r['mae']) for r in group]
            row.update(tail20_mae=tail_mean(maes),p95_mae=float(np.quantile(maes,.95)),max_mae=max(maes),
                       median_mae=float(np.median(maes)),modified_images=sum(v>1e-12 for v in maes))
            summary.append(row)
        write_csv(folder/'summary.csv',summary)
        lookup={(r['image_id'],r['kind'],r['amplitude'],r['method']):r for r in rows}
        comparisons=[];rng=np.random.default_rng(2026111604)
        for kind in sorted({r['kind'] for r in rows}-{'clean'}):
            for baseline in ('v2_frozen','v3_coupled_P0','v3_unprotected'):
                grouped=defaultdict(list)
                for (image_id,k,a,m),r in lookup.items():
                    if k==kind and m=='v3_decoupled':grouped[image_id].append(float(r['psnr_db'])-float(lookup[(image_id,k,a,baseline)]['psnr_db']))
                values=np.array([np.mean(v) for v in grouped.values()]);samples=values[rng.integers(0,len(values),(2000,len(values)))].mean(axis=1)
                lo,hi=np.quantile(samples,[.025,.975]);comparisons.append(dict(kind=kind,baseline=baseline,n_images=len(values),
                    mean_delta_psnr=float(values.mean()),image_bootstrap_low=float(lo),image_bootstrap_high=float(hi)))
        write_csv(folder/'paired_comparisons.csv',comparisons)
        for ex in sorted((folder/'examples').glob('*')):
            reference=np.load(ex/'reference.npy');fig,axs=plt.subplots(2,3,figsize=(10,6.7),layout='constrained')
            for ax,name in zip(axs.flat,('reference',*METHODS)):
                a=np.load(ex/f'{name}.npy');mse=float(np.mean((a-reference)**2))
                ax.imshow(a,cmap='gray',vmin=0,vmax=1);ax.axis('off');ax.set_title(name+(f' {-10*np.log10(mse):.2f} dB' if mse else ' unchanged'),fontsize=9)
            fig.suptitle(ex.name+' | shared range [0,1]');fig.savefig(ex/'comparison.png',dpi=150);plt.close(fig)
        print(stage,[(r['kind'],r['method'],round(r['mean_psnr_db'],3)) for r in summary if r['method'] in ('v2_frozen','v3_decoupled')],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('calibrate','audit','stress','textures','real','summarize'))
    args=p.parse_args();{'calibrate':calibrate,'audit':lambda:run('audit'),'stress':lambda:run('stress'),
        'textures':lambda:run('textures'),'real':real,'summarize':summarize}[args.stage]()


if __name__=='__main__':main()
