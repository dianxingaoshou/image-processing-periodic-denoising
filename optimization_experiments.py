"""Frozen v2 optimization study; never calibrate on regression/audit images.

The original experiment and results are preserved. A fresh audit split is used
because the previous test images have already informed failure analysis.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from run_experiments import (AMPLITUDES, KINDS, json_dump, load_clean, make_sample,
                             metric_row, starter, time_call)

DATA=Path('data')
OUT=Path('results/optimization_v2')
CONFIG=Path('configs/optimized_v2.json')
FIELDS=['image_id','split','kind','amplitude','method','runtime_ms','mse','psnr_db','ssim','mae',
        'contaminated_mse','clean_mse','clean_modification_mae','input_mse','degraded','error']


def records(split):
    manifest=json.loads((DATA/'optimization_manifest.json').read_text(encoding='utf-8'))
    return [r for r in manifest['images'] if r['split']==split]


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def calibrate():
    from optimized_denoiser import V2Config, prepare, restore, config_dict
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=V2Config()
    strengths=(0.,.5,.75,1.,1.25,1.5,2.)
    losses={s:dict(relative=[],clean_mae=[],clean_mse=[],local_clean_mse=[]) for s in strengths}
    detail=[]
    dev=records('dev')
    for index,record in enumerate(dev):
        clean=load_clean(record,DATA)
        for kind in (*KINDS,'clean'):
            observed,_,masks=make_sample(clean,record['id'],kind,.12)
            p=prepare(observed,cfg)
            input_mse=float(np.mean((observed-clean)**2))
            for strength in strengths:
                result=restore(p,strength=strength)
                mse=float(np.mean((result-clean)**2));mae=float(np.abs(result-clean).mean())
                if kind=='clean':
                    losses[strength]['clean_mae'].append(mae)
                    losses[strength]['clean_mse'].append(mse)
                else:
                    losses[strength]['relative'].append(mse/max(input_mse,1e-8))
                    if kind=='local':
                        losses[strength]['local_clean_mse'].append(float(np.mean((result[masks['clean']]-clean[masks['clean']])**2)))
                detail.append(dict(image_id=record['id'],kind=kind,strength=strength,mse=mse,
                                   input_mse=input_mse,mae=mae))
        print(f'v2 calibration {index+1}/{len(dev)} {record["id"]}',flush=True)
    scores={s:dict(mean_relative_mse=float(np.mean(v['relative'])),
                   mean_clean_mae=float(np.mean(v['clean_mae'])),mean_clean_mse=float(np.mean(v['clean_mse'])),
                   mean_local_clean_mse=float(np.mean(v['local_clean_mse'])))
            for s,v in losses.items()}
    feasible=[s for s in strengths if scores[s]['mean_clean_mae']<=5e-4 and scores[s]['mean_local_clean_mse']<=5e-6]
    chosen=min(feasible,key=lambda s:(scores[s]['mean_relative_mse'],scores[s]['mean_clean_mae']))
    cfg=replace(cfg,strength=chosen)
    frozen=dict(schema_version=2,configuration=config_dict(cfg),
        calibration=dict(split='dev',ids=[r['id'] for r in dev],amplitude=.12,cases=[*KINDS,'clean'],
            criterion='minimum mean normalized noisy-image MSE; mean clean-control MAE <= 0.0005 and local clean-region MSE <= 0.000005',
            normalization='mse/max(input_mse,1e-8); each image/condition weighted equally',
            selected_strength=chosen,scores=scores),
        fresh_audit_ids=[r['id'] for r in records('audit_test')],
        source_hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ('local_estimation_v2.py','texture_guard_v2.py','optimized_denoiser.py','optimization_experiments.py')})
    json_dump(CONFIG,frozen)
    write_csv(OUT/'development_calibration.csv',detail)


def frozen_config():
    from optimized_denoiser import config_from_dict
    frozen=json.loads(CONFIG.read_text(encoding='utf-8'))
    for name,digest in frozen['source_hashes'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Frozen source changed: {name}. Recalibrate before reading audit outcomes.')
    dev=set(frozen['calibration']['ids']);audit={r['id'] for r in records('audit_test')}
    if dev&audit:raise ValueError('Development/audit overlap')
    return frozen,config_from_dict(frozen['configuration'])


def outputs(observed,cfg,ablation=False):
    from local_spectral import LocalConfig, denoise as old_denoise
    from optimized_denoiser import prepare,restore
    from published_baseline import denoise as arpenos
    if not ablation:
        yield 'input',observed,0.,None
        old=json.loads(Path('configs/frozen.json').read_text(encoding='utf-8'))
        result,ms=time_call(lambda:old_denoise(observed,replace(LocalConfig(**old['local_config']),strength=old['local_strength'])))
        yield 'v1_full',result,ms,None
        (result,_,_),ms=time_call(lambda:starter.denoise(observed))
        yield 'global_adaptive',result,ms,None
        result,ms=time_call(lambda:arpenos(observed))
        yield 'published_arpenos',np.clip(result,0,1),ms,None
    p,prep_ms=time_call(lambda:prepare(observed,cfg))
    (result,diag),ms=time_call(lambda:restore(p,return_diagnostics=True))
    yield 'v2_full',result,prep_ms+ms,diag
    if ablation:
        (result,diag),ms=time_call(lambda:restore(p,conservative=False,return_diagnostics=True))
        yield 'v2_without_guard',result,prep_ms+ms,diag
        variants=[('v2_fixed_selection128',dict(force_scale=128)),
                  ('v2_without_chirp',dict(enable_chirp=False)),
                  ('v2_constant_amplitude',dict(enable_amplitude_model=False))]
        for method,changes in variants:
            variant=replace(cfg,estimation=replace(cfg.estimation,**changes))
            pv,pv_ms=time_call(lambda:prepare(observed,variant))
            (result,diag),ms=time_call(lambda:restore(pv,return_diagnostics=True))
            yield method,result,pv_ms+ms,diag


def run_audit(ablation=False):
    frozen,cfg=frozen_config()
    audit=records('audit_test')[:20] if ablation else records('audit_test')
    stage='ablations' if ablation else 'audit'
    folder=OUT/stage;folder.mkdir(parents=True,exist_ok=True)
    json_dump(folder/'run.json',dict(config=frozen,ids=[r['id'] for r in audit],
        split='audit_test',stage=stage,manifest_sha256=hashlib.sha256((DATA/'optimization_manifest.json').read_bytes()).hexdigest()))
    all_rows=[];exceptions=[];synthesis_records=[]
    for index,record in enumerate(audit):
        clean=load_clean(record,DATA)
        cases=[(k,.12) for k in KINDS] if ablation else [(k,a) for k in KINDS for a in AMPLITUDES]
        for kind,amplitude in cases+[('clean',0.)]:
            observed,meta,masks=make_sample(clean,record['id'],kind,amplitude)
            synthesis_records.append(dict(image_id=record['id'],**meta))
            try:
                for method,result,ms,diag in outputs(observed,cfg,ablation):
                    row=dict(image_id=record['id'],split='audit_test',kind=kind,amplitude=amplitude,
                             method=method,runtime_ms=ms,error='')
                    row.update(metric_row(clean,observed,result,masks));all_rows.append(row)
                    if index<3 and not ablation and amplitude in (.12,0.):
                        ex=folder/'examples'/f'{record["id"]}_{kind}';ex.mkdir(parents=True,exist_ok=True)
                        np.save(ex/f'{method}.npy',result)
                        if method=='input':np.save(ex/'reference.npy',clean)
                        if method=='v2_full' and diag:
                            np.savez_compressed(ex/'diagnostics.npz',**{k:v for k,v in diag.items() if isinstance(v,np.ndarray)})
                            json_dump(ex/'decisions.json',dict(global_assessment=diag.get('global_assessment'),tiles=diag.get('tiles')))
            except Exception as exc:
                exceptions.append(dict(image_id=record['id'],kind=kind,amplitude=amplitude,error=repr(exc)))
            write_csv(folder/'metrics.csv',all_rows,FIELDS)
            with (folder/'synthesis.jsonl').open('w',encoding='utf-8') as metadata_file:
                for metadata in synthesis_records:metadata_file.write(json.dumps(metadata,ensure_ascii=False)+'\n')
        print(f'v2 {stage} {index+1}/{len(audit)} {record["id"]}',flush=True)
    json_dump(folder/'exceptions.json',exceptions)
    expected=len(audit)*(5 if ablation else 13)*5
    keys={(r['image_id'],r['kind'],r['amplitude'],r['method']) for r in all_rows}
    json_dump(folder/'validation.json',dict(expected=expected,actual=len(all_rows),unique=len(keys),
        complete=len(keys)==len(all_rows)==expected and not exceptions,exceptions=len(exceptions)))
    if exceptions or len(keys)!=expected:raise RuntimeError('Incomplete v2 experiment')


def summarize():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for stage in ('audit','ablations'):
        rows=read_csv(OUT/stage/'metrics.csv');groups=defaultdict(list)
        for r in rows:groups[(r['kind'],r['method'])].append(r)
        summary=[]
        for (kind,method),group in sorted(groups.items()):
            d=dict(kind=kind,method=method,n=len(group))
            for key in ('psnr_db','ssim','mse','mae','contaminated_mse','clean_mse','clean_modification_mae','degraded','runtime_ms'):
                values=[float(r[key]) for r in group if r[key]!='']
                d['mean_'+key]=float(np.mean(values)) if values else ''
            mae=np.array([float(r['mae']) for r in group])
            d.update(median_mae=float(np.median(mae)),p95_mae=float(np.quantile(mae,.95)),max_mae=float(mae.max()),
                     modified_images=int(np.sum(mae>1e-12)))
            summary.append(d)
        write_csv(OUT/stage/'summary.csv',summary)
        print(stage)
        for r in summary:
            print(r['kind'],r['method'],round(r['mean_psnr_db'],3),'MAE',round(r['mean_mae'],6),flush=True)
        lookup={(r['image_id'],r['kind'],r['amplitude'],r['method']):r for r in rows}
        paired=[];rng=np.random.default_rng(2026111603)
        for kind in KINDS:
            for method in sorted({r['method'] for r in rows}-{'input','v2_full'}):
                differences=defaultdict(list)
                for (image_id,k,a,m),r in lookup.items():
                    if k!=kind or m!='v2_full':continue
                    other=lookup.get((image_id,k,a,method))
                    if other:differences[image_id].append(float(r['psnr_db'])-float(other['psnr_db']))
                values=np.array([np.mean(v) for v in differences.values()])
                samples=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(axis=1)
                lo,hi=np.quantile(samples,[.025,.975])
                paired.append(dict(kind=kind,baseline=method,n_images=len(values),mean_delta_psnr=float(values.mean()),
                                   image_bootstrap_low=float(lo),image_bootstrap_high=float(hi)))
        write_csv(OUT/stage/'paired_comparisons.csv',paired)
    folder=OUT/'audit';rows=read_csv(folder/'metrics.csv')
    summary=read_csv(folder/'summary.csv');lookup={(r['kind'],r['method']):r for r in summary}
    methods=['v1_full','global_adaptive','published_arpenos','v2_full']
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');x=np.arange(4)
    for j,method in enumerate(methods):
        gains=[float(lookup[(kind,method)]['mean_psnr_db'])-float(lookup[(kind,'input')]['mean_psnr_db']) for kind in KINDS]
        axs[0].bar(x+(j-1.5)*.19,gains,width=.18,label=method)
    axs[0].set_xticks(x,KINDS);axs[0].set_ylabel('Mean PSNR gain (dB)');axs[0].legend(fontsize=8)
    axs[1].bar(np.arange(4),[float(lookup[('clean',m)]['mean_mae']) for m in methods])
    axs[1].set_xticks(np.arange(4),methods,rotation=22,ha='right');axs[1].set_ylabel('MAE introduced on clean controls')
    fig.savefig(OUT/'overview.png',dpi=170);plt.close(fig)
    for ex in sorted((folder/'examples').glob('*')):
        names=['reference','input','global_adaptive','published_arpenos','v1_full','v2_full']
        fig,axs=plt.subplots(2,3,figsize=(10,6.7),layout='constrained');ref=np.load(ex/'reference.npy')
        for ax,name in zip(axs.flat,names):
            a=np.load(ex/f'{name}.npy');mse=np.mean((a-ref)**2)
            title=name+(f' {-10*np.log10(mse):.2f} dB' if mse else ' unchanged')
            ax.imshow(a,cmap='gray',vmin=0,vmax=1);ax.set_title(title,fontsize=9);ax.axis('off')
        fig.suptitle(ex.name+' | range [0,1]')
        fig.savefig(ex/'comparison.png',dpi=150);plt.close(fig)
        d=np.load(ex/'diagnostics.npz');fig,axs=plt.subplots(2,3,figsize=(10,6),layout='constrained')
        names=['residual','selected_scale','suppression','model_chirp','model_affine_amplitude','frequency_x']
        limits=[(-.18,.18),(0,256),(0,1),(0,1),(0,1),(-.2,.2)]
        for ax,name,limit in zip(axs.flat,names,limits):
            im=ax.imshow(d[name],cmap='coolwarm' if name in ('residual','frequency_x') else 'viridis',vmin=limit[0],vmax=limit[1])
            ax.axis('off');ax.set_title(name,fontsize=9);fig.colorbar(im,ax=ax,shrink=.8)
        fig.suptitle(ex.name+' | overlap-weighted diagnostics; frequency in cycles/pixel')
        fig.savefig(ex/'diagnostics.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('calibrate','audit','ablations','summarize'))
    args=p.parse_args()
    {'calibrate':calibrate,'audit':run_audit,'ablations':lambda:run_audit(True),'summarize':summarize}[args.stage]()


if __name__=='__main__':main()
