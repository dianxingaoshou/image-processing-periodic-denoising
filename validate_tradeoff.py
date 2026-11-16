"""Frozen-parameter supplementary checks: matched removal and texture ambiguity.

Uses the first ten stored manifest test IDs, local/chirp/orientation at amplitude
0.12.  This is a fixed diagnostic subset, not a second tuning set.  All synthesis
seeds come from run_experiments.make_sample, so the samples match the main run.

The strength sweep uses existing restore() semantics: strength*confidence is
capped at one per tile; without_conservative is consequently saturated at
strength>=1.  No new operating point is selected from these results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from data_pipeline import load_manifest
from local_spectral import LocalConfig, prepare, restore
from run_experiments import dataset_records, load_clean, make_sample


STRENGTHS = (0., .125, .25, .375, .5, .625, .75, .875, 1., 1.25, 1.5, 2.)
MODES = (('conservative', True), ('without_conservative', False))


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array,dtype='<f8').tobytes()).hexdigest()


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_or_blank(error, mask):
    return float(error[mask].mean()) if np.any(mask) else ''


def _pareto(rows):
    """Return non-dominated (removal fraction, clean damage) in increasing x.

    Both contaminated MSE and clean MSE are minimized. Strictly worse damage
    points are discarded before interpolation. Equal-damage plateaus are kept
    to retain the sampled achievable removal range (including zero damage).
    Strength is never the matching axis.
    """
    points=sorted((float(r['contaminated_mse']),float(r['clean_mse']),
                   float(r['input_contaminated_mse'])) for r in rows
                  if r['clean_mse']!='' and r['contaminated_mse']!='')
    frontier=[]
    best_damage=float('inf')
    for contaminated,damage,input_mse in points:
        if damage <= best_damage+1e-18:
            frontier.append((1.-contaminated/input_mse,damage))
            best_damage=min(best_damage,damage)
    unique={}
    for removal,damage in frontier:
        unique[removal]=min(unique.get(removal,float('inf')),damage)
    return sorted(unique.items())


def _match(rows, group_label, image_id, kind):
    curves={mode:_pareto([r for r in rows if r['mode']==mode]) for mode,_ in MODES}
    if any(len(v)<2 for v in curves.values()):
        return []
    lo=max(v[0][0] for v in curves.values())
    hi=min(v[-1][0] for v in curves.values())
    if hi<=lo+1e-10:
        return []
    result=[]
    for removal in np.linspace(lo,hi,21):
        damage={}
        for mode,points in curves.items():
            xp,yp=map(np.asarray,zip(*points))
            damage[mode]=float(np.interp(removal,xp,yp))
        conservative=damage['conservative']; unprotected=damage['without_conservative']
        result.append(dict(group=group_label,image_id=image_id,kind=kind,
                           matched_contaminated_mse_reduction_fraction=float(removal),
                           conservative_clean_mse=conservative,
                           without_conservative_clean_mse=unprotected,
                           clean_mse_difference=conservative-unprotected,
                           clean_mse_ratio=conservative/unprotected if unprotected>1e-18 else '',
                           common_removal_min=float(lo),common_removal_max=float(hi)))
    return result


def strength_sweep(data_dir, config, output):
    cfg=LocalConfig(**config.get('local_config',{}))
    records=dataset_records(load_manifest(data_dir),'test')[:10]
    rows=[]
    matched=[]
    coverage=[]
    for index,record in enumerate(records):
        image_id=str(record.get('id',record.get('image_id')))
        clean=load_clean(record,data_dir)
        for kind in ('local','chirp','orientation'):
            observed,metadata,masks=make_sample(clean,image_id,kind,.12)
            prepared=prepare(observed,cfg)
            input_error=(observed-clean)**2
            input_contaminated=_mean_or_blank(input_error,masks['contaminated'])
            case=[]
            for mode,conservative in MODES:
                for strength in STRENGTHS:
                    result=restore(prepared,strength=strength,conservative=conservative)
                    error=(result-clean)**2
                    row=dict(image_id=image_id,kind=kind,amplitude=.12,seed=metadata['seed'],
                             mode=mode,strength=strength,
                             contaminated_mse=_mean_or_blank(error,masks['contaminated']),
                             clean_mse=_mean_or_blank(error,masks['clean']),
                             input_contaminated_mse=input_contaminated,
                             overall_mse=float(error.mean()),
                             clean_fraction=float(masks['clean'].mean()),
                             contaminated_fraction=float(masks['contaminated'].mean()))
                    rows.append(row);case.append(row)
            matches=_match(case,'individual',image_id,kind)
            matched.extend(matches)
            best={mode:max(1-float(r['contaminated_mse'])/float(r['input_contaminated_mse'])
                           for r in case if r['mode']==mode) for mode,_ in MODES}
            coverage.append(dict(image_id=image_id,kind=kind,
                                 clean_fraction=float(masks['clean'].mean()),
                                 conservative_best_reduction=best['conservative'],
                                 without_conservative_best_reduction=best['without_conservative'],
                                 matched_points=len(matches),
                                 reason='matched' if matches else ('no clean region' if not np.any(masks['clean'])
                                        else 'no positive common removal interval')))
        print(f'supplement sweep {index+1}/{len(records)} {image_id}',flush=True)
    _write_csv(output/'strength_curve.csv',rows)
    _write_csv(output/'matched_coverage.csv',coverage)
    aggregate=[]
    groups=defaultdict(list)
    for row in rows:
        if row['clean_mse']!='':
            groups[(row['kind'],row['mode'],row['strength'])].append(row)
    for (kind,mode,strength),group in groups.items():
        aggregate.append(dict(kind=kind,mode=mode,strength=strength,
                              contaminated_mse=float(np.mean([r['contaminated_mse'] for r in group])),
                              clean_mse=float(np.mean([r['clean_mse'] for r in group])),
                              input_contaminated_mse=float(np.mean([r['input_contaminated_mse'] for r in group]))))
    for kind in ('local','chirp','orientation'):
        matched.extend(_match([r for r in aggregate if r['kind']==kind],'subset_mean','all10',kind))
    _write_csv(output/'matched_removal.csv',matched)
    _write_csv(output/'strength_curve_mean.csv',aggregate)
    _plot_tradeoff(aggregate,matched,output)
    return [str(r.get('id',r.get('image_id'))) for r in records],rows,matched


def _plot_tradeoff(aggregate, matched, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for mode,_ in MODES:
        group=sorted([r for r in aggregate if r['kind']=='local' and r['mode']==mode],key=lambda r:r['strength'])
        x=[1-r['contaminated_mse']/r['input_contaminated_mse'] for r in group]
        y=[r['clean_mse'] for r in group]
        axes[0].plot(x,y,'o-',label=mode)
    axes[0].set(xlabel='Contaminated MSE reduction fraction',ylabel='Clean-region MSE',
                title='Actual sweep operating points')
    group=[r for r in matched if r['group']=='subset_mean' and r['kind']=='local']
    if group:
        x=[r['matched_contaminated_mse_reduction_fraction'] for r in group]
        for mode,_ in MODES:
            axes[1].plot(x,[r[f'{mode}_clean_mse'] for r in group],label=mode)
    axes[1].set(xlabel='Matched contaminated MSE reduction',ylabel='Clean-region MSE',
                title='Pareto curves in common removal interval')
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=8);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    fig.savefig(output/'matched_removal.png',dpi=160)
    plt.close(fig)


def _texture_inputs():
    yy,xx=np.indices((256,256),dtype=float)
    flat=np.full((256,256),.5)
    synthetic={
        'starter_failure_exact':flat+.15*np.sin(2*np.pi*(23*yy/256+37*xx/256)+.7),
        'pure_sinusoid':flat+.12*np.sin(2*np.pi*(.083*yy+.137*xx)+.4),
        'grid_multifrequency':flat+.075*np.sin(2*np.pi*(.11*yy+.025*xx))
             +.055*np.cos(2*np.pi*(.017*yy+.145*xx)+.7)
             +.035*np.cos(2*np.pi*(.11*yy+.145*xx)+1.1),
        'chirped_texture':flat+.12*np.sin(2*np.pi*(.075*yy+.10*xx+.5*.055*(xx-127.5)**2/255)+.3),
    }
    cases=[(name,a,flat.copy()) for name,a in synthetic.items()]
    # Never trigger a dataset download for the optional natural texture.
    from skimage import data
    brick=Path(data.__file__).parent/'brick.png'
    if brick.is_file():
        with Image.open(brick) as im:
            natural=np.asarray(im.convert('L').resize((256,256),Image.Resampling.LANCZOS),dtype=float)/255
        cases.append(('brick_packaged_skimage',natural,None))
    return cases


def texture_checks(config, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg=LocalConfig(**config.get('local_config',{}))
    strength=float(config['local_strength'])
    rows=[]; panels=[]; arrays={}; starter_rows=[]
    for name,observed,alternative_clean in _texture_inputs():
        p=prepare(observed,cfg)
        outputs={mode:restore(p,strength=strength,conservative=conservative) for mode,conservative in MODES}
        if name=='starter_failure_exact':
            from run_experiments import starter
            original,_,_=starter.denoise(observed,starter.Config(),adaptive=True)
            check_outputs={'input':observed,'starter_global_adaptive':original,**outputs}
            fig,axes=plt.subplots(1,4,figsize=(11,3.2),layout='constrained')
            for ax,(method,result) in zip(axes,check_outputs.items()):
                error=float(np.mean((result-observed)**2))
                psnr=-10*np.log10(error) if error>0 else float('inf')
                starter_rows.append(dict(case=name,method=method,mse=error,psnr_db=psnr,
                                         modification_mae=float(np.abs(result-observed).mean())))
                ax.imshow(result,cmap='gray',vmin=0,vmax=1);ax.axis('off')
                ax.set_title(f'{method}\nPSNR {psnr:.2f} dB',fontsize=9)
            fig.suptitle('Exact supplied failure: legitimate periodic texture, no added noise',fontsize=11)
            fig.savefig(output/'starter_failure_comparison.png',dpi=150)
            plt.close(fig)
            arrays[f'{name}_starter_global_adaptive']=original
        input_hash=_sha(observed)
        for interpretation,clean in [('legitimate_texture',observed),('noise_on_flat_field',alternative_clean)]:
            if clean is None:
                continue
            for mode,result in outputs.items():
                error=(result-clean)**2
                mse=float(error.mean())
                rows.append(dict(case=name,interpretation=interpretation,mode=mode,strength=strength,
                                 mse=mse,psnr_db=-10*np.log10(mse) if mse>0 else 'inf',
                                 modification_mae=float(np.abs(result-observed).mean()),
                                 input_mse=float(np.mean((observed-clean)**2)),
                                 pure_texture_score=p.pure_texture_score,
                                 input_sha256=input_hash,output_sha256=_sha(result)))
        if alternative_clean is not None:
            for mode in outputs:
                pair=[r for r in rows if r['case']==name and r['mode']==mode]
                assert len(pair)==2 and pair[0]['input_sha256']==pair[1]['input_sha256']
                assert pair[0]['output_sha256']==pair[1]['output_sha256']
        panels.append((name,observed,outputs['conservative'],outputs['without_conservative']))
        arrays[f'{name}_input']=observed
        for mode,result in outputs.items():arrays[f'{name}_{mode}']=result
        if alternative_clean is not None:arrays[f'{name}_alternative_reference']=alternative_clean
    _write_csv(output/'texture_metrics.csv',rows)
    _write_csv(output/'starter_failure_metrics.csv',starter_rows)
    np.savez_compressed(output/'texture_arrays.npz',**arrays)
    fig,axes=plt.subplots(len(panels),4,figsize=(11,2.85*len(panels)),layout='constrained')
    for j,(name,observed,conservative,unprotected) in enumerate(panels):
        imgs=(observed,conservative,unprotected,np.abs(conservative-observed))
        titles=('Legitimate input',f'Conservative s={strength:g}',f'Without conservative s={strength:g}','Absolute change [0, 0.12]')
        for i,(a,title) in enumerate(zip(imgs,titles)):
            axes[j,i].imshow(a,cmap='gray',vmin=0,vmax=.12 if i==3 else 1)
            axes[j,i].set_title(title,fontsize=9);axes[j,i].set_xticks([]);axes[j,i].set_yticks([])
        axes[j,0].set_ylabel(name,fontsize=9)
    fig.savefig(output/'texture_comparison.png',dpi=150)
    plt.close(fig)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='data')
    parser.add_argument('--config',default='configs/frozen.json')
    parser.add_argument('--out',default='results/supplement')
    args=parser.parse_args()
    output=Path(args.out);output.mkdir(parents=True,exist_ok=True)
    config_path=Path(args.config)
    config=json.loads(config_path.read_text(encoding='utf-8'))
    ids,curves,matched=strength_sweep(Path(args.data),config,output)
    textures=texture_checks(config,output)
    note=dict(test_ids=ids,amplitude=.12,kinds=['local','chirp','orientation'],strengths=STRENGTHS,
              configuration_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
              local_config=asdict(LocalConfig(**config.get('local_config',{}))),
              seed_source='run_experiments.make_sample; identical to main evaluation',
              sweep_rows=len(curves),matched_rows=len(matched),texture_rows=len(textures),
              no_parameter_tuning=True,
              matching='Discard points with strictly worse clean damage at worse contaminated MSE; '
                       'retain equal-damage plateau points to represent sampled achievable strengths. '
                       'Linearly interpolate clean MSE '
                       'at equal contaminated-MSE reduction only within both curves common removal interval. '
                       'Individual curves and curve of subset mean losses are both supplied; finite-sweep approximation.',
              no_clean_region='chirp and orientation cover the image; clean_mse is blank and no clean-region '
                              'matched-removal claim is made for them.',
              cap='restore subtracts min(strength*confidence,1) per tile; without_conservative '
                  'saturates at strength >=1. Sweep does not select a replacement operating point.',
              ambiguity='For synthetic texture pairs the observed pixels, configuration and restored outputs '
                        'are byte-identical. Only evaluation reference changes: legitimate texture versus '
                        'an additive interference on a flat field. This demonstrates non-identifiability, '
                        'not an ability to classify these interpretations.',
              brick='Only read the brick.png packaged inside installed skimage.data; no optional download.')
    (output/'validation_notes.json').write_text(json.dumps(note,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'Saved supplement to {output}: {len(curves)} sweep rows, {len(matched)} matched rows, {len(textures)} texture rows',flush=True)


if __name__=='__main__':
    main()
