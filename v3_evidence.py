"""Matched-removal evidence and declared size/frequency diagnostics for v3."""
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image

from v3_experiments import DATA,OUT,load_frozen,records,write_csv,dump,outputs,BASE_FIELDS
from denoiser_v3 import prepare,restore
from evaluation_v3 import metric_row,estimation_metrics
from run_experiments import load_clean
from validation_data_v3 import make_sample

STRENGTHS=(0.,.25,.5,.75,1.,1.25,1.5,2.)


def frontier(rows):
    points=sorted((float(r['contaminated_mse']),float(r['clean_mse']),float(r['input_contaminated_mse'])) for r in rows)
    best=float('inf');curve={}
    for noisy,damage,original in points:
        if damage<=best+1e-18:
            removal=1-noisy/original
            curve[removal]=min(curve.get(removal,float('inf')),damage);best=min(best,damage)
    return sorted(curve.items())


def match(rows,group,image_id,pair):
    modes=('decoupled',pair);curves={m:frontier([r for r in rows if r['mode']==m]) for m in modes}
    if any(len(v)<2 for v in curves.values()):return []
    lo=max(0.,*[v[0][0] for v in curves.values()]);hi=min(v[-1][0] for v in curves.values())
    if hi<=lo+1e-10:return []
    result=[]
    for removal in np.linspace(lo,hi,21):
        values={m:float(np.interp(removal,*map(np.asarray,zip(*v)))) for m,v in curves.items()}
        result.append(dict(group=group,image_id=image_id,comparison='decoupled_vs_'+pair,
            matched_contaminated_mse_reduction_fraction=float(removal),decoupled_clean_mse=values['decoupled'],
            coupled_clean_mse=values.get('coupled',''),unprotected_clean_mse=values.get('unprotected',''),
            clean_mse_difference=values['decoupled']-values[pair],common_removal_min=lo,common_removal_max=hi))
    return result


def matched():
    _,cfg=load_frozen();folder=OUT/'supplement';folder.mkdir(parents=True,exist_ok=True)
    rows=[];matched=[];coverage=[]
    for i,record in enumerate(records('audit_v3')[:10]):
        clean=load_clean(record,DATA);observed,metadata,masks=make_sample(clean,record['id'],'local',.12)
        p=prepare(observed,cfg);baseline=float(np.mean((observed[masks['contaminated']]-clean[masks['contaminated']])**2))
        case=[]
        for mode in ('coupled','decoupled','unprotected'):
            for strength in STRENGTHS:
                result,diag=restore(p,strength=strength,combination='coupled' if mode=='coupled' else 'decoupled',
                    conservative=mode!='unprotected',return_diagnostics=True)
                error=(result-clean)**2
                row=dict(image_id=record['id'],mode=mode,strength=strength,seed=metadata['seed'],
                    contaminated_mse=float(error[masks['contaminated']].mean()),clean_mse=float(error[masks['clean']].mean()),
                    input_contaminated_mse=baseline,risky_full_blocks=diag['risky_full_subtraction_blocks'])
                rows.append(row);case.append(row)
        for pair in ('coupled','unprotected'):
            matches=match(case,'individual',record['id'],pair);matched.extend(matches)
            coverage.append(dict(image_id=record['id'],comparison='decoupled_vs_'+pair,matched_points=len(matches),
                reason='positive common removal interval' if matches else 'no positive common removal interval'))
        print(f'v3 matched {i+1}/10',flush=True)
    groups=defaultdict(list)
    for r in rows:groups[(r['mode'],r['strength'])].append(r)
    means=[]
    for (mode,strength),group in groups.items():
        means.append(dict(mode=mode,strength=strength,**{key:float(np.mean([r[key] for r in group])) for key in
            ('contaminated_mse','clean_mse','input_contaminated_mse')}))
    for pair in ('coupled','unprotected'):matched.extend(match(means,'subset_mean','all10',pair))
    write_csv(folder/'weight_curves.csv',rows);write_csv(folder/'weight_curves_mean.csv',means)
    write_csv(folder/'matched_coverage.csv',coverage)
    if matched:write_csv(folder/'matched_removal.csv',matched)
    else:write_csv(folder/'matched_removal.csv',[],['group','image_id','comparison','matched_contaminated_mse_reduction_fraction',
        'decoupled_clean_mse','coupled_clean_mse','unprotected_clean_mse','clean_mse_difference','common_removal_min','common_removal_max'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for mode in ('coupled','decoupled','unprotected'):
        r=[x for x in means if x['mode']==mode]
        axs[0].plot([1-x['contaminated_mse']/x['input_contaminated_mse'] for x in r],[x['clean_mse'] for x in r],'-o',label=mode)
    axs[0].set_title('Actual operating points; fixed10 subset');axs[0].set_xlabel('Contaminated MSE reduction');axs[0].set_ylabel('Clean-region MSE');axs[0].legend(fontsize=8)
    for ax,pair in zip(axs[1:],('coupled','unprotected')):
        r=[x for x in matched if x['group']=='subset_mean' and x['comparison']=='decoupled_vs_'+pair]
        if r:
            x=[v['matched_contaminated_mse_reduction_fraction'] for v in r]
            ax.plot(x,[v['decoupled_clean_mse'] for v in r],label='decoupled')
            ax.plot(x,[v[pair+'_clean_mse'] for v in r],label=pair);ax.legend(fontsize=8)
        ax.set_title('Common removal: decoupled vs '+pair);ax.set_xlabel('Matched contaminated MSE reduction');ax.set_ylabel('Clean-region MSE')
    for ax in axs:ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(alpha=.2)
    fig.savefig(folder/'matched_removal.png',dpi=170);plt.close(fig)
    dump(folder/'scope.json',dict(ids=[r['id'] for r in records('audit_v3')[:10]],strengths=STRENGTHS,
        matching='Pareto frontier and linear interpolation in common positive-removal interval only; no new operating point selected'))


def boundaries():
    _,cfg=load_frozen();folder=OUT/'boundaries';folder.mkdir(parents=True,exist_ok=True)
    record=records('dev')[0];base=load_clean(record,DATA);rows=[]
    for shape in ((64,64),(96,96),(128,256),(384,512)):
        clean=np.asarray(Image.fromarray(base.astype('float32')).resize((shape[1],shape[0]),Image.Resampling.LANCZOS),dtype=float)
        clean=np.clip(clean,0,1);y,x=np.indices(shape)
        for f in (.03,.25,.38):
            noisy=np.clip(clean+.12*np.sin(2*np.pi*f*x+.7),0,1)
            masks=dict(contaminated=np.ones(shape,bool),clean=np.zeros(shape,bool),true_fy=np.zeros(shape),true_fx=np.full(shape,f))
            for method,result,ms,diag,strength,combination in outputs(noisy,cfg):
                row=dict(image_id=f'dev_{record["id"]}_{shape[0]}x{shape[1]}_f{f}',kind='frequency_size',amplitude=.12,
                    method=method,strength=strength,combination=combination,runtime_ms=ms,error='')
                row.update(metric_row(clean,noisy,result,masks));row.update(estimation_metrics(diag,masks));rows.append(row)
        print('v3 size/frequency diagnostic',shape,flush=True)
    write_csv(folder/'metrics.csv',rows,BASE_FIELDS)
    dump(folder/'scope.json',dict(split='development_diagnostic',source_image_id=record['id'],
        shapes=[[64,64],[96,96],[128,256],[384,512]],frequencies=[.03,.25,.38],
        interpretation='Boundary diagnostic on an already-used development image; not new independent performance evidence'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('matched','boundaries'))
    args=p.parse_args();{'matched':matched,'boundaries':boundaries}[args.stage]()
