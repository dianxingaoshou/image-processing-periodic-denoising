"""Plot measured contrast-context ablations without running or tuning inference."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METHODS=('baseline','ratio_relaxed','context_no_width','context_full')
VARIANTS=METHODS[1:]
LABELS=dict(baseline='Original',ratio_relaxed='Flank ratio 0.9',
    context_no_width='Amplitude context\n(no width cap)',context_full='Amplitude context\n(width cap 32)')
COLORS=dict(baseline='#34465f',ratio_relaxed='#c88636',context_no_width='#9472a1',context_full='#258978')


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='')as stream:return list(csv.DictReader(stream))


def truth(value):return str(value).lower()=='true'


def unique(rows,**match):
    selected=[r for r in rows if all(str(r.get(k))==str(v)for k,v in match.items())]
    if len(selected)!=1:raise ValueError(f'Expected one row for {match}, got {len(selected)}')
    return selected[0]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results/context'))
    parser.add_argument('--figure',type=Path,default=Path('docs/figures/context-ablation.png'))
    parser.add_argument('--diagnosis',type=Path,default=Path('results/rejection_diagnosis'))
    args=parser.parse_args()
    regression=[r for r in read_csv(args.out/'regression/metrics.csv')if float(r['strength'])==1.]
    intervals=read_csv(args.out/'regression/intervals.csv')
    paired=read_csv(args.out/'check/paired_deltas.csv')
    controls=[r for r in read_csv(args.out/'controls/metrics.csv')if float(r['strength'])==1.]
    diagnostic=read_csv(args.diagnosis/'case_rows.csv')
    case=unique(read_csv(args.diagnosis/'case_metrics.csv'),image_id='69022',kind='edge8',variant=3)
    frequency=float(case['estimated_frequency'])
    old=[r for r in intervals if r['image_id']=='69022'and r['kind']=='edge8'and r['method']=='baseline'
         and r['axis']=='0'and abs(float(r['frequency'])-frequency)<1e-7]
    if len(old)!=1:raise ValueError('69022 diagnostic interval is missing or ambiguous')
    old=old[0]
    current=unique(intervals,image_id='69022',kind='edge8',method='context_full',axis=0,
                   frequency=old['frequency'],start=old['start'],stop=old['stop'])
    rows=sorted([r for r in diagnostic if r['image_id']=='69022'and r['kind']=='edge8'and r['variant']=='3'
                 and r['axis']=='0'and int(r['row'])>=244],key=lambda r:int(r['row']))
    if len({r['row']for r in rows})!=len(rows)or len(rows)<8:raise ValueError('Diagnostic rows incomplete/duplicated')
    x=np.array([int(r['row'])for r in rows]);amp=np.array([float(r['amplitude'])for r in rows])
    qualified=np.array([truth(r['qualified'])for r in rows])
    fig,axes=plt.subplots(2,2,figsize=(15,10),constrained_layout=True)
    axis=axes[0,0]
    axis.axvspan(int(current['context_start'])-.5,int(current['context_stop'])-.5,color='#e0b575',alpha=.22,
                 label='Amplitude context: contrast only')
    axis.axvspan(int(current['start'])-.5,int(current['stop'])-.5,color='#258978',alpha=.20,
                 label='Qualified rows: eligibility unchanged')
    axis.plot(x,amp,'o-',color='#34465f',markersize=4,label='Observed row amplitude')
    axis.axhline(.02,color='#777777',linestyle='--',linewidth=1,label='Same amplitude floor 0.02')
    axis.scatter(x[qualified],amp[qualified],color='#258978',s=35,zorder=4)
    axis.scatter([255],[float(next(r['amplitude']for r in rows if r['row']=='255'))],marker='x',
                 color='#bd4b42',s=75,linewidth=2,zorder=5)
    axis.annotate('Row 255 fails stability;\nits weight remains zero',xy=(255,amp[-1]),xytext=(246,.113),
                  arrowprops=dict(arrowstyle='->',color='#bd4b42'),fontsize=9,color='#823b36')
    axis.set(xlabel='Normal row index',ylabel='Observed amplitude',xlim=(244,256),ylim=(0,.165))
    axis.set_title('(a) 69022 / edge8: context extent is not output support',loc='left',fontweight='bold',fontsize=11)
    axis.legend(fontsize=8,loc='upper left',bbox_to_anchor=(0.,.61),framealpha=.95)
    axis.text(.02,.98,f"Outside / inside: {float(old['outside_to_inside_ratio']):.3f} → {float(current['outside_to_inside_ratio']):.3f}\nFlank-ratio threshold stays 0.35",transform=axis.transAxes,va='top',fontsize=9)
    axis.grid(axis='y',alpha=.2)

    axis=axes[0,1];kinds=('edge8','edge16','edge32');positions=np.arange(3);maximum=0.
    for i,method in enumerate(METHODS):
        values=[float(unique(regression,image_id='69022',kind=kind,method=method,variant=3)['contaminated_mse'])for kind in kinds]
        maximum=max(maximum,max(values))
        axis.bar(positions+(i-1.5)*.19,values,width=.18,color=COLORS[method],label=LABELS[method].replace('\n',' '))
    axis.set_xticks(positions,['8px edge','16px edge','32px edge'])
    axis.set_ylabel('Contaminated-region MSE (lower is better)')
    axis.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
    axis.set_ylim(0,maximum*1.55)
    axis.set_title('(b) 69022 regression: fixed proposals and row gate',loc='left',fontweight='bold',fontsize=11)
    axis.legend(fontsize=8,ncols=2,loc='upper right')
    axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)

    axis=axes[1,0];fields=('new_recovery','old_success_regressed','clean_increased','adjacent_increased')
    names=('New recovery','Previous success regressed','New clean damage','New adjacent damage')
    bar_colors=('#258978','#8294a6','#bd7848','#9b668d')
    positions=np.arange(len(VARIANTS));denominator=None;previous=None
    for fi,(field,label,color)in enumerate(zip(fields,names,bar_colors)):
        counts=[]
        for method in VARIANTS:
            measured=[r for r in paired if r['method']==method and truth(r['local_evaluable'])]
            identities={(r['image_id'],r['kind'])for r in measured}
            if len(measured)!=20 or len(identities)!=20:raise ValueError('Expected all20 paired local inputs')
            denominator=len(measured);previous=sum(truth(r['old_success'])for r in measured)
            counts.append(sum(truth(r[field])for r in measured))
        offset=(fi-1.5)*.18
        axis.bar(positions+offset,counts,.17,label=label,color=color)
        for index,count in enumerate(counts):axis.text(index+offset,count+.24,str(count),ha='center',fontsize=9)
    axis.set_xticks(positions,[LABELS[m]for m in VARIANTS],fontsize=9)
    axis.set_ylim(0,20)
    axis.set_ylabel('Number of image-condition pairs')
    axis.set_title('(c) New 4-image check: only 2 / 20 newly recovered',loc='left',fontweight='bold',fontsize=11)
    axis.legend(fontsize=8,ncols=2,loc='upper center')
    axis.text(.02,.64,f'20 local inputs; {previous} previous successes checked.\nFull method: 2 new recoveries, 0 previous regressions.',
              transform=axis.transAxes,fontsize=9,color='#34465f')
    axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)

    axis=axes[1,1]
    values=[float(unique(controls,kind='broad_phase_banded_clean',method=method)['mae'])for method in METHODS]
    compact=[float(unique(controls,kind='compact_legitimate_stripe_clean',method=method)['mae'])for method in METHODS]
    axis.bar(np.arange(4),values,color=[COLORS[m]for m in METHODS],width=.62)
    axis.set_xticks(np.arange(4),[LABELS[m]for m in METHODS],fontsize=9)
    axis.set_ylim(0,max(values)*1.6)
    axis.set_ylabel('Clean-control modification MAE')
    axis.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
    axis.set_title('(d) Broad phase-banded synthetic texture: no added noise',loc='left',fontweight='bold',fontsize=11)
    axis.text(.02,.96,f'Compact legitimate stripes remain ambiguous:\nall four methods modify them (MAE ≈ {np.mean(compact):.6f}).',
              transform=axis.transAxes,va='top',fontsize=9,color='#823b36')
    axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
    for a in axes.ravel():a.spines[['top','right']].set_visible(False)
    fig.suptitle('Contrast-context refinement: same row qualification and bounded contrast context\n'
                 '69022 regression, one new 4-image check, and synthetic texture stress controls; strength = 1',fontsize=13)
    fig.supxlabel('Amplitude context only determines where contrast is measured. It never fills qualification gaps or proves semantic noise identity. Repeated cases are not independent photos.',fontsize=9)
    args.figure.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.figure,dpi=160);plt.close(fig)
    print(args.figure.resolve())


if __name__=='__main__':main()
