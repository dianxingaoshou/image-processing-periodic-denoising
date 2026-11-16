"""Plot measured narrow-band ablations, with image-paired normalization.

This module only reads result CSVs. It neither invokes inference nor chooses
parameters, and it keeps BSDS checks and DTD regression controls separate.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


KINDS=('edge8','edge16','edge32','interior16','patch32')
KIND_LABELS=('Edge 8px','Edge 16px','Edge 32px','Interior 16px','Patch 32px')
SELECTED=('legacy32_grid','old_peaks_new_support','proposal_only','full')
VARIANTS=('old_peaks_new_support','proposal_only','profile_gate','compact_only','full')
TEXTURE=('legacy64','legacy32_grid','proposal_only','profile_gate','compact_only','full')
LABELS=dict(legacy64='Legacy 64px',legacy32_grid='Legacy 32px + grid',
    old_peaks_new_support='Old 2D + new fit',proposal_only='1D proposal only',
    profile_gate='1D + profile gate',compact_only='1D + compact support',full='1D + gate + support')
COLORS=dict(legacy32_grid='#34465f',old_peaks_new_support='#9472a1',
    proposal_only='#cb8439',full='#258978')


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def primary(rows,dataset):
    return [row for row in rows if row['dataset']==dataset and
        (row['method']=='input' or float(row['strength'])==(2. if row['method'].startswith('legacy') else 1.))]


def index_rows(rows):
    lookup={}
    for row in rows:
        key=(row['image_id'],row['kind'],row['method'])
        if key in lookup:raise ValueError(f'Duplicated primary row: {key}')
        lookup[key]=row
    return lookup


def ratios(lookup,image_ids,kind,method):
    values=[]
    for image_id in image_ids:
        observed=float(lookup[image_id,kind,'input']['contaminated_mse'])
        restored=float(lookup[image_id,kind,method]['contaminated_mse'])
        if not np.isfinite(observed) or observed<=0 or not np.isfinite(restored):
            raise ValueError('Normalized dirty MSE requires finite values and positive input error')
        values.append(restored/observed)
    return values


def truth(value):return str(value).lower()=='true'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results/narrow'))
    parser.add_argument('--figure',type=Path,default=Path('docs/figures/narrow-ablation.png'))
    args=parser.parse_args()
    check=primary(read_csv(args.out/'check/metrics.csv'),'BSDS')
    texture=[row for row in primary(read_csv(args.out/'regression/metrics.csv'),'DTD_regression') if row['kind']=='clean']
    paired=[row for row in read_csv(args.out/'check/paired_deltas.csv') if row['dataset']=='BSDS' and row['kind'] in KINDS]
    lookup=index_rows(check)
    image_ids=sorted({row['image_id'] for row in check});n_images=len(image_ids)
    texture_ids=sorted({row['image_id'] for row in texture})
    if not n_images or not texture_ids:raise ValueError('Check and texture control records must be nonempty')
    # Fail on incomplete cases instead of quietly changing per-image weights.
    for method in ('input',*SELECTED):
        for kind in KINDS:
            for image_id in image_ids:
                if (image_id,kind,method) not in lookup:
                    raise ValueError(f'Missing fixed case: {image_id}/{kind}/{method}')
    for method in TEXTURE:
        values=[row for row in texture if row['method']==method]
        if len(values)!=len(texture_ids) or len({row['image_id'] for row in values})!=len(texture_ids):
            raise ValueError(f'Incomplete or duplicated DTD cases: {method}')
    fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)

    axis=axes[0,0]
    for method in SELECTED:
        means=[float(np.mean(ratios(lookup,image_ids,kind,method))) for kind in KINDS]
        axis.plot(np.arange(len(KINDS)),means,marker='o',color=COLORS[method],linewidth=1.8,
                  markersize=5,label=LABELS[method])
    axis.axhline(1.,color='#777777',linestyle='--',linewidth=1,label='Noisy input = 1')
    axis.set_xticks(np.arange(len(KINDS)),KIND_LABELS)
    axis.tick_params(axis='x',labelsize=9)
    axis.set_ylabel('Mean of per-image dirty MSE / input dirty MSE')
    axis.set_ylim(bottom=0)
    axis.set_title('(a) Independent check: normalized regional restoration',loc='left',fontweight='bold',fontsize=11)
    axis.legend(fontsize=8,ncols=2,framealpha=.9)
    axis.grid(axis='y',alpha=.22)

    axis=axes[0,1]
    markers=('o','s','^','D','v');kind_colors=('#3775aa','#d2872f','#2c9974','#a36a9f','#697680')
    adjacent=[]
    for kind,label,marker,color in zip(KINDS,KIND_LABELS,markers,kind_colors):
        x=ratios(lookup,image_ids,kind,'full')
        y=[float(lookup[image_id,kind,'full']['adjacent_clean_mse']) for image_id in image_ids]
        adjacent.extend(y)
        if not np.isfinite(y).all() or np.any(np.asarray(y)<0):raise ValueError('Adjacent MSE must be finite and nonnegative')
        axis.scatter(x,y,label=label,marker=marker,color=color,s=43,alpha=.85,edgecolor='white',linewidth=.4)
    axis.axvline(1.,color='#777777',linestyle='--',linewidth=1)
    if any(value>1e-12 for value in adjacent):
        axis.set_yscale('symlog',linthresh=1e-12)
        axis.set_ylabel('Adjacent clean MSE (symlog; linear below 1e-12)')
    else:
        axis.set_ylabel('Adjacent clean MSE')
        axis.set_yticks([0.])
        axis.text(.04,.58,f'All {len(adjacent)} measured adjacent-region MSE values are zero.\nZero-error points may overlap.',
                  transform=axis.transAxes,fontsize=10,color='#34465f')
    axis.set_ylim(bottom=0)
    axis.set_xlim(left=0)
    axis.set_xlabel('Full method dirty MSE / noisy-input dirty MSE')
    axis.set_title('(b) Full method: each image-condition pair',loc='left',fontweight='bold',fontsize=11)
    axis.legend(fontsize=8,ncols=2,framealpha=.9)
    axis.grid(alpha=.18)

    axis=axes[1,0];strict=[];relative=[];denominators=[]
    for method in VARIANTS:
        values=[row for row in paired if row['method']==method and row.get('contaminated_mse_delta','')!=''
                and row.get('clean_mse_delta','')!='' and row.get('adjacent_clean_mse_delta','')!='']
        identities={(row['image_id'],row['kind']) for row in values}
        if len(values)!=len(KINDS)*n_images or len(identities)!=len(values):
            raise ValueError(f'Incomplete or duplicated paired outcomes: {method}')
        denominators.append(len(values));strict.append(sum(truth(r['input_net_joint']) for r in values))
        relative.append(sum(truth(r['relative_joint']) for r in values))
    positions=np.arange(len(VARIANTS))
    axis.barh(positions-.17,strict,height=.3,color='#258978')
    axis.barh(positions+.17,relative,height=.3,color='#8294a6')
    for index,(s,r,d) in enumerate(zip(strict,relative,denominators)):
        axis.text(s+.25,index-.17,f'{s}/{d}',va='center',fontsize=8)
        axis.text(r+.25,index+.17,f'{r}/{d}',va='center',fontsize=8)
    axis.set_yticks(positions,[LABELS[m] for m in VARIANTS]);axis.invert_yaxis()
    axis.tick_params(axis='y',labelsize=9)
    axis.set_xlim(0,max(denominators)+2.5)
    axis.set_xlabel('Green: input-net joint; gray: legacy-relative joint',fontsize=9)
    axis.set_title('(c) Restoration with both clean-region constraints',loc='left',fontweight='bold',fontsize=11)
    axis.grid(axis='x',alpha=.22);axis.set_axisbelow(True)

    axis=axes[1,1];means=[];tails=[]
    for method in TEXTURE:
        values=sorted(float(row['mae']) for row in texture if row['method']==method)
        means.append(float(np.mean(values)));tails.append(float(np.mean(values[-max(1,int(np.ceil(.2*len(values)))):])))
    positions=np.arange(len(TEXTURE))
    axis.barh(positions-.17,means,height=.3,color='#587a9b')
    axis.barh(positions+.17,tails,height=.3,color='#bf8953')
    axis.set_yticks(positions,[LABELS[m] for m in TEXTURE]);axis.invert_yaxis()
    axis.tick_params(axis='y',labelsize=9)
    axis.set_xlabel('Modification MAE: blue = mean; orange = worst 20% mean',fontsize=9)
    axis.set_title(f'(d) DTD regression: {len(texture_ids)} photos, no added noise',loc='left',fontweight='bold',fontsize=11)
    axis.grid(axis='x',alpha=.22);axis.set_axisbelow(True)
    for axis in axes.ravel():axis.spines[['top','right']].set_visible(False)
    fig.suptitle(f'Narrow-band diagnosis-led ablations: {n_images} independent BSDS images; amplitude = 0.12\n'
                 'Legacy strength = 2; new-channel strength = 1. The new channel is a specialized input-only method.',fontsize=13)
    fig.supxlabel('Equal image weights within each condition. Repeated conditions are not independent photos; zero-error scatter points may overlap. No universal safety claim.',fontsize=9)
    args.figure.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.figure,dpi=160);plt.close(fig)
    print(args.figure.resolve())


if __name__=='__main__':main()
