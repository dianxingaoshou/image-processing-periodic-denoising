"""Re-run and visualize the worst full-method regressions, with frozen settings.

These cases are deliberately selected by error, unlike the predetermined main
example gallery. No parameter is selected or changed by this script.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from data_pipeline import load_manifest
from local_spectral import LocalConfig, denoise
from run_experiments import load_clean, make_sample


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--results',type=Path,default=Path('results/main'))
    args=p.parse_args()
    with (args.results/'metrics.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['method']=='local_adaptive' and not r['error']]
    cfg=json.loads((args.results/'run_config.json').read_text(encoding='utf-8'))['config']
    from dataclasses import replace
    local_cfg=replace(LocalConfig(**cfg['local_config']),strength=cfg['local_strength'])
    records={str(r['id']):r for r in load_manifest(args.data)['images']}
    selected=[]
    fig,axes=plt.subplots(5,4,figsize=(11,13),layout='constrained')
    folder=args.results/'failure_audit';folder.mkdir(exist_ok=True)
    for i,kind in enumerate(('global','local','chirp','orientation','clean')):
        group=[r for r in rows if r['kind']==kind]
        worst=max(group,key=lambda r:float(r['mse'])-float(r['input_mse']))
        clean=load_clean(records[worst['image_id']],args.data)
        observed,meta,_=make_sample(clean,worst['image_id'],kind,float(worst['amplitude']))
        restored,diag=denoise(observed,local_cfg,return_diagnostics=True)
        rerun_mse=float(np.mean((restored-clean)**2))
        if not np.isclose(rerun_mse,float(worst['mse']),rtol=1e-9,atol=1e-12):
            raise RuntimeError('Frozen-run reproduction mismatch')
        selected.append(dict(**worst,rerun_mse=rerun_mse,
            input_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),synthesis=meta))
        np.savez_compressed(folder/f'{kind}_worst.npz',reference=clean,input=observed,
                            output=restored,residual=observed-restored,scale=diag['selected_scale'])
        for j,(a,title) in enumerate(((clean,'Reference'),(observed,'Input'),(restored,'Full method'),
                                     (observed-restored,'Subtracted residual [-0.18,0.18]'))):
            axes[i,j].imshow(a,cmap='coolwarm' if j==3 else 'gray',vmin=-.18 if j==3 else 0,vmax=.18 if j==3 else 1)
            axes[i,j].set_title(title,fontsize=9)
            axes[i,j].set_xticks([]);axes[i,j].set_yticks([])
        axes[i,0].set_ylabel(f'{kind}\nID {worst["image_id"]}\nA={worst["amplitude"]}',fontsize=9)
    fig.suptitle('Largest output MSE minus input MSE per condition | settings unchanged',fontsize=11)
    fig.savefig(folder/'worst_cases.png',dpi=150);plt.close(fig)
    (folder/'selection.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Reproduced all five worst-case selections without changing settings')


if __name__=='__main__':main()
