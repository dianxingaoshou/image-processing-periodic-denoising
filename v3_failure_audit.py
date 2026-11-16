"""Reproduce the three largest clean-control modifications per validation set."""
import hashlib

import numpy as np

from v3_experiments import OUT,DATA,records,read_csv,load_clean,load_frozen,outputs,dump


def main():
    _,cfg=load_frozen();folder=OUT/'failure_audit';folder.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selections=[]
    for stage,split in (('audit','audit_v3'),('textures','texture_test')):
        candidates=[r for r in read_csv(OUT/stage/'metrics.csv') if r['kind']=='clean' and r['method']=='v3_decoupled']
        chosen=sorted(candidates,key=lambda r:(-float(r['mae']),r['image_id']))[:3]
        by_id={r['id']:r for r in records(split)}
        for row in chosen:
            observed=load_clean(by_id[row['image_id']],DATA)
            fig,axs=plt.subplots(2,3,figsize=(10,6.7),layout='constrained')
            for ax,(method,result,ms,diag,strength,combination) in zip(axs.flat,outputs(observed,cfg)):
                mae=float(np.abs(result-observed).mean())
                ax.imshow(result,cmap='gray',vmin=0,vmax=1);ax.axis('off');ax.set_title(f'{method} MAE={mae:.4g}',fontsize=9)
                if method=='v3_decoupled':
                    if not np.isclose(mae,float(row['mae']),rtol=1e-10,atol=1e-12):raise RuntimeError('Worst-case reproduction mismatch')
                    change=result-observed
            im=axs.flat[-1].imshow(change,cmap='coolwarm',vmin=-.15,vmax=.15)
            axs.flat[-1].set_title('v3 change [-0.15,0.15]',fontsize=9);axs.flat[-1].axis('off');fig.colorbar(im,ax=axs.flat[-1],shrink=.8)
            fig.suptitle(f'{stage} | {row["image_id"]} | selected by largest v3 modification MAE')
            filename=f'{stage}_{row["image_id"]}.png';fig.savefig(folder/filename,dpi=150);plt.close(fig)
            selections.append(dict(stage=stage,image_id=row['image_id'],mae=float(row['mae']),figure=filename,reproduced=True))
    dump(folder/'selection.json',dict(criterion='top3 clean-control MAE per set; selected after evaluation for descriptive failure analysis',
        script_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),cases=selections))
    print('Reproduced6 worst-control examples; no parameter changes')


if __name__=='__main__':main()
