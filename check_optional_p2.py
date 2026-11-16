"""Development-only diagnostic for the optional V3 candidate reranking.

The first five stored development IDs, four middle-amplitude degradations and
clean controls are fixed before execution. All methods subtract eligible raw
residuals with weight one: this isolates estimation, not the complete guarded
denoiser. No operating parameter is selected by this script.

P2 proposes FFT candidates using each complete analysis window. Only subsequent
phase/amplitude fitting and scoring use a fixed spatially stratified holdout.
This is a prediction diagnostic conditional on full-window FFT candidates, not
fully independent cross-validation and not an ability to identify noise.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from local_estimation_v2 import EstimationConfig as ConfigV2
from local_estimation_v2 import prepare as prepare_v2, restore as restore_v2
from local_estimation_v3 import EstimationConfig as ConfigV3
from local_estimation_v3 import prepare as prepare_v3, restore as restore_v3
from run_experiments import load_clean, make_sample


ROOT=Path(__file__).resolve().parent
SCOPE=('FFT candidates use all analysis-window pixels; only subsequent parameter fitting '
       'uses fixed spatially stratified held-out pixels. Conditional prediction diagnostic, '
       'not independent cross-validation or noise identification. Raw residuals only; no guard.')
METHODS=(
    ('v2_raw',ConfigV2(),prepare_v2,restore_v2,'local_estimation_v2.py'),
    ('v3_P0_raw',ConfigV3(),prepare_v3,restore_v3,'local_estimation_v3.py'),
    ('v3_top2_raw',ConfigV3(refine_candidates=2),prepare_v3,restore_v3,'local_estimation_v3.py'),
    ('v3_top2_overlap_raw',ConfigV3(refine_candidates=2,use_overlap_reranking=True),
     prepare_v3,restore_v3,'local_estimation_v3.py'),
)
FIELDS=('image_id','split','kind','amplitude','seed','method','input_mse','mse','mae','runtime_ms',
        'tile_count','selected_tiles','overlap_reranked_tiles','analysis_count','estimator_sha256',
        'manifest_sha256','timing_scope','validation_scope')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS)
        writer.writeheader();writer.writerows(rows)


def run(data,out):
    manifest_path=data/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    records=[r for r in manifest['images'] if r['split']=='dev'][:5]
    if len(records)!=5:raise ValueError('Expected at least five stored development records')
    manifest_hash=sha(manifest_path)
    source_hashes={name:sha(ROOT/name) for *_,name in METHODS}
    rows=[]
    for index,record in enumerate(records):
        clean=load_clean(record,data)
        for kind in ('global','local','chirp','orientation','clean'):
            amplitude=0. if kind=='clean' else .12
            observed,metadata,_=make_sample(clean,record['id'],kind,amplitude)
            input_mse=float(np.mean((observed-clean)**2))
            for method,config,prepare,restore,source in METHODS:
                start=time.perf_counter()
                p=prepare(observed,config)
                result=restore(p,weights=np.ones(len(p.tiles)),strength=1.)
                elapsed=(time.perf_counter()-start)*1000
                if result.shape!=clean.shape or not np.isfinite(result).all():
                    raise ValueError('Invalid estimation output')
                row=dict(image_id=record['id'],split='dev',kind=kind,amplitude=amplitude,
                    seed=metadata['seed'],method=method,input_mse=input_mse,
                    mse=float(np.mean((result-clean)**2)),mae=float(np.mean(np.abs(result-clean))),
                    runtime_ms=elapsed,tile_count=len(p.tiles),
                    selected_tiles=sum(t['selected'] is not None for t in p.tiles),
                    overlap_reranked_tiles=sum(bool(t.get('overlap_reranked',False)) for t in p.tiles),
                    analysis_count=p.analysis_count,estimator_sha256=source_hashes[source],
                    manifest_sha256=manifest_hash,
                    timing_scope='single-run wall-clock prepare plus raw restore; approximate local-machine time',
                    validation_scope=SCOPE)
                rows.append(row)
        write_csv(out,rows)
        print(f'optional P2 development {index+1}/5 {record["id"]}',flush=True)
    if len(rows)!=100 or len({(r['image_id'],r['kind'],r['method']) for r in rows})!=100:
        raise RuntimeError('Expected exactly100 unique development diagnostic rows')
    if sha(manifest_path)!=manifest_hash or any(sha(ROOT/n)!=h for n,h in source_hashes.items()):
        raise RuntimeError('Source or manifest changed during diagnostic')
    for method,*_ in METHODS:
        group=[r for r in rows if r['method']==method]
        print(method,'mean_runtime_ms',float(np.mean([r['runtime_ms'] for r in group])),
              'overlap_reranked_tiles',sum(r['overlap_reranked_tiles'] for r in group))
        for kind in ('global','local','chirp','orientation','clean'):
            print(' ',kind,'MSE',float(np.mean([r['mse'] for r in group if r['kind']==kind])))
    print(f'Saved100 development-only rows to {out}')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data')
    parser.add_argument('--out',type=Path,default=ROOT/'results/v3/optional_p2_development.csv')
    args=parser.parse_args();run(args.data,args.out)


if __name__=='__main__':main()
