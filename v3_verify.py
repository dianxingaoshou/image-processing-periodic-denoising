"""Check completed v3 artifacts without tuning parameters or certifying quality.

Run only after audit/stress/texture experiments, weight curves and unpaired
real examples have completed. Artifact consistency, algorithmic safety and
scientific superiority are separate questions. Any failed check exits nonzero.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import itertools
import json
import math
import sys
from collections import Counter,defaultdict
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent
METHODS={'input','v2_frozen','v3_coupled_P0','v3_decoupled','v3_unprotected'}
V3_METHODS={'v3_coupled_P0','v3_decoupled','v3_unprotected'}
CORE=('global','local','chirp','orientation')
STRESS=('cubic','random_phase','corner','edge_strip','irregular')
AMPLITUDES=(.06,.12,.18)
STRENGTHS=(0.,.25,.5,.75,1.,1.25,1.5,2.)
REQUIRED_FROZEN={
    'data_pipeline.py','run_experiments.py','evaluation_v3.py','published_baseline.py',
    'local_spectral.py','local_estimation_v2.py','texture_guard_v2.py','optimized_denoiser.py',
    'local_estimation_v3.py','texture_guard_v3.py','denoiser_v3.py','image_io_v3.py',
    'validation_data_v3.py','reproducibility_v3.py','v3_experiments.py',
    'v3_protocol.json','data/v3_manifest.json',
}
RATES=('candidate_block_coverage','correct_candidate_coverage','selected_block_coverage',
       'processed_block_coverage','clean_block_processing_rate','clean_candidate_rate')


def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def require(value,message):
    if not value:raise AssertionError(message)


def near(actual,expected,label,rtol=1e-9,atol=1e-12):
    require(np.isclose(float(actual),float(expected),rtol=rtol,atol=atol),f'{label}: {actual} != {expected}')


def count(value,label):
    number=float(value)
    require(np.isfinite(number) and number>=0 and abs(number-round(number))<1e-8,f'Invalid count {label}: {value}')
    return int(round(number))


def key(row):return str(row['image_id']),row['kind'],float(row.get('amplitude',0.))


def find_snapshot(value):
    if isinstance(value,dict):
        if {'files','audit_ids','snapshot_sha256'}<=set(value):return value
        for child in value.values():
            result=find_snapshot(child)
            if result is not None:return result
    if isinstance(value,list):
        for child in value:
            result=find_snapshot(child)
            if result is not None:return result
    return None


class Verifier:
    def __init__(self,root=ROOT):
        self.root=Path(root).resolve();self.output=self.root/'results/v3';self.data=self.root/'data'
        self.checks=[];self.details={};self.metrics={}

    def check(self,name,function):
        try:
            detail=function()
            self.checks.append(dict(name=name,passed=True,detail=detail))
            print('PASS',name,flush=True)
        except Exception as exc:
            self.checks.append(dict(name=name,passed=False,error=f'{type(exc).__name__}: {exc}'))
            print('FAIL',name,repr(exc),flush=True)

    def load(self):
        self.manifest=read_json(self.data/'v3_manifest.json')
        self.protocol=read_json(self.root/'v3_protocol.json')
        self.records={r['id']:r for r in self.manifest['images']+self.manifest['textures']}
        self.audit=[r['id'] for r in self.manifest['images'] if r['split']=='audit_v3']
        self.texture_ids=[r['id'] for r in self.manifest['textures']]
        self.experiment=importlib.import_module('v3_experiments')
        self.frozen,self.cfg=self.experiment.load_frozen()
        for stage in ('audit','stress','textures'):
            self.metrics[stage]=read_csv(self.output/stage/'metrics.csv')
        return dict(audit_images=len(self.audit),textures=len(self.texture_ids),real_inputs=len(self.manifest['real_examples']))

    def splits_and_data(self):
        expected={'dev':30,'regression_v1':60,'regression_v2':60,'audit_v3':60}
        rows=self.manifest['images'];require(Counter(r['split'] for r in rows)==expected,'Incorrect210-image split inventory')
        require(len({r['id'] for r in rows})==210,'Image-level split overlap')
        require(self.audit==self.manifest['selection']['audit_v3_ids'],'Audit ID ordering changed')
        require(self.audit==self.protocol['selection']['audit_v3_ids'],'Protocol audit ID ordering changed')
        require(len(self.texture_ids)==len(set(self.texture_ids))==12,'Expected12 unique DTD texture images')
        require(all(r['dataset']=='DTD' and r['official_split']=='test1' for r in self.manifest['textures']),'DTD subset mislabeled')
        require(len(self.manifest['real_examples'])==2,'Expected2 unpaired real inputs')
        checked=0
        for row in rows+self.manifest['textures']:
            require(sha(self.data/row['source_jpg'])==row['source_sha256'],'Image source hash changed: '+row['id'])
            require(sha(self.data/row['npy_path'])==row['clean_sha256'],'Float reference hash changed: '+row['id'])
            checked+=1
        for row in self.manifest['real_examples']:
            require(row['reference_available'] is False,'Real input unexpectedly claims paired reference')
            require(sha(self.root/row['path'])==row['sha256'],'Real source changed: '+row['id'])
        return dict(reference_sources_checked=checked,unpaired_real_inputs=2,all_splits_disjoint=True)

    def frozen_and_legacy(self):
        require(sha(self.data/'manifest.json')==self.manifest['source_manifest_sha256'],'v1 manifest changed')
        require(sha(self.data/'optimization_manifest.json')==self.manifest['v2_manifest_sha256'],'v2 manifest changed')
        require(sha(self.root/'v3_protocol.json')==self.manifest['protocol_sha256'],'v3 protocol changed after manifest creation')
        legacy_maps=[read_json(self.root/'results/main/source_hashes.json'),
                     read_json(self.root/'configs/optimized_v2.json')['source_hashes']]
        for mapping in legacy_maps:
            for name,digest in mapping.items():require(sha(self.root/name)==digest,'Frozen legacy source changed: '+name)
        snapshot=find_snapshot(self.frozen)
        require(snapshot is not None,'Frozen configuration has no full snapshot')
        labels={str(name).replace('\\','/') for name in snapshot['files']}
        require(REQUIRED_FROZEN<=labels,'Freeze omits transitive dependencies: '+str(sorted(REQUIRED_FROZEN-labels)))
        require(snapshot['audit_ids']==self.audit,'Frozen audit IDs differ in content/order')
        require(snapshot.get('hash_dependencies') is True,'Numeric dependencies frozen by version only, missing artifact hashes')
        from reproducibility_v3 import verify_snapshot
        result=verify_snapshot(snapshot,audit_ids=self.audit,base_dir=self.root)
        require(result['all_passed'],'Snapshot verification failed')
        return dict(snapshot_sha256=snapshot['snapshot_sha256'],files_frozen=len(snapshot['files']),
                    numerical_dependencies=list(snapshot['dependencies']),legacy_sources_unchanged=True)

    def expected_samples(self,stage):
        if stage=='audit':
            return [(image_id,kind,amp) for image_id in self.audit for kind in CORE for amp in AMPLITUDES]+[(i,'clean',0.) for i in self.audit]
        if stage=='stress':return [(i,k,.12) for i in self.audit[:20] for k in STRESS]
        return [(i,k,0. if k=='clean' else .12) for i in self.texture_ids for k in ('clean','global','chirp')]

    def inventory(self):
        counts={}
        for stage,expected_count in (('audit',3900),('stress',500),('textures',180)):
            rows=self.metrics[stage]
            actual=[(*key(r),r['method']) for r in rows]
            expected={(*sample,method) for sample in self.expected_samples(stage) for method in METHODS}
            require(len(rows)==len(set(actual))==expected_count,f'{stage}: incomplete or duplicate result inventory')
            require(set(actual)==expected,f'{stage}: missing/unexpected method or sample combinations')
            require(all(not r.get('error') for r in rows),f'{stage}: algorithm exception rows')
            order=list(dict.fromkeys(r['image_id'] for r in rows))
            expected_order=self.audit if stage=='audit' else self.audit[:20] if stage=='stress' else self.texture_ids
            require(order==expected_order,f'{stage}: sample image order changed')
            counts[stage]=dict(rows=len(rows),per_method=dict(Counter(r['method'] for r in rows)))
            path=self.output/stage/'exceptions.json'
            if path.exists():require(read_json(path)==[],f'{stage}: stored exceptions are not empty')
        return counts

    def numeric_metrics(self):
        quality_regressions={}
        for stage,rows in self.metrics.items():
            regressions=Counter()
            for row in rows:
                label=(stage,*key(row),row['method'])
                mse=float(row['mse']);mae=float(row['mae']);ssim=float(row['ssim']);psnr=float(row['psnr_db'])
                require(np.isfinite(mse) and mse>=0,f'Invalid MSE {label}')
                require(np.isfinite(mae) and 0<=mae<=1,f'Invalid MAE {label}')
                require(np.isfinite(ssim) and -1<=ssim<=1,f'Invalid SSIM {label}')
                if mse==0:require(psnr==float('inf'),f'Zero-MSE PSNR must be +inf {label}')
                else:near(psnr,-10*np.log10(mse),f'PSNR data_range1 relation {label}',atol=1e-9)
                if row.get('degraded','')!='' and row.get('input_mse','')!='':
                    degraded=int(mse>float(row['input_mse'])+1e-8)
                    require(int(row['degraded'])==degraded,f'Quality regression flag inconsistent {label}')
                    regressions[row['method']]+=degraded
                if row['method']=='input' and row['kind']=='clean':
                    require(mse==0 and mae==0 and ssim==1,'Clean control input was modified')
            quality_regressions[stage]=dict(regressions)
        return dict(quality_regressions=quality_regressions,
                    interpretation='Regressions are descriptive scientific outcomes; zero program exceptions does not imply good restoration.')

    def synthesis_pairing(self):
        summaries={}
        for stage in self.metrics:
            path=self.output/stage/'synthesis.jsonl'
            metadata=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            actual=[key(row) for row in metadata]
            require(len(actual)==len(set(actual)) and set(actual)==set(self.expected_samples(stage)),f'{stage}: synthesis metadata missing or duplicated')
            groups=defaultdict(list)
            for row in metadata:
                if row['kind']=='clean':continue
                require('seed' in row and 'parameters' in row,f'{stage}: structure recipe missing')
                require(0<=float(row['clipping_fraction'])<=1,f'{stage}: invalid clipping fraction')
                groups[(row['image_id'],row['kind'])].append(row)
                if row['kind'] in STRESS:
                    require(.025<=float(row['frequency_min'])<=float(row['frequency_max'])<=.45,'OOD frequency bounds violated')
            for label,group in groups.items():
                require(len({r['seed'] for r in group})==1,f'Amplitude-dependent structure seed {stage} {label}')
                require(len({json.dumps(r['parameters'],sort_keys=True) for r in group})==1,f'Amplitude-dependent phase/geometry {stage} {label}')
            if stage=='stress':
                corners=Counter(r['parameters']['corner_index'] for r in metadata if r['kind']=='corner')
                edges=Counter(r['parameters']['edge_index'] for r in metadata if r['kind']=='edge_strip')
                require(corners=={0:5,1:5,2:5,3:5},'Stress did not cover four corners equally')
                require(edges=={0:5,1:5,2:5,3:5},'Stress did not cover four edges equally')
            summaries[stage]=dict(recipes=len(metadata),paired_structure_groups=len(groups))
        return summaries

    def block_metrics_and_safety(self):
        from denoiser_v3 import effective_gains
        q=np.linspace(0,1,101);s=np.linspace(1,0,101)
        for alpha in (0.,.5,1.,1.5,2.):
            gains=effective_gains(q,s,alpha,'decoupled')
            require(np.all(gains<=s+1e-15),'Strength cancels the safety cap')
        near(effective_gains([1],[.7],2.,'decoupled')[0],.7,'Safety counterexample')
        near(effective_gains([1],[.7],2.,'coupled')[0],1.,'Coupled counterexample')
        totals=Counter();risky=Counter()
        for stage,rows in self.metrics.items():
            for row in rows:
                if row['method']=='input':continue
                label=(stage,*key(row),row['method'])
                for field in ('dirty_block_centres','clean_block_centres','frequency_error_count','risky_full_blocks','eligible_blocks'):
                    require(row.get(field,'')!='',f'Missing block-metric count {field} {label}')
                dirty=count(row['dirty_block_centres'],'dirty');clean=count(row['clean_block_centres'],'clean')
                frequency_count=count(row['frequency_error_count'],'frequency_error_count')
                risky_count=count(row['risky_full_blocks'],'risky_full_blocks');eligible=count(row['eligible_blocks'],'eligible_blocks')
                require(frequency_count<=dirty,'Frequency error count exceeds all dirty centers')
                # All main/stress/DTD references are256x256. The synthesis
                # geometry is explicit in the frozen estimator config.
                size=self.cfg.estimation.synthesis_size;stride=self.cfg.estimation.stride
                def starts(length):
                    if length<=size:return [0]
                    seq=list(range(0,length-size+1,stride))
                    if seq[-1]!=length-size:seq.append(length-size)
                    return seq
                blocks=len(starts(256))**2
                require(dirty+clean<=blocks and eligible<=blocks and risky_count<=blocks,'Invalid block-center denominator')
                for field in RATES:
                    value=row.get(field,'')
                    denominator=clean if field in ('clean_block_processing_rate','clean_candidate_rate') else dirty
                    if denominator==0:require(value=='',f'Empty denominator must remain blank: {field} {label}')
                    else:
                        require(value!='',f'Missing unconditional coverage: {field} {label}')
                        rate=float(value);require(np.isfinite(rate) and 0<=rate<=1,f'Coverage outside[0,1]: {field} {label}')
                        count(rate*denominator,field+' numerator')
                if dirty:
                    candidate=float(row['candidate_block_coverage']);correct=float(row['correct_candidate_coverage'])
                    selected=float(row['selected_block_coverage']);processed=float(row['processed_block_coverage'])
                    require(correct<=candidate+1e-12 and processed<=selected+1e-12 and selected<=candidate+1e-12,'Invalid candidate/selected/processed nesting')
                    near(frequency_count,selected*dirty,'Frequency count must include all selected dirty blocks, not only processed blocks',atol=1e-8)
                    if frequency_count:require(row.get('frequency_error_mean','')!='' and np.isfinite(float(row['frequency_error_mean'])) and float(row['frequency_error_mean'])>=0,'Missing unconditional selected frequency error')
                    else:require(row.get('frequency_error_mean','')=='','Empty selected frequency error must be blank')
                if clean:
                    require(float(row['clean_block_processing_rate'])<=float(row['clean_candidate_rate'])+1e-12,'Clean processing exceeds clean candidates')
                if row['kind']=='clean':require(dirty==0 and clean==blocks,'Clean control denominator omits unprocessed blocks')
                if row['method']=='v3_decoupled':require(risky_count==0,f'Decoupled risky block reached full subtraction {label}')
                totals[(stage,row['method'])]+=1;risky[(stage,row['method'])]+=risky_count
        return dict(rows_checked={f'{s}/{m}':n for (s,m),n in totals.items()},
                    risky_full_subtraction_blocks={f'{s}/{m}':n for (s,m),n in risky.items()},
                    frequency_scope='All dirty/clean tile centers are denominators. Blended display maps are not frequency truth fields.')

    def weight_curves(self):
        folder=self.output/'supplement'
        rows=read_csv(folder/'weight_curves.csv')
        require(len(rows)==240,'Expected10images x3modes x8strengths')
        modes=('coupled','decoupled','unprotected')
        mode_key=next((k for k in ('mode','combination') if k in rows[0]),None)
        require(mode_key is not None,'Weight curve lacks mode/combination')
        actual={(r['image_id'],r[mode_key],float(r['strength'])) for r in rows}
        expected=set(itertools.product(self.audit[:10],modes,STRENGTHS))
        require(actual==expected and len(actual)==240,'Weight curve grid missing or duplicated')
        for row in rows:
            for field in ('contaminated_mse','clean_mse'):
                require(row.get(field,'')!='' and np.isfinite(float(row[field])) and float(row[field])>=0,'Invalid matched-removal loss')
            if float(row['strength'])==0:near(row['clean_mse'],0,'Zero strength must leave clean pixels untouched')
        coverage=read_csv(folder/'matched_coverage.csv')
        require(len(coverage)==20 and Counter(r['image_id'] for r in coverage)=={i:2 for i in self.audit[:10]},'Matched coverage excludes one of the10images/two comparisons')
        pair_key=next((k for k in ('comparison','pair','baseline','mode') if k in coverage[0]),None)
        require(pair_key is not None,'Matched coverage has no comparison identity')
        require(len({(r['image_id'],r[pair_key]) for r in coverage})==20,'Matched coverage has duplicate pairs')
        points_key=next((k for k in ('matched_points','n_matched','n_points') if k in coverage[0]),None)
        require(points_key is not None,'Matched coverage must report accepted point counts, including zero')
        matched=Counter()
        for row in coverage:
            points=count(row[points_key],'matched_points')
            if points:matched[row[pair_key]]+=1
            else:require(bool(row.get('reason') or row.get('status')),'Unmatched case lacks an explicit reason')
        return dict(sweep_rows=240,coverage_rows=20,matched_images_by_comparison=dict(matched),
                    scope='Finite common-removal intervals only; artifact completeness is not a claim that all images improve.')

    def _resolve_output(self,value,case_id):
        candidates=[]
        def visit(v):
            if isinstance(v,str) and Path(v).suffix.lower() in ('.png','.npy','.jpg','.tif','.tiff'):
                p=Path(v)
                candidates.extend([p if p.is_absolute() else self.root/p,self.output/'real'/p,self.output/'real'/case_id/p])
            elif isinstance(v,dict):
                for child in v.values():visit(child)
            elif isinstance(v,list):
                for child in v:visit(child)
        visit(value)
        for candidate in candidates:
            if candidate.is_file():return candidate
        raise FileNotFoundError(f'No saved image/array found for real case {case_id}: {value}')

    def unpaired_real_examples(self):
        metadata=read_json(self.output/'real/metadata.json')
        if isinstance(metadata,dict):metadata=metadata.get('cases',metadata.get('records',metadata.get('examples')))
        require(isinstance(metadata,list) and len(metadata)==2,'Real metadata must contain two cases')
        expected_ids={r['id'] for r in self.manifest['real_examples']}
        actual_ids={str(r.get('id',r.get('image_id'))) for r in metadata}
        require(actual_ids==expected_ids,'Real-case inventory changed')
        def forbidden(v):
            if isinstance(v,dict):
                for k,child in v.items():
                    if 'psnr' in k.lower() or 'ssim' in k.lower():return True
                    if forbidden(child):return True
            elif isinstance(v,list):return any(forbidden(child) for child in v)
            return False
        shapes={}
        for record in metadata:
            case_id=str(record.get('id',record.get('image_id')))
            require(not forbidden(record),'Unpaired real image has unsupported PSNR/SSIM fields')
            require(record.get('reference_available',False) is False,'Real result claims a paired reference')
            outputs=record.get('outputs',record.get('methods'))
            if isinstance(outputs,list):outputs={r['method']:r for r in outputs}
            require(isinstance(outputs,dict) and set(outputs)==METHODS,'Real case missing method output')
            case_shapes=[]
            for method,value in outputs.items():
                path=self._resolve_output(value,case_id)
                if path.suffix.lower()=='.npy':shape=np.load(path).shape
                else:
                    with Image.open(path) as image:shape=(image.height,image.width)
                require(len(shape)==2 and max(shape)==512,'Every real method must use same fixed longest-side512 preprocessing')
                case_shapes.append(tuple(shape))
            require(len(set(case_shapes))==1,'Real methods use different spatial preprocessing')
            shapes[case_id]=case_shapes[0]
        return dict(cases=2,method_outputs=10,processed_shapes=shapes,reference_metrics=False)

    def deterministic_reruns(self):
        from denoiser_v3 import denoise
        from run_experiments import load_clean
        from validation_data_v3 import make_sample
        record=self.records[self.audit[0]];clean=load_clean(record,self.data)
        cfg=replace(self.cfg,combination='decoupled')
        lookup={(*key(r),r['method']):r for r in self.metrics['audit']}
        cases=[]
        for kind in (*CORE,'clean'):
            amplitude=0. if kind=='clean' else .12
            observed,_,_=make_sample(clean,record['id'],kind,amplitude)
            restored=denoise(observed,cfg)
            require(restored.shape==clean.shape and np.isfinite(restored).all() and restored.min()>=0 and restored.max()<=1,'Invalid deterministic rerun output')
            mse=float(np.mean((restored-clean)**2));row=lookup[(record['id'],kind,amplitude,'v3_decoupled')]
            near(mse,row['mse'],'Frozen deterministic rerun '+kind,rtol=1e-10,atol=1e-12)
            cases.append(dict(image_id=record['id'],kind=kind,mse=mse,output_sha256=hashlib.sha256(restored.tobytes()).hexdigest()))
        return cases

    def run(self):
        for name,function in (
            ('load_completed_artifacts',self.load),('data_sources_and_disjoint_splits',self.splits_and_data),
            ('full_freeze_and_legacy_preservation',self.frozen_and_legacy),('exact_method_sample_inventory',self.inventory),
            ('numeric_quality_metrics',self.numeric_metrics),('structure_pairing_and_placement',self.synthesis_pairing),
            ('unconditional_frequency_coverage_and_safety_caps',self.block_metrics_and_safety),
            ('matched_removal_inventory_and_coverage',self.weight_curves),('unpaired_real_cases_no_reference_metrics',self.unpaired_real_examples),
            ('five_frozen_deterministic_reruns',self.deterministic_reruns)):
            self.check(name,function)
        passed=sum(check['passed'] for check in self.checks)
        return dict(schema_version=3,verified_at_utc=datetime.now(timezone.utc).isoformat(),
            all_passed=passed==len(self.checks),checks_passed=passed,checks_total=len(self.checks),checks=self.checks,
            verifier_sha256=sha(__file__),scope='Artifact correctness, frozen identity, coverage accounting and reproducibility; does not certify universal denoising or texture preservation.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--out',type=Path,default=ROOT/'results/v3/verification.json')
    args=parser.parse_args();report=Verifier(args.root).run()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(f'{report["checks_passed"]}/{report["checks_total"]} checks passed; saved {args.out}',flush=True)
    return 0 if report['all_passed'] else 1


if __name__=='__main__':sys.exit(main())
