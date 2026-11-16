"""Rebuild narrow-study inputs and validate source/config/result pairing."""
import argparse
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import narrow_experiments as study


def verify(out):
    study.OUT=Path(out).resolve();frozen=study.verify_frozen();protocol=study.base.read(study.PLAN)
    old=study.base.read(study.DATA/'v3_manifest.json')
    used={r['id'] for r in old['images']}
    for name in ('refinement_protocol.json','support_protocol.json'):
        used|={r['id'] for r in study.base.read(study.ROOT/name)['independent_check']}
    ids=[r['id'] for r in protocol['independent_check']]
    if len(ids)!=4 or len(set(ids))!=4 or used&set(ids):raise ValueError('Check images overlap previous data')
    report=dict(all_passed=True,snapshot_sha256=frozen['snapshot']['snapshot_sha256'],new_check_disjoint=True,stages={},
        note='Input reconstruction only; inference is not rerun and parameters are not retuned. This is an integrity check, not proof of texture safety.')
    sizes=dict(development=len(protocol['development_ids'])*len(protocol['kinds']),
        regression=len(protocol['regression_textures'])+4*len(protocol['regression_images']),check=len(ids)*len(protocol['kinds']))
    for stage,size in sizes.items():
        root=study.OUT/stage;inputs=study.base.read(root/'inputs.json');rows=study.base.read_csv(root/'metrics.csv');run=study.base.read(root/'run.json')
        if run['source_hashes']!=frozen['snapshot']['files']:raise ValueError('Source hashes differ from freeze')
        if len(inputs)!=size or len(rows)!=size*22:raise ValueError('Result count differs from protocol')
        actual=[]
        for row in rows:
            actual.append((*[row[k] for k in ('dataset','image_id','kind','variant','method')],float(row['strength'])))
            if any(not math.isfinite(float(row[field])) for field in ('mse','mae','ssim')):raise ValueError('Nonfinite quality metrics')
        expected=set()
        for item in inputs:
            identity=tuple(str(item[k]) for k in ('dataset','image_id','kind','variant'))
            expected.add((*identity,'input',0.))
            for method in study.METHODS:
                strengths=study.OLD_STRENGTHS if method.startswith('legacy') else study.NEW_STRENGTHS
                expected.update((*identity,method,strength) for strength in strengths)
        if len(set(actual))!=len(actual) or set(actual)!=expected:raise ValueError('Missing/duplicate method-input pairs')
        for item,(image_id,clean,kind,variant,_,dataset) in zip(inputs,study.samples(stage),strict=True):
            if (item['image_id'],item['kind'],item['variant'],item['dataset'])!=(image_id,kind,variant,dataset):raise ValueError('Input order changed')
            observed,_,_=study.make_sample(clean,image_id,kind,0. if kind=='clean' else .12,variant)
            if hashlib.sha256(observed.tobytes()).hexdigest()!=item['float_input_sha256']:raise ValueError('Reconstructed input changed')
        report['stages'][stage]=dict(inputs=len(inputs),metric_rows=len(rows),source_hashes_match=True,unique_complete_pairing=True,reconstructed_inputs=len(inputs))
    tests=study.OUT/'tests.xml'
    if tests.exists():
        suites=list(ET.parse(tests).getroot().iter('testsuite'))
        report['tests']={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        if any(report['tests'][k] for k in ('failures','errors','skipped')):raise ValueError('Full tests contain failures or skips')
    else:report['tests']=dict(status='not checked: tests.xml absent')
    # Historical baselines are held byte-for-byte unchanged in this iteration.
    diagnosis=study.base.read(study.ROOT/'results/narrow_diagnosis/source_state.json')
    core=('local_estimation_v3.py','texture_guard_v3.py','denoiser_v3.py','support_data.py')
    report['historical_core_unchanged']=all(study.sha256_file(study.ROOT/p)==diagnosis['source_hashes'][p] for p in core)
    if not report['historical_core_unchanged']:raise ValueError('Old production core changed')
    study.base.dump(study.OUT/'verification.json',report)
    print(f"Verified {sum(s['inputs'] for s in report['stages'].values())} inputs and {sum(s['metric_rows'] for s in report['stages'].values())} metric rows")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=Path('results/narrow'))
    verify(parser.parse_args().out)
