"""Verify context-study inputs, frozen sources, and complete paired outputs."""
import argparse
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import rejection_experiments as study


def verify(out):
    study.OUT=Path(out).resolve();frozen=study.verify_frozen();protocol=study.base.read(study.PLAN)
    used={r['id'] for r in study.base.read(study.DATA/'v3_manifest.json')['images']}
    for name in ('refinement_protocol.json','support_protocol.json','narrow_protocol.json'):
        used|={r['id'] for r in study.base.read(study.ROOT/name)['independent_check']}
    ids=[r['id'] for r in protocol['independent_check']]
    if len(ids)!=4 or len(set(ids))!=4 or used&set(ids):raise ValueError('New check overlaps old images')
    report=dict(all_passed=True,snapshot_sha256=frozen['snapshot']['snapshot_sha256'],new_check_disjoint=True,stages={},
        note='Input reconstruction without rerunning inference or retuning. Integrity verification is not a universal texture-safety guarantee.')
    sizes=dict(development=len(protocol['development_ids'])*len(protocol['kinds']),controls=len(study.make_controls()),
        regression=len(protocol['regression_textures'])+4*len(protocol['regression_images']),check=len(ids)*len(protocol['kinds']))
    for stage,expected_size in sizes.items():
        root=study.OUT/stage;inputs=study.base.read(root/'inputs.json');rows=study.base.read_csv(root/'metrics.csv');run=study.base.read(root/'run.json')
        if len(inputs)!=expected_size or len(rows)!=expected_size*(1+len(study.METHODS)*len(study.STRENGTHS)):
            raise ValueError('Input/result counts differ from protocol')
        if run['source_hashes']!=frozen['snapshot']['files']:raise ValueError('Recorded source hashes differ from freeze')
        expected=set();actual=[]
        for item in inputs:
            identity=tuple(str(item[k]) for k in ('dataset','image_id','kind','variant'))
            expected.add((*identity,'input',0.))
            expected.update((*identity,method,strength) for method in study.METHODS for strength in study.STRENGTHS)
        for row in rows:
            actual.append((*[row[k] for k in ('dataset','image_id','kind','variant','method')],float(row['strength'])))
            if any(not math.isfinite(float(row[field])) for field in ('mse','mae','ssim')):raise ValueError('Nonfinite metric')
        if len(set(actual))!=len(actual) or set(actual)!=expected:raise ValueError('Incomplete/duplicate pairing')
        interval_path=root/'intervals.csv'
        interval_sets={method:[] for method in study.METHODS}
        if interval_path.exists():
            for row in study.base.read_csv(interval_path):
                interval_sets[row['method']].append(tuple(row[k] for k in
                    ('dataset','image_id','kind','variant','axis','frequency','start','stop')))
        original=sorted(interval_sets['baseline'])
        if any(sorted(values)!=original for values in interval_sets.values()):
            raise ValueError('A method changed the strict qualified interval inventory')
        for saved,(image_id,_,observed,_,metadata,_,dataset,variant) in zip(inputs,study.samples(stage),strict=True):
            if (saved['image_id'],saved['kind'],saved['dataset'],saved['variant'])!=(image_id,metadata.get('kind',image_id),dataset,variant):
                raise ValueError('Input identity/order changed')
            if hashlib.sha256(observed.tobytes()).hexdigest()!=saved['float_input_sha256']:raise ValueError('Input reconstruction changed')
        report['stages'][stage]=dict(inputs=len(inputs),metric_rows=len(rows),source_hashes_match=True,
            complete_unique_pairs=True,reconstructed_inputs=len(inputs),same_qualified_interval_inventory=True)
    tests=study.OUT/'tests.xml'
    if tests.exists():
        suites=list(ET.parse(tests).getroot().iter('testsuite'))
        report['tests']={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        if any(report['tests'][k] for k in ('failures','errors','skipped')):raise ValueError('Full tests have failures/skips')
    else:report['tests']=dict(status='not verified: tests.xml absent')
    baseline=study.load_configs()['baseline'];full=study.load_configs()['context_full']
    report['same_numeric_gates']=all(getattr(baseline.support,k)==getattr(full.support,k) for k in
        ('min_amplitude','min_stability','min_validation_gain','min_width','max_width','flank_width','max_flank_ratio'))
    if not report['same_numeric_gates']:raise ValueError('Full context changed original numeric gates')
    mapping=dict(min_amplitude='amplitude',min_stability='stability',min_validation_gain='validation_gain',
        min_width='qualified_min_width',max_width='max_width',flank_width='flank_width',max_flank_ratio='flank_ratio')
    report['gates_match_protocol']=all(getattr(full.support,k)==protocol['fixed_thresholds'][v] for k,v in mapping.items())
    if not report['gates_match_protocol']:raise ValueError('Original numerical gates differ from protocol')
    study.base.dump(study.OUT/'verification.json',report)
    print(f"Verified {sum(s['inputs'] for s in report['stages'].values())} inputs and {sum(s['metric_rows'] for s in report['stages'].values())} metric rows")


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,default=Path('results/context'))
    verify(parser.parse_args().out)
