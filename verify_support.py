"""Verify frozen support-study artifacts and rebuild inputs without inference."""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import support_experiments as study
from support_data import make_sample


def verify(out):
    study.OUT=Path(out).resolve()
    frozen=study.verify_frozen();protocol=study.read(study.PLAN)
    manifest=study.read(study.DATA/'v3_manifest.json')
    previous=study.read(study.ROOT/'refinement_protocol.json')
    used={r['id'] for r in manifest['images']}|{r['id'] for r in previous['independent_check']}
    check_ids=[r['id'] for r in protocol['independent_check']]
    if len(set(check_ids))!=4 or set(check_ids)&used:raise ValueError('Check IDs overlap previously used images')
    report=dict(all_passed=True,snapshot_sha256=frozen['snapshot']['snapshot_sha256'],
        independent_ids_disjoint_from_old210_and_previous4=True,stages={},
        note='Input reconstruction and artifact checks, without rerunning inference or retuning parameters. Not a performance/safety claim.')
    sizes=dict(development=len(protocol['development_ids'])*len(protocol['kinds']),
        regression=len(protocol['regression_texture_ids'])+len(protocol['regression_edge_ids']),
        check=len(check_ids)*len(protocol['kinds']))
    for stage,expected in sizes.items():
        directory=study.OUT/stage;run=study.read(directory/'run.json')
        rows=study.read_csv(directory/'metrics.csv');inputs=study.read(directory/'inputs.json')
        if run['source_hashes']!=frozen['snapshot']['files']:raise ValueError(f'{stage}: source inventory mismatch')
        multiplier=len(study.FACTORS)*len(study.STRENGTHS)+1
        if len(rows)!=expected*multiplier or len(inputs)!=expected:raise ValueError(f'{stage}: result count mismatch')
        keys=[tuple(row[k] for k in ('dataset','image_id','kind','variant','method','strength')) for row in rows]
        if len(set(keys))!=len(keys):raise ValueError(f'{stage}: duplicate result keys')
        expected_keys=set()
        for item in inputs:
            identity=tuple(str(item[k]) for k in ('dataset','image_id','kind','variant'))
            expected_keys.add((*identity,'input',0.))
            expected_keys.update((*identity,method,strength) for method in study.FACTORS for strength in study.STRENGTHS)
        actual_keys={(*key[:-1],float(key[-1])) for key in keys}
        if actual_keys!=expected_keys:raise ValueError(f'{stage}: method/input pairing mismatch')
        for row in rows:
            if row['method']!='input' and (row['method'] not in study.FACTORS or float(row['strength']) not in study.STRENGTHS):
                raise ValueError(f'{stage}: unexpected method/strength')
            for field in ('mse','mae','ssim'):
                if not math.isfinite(float(row[field])):raise ValueError(f'{stage}: nonfinite {field}')
        reconstructed=0
        for saved,(image_id,clean,kind,variant,_,dataset) in zip(inputs,study.samples(stage),strict=True):
            if (saved['image_id'],saved['kind'],saved['variant'],saved['dataset'])!=(image_id,kind,variant,dataset):
                raise ValueError(f'{stage}: input order or identity changed')
            observed,_,_=make_sample(clean,image_id,kind,0. if kind=='clean' else .12,variant)
            if hashlib.sha256(observed.tobytes()).hexdigest()!=saved['float_input_sha256']:
                raise ValueError(f'{stage}: input pixels changed')
            reconstructed+=1
        report['stages'][stage]=dict(inputs=len(inputs),metric_rows=len(rows),
            unique_result_keys=True,source_hashes_match_freeze=True,reconstructed_inputs=reconstructed)
    tests=study.OUT/'tests.xml'
    if tests.exists():
        suites=list(ET.parse(tests).getroot().iter('testsuite'))
        report['tests']={key:sum(int(s.attrib.get(key,0)) for s in suites) for key in ('tests','failures','errors','skipped')}
        if report['tests']['failures'] or report['tests']['errors'] or report['tests']['skipped']:
            raise ValueError('Full-data test record has failures/errors/skips')
    else:
        report['tests']=dict(status='not verified: tests.xml absent')
    study.dump(study.OUT/'verification.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('results/support'))
    args=parser.parse_args();result=verify(args.out)
    print(f"Verified {sum(s['inputs'] for s in result['stages'].values())} inputs and "
          f"{sum(s['metric_rows'] for s in result['stages'].values())} metric rows")


if __name__=='__main__':main()
