"""Model-mismatch, placement, pairing and untouched-image audit checks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from validation_data_v3 import EXTRA_KINDS,make_sample,synthesize


@pytest.mark.parametrize('kind',EXTRA_KINDS)
def test_ood_amplitude_pairing_and_truth_boundaries(kind):
    clean=np.full((96,128),.5)
    low,ml,tl=make_sample(clean,'development_analytic',kind,.06,variant=1 if kind in ('corner','edge_strip') else None)
    high,mh,th=make_sample(clean,'development_analytic',kind,.18,variant=1 if kind in ('corner','edge_strip') else None)
    assert ml['seed']==mh['seed']
    assert ml['parameters']==mh['parameters']
    np.testing.assert_allclose(high-clean,3*(low-clean),atol=3e-16)
    np.testing.assert_array_equal(tl['true_fy'],th['true_fy'])
    np.testing.assert_array_equal(tl['true_fx'],th['true_fx'])
    frequency=np.hypot(tl['true_fy'],tl['true_fx'])
    assert frequency.min()>=.025 and frequency.max()<=.45
    assert np.isfinite(low).all() and 0<=low.min()<=low.max()<=1
    np.testing.assert_array_equal(low[tl['clean']],clean[tl['clean']])
    assert np.all(tl['contaminated']|tl['clean']|tl['transition'])
    assert not np.any(tl['contaminated']&tl['clean'])
    json.dumps(ml,allow_nan=False)


def test_corners_and_strips_explicitly_cover_four_image_edges():
    clean=np.full((96,128),.5)
    locations=((0,0),(0,-1),(-1,0),(-1,-1))
    for variant,position in enumerate(locations):
        _,meta,masks=make_sample(clean,'placement_control','corner',.12,variant=variant)
        assert meta['parameters']['corner_index']==variant
        assert masks['envelope'][position]==pytest.approx(1)
        opposite=locations[3-variant]
        assert masks['clean'][opposite]
    for variant in range(4):
        _,meta,masks=make_sample(clean,'placement_control','edge_strip',.12,variant=variant)
        assert meta['parameters']['edge_index']==variant
        assert masks['clean'].mean()>.7


def test_random_phase_is_reproducible_and_not_a_quadratic_field():
    clean=np.full((96,128),.5)
    a,ma,ta=make_sample(clean,'nonquadratic_control','random_phase',.12)
    b,mb,tb=make_sample(clean,'nonquadratic_control','random_phase',.12)
    np.testing.assert_array_equal(a,b)
    assert ma==mb
    # A quadratic phase has an affine derivative. A spatially varying second
    # derivative here ensures the truth is not confined to that model family.
    assert np.std(np.diff(ta['true_fx'],n=2,axis=1))>1e-7


def test_generator_preserves_global_rng():
    clean=np.full((64,64),.5)
    np.random.seed(31);state=np.random.get_state()
    make_sample(clean,'rng_control','random_phase',.12)
    after=np.random.get_state()
    assert state[0]==after[0] and state[2:]==after[2:]
    np.testing.assert_array_equal(state[1],after[1])


def test_new_audit_is_disjoint_and_frozen_legacy_manifests_unchanged():
    root=Path('data')
    if not (root/'v3_manifest.json').exists():pytest.skip('v3 data not built')
    m=json.loads((root/'v3_manifest.json').read_text(encoding='utf-8'))
    groups={s:{r['id'] for r in m['images'] if r['split']==s} for s in ('dev','regression_v1','regression_v2','audit_v3')}
    assert [len(groups[s]) for s in groups]==[30,60,60,60]
    all_ids=[r['id'] for r in m['images']]
    assert len(set(all_ids))==len(all_ids)==210
    assert [r['id'] for r in m['images'] if r['split']=='audit_v3']==m['selection']['audit_v3_ids']
    assert hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest()==m['source_manifest_sha256']
    assert hashlib.sha256((root/'optimization_manifest.json').read_bytes()).hexdigest()==m['v2_manifest_sha256']
    for r in m['images']:
        if r['split']=='audit_v3':
            assert r['official_split']=='test'
            assert hashlib.sha256((root/r['npy_path']).read_bytes()).hexdigest()==r['clean_sha256']
