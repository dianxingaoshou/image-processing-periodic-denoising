"""Numerical boundary tests for V3; no reference image enters the estimator."""
import json
from dataclasses import replace

import numpy as np
import pytest

from local_estimation_v3 import (EstimationConfig,prepare,restore,_fit_constant,
    _refine_phase,_holdout_mask,_overlap_discrepancy)


FREQUENCIES=((0.,.25),(0.,.2495),(0.,.2505),(.25,0.),(.15,.20),(.1767766953,.1767766953))


@pytest.mark.parametrize('frequency',FREQUENCIES)
@pytest.mark.parametrize('offset',(0.,.37,1.5707963267948966))
def test_full_pixel_fit_does_not_lose_quarter_pixel_frequency(frequency,offset):
    y,x=np.indices((64,80));fy,fx=frequency
    phase=2*np.pi*(fy*y+fx*x)+offset
    observed=.5+.1*np.sin(phase)
    for compatibility_subsample in (1,2):
        coef,prediction,_=_fit_constant(observed,phase,compatibility_subsample)
        assert np.sqrt(np.mean((prediction-observed)**2))<2e-13
        assert np.hypot(coef[0],coef[1])==pytest.approx(.1,abs=2e-13)


@pytest.mark.parametrize('frequency',((0.,.25),(.25,0.),(0.,.2495),(0.,.2505)))
def test_quarter_frequency_candidate_ranking_and_amplitude(frequency):
    y,x=np.indices((64,64));fy,fx=frequency
    main=.12*np.sin(2*np.pi*(fy*y+fx*x))
    secondary=.035*np.sin(2*np.pi*(.11*y+.13*x)+.7)
    observed=.5+main+secondary
    p=prepare(observed,EstimationConfig(scales=(64,),enable_chirp=False,enable_amplitude_model=False))
    selected=p.tiles[0]['selected']
    distance=min(np.hypot(selected['fy']-fy,selected['fx']-fx),np.hypot(selected['fy']+fy,selected['fx']+fx))
    assert distance<.0005
    assert p.tiles[0]['amplitude']==pytest.approx(.12,abs=.002)
    assert np.corrcoef(p.tiles[0]['residual'].ravel(),main.ravel())[0,1]>.999


@pytest.mark.parametrize('offset',(0.,.7,1.5707963267948966))
def test_phase_refinement_near_quarter_frequency(offset):
    y,x=np.indices((64,64));a=.5+.1*np.sin(2*np.pi*.25*x+offset)
    fitted=_refine_phase(a,0.,.2475,False,EstimationConfig())
    assert fitted['fx']==pytest.approx(.25,abs=1e-6)
    assert fitted['fy']==pytest.approx(0.,abs=1e-6)
    assert np.hypot(fitted['coef'][0],fitted['coef'][1])==pytest.approx(.1,abs=1e-7)


def test_small_image_has_one_effective_analysis_window():
    y,x=np.indices((64,64));image=.5+.1*np.sin(2*np.pi*(.12*y+.16*x)+.4)
    p=prepare(image,EstimationConfig(force_scale=128))
    assert p.analysis_count==1
    tile=p.tiles[0]
    assert tile['effective_window_count']==1
    assert tile['selected']['requested_scales']==[64,128,256]
    assert tile['selected']['analysis_window_id']==(0,0,64,64)
    assert not tile['different_window_support']
    assert tile['cross_window']==tile['cross_scale']==0.
    assert len({c['analysis_window_id'] for c in tile['candidates']})==1


@pytest.mark.parametrize('shape',((96,96),(128,256)))
def test_truncated_scales_do_not_duplicate_window_evidence(shape):
    y,x=np.indices(shape);a=.5+.1*np.sin(2*np.pi*(.09*y+.14*x)+.3)
    p=prepare(a)
    for tile in p.tiles:
        candidates=tile['candidates']
        keys=[(c['analysis_window_id'],c['fy'],c['fx']) for c in candidates]
        assert len(keys)==len(set(keys))
        ids={c['analysis_window_id'] for c in candidates}
        if shape==(96,96):
            assert tile['effective_window_count']==2
            whole=[c for c in candidates if c['analysis_window_id']==(0,0,96,96)]
            assert all(c['requested_scales']==[128,256] for c in whole)
        if tile['selected'] is not None:
            assert tile['different_window_support']==any(key!=tile['selected']['analysis_window_id'] for key in ids)


def test_validation_sampling_is_stratified_deterministic_and_rng_independent():
    np.random.seed(79);state=np.random.get_state()
    mask=_holdout_mask((65,79),.25,1729)
    np.testing.assert_array_equal(state[1],np.random.get_state()[1])
    np.testing.assert_array_equal(mask,_holdout_mask((65,79),.25,1729))
    assert not mask.flags.writeable
    for y in range(0,64,4):
        for x in range(0,76,4):assert mask[y:y+4,x:x+4].sum()==4
    # Training includes all parity classes; this is not another ::2 lattice.
    assert all(np.any((~mask)[dy::2,dx::2]) for dy in (0,1) for dx in (0,1))
    y,x=np.indices(mask.shape);phase=2*np.pi*.25*x
    a=.5+.1*np.sin(phase)
    coef,prediction,_=_fit_constant(a,phase,fit_mask=~mask)
    assert np.sqrt(np.mean((prediction[mask]-a[mask])**2))<1e-12
    assert np.hypot(coef[0],coef[1])==pytest.approx(.1,abs=1e-12)


def test_reranking_refines_distinct_candidates_and_exports_finite_diagnostics():
    y,x=np.indices((96,96))
    a=.5+.09*np.sin(2*np.pi*(.12*y+.16*x))+.06*np.cos(2*np.pi*(.19*y-.11*x)+.4)
    cfg=EstimationConfig(scales=(64,128),refine_candidates=2,use_overlap_reranking=True)
    p=prepare(a,cfg)
    assert any(len(t['candidate_models'])==2 for t in p.tiles)
    for tile in p.tiles:
        for candidate in tile['candidate_models']:
            assert np.isfinite(candidate['validation_mse'])
            assert np.isfinite(candidate['validation_score'])
            assert 0<=candidate['overlap_discrepancy']<=2
    b,d=restore(p,weights=np.ones(len(p.tiles)),return_diagnostics=True)
    assert np.isfinite(b).all()
    json.dumps(d['tiles'],allow_nan=False)
    np.testing.assert_array_equal(restore(p,weights=np.zeros(len(p.tiles))),a)


def test_overlap_checks_phase_but_allows_absent_boundary_support():
    tile=dict(y=0,x=0,h=64,w=64);neighbor=dict(y=0,x=32,h=64,w=64)
    y,x=np.indices((64,96));noise=.1*np.sin(2*np.pi*(.12*y+.16*x)+.7)
    left=noise[:,:64];right=noise[:,32:]
    assert _overlap_discrepancy(tile,left,neighbor,right)==pytest.approx(0.,abs=1e-12)
    assert _overlap_discrepancy(tile,left,neighbor,-right)>1.99
    assert _overlap_discrepancy(tile,left,neighbor,np.zeros_like(right))==0.


def test_constant_odd_image_and_invalid_optional_settings():
    a=np.full((79,113),.42);p=prepare(a)
    np.testing.assert_array_equal(restore(p,weights=np.ones(len(p.tiles))),a)
    with pytest.raises(ValueError):EstimationConfig(use_overlap_reranking=True)
    with pytest.raises(ValueError):EstimationConfig(refine_candidates=3)
