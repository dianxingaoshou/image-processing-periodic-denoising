"""Analytic tests for input-only narrow-profile proposals and validation."""
from dataclasses import asdict
import json

import numpy as np
import pytest

from narrow_profiles import ProfileConfig,propose_profiles,profiles_from_frequencies


def strip(axis=0,frequency=.13725,offset=.3):
    normal,tangent=np.indices((64,128))
    amplitude=.12*np.maximum(1-normal/8,0)
    phase=2*np.pi*frequency*tangent+2*np.pi*.073*normal+offset
    background=.45+.03*normal/64+.02*tangent/128
    image=background+amplitude*np.sin(phase)
    return (image if axis==0 else image.T),amplitude[:,0],phase[:,0]


@pytest.mark.parametrize('axis',(0,1))
@pytest.mark.parametrize('frequency',(.13725,.25,.026))
def test_narrow_band_is_not_diluted_by_other_normal_profiles(axis,frequency):
    image,amplitude,phase0=strip(axis,frequency)
    pool=propose_profiles(image,axis)
    selected=min(pool,key=lambda p:abs(p['frequency']-frequency))
    assert selected['frequency']==pytest.approx(frequency,abs=2e-6)
    assert selected['axis']==axis and selected['tangent_length']==128
    np.testing.assert_allclose(selected['amplitude'],amplitude,atol=2e-5)
    np.testing.assert_allclose(selected['coefficients'][:,0],amplitude*np.cos(phase0),atol=2e-5)
    np.testing.assert_allclose(selected['coefficients'][:,1],amplitude*np.sin(phase0),atol=2e-5)
    assert np.min(selected['stability'][:6])>.99
    assert np.min(selected['validation_gain'][:6])>.99
    assert selected['coefficients'].shape==(64,2)
    assert selected['split_coefficients'].shape==(2,64,2)
    # The same absolute coordinate origin is used for both tangent halves.
    np.testing.assert_allclose(selected['split_coefficients'][0,:6],selected['split_coefficients'][1,:6],atol=2e-5)


def test_broad_periodic_content_is_only_a_proposal_not_a_noise_label():
    normal,tangent=np.indices((64,128))
    image=.5+.1*np.sin(2*np.pi*(.13725*tangent+.08*normal))
    pool=propose_profiles(image,0)
    p=min(pool,key=lambda p:abs(p['frequency']-.13725))
    assert np.all(p['amplitude']>.09)
    assert np.all(p['stability']>.99)
    assert 'noise_probability' not in p and 'support_mask' not in p


def test_opposite_tangent_phase_is_not_mistaken_for_replicated_coefficients():
    normal,tangent=np.indices((64,128))
    amplitude=.1*np.maximum(1-normal/8,0)
    image=.5+amplitude*np.sin(2*np.pi*.13725*tangent+np.where(tangent>=64,np.pi,0))
    pool=propose_profiles(image,0)
    nearby=[p for p in pool if abs(p['frequency']-.13725)<.015]
    assert nearby
    assert all(float(np.median(p['stability'][:4]))<.8 for p in nearby)


def test_frequency_change_fails_opposite_half_prediction():
    normal,tangent=np.indices((64,128))
    phase=np.where(tangent<64,2*np.pi*.13725*tangent,2*np.pi*.19*tangent)
    image=.5+.1*np.maximum(1-normal/8,0)*np.sin(phase)
    pool=propose_profiles(image,0)
    nearby=[p for p in pool if min(abs(p['frequency']-.13725),abs(p['frequency']-.19))<.01]
    assert nearby and all(np.max(p['validation_gain'][:5])<.08 for p in nearby)


def test_constant_linear_and_short_tangent_abstain():
    normal,tangent=np.indices((64,128))
    for image in (np.full((64,128),.5),.4+.02*normal/64+.03*tangent/128):
        assert propose_profiles(image,0)==[]
    image,_,_=strip()
    assert propose_profiles(image[:,:48],0)==[]


def test_repeatability_does_not_touch_global_rng_or_input():
    image,_,_=strip();before=image.copy()
    np.random.seed(42);state=np.random.get_state()
    one=propose_profiles(image,0);two=propose_profiles(image,0)
    np.testing.assert_array_equal(np.random.get_state()[1],state[1])
    np.testing.assert_array_equal(image,before)
    assert len(one)==len(two)<=ProfileConfig().max_candidates
    for a,b in zip(one,two):
        assert a['frequency']==b['frequency']
        np.testing.assert_array_equal(a['coefficients'],b['coefficients'])
    json.dumps(asdict(ProfileConfig()),allow_nan=False)


def test_external_input_candidates_use_identical_fit_without_frequency_refinement():
    image,amplitude,phase0=strip()
    frequency=.13725
    pool=profiles_from_frequencies(image,0,[-frequency,frequency,frequency+.0001,.001,.49])
    assert len(pool)==1 and pool[0]['frequency']==frequency
    candidate=pool[0]
    assert candidate['proposal_source']=='external_frequency'
    np.testing.assert_allclose(candidate['coefficients'][:,0],amplitude*np.cos(phase0),atol=2e-12)
    np.testing.assert_allclose(candidate['coefficients'][:,1],amplitude*np.sin(phase0),atol=2e-12)
    np.testing.assert_allclose(candidate['split_coefficients'][0],candidate['split_coefficients'][1],atol=2e-12)
    fitted=propose_profiles(image,0)
    same=min(fitted,key=lambda p:abs(p['frequency']-frequency))
    np.testing.assert_allclose(candidate['validation_gain'],same['validation_gain'],atol=1e-8)


def test_external_candidate_units_cycles_and_validation_are_explicit():
    image,_,_=strip(axis=1)
    assert profiles_from_frequencies(image,1,[0.,.001])==[]
    pool=profiles_from_frequencies(image,1,[.1,.15,.2,.25,.3,.35],ProfileConfig(max_candidates=2))
    assert len(pool)==2
    with pytest.raises(ValueError):profiles_from_frequencies(image,1,[np.nan])
    with pytest.raises(ValueError):profiles_from_frequencies(image,1,[[.1,.2]])


def test_external_frequency_refinement_is_identical_to_new_pool_refinement():
    image,_,_=strip()
    proposed=propose_profiles(image,0)
    selected=min(proposed,key=lambda p:abs(p['frequency']-.13725))
    coarse=selected['initial_frequency']
    unchanged=profiles_from_frequencies(image,0,[coarse])[0]
    refined=profiles_from_frequencies(image,0,[coarse],refine=True)[0]
    assert unchanged['frequency']==coarse
    assert refined['frequency']==selected['frequency']
    assert refined['frequency']==pytest.approx(.13725,abs=2e-6)
    assert refined['pooling_rows']==selected['pooling_rows']
    for field in ('coefficients','split_coefficients','stability','validation_gain'):
        np.testing.assert_array_equal(refined[field],selected[field])


@pytest.mark.parametrize('kwargs',[dict(min_frequency=-1),dict(max_frequency=.5),dict(minimum_cycles=0),
    dict(minimum_tangent_length=16),dict(max_candidates=0),dict(pooling_rows=0),dict(fft_oversampling=3),
    dict(min_frequency=float('nan'))])
def test_invalid_profile_configuration_fails(kwargs):
    with pytest.raises(ValueError):ProfileConfig(**kwargs)


@pytest.mark.parametrize('image,axis',[(np.zeros((8,64)),0),(np.zeros((64,64,3)),0),
    (np.full((64,64),np.nan),0),(np.full((64,64),1.1),0),(np.zeros((64,64)),2)])
def test_invalid_inputs_fail(image,axis):
    with pytest.raises(ValueError):propose_profiles(image,axis)
