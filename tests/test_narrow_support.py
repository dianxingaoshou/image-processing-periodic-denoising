"""Coherence and compactness require distinct input evidence."""
import json

import numpy as np
import pytest

from narrow_support import SupportConfig,build_support


def profile(amplitude,stability=None,gain=None):
    amplitude=np.asarray(amplitude,float)
    return dict(amplitude=amplitude,
        stability=np.ones_like(amplitude) if stability is None else stability,
        validation_gain=np.full_like(amplitude,.8) if gain is None else gain)


@pytest.mark.parametrize('start,width',[(0,6),(0,12),(25,8),(116,12)])
def test_compact_edge_and_interior_regions_have_exact_support(start,width):
    amplitude=np.full(128,.003);amplitude[start:start+width]=.1
    result=build_support(profile(amplitude))
    expected=np.zeros(128,bool);expected[start:start+width]=True
    np.testing.assert_array_equal(result['mask'],expected)
    np.testing.assert_array_equal(result['weights'],expected.astype(float))
    json.dumps(result['intervals'],allow_nan=False)


def test_broad_periodicity_is_rejected_but_is_explicit_in_ablation():
    image_profile=profile(np.full(128,.1))
    result=build_support(image_profile)
    assert not result['mask'].any()
    assert result['intervals'][0]['reason']=='broad_periodic_region'
    assert build_support(image_profile,compact_support=False)['mask'].all()


def test_locally_qualified_group_requires_actual_flank_contrast():
    amplitude=np.full(128,.06);stability=np.zeros(128)
    amplitude[40:48]=.1;stability[40:48]=1
    result=build_support(profile(amplitude,stability))
    assert not result['mask'].any()
    assert result['intervals'][0]['reason']=='insufficient_outside_contrast'
    assert result['intervals'][0]['outside_to_inside_ratio']==pytest.approx(.6)


def test_stronger_of_two_observed_flanks_controls_acceptance():
    amplitude=np.full(128,.003);stability=np.zeros(128)
    amplitude[40:48]=.1;stability[40:48]=1
    amplitude[48:64]=.06
    result=build_support(profile(amplitude,stability))
    assert not result['mask'].any()
    assert result['intervals'][0]['right_flank_median']==pytest.approx(.06)


@pytest.mark.parametrize('field,value',[('stability',.7),('validation_gain',-.1)])
def test_high_amplitude_alone_cannot_bypass_coherence_gate(field,value):
    amplitude=np.full(128,.002);amplitude[:8]=.1
    p=profile(amplitude);p[field][:8]=value
    assert not build_support(p)['mask'].any()
    assert build_support(p,use_gate=False)['mask'][:8].all()


def test_no_gap_filling_or_dilation_outside_qualified_rows():
    amplitude=np.full(128,.002);amplitude[30:34]=.1;amplitude[35:39]=.1
    result=build_support(profile(amplitude))
    assert result['mask'][30:34].all() and result['mask'][35:39].all()
    assert not result['mask'][34]
    assert not result['weights'][[29,34,39]].any()


def test_entire_short_image_has_no_observed_flank_and_is_rejected():
    result=build_support(profile(np.full(16,.1)))
    assert not result['mask'].any()
    assert result['intervals'][0]['reason']=='no_observed_flank'


@pytest.mark.parametrize('kwargs',[dict(min_amplitude=np.nan),dict(min_width=0),dict(max_width=2),
    dict(max_width=32.5),dict(flank_width=True),dict(max_flank_ratio=1.),dict(min_validation_gain=-.1)])
def test_invalid_configuration_fails(kwargs):
    with pytest.raises(ValueError):SupportConfig(**kwargs)


@pytest.mark.parametrize('p',[dict(amplitude=[.1],stability=[np.nan],validation_gain=[.5]),
    dict(amplitude=[-.1],stability=[1.],validation_gain=[.5]),
    dict(amplitude=[.1,.2],stability=[1.],validation_gain=[.5]),dict(amplitude=[.1])])
def test_invalid_profiles_fail(p):
    with pytest.raises(ValueError):build_support(p)
