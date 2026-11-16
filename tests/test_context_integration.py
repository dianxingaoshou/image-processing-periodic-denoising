"""Contrast context must not turn rejected rows into output support."""
from dataclasses import replace
import json

import numpy as np
import pytest

from narrow_support import SupportConfig,build_support


def profile(amplitude,stability=None,gain=None):
    amplitude=np.asarray(amplitude,float)
    return dict(amplitude=amplitude,
        stability=np.ones_like(amplitude) if stability is None else np.asarray(stability,float),
        validation_gain=np.full_like(amplitude,.5) if gain is None else np.asarray(gain,float))


def context_config(**kwargs):
    return replace(SupportConfig(),contrast_context='amplitude',**kwargs)


def test_high_amplitude_rejected_endpoint_is_context_not_output_support():
    # Mirrors the causal structure of 69022/edge8, without reading a dataset.
    amplitude=np.full(256,.003)
    amplitude[250:]=[.024,.060,.084,.115,.130,.143]
    stability=np.ones(256);stability[250]=.54;stability[255]=.778
    gain=np.full(256,.5);gain[250]=.02
    candidate=profile(amplitude,stability,gain)
    old=build_support(candidate)
    current=build_support(candidate,context_config())
    assert not old['mask'].any()
    assert old['intervals'][0]['reason']=='insufficient_outside_contrast'
    expected=np.zeros(256,bool);expected[251:255]=True
    np.testing.assert_array_equal(current['mask'],expected)
    np.testing.assert_array_equal(current['qualified'],old['qualified'])
    assert not current['weights'][[250,255]].any()
    interval=current['intervals'][0]
    assert (interval['context_start'],interval['context_stop'])==(250,256)
    assert interval['right_flank_median'] is None
    assert interval['outside_to_inside_ratio']<.35
    json.dumps(current['intervals'],allow_nan=False)


def test_fragmented_qualification_does_not_fill_holes_or_grow_output():
    amplitude=np.full(96,.003);amplitude[30:42]=.1
    stability=np.zeros(96);stability[30:34]=1.;stability[36:40]=1.
    candidate=profile(amplitude,stability)
    result=build_support(candidate,context_config())
    expected=np.zeros(96,bool);expected[30:34]=True;expected[36:40]=True
    np.testing.assert_array_equal(result['mask'],expected)
    assert len(result['intervals'])==2
    assert all((i['context_start'],i['context_stop'])==(30,42) for i in result['intervals'])
    assert not result['weights'][[29,34,35,40,41,42]].any()
    assert not np.any(result['mask']&~result['qualified'])


@pytest.mark.parametrize('width',(1,2))
def test_wide_amplitude_context_cannot_rescue_too_few_qualified_rows(width):
    amplitude=np.full(96,.003);amplitude[30:42]=.1
    stability=np.zeros(96);stability[34:34+width]=1.
    result=build_support(profile(amplitude,stability),context_config())
    assert not result['mask'].any()
    assert result['intervals'][0]['reason']=='too_few_supported_rows'
    assert result['intervals'][0]['context_width']==12


def test_broad_amplitude_context_is_rejected_even_when_qualified_run_is_short():
    amplitude=np.full(96,.003);amplitude[20:70]=.1
    stability=np.zeros(96);stability[30:34]=1.
    candidate=profile(amplitude,stability)
    full=build_support(candidate,context_config())
    ablation=build_support(candidate,context_config(limit_context_width=False))
    assert full['intervals'][0]['context_width']==50
    assert full['intervals'][0]['reason']=='broad_amplitude_context'
    assert not full['mask'].any()
    expected=np.zeros(96,bool);expected[30:34]=True
    np.testing.assert_array_equal(ablation['mask'],expected)
    assert not np.any(ablation['mask']&~full['qualified'])


@pytest.mark.parametrize('field,value',[('amplitude',.019999),('stability',.799999),('validation_gain',.079999)])
def test_each_original_row_requirement_remains_mandatory(field,value):
    amplitude=np.full(96,.003);amplitude[32:40]=.1
    candidate=profile(amplitude)
    candidate[field][35]=value
    result=build_support(candidate,context_config())
    assert not result['qualified'][35]
    assert not result['mask'][35] and result['weights'][35]==0.
    assert result['mask'][32:35].all() and result['mask'][36:40].all()
    assert not np.any(result['mask']&~result['qualified'])


def test_new_context_keeps_exact_old_gate_thresholds():
    old=SupportConfig();new=context_config()
    for name in ('min_amplitude','min_stability','min_validation_gain','min_width','max_width','flank_width','max_flank_ratio'):
        assert getattr(new,name)==getattr(old,name)
