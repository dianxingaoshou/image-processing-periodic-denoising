"""Reproducible control pixels, paired carriers and honest ambiguity labels."""
import json

import numpy as np
import pytest

from context_controls import make_controls


def test_controls_are_deterministic_and_do_not_change_global_rng():
    np.random.seed(19);before=np.random.get_state()
    first=make_controls(128);after=np.random.get_state();second=make_controls(128)
    np.testing.assert_array_equal(before[1],after[1])
    assert len(first)==10 and len({row[0] for row in first})==10
    for a,b in zip(first,second):
        assert a[0]==b[0] and a[4]==b[4]
        np.testing.assert_array_equal(a[1],b[1]);np.testing.assert_array_equal(a[2],b[2])
        for key in a[3]:np.testing.assert_array_equal(a[3][key],b[3][key])


def test_masks_and_reference_interpretations_are_explicit():
    for identifier,clean,observed,masks,metadata in make_controls(128):
        assert clean.shape==observed.shape==(128,128)
        assert clean.dtype==observed.dtype==np.float64
        assert np.isfinite(clean).all() and np.isfinite(observed).all()
        assert 0<=clean.min()<=clean.max()<=1 and 0<=observed.min()<=observed.max()<=1
        total=masks['clean'].astype(int)+masks['contaminated']+masks['transition']
        np.testing.assert_array_equal(total,np.ones(clean.shape))
        np.testing.assert_array_equal(observed[masks['clean']],clean[masks['clean']])
        np.testing.assert_array_equal(masks['true_noise'],observed-clean)
        assert np.all(masks['adjacent_clean']<=masks['clean'])
        if identifier.endswith('_clean'):
            np.testing.assert_array_equal(clean,observed)
            assert masks['clean'].all() and not masks['contaminated'].any()
        json.dumps(metadata,allow_nan=False)
    ambiguous=next(row for row in make_controls(128) if row[0]=='compact_legitimate_stripe_clean')
    assert ambiguous[4]['ambiguity_control'] is True
    assert 'not identifiable' in ambiguous[4]['caveat']


def test_added_width_pair_keeps_background_and_carrier():
    controls={row[0]:row for row in make_controls(128)}
    low=controls['broad_texture_edge8'];wide=controls['broad_texture_edge16']
    np.testing.assert_array_equal(low[1],wide[1])
    for field in ('carrier_phase','normal_frequency','tangent_frequency','amplitude','pairing_id'):
        assert low[4][field]==wide[4][field]
    np.testing.assert_array_equal(low[3]['true_fx'],wide[3]['true_fx'])
    np.testing.assert_array_equal(low[3]['true_fy'],wide[3]['true_fy'])
    assert not low[3]['contaminated'][8:].any()
    assert not wide[3]['contaminated'][16:].any()
    assert wide[3]['contaminated'].sum()>low[3]['contaminated'].sum()


@pytest.mark.parametrize('size',(64,127,True,128.5))
def test_invalid_geometry_is_rejected(size):
    with pytest.raises(ValueError):make_controls(size)
