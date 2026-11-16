"""Support widths and evaluation masks must be fixed and pairable."""
import json

import numpy as np
import pytest

from support_data import KINDS, make_sample


@pytest.mark.parametrize('kind',KINDS)
def test_support_masks_partition_and_reproduce_actual_clipped_noise(kind):
    reference=np.full((96,128),.5)
    observed,meta,masks=make_sample(reference,'analytic-support',kind,0 if kind=='clean' else .12,2)
    count=masks['contaminated'].astype(int)+masks['clean']+masks.get('transition',np.zeros(reference.shape,bool))
    np.testing.assert_array_equal(count,np.ones(reference.shape))
    np.testing.assert_array_equal(observed[masks['clean']],reference[masks['clean']])
    if kind not in ('clean','global'):
        np.testing.assert_allclose(masks['true_noise'],observed-reference,atol=0)
        assert np.all(masks['adjacent_clean']<=masks['clean'])
        assert masks['adjacent_clean'].any()
    json.dumps(meta,allow_nan=False)


def test_width_and_amplitude_changes_preserve_carrier_and_rng():
    clean=np.full((96,128),.5)
    np.random.seed(7);state=np.random.get_state()
    low,ml,tl=make_sample(clean,'paired','edge8',.06,0)
    high,mh,th=make_sample(clean,'paired','edge8',.18,0)
    wide,mw,tw=make_sample(clean,'paired','edge32',.06,0)
    np.testing.assert_array_equal(np.random.get_state()[1],state[1])
    np.testing.assert_allclose(high-clean,3*(low-clean),atol=2e-16)
    for name in ('seed','fy','fx','phase_offset'):
        assert ml[name]==mh[name]==mw[name]
    assert not tl['contaminated'][:,8:].any()
    assert not tw['contaminated'][:,32:].any()
    assert tw['contaminated'].sum()>tl['contaminated'].sum()
