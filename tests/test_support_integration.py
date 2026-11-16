"""Integration checks: support extensions are input-only and independently switched."""
from dataclasses import replace
import json

import numpy as np
import pytest

from local_estimation_v3 import EstimationConfig, prepare, restore, fit_candidate


def test_adding_small_window_keeps_large_window_fallback_for_too_few_cycles():
    yy,xx=np.indices((96,96))
    image=.5+.12*np.sin(2*np.pi*(.015*yy+.04*xx)+.2)
    prepared=prepare(image,EstimationConfig(scales=(32,64,128)))
    checks=[check for tile in prepared.tiles for check in tile['small_window_checks']]
    assert any(check['reason']=='insufficient_cycles' for check in checks)
    assert any(tile['selected'] is not None and tile['selected']['scale']>=64 for tile in prepared.tiles)
    for tile in prepared.tiles:
        assert all(candidate['small_window_evidence']['eligible'] for candidate in tile['candidates'] if candidate['scale']==32)


def test_32_synthesis_covers_odd_image_and_zero_weights_preserve_pixels():
    yy,xx=np.indices((79,113));image=.5+.08*np.sin(2*np.pi*(.12*yy+.16*xx))
    prepared=prepare(image,EstimationConfig(scales=(32,64,128),synthesis_size=32,stride=16))
    output,diag=restore(prepared,weights=np.zeros(len(prepared.tiles)),return_diagnostics=True)
    np.testing.assert_array_equal(output,image)
    assert np.all(diag['weight_sum']>0)
    assert all(tile['h']==tile['w']==32 for tile in prepared.tiles)
    json.dumps(diag['tiles'],allow_nan=False)


def test_grid_alternative_keeps_phase_and_can_select_nonaffine_envelope():
    yy,xx=np.indices((64,64));fy,fx=.14,.17
    envelope=.12*np.maximum(1-np.abs(xx-31.5)/31.5,0)
    image=.5+envelope*np.sin(2*np.pi*(fy*yy+fx*xx)+.3)
    candidate=dict(fy=fy,fx=fx,scale=64,width_bins=2.)
    cfg=EstimationConfig(enable_chirp=False)
    baseline=fit_candidate(image,candidate,cfg)
    grid=fit_candidate(image,candidate,replace(cfg,amplitude_model='grid'))
    assert grid['fy']==baseline['fy'] and grid['fx']==baseline['fx']
    assert 'grid_amplitude' in grid['model']
    assert grid['final_sse']<baseline['final_sse']*.2
    assert grid['envelope_diagnostics']['selected']
    assert np.isfinite(grid['prediction']).all()


@pytest.mark.parametrize('kwargs',[dict(amplitude_model='pixel'),dict(envelope_grid_size=5),
    dict(envelope_ridge=-1),dict(minimum_small_window_cycles=np.nan)])
def test_invalid_support_settings_fail(kwargs):
    with pytest.raises(ValueError): EstimationConfig(**kwargs)
