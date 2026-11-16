"""Regression for frequency minima hidden inside a quadratic-phase tile."""
from dataclasses import replace

import numpy as np
import pytest

from local_estimation_v3 import EstimationConfig, _geometry, fit_candidate


def _chirp(center_fy, curvature=4.):
    y,x,u,v,*_=_geometry((64,64))
    observed=.5+.1*np.sin(2*np.pi*(center_fy*y+curvature*u*v)+.4)
    candidate=dict(fy=center_fy,fx=0.,scale=64,width_bins=2.)
    return observed,candidate


def _fitted_radius(result):
    y,x=np.indices((64,64),dtype=float)
    dy=y-31.5;dx=x-31.5
    fy=result['fy']+result['hessian_yy']*dy+result['hessian_yx']*dx
    fx=result['fx']+result['hessian_yx']*dy+result['hessian_xx']*dx
    return np.hypot(fy,fx)


@pytest.mark.parametrize('center_fy',(.025,.053))
def test_in_band_corners_cannot_hide_interior_frequency_violation(center_fy):
    observed,candidate=_chirp(center_fy)
    cfg=EstimationConfig()
    result=fit_candidate(observed,candidate,cfg)
    radius=_fitted_radius(result)
    corners=radius[np.ix_([0,63],[0,63])]
    assert np.all((corners>=cfg.min_frequency)&(corners<=cfg.max_frequency))
    assert radius.min()<cfg.min_frequency
    if center_fy==.025:assert radius.min()<.001
    assert result['model']=='rejected_frequency_range'
    assert result['frequency_min']==pytest.approx(radius.min(),abs=1e-14)
    assert result['frequency_max']==pytest.approx(radius.max(),abs=1e-14)
    assert result['frequency_range_scope']=='all discrete pixel centres'
    np.testing.assert_array_equal(result['residual'],np.zeros_like(observed))
    assert result['amplitude']==result['fit_ratio']==result['fit_improvement']==0.
    weight=_geometry(observed.shape)[4]
    actual_sse=np.sum(weight*(observed-result['prediction'])**2)
    assert result['final_sse']==pytest.approx(actual_sse,abs=1e-12)
    assert result['final_sse']==result['baseline_sse']


def test_rejected_holdout_fit_returns_its_own_background_error():
    observed,candidate=_chirp(.025)
    y,x=np.indices(observed.shape)
    mask=(y+x)%3!=0
    result=fit_candidate(observed,candidate,EstimationConfig(),fit_mask=mask)
    assert result['model']=='rejected_frequency_range'
    actual_sse=np.sum(_geometry(observed.shape)[4]*mask*(observed-result['prediction'])**2)
    assert result['final_sse']==pytest.approx(actual_sse,abs=1e-12)


@pytest.mark.parametrize('fy,curvature',((.03,0.),(.14,1.6)))
def test_valid_low_frequency_and_normal_chirp_remain_accepted(fy,curvature):
    observed,candidate=_chirp(fy,curvature)
    cfg=EstimationConfig()
    result=fit_candidate(observed,candidate,cfg)
    assert not result['model'].startswith('rejected')
    if curvature:assert 'chirp' in result['model']
    assert result['frequency_min']>=cfg.min_frequency
    assert result['frequency_max']<=cfg.max_frequency
    assert result['fit_improvement']>.99
    assert np.sqrt(np.mean((result['prediction']-observed)**2))<.001


def test_upper_frequency_violation_also_synchronizes_fallback():
    observed,candidate=_chirp(.14,1.6)
    cfg=replace(EstimationConfig(),max_frequency=.145)
    result=fit_candidate(observed,candidate,cfg)
    assert result['frequency_max']>cfg.max_frequency
    assert result['model']=='rejected_frequency_range'
    assert result['final_sse']==result['baseline_sse']
