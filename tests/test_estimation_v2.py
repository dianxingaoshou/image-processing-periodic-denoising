"""Model and numerical checks for candidate residual estimation only."""
import json
from dataclasses import replace

import numpy as np
import pytest

from local_estimation_v2 import EstimationConfig,prepare,restore


def unweighted(image,cfg):
    p=prepare(image,cfg)
    return restore(p,weights=np.ones(len(p.tiles))),p


def test_affine_amplitude_reduces_edge_bias():
    y,x=np.indices((128,160))
    clean=.4+.03*x/160+.02*y/128
    noise=.12*(.2+.8*x/159)*np.sin(2*np.pi*(.11*y+.15*x)+.4)
    cfg=EstimationConfig(scales=(64,),enable_chirp=False)
    result,p=unweighted(clean+noise,cfg)
    constant,_=unweighted(clean+noise,replace(cfg,enable_amplitude_model=False))
    assert np.mean((result-clean)**2)<.2*np.mean((constant-clean)**2)
    assert any('affine_amplitude' in t.get('model','') for t in p.tiles)


def test_chirp_model_reduces_curved_phase_residual():
    y,x=np.indices((128,128))
    clean=np.full(y.shape,.45)
    phase=.12*y+.15*x+.00025*(x-63.5)**2+.0004*(x-63.5)*(y-63.5)
    observed=clean+.1*np.sin(2*np.pi*phase+.7)
    cfg=EstimationConfig(scales=(64,),enable_amplitude_model=False)
    result,p=unweighted(observed,cfg)
    stationary,_=unweighted(observed,replace(cfg,enable_chirp=False))
    assert np.mean((result-clean)**2)<.75*np.mean((stationary-clean)**2)
    assert any('chirp' in t.get('model','') for t in p.tiles)


def test_force_scale_keeps_evidence_and_json_diagnostics():
    y,x=np.indices((256,256));image=.5+.12*np.sin(2*np.pi*(.09*y+.14*x)+.7)
    p=prepare(image,EstimationConfig(force_scale=128))
    selected=[t for t in p.tiles if t['selected'] is not None]
    assert selected
    assert all(t['selected']['requested_scale']==128 for t in selected)
    assert any({c['requested_scale'] for c in t['candidates']}=={64,128,256} for t in selected)
    _,d=restore(p,return_diagnostics=True)
    json.dumps(d['tiles'],allow_nan=False)


def test_odd_boundaries_and_explicit_abstention():
    y,x=np.indices((79,113));a=.45+.05*np.sin(2*np.pi*(.10*y+.16*x))
    p=prepare(a)
    zero,d=restore(p,weights=np.zeros(len(p.tiles)),return_diagnostics=True)
    np.testing.assert_array_equal(zero,a)
    assert np.min(d['weight_sum'])>0
    assert np.isfinite(d['weight_sum']).all()
    for weights in ([1.],np.full(len(p.tiles),np.nan),np.full(len(p.tiles),1.1)):
        with pytest.raises(ValueError):restore(p,weights=weights)


def test_constant_image_has_no_candidate_modification():
    a=np.full((97,131),.42)
    b,p=unweighted(a,EstimationConfig())
    np.testing.assert_array_equal(a,b)
    assert all(t['selected'] is None for t in p.tiles)
