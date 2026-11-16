import numpy as np
import pytest

from denoiser_v3 import effective_gains
from texture_guard_v3 import guard_weights


def test_safety_is_not_cancelled_by_strength():
    assert effective_gains([1.],[.7],2.,'coupled')[0]==1.
    assert effective_gains([1.],[.7],2.,'decoupled')[0]==pytest.approx(.7)
    rng=np.random.default_rng(3);q=rng.random(100);s=rng.random(100)
    for alpha in (0.,.5,1.,1.5,2.):assert np.all(effective_gains(q,s,alpha)<=s+1e-15)


def test_duplicate_requested_scales_are_not_window_support():
    y,x=np.indices((64,64));image=.5+.1*np.sin(2*np.pi*(.1*y+.15*x))
    selected=dict(score=1.,evidence=1.,requested_scale=64,analysis_window_id=(0,0,64,64))
    tile=dict(y=0,x=0,h=64,w=64,residual=image-.5,selected=selected,
        amplitude=.1,fit_improvement=1.,neighbor=1.,cross_scale=1.,
        candidates=[dict(selected,requested_scale=s) for s in (64,128,256)])
    q,_,_,decisions=guard_weights(image,[tile])
    assert not decisions[0]['different_window_support']
    assert not decisions[0]['candidate_gate']
    assert q[0]==0.
