"""Protection tests exercise ambiguity and spatial decisions, not named cases."""
from dataclasses import replace

import numpy as np
import pytest

from texture_guard_v2 import GuardConfig,assess_global,guard_weights,score_tile


@pytest.mark.parametrize('kind',['single','grid','chirp','rotating'])
def test_general_low_dimensional_periodic_images_abstain(kind):
    rng=np.random.default_rng(703)
    yy,xx=np.indices((192,224),dtype=float)
    fy,fx=rng.uniform(.07,.13,size=2)
    phase=fy*yy+fx*xx
    if kind=='chirp':phase+=.5*.04*(xx-111.5)**2/223
    elif kind=='rotating':phase+=.06*(xx-111.5)*(yy-95.5)/223
    a=.45+.12*np.sin(2*np.pi*phase+.41)
    if kind=='grid':a+=.06*np.cos(2*np.pi*(-.12*yy+.04*xx)+.2)
    diag=assess_global(a)
    assert diag['abstain']
    assert diag['ambiguity_risk']==1.


def test_broadband_structure_is_not_declared_pure_periodic():
    yy,xx=np.indices((128,160),dtype=float)
    a=.4+.08*np.sin(2*np.pi*(.091*yy+.132*xx))
    a+=.06*np.random.default_rng(19).normal(size=a.shape)
    a=np.clip(a,0,1)
    diag=assess_global(a)
    assert not diag['abstain']
    assert diag['ambiguity_risk']==0.


def test_same_observation_decision_has_no_reference_argument():
    yy,xx=np.indices((64,64),dtype=float)
    candidate=.1*np.sin(2*np.pi*(.1*yy+.14*xx))
    image=.5+candidate
    ambiguity=assess_global(image)
    decision=score_tile(image,candidate,evidence=1,neighbor_consistency=1,
                        cross_scale_consistency=1,global_assessment=ambiguity)
    assert decision['suppression']==0.
    with pytest.raises(TypeError):
        score_tile(image,candidate,clean_reference=np.full_like(image,.5))


def test_texture_penalty_changes_spatial_decision_not_constant_gain():
    y,x=np.indices((64,64),dtype=float)
    candidate=.05*np.sin(2*np.pi*(.11*y+.15*x))
    smooth=.5+.02*x/64
    structured=.5+.12*np.sin(2*np.pi*(.035*y+.04*x))
    kwargs=dict(evidence=1,peak_score=6.,neighbor_consistency=1,cross_scale_consistency=1,
                amplitude=.05,fit_improvement=.4)
    ds=score_tile(smooth+candidate,candidate,**kwargs)
    dt=score_tile(structured+candidate,candidate,**kwargs)
    assert ds['suppression']>dt['suppression']
    assert ds['texture_risk']<dt['texture_risk']


def test_adapter_bounds_and_missing_candidate():
    a=np.full((64,64),.5)
    weights,diag,records=guard_weights(a,[dict(selected=None)])
    assert weights.tolist()==[0.]
    assert not diag['abstain']
    assert records[0]['reason']=='no candidate'
    with pytest.raises(ValueError):GuardConfig(reliability_low=.9,reliability_high=.8)


def test_ablations_are_explicit_and_diagnostics_preserve_raw_risk():
    yy,xx=np.indices((64,64),dtype=float)
    a=.5+.1*np.sin(2*np.pi*(.11*yy+.15*xx))
    config=GuardConfig(use_ambiguity_guard=False,use_texture_risk=False,use_consistency=False)
    diag=assess_global(a,config)
    assert diag['raw_ambiguity_risk']==1.
    assert diag['ambiguity_risk']==0.
    decision=score_tile(a,a-.5,evidence=1,peak_score=6.,neighbor_consistency=0,cross_scale_consistency=0,
                        global_assessment=diag,config=config)
    assert decision['reliability']==1.
    assert decision['texture_risk']==0.
    assert decision['suppression']==1.


def test_energy_only_low_score_candidate_cannot_subtract_without_independent_support():
    yy,xx=np.indices((64,64),dtype=float)
    residual=.1*np.sin(2*np.pi*(.11*yy+.15*xx))
    image=.5+residual
    kwargs=dict(evidence=1,peak_score=3.,neighbor_consistency=1.,cross_scale_consistency=.5)
    decision=score_tile(image,residual,**kwargs)
    assert decision['suppression']==0.
    assert not decision['spectral_gate']
    assert not decision['candidate_gate']
    assert decision['support_reason']=='insufficient_periodic_evidence'
    # A high consistency number alone does not prove independent-scale support.
    kwargs['cross_scale_consistency']=1.
    assert score_tile(image,residual,**kwargs)['suppression']==0.
    supported=score_tile(image,residual,independent_scale_support=True,**kwargs)
    assert supported['suppression']>0
    assert supported['support_reason']=='independent_scale_and_neighbor'
