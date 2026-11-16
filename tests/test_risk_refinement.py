"""Mechanism checks for opt-in structure risk, without clean-reference access."""
import inspect
import json

import numpy as np
import pytest

from texture_guard_v3 import GuardConfig, score_tile


def periodic_patch():
    y,x=np.indices((64,64),dtype=float)
    carrier=.10*np.sin(2*np.pi*(.109375*y+.15625*x)+.37)
    return .5+carrier,carrier


def decision(image,residual,config,**kwargs):
    arguments=dict(evidence=1.,peak_score=8.,neighbor_consistency=1.,
                   cross_scale_consistency=1.,different_window_support=True,
                   amplitude=.1,fit_improvement=.8)
    arguments.update(kwargs)
    return score_tile(image,residual,config=config,**arguments)


def test_structure_risk_cannot_be_waived_by_high_fit_improvement():
    image,residual=periodic_patch()
    # Add some nonperiodic structure so the legacy risk is nonzero.
    image+=.025*np.random.default_rng(7).normal(size=image.shape)
    cfg=GuardConfig(risk_mode='structure')
    low=decision(image,residual,cfg,fit_improvement=0.)
    high=decision(image,residual,cfg,fit_improvement=1.)
    assert low['texture_risk']==high['texture_risk']
    assert low['fit_weight']==high['fit_weight']
    assert low['legacy_texture_risk']>high['legacy_texture_risk']
    assert high['legacy_texture_risk']==0.


def test_local_periodic_ambiguity_is_not_diluted_by_other_image_regions():
    image,residual=periodic_patch()
    cfg=GuardConfig(risk_mode='structure')
    pure=decision(image,residual,cfg)
    cluttered=image+.075*np.random.default_rng(19).normal(size=image.shape)
    clutter=decision(cluttered,residual,cfg)
    assert pure['local_periodic_fraction']>.99
    assert pure['local_ambiguity']==1.
    assert pure['structure_loss']==pytest.approx(1.)
    assert pure['texture_risk']==pytest.approx(cfg.texture_penalty)
    assert clutter['local_periodic_fraction']<pure['local_periodic_fraction']
    assert clutter['texture_risk']<pure['texture_risk']


def test_observed_harmonics_are_reported_without_truth_frequency():
    y,x=np.indices((64,64),dtype=float)
    phase=2*np.pi*(.0625*y+.09375*x)+.37
    residual=.10*np.sin(phase)
    cfg=GuardConfig(risk_mode='structure')
    plain=decision(.5+residual,residual,cfg)
    harmonics=.5+residual+.07*np.sin(2*phase)+.05*np.sin(3*phase)
    mixed=decision(harmonics,residual,cfg)
    assert mixed['harmonic_support']>.9
    assert mixed['harmonic_support']>plain['harmonic_support']
    assert 'clean_reference' not in inspect.signature(score_tile).parameters
    with pytest.raises(TypeError):
        score_tile(harmonics,residual,clean_reference=.5)


def test_risk_modes_share_exact_candidate_fit_weights():
    image,residual=periodic_patch()
    image+=.03*np.random.default_rng(11).normal(size=image.shape)
    old=decision(image,residual,GuardConfig())
    new=decision(image,residual,GuardConfig(risk_mode='structure'))
    for field in ('fit_weight','reliability','reliability_weight','amplitude_gate',
                  'extrapolation_gate','candidate_gate'):
        assert old[field]==new[field]
    assert old['texture_risk']==old['legacy_texture_risk']
    assert new['texture_risk']==new['structure_texture_risk']


def test_zero_candidate_and_flat_input_have_finite_zero_structure_features():
    cfg=GuardConfig(risk_mode='structure')
    result=decision(np.full((64,64),.5),np.zeros((64,64)),cfg,amplitude=0.)
    for name in ('structure_loss','local_ambiguity','harmonic_support',
                 'structure_texture_risk','fit_weight'):
        assert result[name]==0.
    json.dumps(result,allow_nan=False)


def test_texture_ablation_preserves_raw_features_and_legacy_default():
    image,residual=periodic_patch()
    cfg=GuardConfig(risk_mode='structure',use_texture_risk=False)
    result=decision(image,residual,cfg)
    assert result['texture_risk']==0.
    assert result['structure_texture_risk']>0.
    assert result['safety_weight']==1.
    assert GuardConfig().risk_mode=='legacy'


@pytest.mark.parametrize('changes',[
    {'risk_mode':'unknown'}, {'local_ambiguity_low':.99,'local_ambiguity_high':.9},
    {'local_ambiguity_high':float('nan')}, {'harmonic_reference':float('inf')},
    {'harmonic_reference':0.}, {'amplitude_high':float('inf')},
])
def test_invalid_risk_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        GuardConfig(**changes)


@pytest.mark.parametrize('field',['evidence','peak_score','neighbor_consistency',
                                 'cross_scale_consistency','amplitude','fit_improvement'])
def test_nonfinite_candidate_scalars_are_rejected(field):
    image,residual=periodic_patch()
    with pytest.raises(ValueError,match='finite'):
        decision(image,residual,GuardConfig(risk_mode='structure'),**{field:float('nan')})


def test_invalid_global_risk_cannot_propagate_nan_to_weights():
    image,residual=periodic_patch()
    with pytest.raises(ValueError,match='Global ambiguity risk'):
        decision(image,residual,GuardConfig(risk_mode='structure'),
                 global_assessment={'ambiguity_risk':float('nan')})
