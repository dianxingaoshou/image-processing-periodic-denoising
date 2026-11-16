"""V3 separates candidate fitting evidence from non-cancellable safety caps.

These are explicit heuristic scores, not posterior noise probabilities.  The
global guard recognizes images almost completely explained by a small number
of local narrow bands, including smooth chirps.  Such pixels can equally be
legitimate texture or noise on a smooth field, so the policy is to abstain.
It does not claim to distinguish these indistinguishable interpretations.

Only observed pixels and candidate estimates enter this module.  Synthetic
frequencies, clean references and degradation masks are never accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class GuardConfig:
    use_ambiguity_guard: bool = True
    use_texture_risk: bool = True
    use_consistency: bool = True
    use_candidate_gate: bool = True
    minimum_peak_score: float = 4.5
    reliability_low: float = .35
    reliability_high: float = .80
    texture_penalty: float = .65
    ambiguity_low: float = .985
    ambiguity_high: float = .998
    local_size: int = 64
    local_stride: int = 64
    global_pairs: int = 4
    local_pairs: int = 3
    band_radius_bins: float = 3.
    min_frequency: float = .025
    amplitude_low: float = .006
    amplitude_high: float = .024
    gradient_scale: float = .025

    def __post_init__(self):
        if not 0 <= self.reliability_low < self.reliability_high <= 1:
            raise ValueError('Invalid reliability ramp')
        if not 0 <= self.ambiguity_low < self.ambiguity_high <= 1:
            raise ValueError('Invalid ambiguity ramp')
        if not 0 <= self.texture_penalty <= 1:
            raise ValueError('texture_penalty must be in [0,1]')
        if not np.isfinite(self.minimum_peak_score) or self.minimum_peak_score<=0:
            raise ValueError('minimum_peak_score must be positive and finite')
        if self.local_size < 16 or not 0 < self.local_stride <= self.local_size:
            raise ValueError('Invalid local coverage geometry')
        if self.global_pairs < 1 or self.local_pairs < 1 or self.band_radius_bins <= 0:
            raise ValueError('Invalid sparse band settings')
        if not 0 <= self.min_frequency < .5:
            raise ValueError('Invalid protected frequency')
        if not 0 <= self.amplitude_low < self.amplitude_high or self.gradient_scale <= 0:
            raise ValueError('Invalid intensity scales')


def _checked(image):
    a=np.asarray(image,dtype=np.float64)
    if a.ndim!=2 or min(a.shape)<16 or not np.isfinite(a).all():
        raise ValueError('Expected a finite 2-D image, dimensions >=16')
    if a.min()<0 or a.max()>1:
        raise ValueError('Expected pixels in [0,1]')
    return a


def _detrend(a):
    h,w=a.shape
    y=np.arange(h)-(h-1)/2
    x=np.arange(w)-(w-1)/2
    b=a-a.mean()
    b-=y[:,None]*(b.mean(axis=1)@y)/max(float(y@y),1.)
    b-=x[None,:]*(b.mean(axis=0)@x)/max(float(x@x),1.)
    return b


def _ramp(value,low,high):
    return float(np.clip((value-low)/(high-low),0,1))


def _starts(length,size,stride):
    if length<=size:return [0]
    values=list(range(0,length-size+1,stride))
    if values[-1]!=length-size:values.append(length-size)
    return values


def _band_fraction(image,pairs,radius,min_frequency):
    """Fraction of detrended Hann energy in <=pairs conjugate peak disks.

    Fixed original FFT-bin disk size is used only for within-window energy
    concentration.  Physical frequencies are never compared in these bins.
    Energy near DC is included in the denominator and protected from selection.
    """
    a=_detrend(image)
    rms=float(np.sqrt(np.mean(a*a)))
    if rms<1e-8:return 0.,rms
    h,w=a.shape
    window=np.outer(np.hanning(h),np.hanning(w))
    power=np.abs(np.fft.fft2(a*window))**2
    fy=np.fft.fftfreq(h)[:,None]
    fx=np.fft.fftfreq(w)[None,:]
    work=np.where(np.hypot(fy,fx)>=min_frequency,power,0).copy()
    selected=np.zeros(a.shape,dtype=bool)
    yy,xx=np.indices(a.shape)
    for _ in range(pairs):
        py,px=np.unravel_index(np.argmax(work),work.shape)
        if work[py,px]<=0:break
        for cy,cx in ((py,px),((-py)%h,(-px)%w)):
            dy=np.minimum(np.abs(yy-cy),h-np.abs(yy-cy))
            dx=np.minimum(np.abs(xx-cx),w-np.abs(xx-cx))
            selected|=(dy*dy+dx*dx<=radius*radius)
        work[selected]=0
    return float(np.clip(power[selected].sum()/max(float(power.sum()),1e-30),0,1)),rms


def structural_complexity(image,gradient_scale=.025):
    """Intensity-scale gradient complexity after suppressing pixel oscillations.

    This measures residual structure; it is not a semantic texture detector.
    Clipping is deliberately not applied to the candidate-subtracted image.
    """
    a=np.asarray(image,dtype=np.float64)
    smooth=ndi.gaussian_filter(a,1.,mode='reflect')
    dy,dx=np.gradient(smooth)
    gradient_rms=float(np.sqrt(np.mean(dx*dx+dy*dy)))
    return dict(gradient_rms=gradient_rms,score=float(np.clip(gradient_rms/gradient_scale,0,1)))


def assess_global(image,config:GuardConfig|None=None):
    """Assess near-pure periodic ambiguity using only the observed image.

    A local lower quantile requires broad spatial coverage rather than allowing
    one periodic patch to veto a photograph.  The few-band test also handles a
    grid, while local bands permit frequency/direction to change smoothly.
    """
    cfg=config or GuardConfig()
    a=_checked(image)
    global_fraction,rms=_band_fraction(a,cfg.global_pairs,cfg.band_radius_bins,cfg.min_frequency)
    h,w=a.shape
    sh,sw=min(h,cfg.local_size),min(w,cfg.local_size)
    local=[]
    for y in _starts(h,sh,cfg.local_stride):
        for x in _starts(w,sw,cfg.local_stride):
            fraction,patch_rms=_band_fraction(a[y:y+sh,x:x+sw],cfg.local_pairs,
                                             cfg.band_radius_bins,cfg.min_frequency)
            local.append(dict(y=y,x=x,fraction=fraction,detrended_rms=patch_rms))
    fractions=np.array([r['fraction'] for r in local])
    q10=float(np.quantile(fractions,.1))
    global_risk=_ramp(global_fraction,cfg.ambiguity_low,cfg.ambiguity_high)
    local_risk=_ramp(q10,cfg.ambiguity_low,cfg.ambiguity_high)
    raw_risk=max(global_risk,local_risk) if rms>1e-8 else 0.
    risk=raw_risk if cfg.use_ambiguity_guard else 0.
    return dict(ambiguity_risk=float(risk),abstain=bool(risk>=1-1e-12),
                raw_ambiguity_risk=float(raw_risk),sparse_periodic_fraction=global_fraction,
                local_periodic_fraction_q10=q10,
                local_periodic_fraction_mean=float(fractions.mean()),
                detrended_rms=rms,residual_complexity=structural_complexity(a,cfg.gradient_scale),
                local_coverage=local,
                interpretation='reliability and ambiguity scores, not noise probabilities')


def score_tile(image_tile,candidate_residual,*,evidence=0.,peak_score=None,
               neighbor_consistency=0.,cross_scale_consistency=0.,
               different_window_support=False,
               amplitude=None,fit_improvement=None,global_assessment=None,
               config:GuardConfig|None=None):
    """Return an explicit reliability/risk decision for one candidate residual.

    Subtractive candidates that leave strong structure receive a penalty unless
    they explain a substantial part of the detrended patch.  This spatially
    varying risk permits a matched-removal comparison, unlike a constant gain.
    """
    cfg=config or GuardConfig()
    a=_checked(image_tile)
    n=np.asarray(candidate_residual,dtype=np.float64)
    if n.shape!=a.shape or not np.isfinite(n).all():
        raise ValueError('Candidate residual must be finite and match image shape')
    b=a-n
    complexity_before=structural_complexity(a,cfg.gradient_scale)
    complexity_after=structural_complexity(b,cfg.gradient_scale)
    centered=_detrend(a)
    after=_detrend(b)
    improvement=float(np.clip(1-np.mean(after*after)/max(float(np.mean(centered*centered)),1e-20),0,1))
    # The estimate's own weighted score is supplied for audit, while the actual
    # unweighted reduction is used here to detect damaging extrapolation edges.
    model_improvement=improvement if fit_improvement is None else float(np.clip(fit_improvement,0,1))
    if amplitude is None:amplitude=float(np.sqrt(2*np.mean(n*n)))
    amp_gate=_ramp(float(amplitude),cfg.amplitude_low,cfg.amplitude_high)
    e=float(np.clip(evidence,0,1))
    neighbor=float(np.clip(neighbor_consistency,0,1))
    cross=float(np.clip(cross_scale_consistency,0,1))
    spectral_gate=bool(peak_score is not None and float(peak_score)>=cfg.minimum_peak_score)
    supported_fallback=bool(cfg.use_consistency and different_window_support
                            and cross>=.8 and neighbor>=.8)
    eligible=bool(not cfg.use_candidate_gate or spectral_gate or supported_fallback)
    reason=('gate_disabled' if not cfg.use_candidate_gate else
            'spectral_peak' if spectral_gate else
            'different_window_and_neighbor' if supported_fallback else
            'insufficient_periodic_evidence')
    reliability=(.60*e+.25*neighbor+.15*cross) if cfg.use_consistency else e
    reliability_weight=_ramp(reliability,cfg.reliability_low,cfg.reliability_high)
    # A residual that explains a window-center fit but damages its spatial
    # support is not granted a high weight merely for being spectrally strong.
    extrapolation_gate=float(np.clip(improvement/.02,0,1))
    texture_risk=(cfg.texture_penalty*complexity_after['score']*(1-model_improvement)
                  if cfg.use_texture_risk else 0.)
    global_risk=float((global_assessment or {}).get('ambiguity_risk',0.))
    fit_weight=float(np.clip(reliability_weight*amp_gate*extrapolation_gate,0,1))
    safety_weight=float(np.clip((1-texture_risk)*(1-global_risk),0,1))
    weight=fit_weight*safety_weight if eligible else 0.
    return dict(reliability=float(reliability),reliability_weight=reliability_weight,
                texture_risk=float(texture_risk),global_ambiguity_risk=global_risk,
                amplitude_gate=amp_gate,extrapolation_gate=extrapolation_gate,
                fit_improvement=model_improvement,unweighted_improvement=improvement,
                complexity_before=complexity_before,complexity_after=complexity_after,
                peak_score=None if peak_score is None else float(peak_score),
                spectral_gate=spectral_gate,candidate_gate=eligible,support_reason=reason,
                different_window_support=bool(different_window_support),
                fit_weight=fit_weight,safety_weight=safety_weight,
                suppression=weight)


def guard_weights(image,tiles,config:GuardConfig|None=None):
    """Batch adapter for actual-window evidence, never requested scale labels.

    Returns ``(fit_weights, safety_weights, global_diagnostics, decisions)``.
    Different overlapping windows are corroborating views, not statistically
    independent observations. Gate failures zero fitting weights explicitly.
    """
    cfg=config or GuardConfig()
    a=_checked(image)
    global_diag=assess_global(a,cfg)
    fit_weights=[];safety_weights=[];records=[]
    for tile in tiles:
        selected=tile.get('selected')
        if selected is None:
            fit_weights.append(0.);safety_weights.append(1.)
            records.append(dict(fit_weight=0.,safety_weight=1.,candidate_gate=False,suppression=0.,reason='no candidate'))
            continue
        y,x=int(tile['y']),int(tile['x'])
        residual=np.asarray(tile['residual'],dtype=float)
        h,w=residual.shape
        selected_id=tuple(selected.get('analysis_window_id',()))
        window_ids={tuple(c.get('analysis_window_id',())) for c in tile.get('candidates',())}
        window_ids.discard(())
        different_window=bool(selected_id and any(window_id!=selected_id for window_id in window_ids))
        decision=score_tile(a[y:y+h,x:x+w],residual,
            evidence=selected.get('evidence',0.),peak_score=selected.get('score'),
            neighbor_consistency=tile.get('neighbor',0.),cross_scale_consistency=tile.get('cross_scale',0.),
            different_window_support=different_window,
            amplitude=tile.get('amplitude'),fit_improvement=tile.get('fit_improvement',tile.get('fit_ratio')),
            global_assessment=global_diag,config=cfg)
        fit_weights.append(decision['fit_weight'] if decision['candidate_gate'] else 0.)
        safety_weights.append(decision['safety_weight'])
        records.append(decision)
    return np.asarray(fit_weights,dtype=float),np.asarray(safety_weights,dtype=float),global_diag,records


__all__=['GuardConfig','assess_global','score_tile','guard_weights','structural_complexity']
