"""Opt-in restoration of extended, axis-aligned narrow periodic bands.

Unlike the generic local estimator, this branch proposes a tangent frequency
and fits complex coefficients per normal row. It subtracts only on estimated
compact coherent support; it does not run legacy residuals elsewhere. This is
a specialized alternative, not a default/general periodic denoiser.
"""
from __future__ import annotations

from dataclasses import asdict,dataclass,field

import numpy as np

from narrow_profiles import ProfileConfig,propose_profiles
from narrow_support import SupportConfig,build_support


@dataclass(frozen=True)
class Config:
    profiles:ProfileConfig=field(default_factory=ProfileConfig)
    support:SupportConfig=field(default_factory=SupportConfig)
    use_gate:bool=True
    compact_support:bool=True
    strength:float=1.

    def __post_init__(self):
        if not isinstance(self.use_gate,bool) or not isinstance(self.compact_support,bool):
            raise ValueError('Gate/support switches must be boolean')
        if not np.isfinite(self.strength) or not 0<=self.strength<=1:
            raise ValueError('strength must be finite in [0,1]')


@dataclass
class Prepared:
    image:np.ndarray
    config:Config
    profiles:list


def config_dict(config):return asdict(config)


def config_from_dict(data):
    return Config(profiles=ProfileConfig(**data.get('profiles',{})),
        support=SupportConfig(**data.get('support',{})),
        use_gate=data.get('use_gate',True),compact_support=data.get('compact_support',True),
        strength=data.get('strength',1.))


def prepare(image,config=None):
    cfg=config or Config();a=np.asarray(image,dtype=float)
    if a.ndim!=2 or min(a.shape)<32 or not np.isfinite(a).all() or a.min()<0 or a.max()>1:
        raise ValueError('Expected a finite [0,1] 2-D image, dimensions >=32')
    return Prepared(a.copy(),cfg,[p for axis in (0,1) for p in propose_profiles(a,axis,cfg.profiles)])


def restore(prepared,strength=None,*,use_gate=None,compact_support=None,return_diagnostics=False):
    p=prepared;cfg=p.config;alpha=cfg.strength if strength is None else strength
    if not np.isfinite(alpha) or not 0<=alpha<=1:raise ValueError('strength must be in [0,1]')
    gate=cfg.use_gate if use_gate is None else use_gate
    compact=cfg.compact_support if compact_support is None else compact_support
    raw=np.zeros_like(p.image);score_map=np.zeros_like(raw)
    axis_map=np.full(raw.shape,-1,dtype=np.int8);frequency_map=np.zeros_like(raw)
    records=[]
    for index,profile in enumerate(p.profiles):
        support=build_support(profile,cfg.support,use_gate=gate,compact_support=compact)
        mask=support['mask'];axis=profile['axis'];frequency=profile['frequency']
        tangent_length=p.image.shape[1-axis]
        phase=2*np.pi*frequency*np.arange(tangent_length)
        coefficients=profile['coefficients']
        candidate=coefficients[:,0,None]*np.sin(phase)+coefficients[:,1,None]*np.cos(phase)
        # One winner per pixel avoids summing several fits of the same stripe.
        # The ranking is observed candidate energy, never reference benefit.
        row_score=np.asarray(profile['amplitude'])**2
        scores=np.broadcast_to((mask*row_score)[:,None],candidate.shape)
        if axis==1:candidate=candidate.T;scores=scores.T
        update=scores>score_map
        raw[update]=candidate[update];score_map[update]=scores[update]
        axis_map[update]=axis;frequency_map[update]=frequency
        records.append(dict(index=index,axis=int(axis),frequency_cycles_per_pixel=float(frequency),
            spectral_score=float(profile['spectral_score']),
            amplitude=np.asarray(profile['amplitude']).tolist(),stability=np.asarray(profile['stability']).tolist(),
            validation_gain=np.asarray(profile['validation_gain']).tolist(),
            supported_rows=support['supported_rows'],qualified_rows=support['qualified_rows'],
            amplitude_pass_rows=support['amplitude_pass_rows'],stability_pass_rows=support['stability_pass_rows'],
            validation_pass_rows=support['validation_pass_rows'],intervals=support['intervals']))
    raw*=float(alpha);output=np.clip(p.image-raw,0,1)
    if not return_diagnostics:return output
    return output,dict(raw_residual=raw,residual=p.image-output,support_mask=score_map>0,
        tangent_frequency=frequency_map,axis=axis_map,profiles=records,
        use_gate=bool(gate),compact_support=bool(compact),strength=float(alpha),
        interpretation='input-only directional support; not a noise probability or a full 2-D frequency map')


def denoise(image,config=None,return_diagnostics=False):
    return restore(prepare(image,config),return_diagnostics=return_diagnostics)
