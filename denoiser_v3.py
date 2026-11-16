"""Conservative nonstationary periodic residual subtraction with safety caps."""
from dataclasses import asdict,dataclass,field

import numpy as np

from local_estimation_v3 import EstimationConfig,prepare as estimate,restore as reconstruct
from texture_guard_v3 import GuardConfig,guard_weights


@dataclass(frozen=True)
class Config:
    estimation:EstimationConfig=field(default_factory=EstimationConfig)
    guard:GuardConfig=field(default_factory=GuardConfig)
    strength:float=2.
    combination:str='decoupled'

    def __post_init__(self):
        if not np.isfinite(self.strength) or not 0<=self.strength<=2:raise ValueError('strength must be in [0,2]')
        if self.combination not in ('coupled','decoupled'):raise ValueError('Unknown weight combination')


@dataclass
class Prepared:
    config:Config
    estimation:object
    fit_weights:np.ndarray
    safety_weights:np.ndarray
    global_assessment:dict
    decisions:list


def config_dict(config):return asdict(config)


def config_from_dict(data):
    return Config(estimation=EstimationConfig(**data.get('estimation',{})),
        guard=GuardConfig(**data.get('guard',{})),strength=data.get('strength',2.),
        combination=data.get('combination','decoupled'))


def effective_gains(fit_weights,safety_weights,strength,combination='decoupled'):
    q=np.asarray(fit_weights,dtype=float);s=np.asarray(safety_weights,dtype=float)
    if q.shape!=s.shape or not np.isfinite(q).all() or not np.isfinite(s).all() or np.any((q<0)|(q>1)|(s<0)|(s>1)):
        raise ValueError('Fitting and safety weights must be finite [0,1] arrays of identical shape')
    if not np.isfinite(strength) or not 0<=strength<=2:raise ValueError('strength must be in [0,2]')
    if combination=='coupled':return np.minimum(strength*q*s,1.)
    if combination=='decoupled':return np.minimum(strength*q,1.)*s
    raise ValueError('Unknown combination')


def prepare(image,config=None):
    cfg=config or Config();p=estimate(image,cfg.estimation)
    q,s,global_diag,decisions=guard_weights(p.image,p.tiles,cfg.guard)
    return Prepared(cfg,p,q,s,global_diag,decisions)


def restore(prepared,strength=None,combination=None,conservative=True,return_diagnostics=False):
    p=prepared;alpha=p.config.strength if strength is None else strength
    mode=p.config.combination if combination is None else combination
    if not np.isfinite(alpha) or not 0<=alpha<=2:raise ValueError('strength must be in [0,2]')
    if mode not in ('coupled','decoupled'):raise ValueError('Unknown weight combination')
    gains=effective_gains(p.fit_weights,p.safety_weights,alpha,mode) if conservative else np.array([
        min(float(alpha),1.) if t['selected'] is not None else 0. for t in p.estimation.tiles])
    output=reconstruct(p.estimation,weights=gains,strength=1.,return_diagnostics=return_diagnostics)
    if not return_diagnostics:return output
    result,diag=output
    risky=(p.safety_weights<1-1e-8)&(gains>=1-1e-8)&(p.fit_weights>0)
    diag.update(global_assessment=p.global_assessment,combination=mode,
        risky_full_subtraction_blocks=int(np.sum(risky)),eligible_blocks=int(np.sum(p.fit_weights>0)),
        total_blocks=len(p.estimation.tiles))
    maps={k:np.zeros_like(result) for k in ('fit_weight','safety_weight','effective_gain')}
    for i,(t,record,decision) in enumerate(zip(p.estimation.tiles,diag['tiles'],p.decisions)):
        record.update(guard_decision=decision,fit_weight=float(p.fit_weights[i]),
            safety_weight=float(p.safety_weights[i]),effective_gain=float(gains[i]),
            risky_full_subtraction=bool(risky[i]))
        sy,sx=t['h'],t['w'];sl=np.s_[t['y']:t['y']+sy,t['x']:t['x']+sx]
        w=np.outer(np.hanning(sy+2)[1:-1],np.hanning(sx+2)[1:-1])
        for name,value in (('fit_weight',p.fit_weights[i]),('safety_weight',p.safety_weights[i]),('effective_gain',gains[i])):
            maps[name][sl]+=w*value
    for name,array in maps.items():diag[name]=array/diag['weight_sum']
    return result,diag


def denoise(image,config=None,return_diagnostics=False):
    return restore(prepare(image,config),return_diagnostics=return_diagnostics)
