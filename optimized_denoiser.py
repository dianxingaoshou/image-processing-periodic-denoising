"""Second-version estimator plus explicit texture-risk decisions.

All decisions use observed pixels only. Candidate estimates are retained before
the guard, so the guard ablation truly disables both global/local decisions.
"""
from dataclasses import asdict,dataclass,field

import numpy as np

from local_estimation_v2 import EstimationConfig,prepare as estimate,restore as reconstruct
from texture_guard_v2 import GuardConfig,guard_weights


@dataclass(frozen=True)
class V2Config:
    estimation:EstimationConfig=field(default_factory=EstimationConfig)
    guard:GuardConfig=field(default_factory=GuardConfig)
    strength:float=1.

    def __post_init__(self):
        if not np.isfinite(self.strength) or not 0<=self.strength<=2:
            raise ValueError('strength must be finite and in [0,2]')


@dataclass
class Prepared:
    config:V2Config
    estimation:object
    weights:np.ndarray
    global_assessment:dict
    decisions:list


def config_dict(config):return asdict(config)


def config_from_dict(data):
    return V2Config(estimation=EstimationConfig(**data.get('estimation',{})),
                    guard=GuardConfig(**data.get('guard',{})),strength=data.get('strength',1.))


def prepare(image,config=None):
    cfg=config or V2Config()
    p=estimate(image,cfg.estimation)
    weights,assessment,decisions=guard_weights(p.image,p.tiles,cfg.guard)
    return Prepared(cfg,p,weights,assessment,decisions)


def restore(prepared,strength=None,conservative=True,return_diagnostics=False):
    p=prepared
    # No guard means raw eligible residuals, including ambiguous global textures.
    weights=p.weights if conservative else np.array([float(t['selected'] is not None) for t in p.estimation.tiles])
    result=reconstruct(p.estimation,weights=weights,
                       strength=p.config.strength if strength is None else strength,
                       return_diagnostics=return_diagnostics)
    if not return_diagnostics:return result
    image,diag=result
    diag['global_assessment']=p.global_assessment
    for record,decision in zip(diag['tiles'],p.decisions):record['guard_decision']=decision
    return image,diag


def denoise(image,config=None,return_diagnostics=False):
    return restore(prepare(image,config),return_diagnostics=return_diagnostics)
