"""Input-only row-coherence gate and compact support for directional proposals.

This module receives estimated profile features, never a clean reference,
frequency truth or degradation mask. A compact coherent periodic region can
still be legitimate texture: the tests below establish support evidence only.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SupportConfig:
    min_amplitude: float = .02
    min_stability: float = .8
    min_validation_gain: float = .08
    min_width: int = 3
    max_width: int = 32
    flank_width: int = 16
    max_flank_ratio: float = .35

    def __post_init__(self):
        if not np.isfinite(self.min_amplitude) or self.min_amplitude <= 0:
            raise ValueError('min_amplitude must be finite and positive')
        if not np.isfinite(self.min_stability) or not 0 <= self.min_stability <= 1:
            raise ValueError('min_stability must be finite in [0,1]')
        if not np.isfinite(self.min_validation_gain) or not 0 <= self.min_validation_gain <= 1:
            raise ValueError('min_validation_gain must be finite in [0,1]')
        for name in ('min_width','max_width','flank_width'):
            value=getattr(self,name)
            if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.max_width < self.min_width:
            raise ValueError('max_width must not be smaller than min_width')
        if not np.isfinite(self.max_flank_ratio) or not 0 <= self.max_flank_ratio < 1:
            raise ValueError('max_flank_ratio must be finite in [0,1)')


def _profile_arrays(profile):
    arrays={}
    for name in ('amplitude','stability','validation_gain'):
        if name not in profile:
            raise ValueError(f'Missing profile feature: {name}')
        array=np.asarray(profile[name],dtype=float)
        if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
            raise ValueError(f'{name} must be a nonempty finite 1-D array')
        arrays[name]=array
    if len({len(a) for a in arrays.values()})!=1:
        raise ValueError('Profile feature arrays must have the same length')
    if np.any(arrays['amplitude']<0):
        raise ValueError('Profile amplitudes must be nonnegative')
    if np.any((arrays['stability']<0)|(arrays['stability']>1)):
        raise ValueError('Profile stability must be in [0,1]')
    # Negative held-out gains are valid evidence against subtracting a model.
    return arrays['amplitude'],arrays['stability'],arrays['validation_gain']


def _runs(mask):
    starts=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(np.int8))==1)
    stops=np.flatnonzero(np.diff(np.r_[False,mask,False].astype(np.int8))==-1)
    return zip(starts.tolist(),stops.tolist())


def build_support(profile,config:SupportConfig|None=None,*,use_gate=True,compact_support=True):
    """Return a row mask, binary weights and auditable interval decisions.

    A row must exceed the amplitude floor. With ``use_gate`` it must also show
    stable split coefficients and positive cross-fit improvement. These are
    hard input-evidence tests; weights are one, so gain-strength saturation is
    not mixed into this ablation.

    With ``compact_support`` only contiguous groups of 3..32 qualified rows
    survive, and observed neighboring flanks must be substantially weaker. The
    larger median among the available left/right flanks is divided by the
    group's median amplitude. At an image edge one observed flank suffices;
    without any flank compactness is unidentifiable, so the group is rejected.
    No gap filling, dilation or extrapolation is performed. Outside the final
    mask weights are exactly zero.

    Disabling compact support preserves the qualified-row mask while skipping
    group width and flank contrast. Disabling the gate skips only stability and
    validation-gain checks. Neither ablation accepts a ground-truth mask.
    """
    cfg=config or SupportConfig()
    if not isinstance(use_gate,(bool,np.bool_)) or not isinstance(compact_support,(bool,np.bool_)):
        raise ValueError('use_gate and compact_support must be boolean')
    amplitude,stability,gain=_profile_arrays(profile)
    amplitude_ok=amplitude>=cfg.min_amplitude
    stable_ok=stability>=cfg.min_stability
    validation_ok=gain>=cfg.min_validation_gain
    qualified=amplitude_ok & (stable_ok & validation_ok if use_gate else True)
    mask=np.zeros(len(amplitude),dtype=bool);intervals=[]
    for start,stop in _runs(qualified):
        width=stop-start
        inside=float(np.median(amplitude[start:stop]))
        left=amplitude[max(0,start-cfg.flank_width):start]
        right=amplitude[stop:min(len(amplitude),stop+cfg.flank_width)]
        left_median=float(np.median(left)) if len(left) else None
        right_median=float(np.median(right)) if len(right) else None
        flanks=[value for value in (left_median,right_median) if value is not None]
        ratio=max(flanks)/max(inside,1e-15) if flanks else None
        if not compact_support:reason='compact_support_disabled'
        elif width<cfg.min_width:reason='too_few_supported_rows'
        elif width>cfg.max_width:reason='broad_periodic_region'
        elif ratio is None:reason='no_observed_flank'
        elif ratio>cfg.max_flank_ratio:reason='insufficient_outside_contrast'
        else:reason='compact_supported_region'
        accepted=bool(reason in ('compact_support_disabled','compact_supported_region'))
        if accepted:mask[start:stop]=True
        intervals.append(dict(start=start,stop=stop,width=width,accepted=accepted,reason=reason,
            touches_image_edge=bool(start==0 or stop==len(amplitude)),
            median_amplitude=inside,left_flank_median=left_median,right_flank_median=right_median,
            outside_to_inside_ratio=ratio,
            median_stability=float(np.median(stability[start:stop])),
            median_validation_gain=float(np.median(gain[start:stop]))))
    return dict(weights=mask.astype(float),mask=mask,qualified=qualified,
        intervals=intervals,qualified_rows=int(qualified.sum()),supported_rows=int(mask.sum()),
        amplitude_pass_rows=int(amplitude_ok.sum()),stability_pass_rows=int(stable_ok.sum()),
        validation_pass_rows=int(validation_ok.sum()),use_gate=bool(use_gate),compact_support=bool(compact_support),
        interpretation='input-only coherence and support evidence; not semantic noise recognition')


__all__=['SupportConfig','build_support']
