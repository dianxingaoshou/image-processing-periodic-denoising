"""Observed amplitude context for contrast estimation, never output support.

A coherence-qualified run can end before the surrounding periodic amplitude
does. Its adjacent rejected rows are therefore not automatically background.
These helpers locate the contiguous amplitude context and measure flanks beyond
it. They do not expand the restoration mask or accept reference/noise masks.
"""
from __future__ import annotations

import numpy as np


def _checked_interval(amplitude, start, stop):
    values=np.asarray(amplitude,dtype=float)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or np.any(values<0):
        raise ValueError('amplitude must be a nonempty finite nonnegative 1-D array')
    if any(isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) for value in (start,stop)):
        raise ValueError('start and stop must be integer half-open bounds')
    if not 0<=start<stop<=len(values):
        raise ValueError('Require 0 <= start < stop <= amplitude length')
    return values,int(start),int(stop)


def amplitude_context(amplitude,start,stop,threshold):
    """Return the contiguous amplitude interval containing ``[start, stop)``.

    Every seed row must meet the same observed amplitude floor. Expansion stops
    at the first below-floor row and never bridges a gap. Returned bounds are
    used only to select contrast flanks; they are not restoration support.
    """
    values,start,stop=_checked_interval(amplitude,start,stop)
    if not np.isscalar(threshold) or not np.isfinite(threshold) or threshold<=0:
        raise ValueError('threshold must be finite and positive')
    if np.any(values[start:stop]<threshold):
        raise ValueError('The seed interval must be contiguous above-threshold amplitude')
    while start>0 and values[start-1]>=threshold:start-=1
    while stop<len(values) and values[stop]>=threshold:stop+=1
    return start,stop


def context_flanks(amplitude,start,stop,flank_width):
    """Return observed left/right flank medians outside half-open context.

    A missing side is ``None``. No inferred values, ratios, masks or weights are
    created, including when the context reaches both image boundaries.
    """
    values,start,stop=_checked_interval(amplitude,start,stop)
    if (isinstance(flank_width,(bool,np.bool_)) or not isinstance(flank_width,(int,np.integer))
            or flank_width<1):
        raise ValueError('flank_width must be a positive integer')
    left=values[max(0,start-flank_width):start]
    right=values[stop:min(len(values),stop+flank_width)]
    return (float(np.median(left)) if len(left) else None,
            float(np.median(right)) if len(right) else None)


__all__=['amplitude_context','context_flanks']
