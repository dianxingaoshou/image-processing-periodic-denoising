"""Input-only adequacy checks for optional small analysis windows.

The score asks whether a fixed-frequency component has enough observed periods
and reproducible coefficients. It does not distinguish periodic texture from
noise. Complementary spatially stratified masks retain all sampling parities,
unlike an even-pixel lattice which is degenerate near 0.25 cycles/pixel.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=16)
def _balanced_mask(shape):
    """Half of every 4x4 cell, balanced over parities, fixed local RNG only."""
    h, w = shape
    rng = np.random.default_rng(1729)
    mask = np.zeros(shape, dtype=bool)
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            for dy in (0, 1):
                for dx in (0, 1):
                    yy, xx = np.meshgrid(np.arange(y+dy, min(h, y+4), 2),
                                         np.arange(x+dx, min(w, x+4), 2), indexing='ij')
                    size = yy.size
                    if not size:
                        continue
                    chosen = rng.choice(size, size//2, replace=False)
                    mask[yy.ravel()[chosen], xx.ravel()[chosen]] = True
    mask.setflags(write=False)
    return mask


def small_window_evidence(patch, candidate, minimum_cycles=3., minimum_stability=.8):
    """Return JSON-safe evidence for admitting a small-window candidate.

    ``cycles`` is radial frequency times the shorter pixel-center extent. This
    is rotation-invariant and more conservative than counting the full diagonal
    phase span. Both measures are returned, in cycles, not FFT bins.

    Two disjoint masks, spatially distributed within 4x4 cells and balanced over
    pixel parities, fit sin/cos plus a plane at the same proposed frequency.
    Distributed masks permit amplitude to vary spatially, including a narrow
    strip; opposite spatial halves would penalize an absent strip on one side.
    Coefficient agreement is only a heuristic stability check, not independent
    statistical validation. Mask generation has a fixed local RNG seed.
    """
    image = np.asarray(patch, dtype=np.float64)
    if (image.ndim != 2 or min(image.shape) < 16 or not np.isfinite(image).all()
            or image.min() < 0 or image.max() > 1):
        raise ValueError('Expected a finite 2-D patch in [0,1], dimensions >=16')
    try:
        fy, fx = float(candidate['fy']), float(candidate['fx'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Candidate must provide finite fy and fx in cycles/pixel') from exc
    if not np.isfinite([fy, fx]).all() or not 0 < np.hypot(fy, fx) < .5:
        raise ValueError('Candidate radial frequency must be finite and in (0,.5)')
    if (not np.isfinite(minimum_cycles) or minimum_cycles <= 0
            or not np.isfinite(minimum_stability) or not 0 <= minimum_stability <= 1):
        raise ValueError('Invalid small-window cycles or stability threshold')

    h, w = image.shape
    cycles = float(np.hypot(fy, fx) * (min(h, w)-1))
    phase_span = float(abs(fy)*(h-1) + abs(fx)*(w-1))
    result = dict(eligible=False, cycles=cycles, phase_span=phase_span,
                  stability=0., amplitude=0., amplitude_balance=0., fit_rank_min=0,
                  reason='insufficient_cycles')
    if cycles < minimum_cycles:
        return result

    yy, xx = np.indices(image.shape, dtype=float)
    u, v = (yy-(h-1)/2)/h, (xx-(w-1)/2)/w
    phase = 2*np.pi*(fy*yy+fx*xx)
    design = np.stack((np.sin(phase), np.cos(phase), np.ones_like(image), u, v), axis=-1)
    weight = np.outer(np.hanning(h+2)[1:-1], np.hanning(w+2)[1:-1])
    first = _balanced_mask(image.shape)
    coefficients = []; ranks = []
    for mask in (first, ~first):
        root = np.sqrt(weight[mask])
        coef, _, rank, _ = np.linalg.lstsq(design[mask]*root[:, None], image[mask]*root, rcond=None)
        coefficients.append(coef[:2]); ranks.append(int(rank))
    a, b = coefficients
    amp_a, amp_b = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    mass = amp_a+amp_b
    stability = float(np.clip(1-np.linalg.norm(a-b)/max(mass, 1e-15), 0, 1)) if mass > 1e-8 else 0.
    balance = float(min(amp_a, amp_b)/max(amp_a, amp_b, 1e-15))
    valid = min(ranks) == 5 and mass > 1e-8
    eligible = bool(valid and stability >= minimum_stability)
    result.update(eligible=eligible, stability=stability, amplitude=mass/2,
                  amplitude_balance=balance, fit_rank_min=min(ranks),
                  reason='stable_small_window' if eligible else
                         'no_identifiable_component' if not valid else 'unstable_coefficients')
    return result


__all__ = ['small_window_evidence']
