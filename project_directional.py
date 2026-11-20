"""Directional path for the integrated restorer, with early negative evidence checks."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d

from narrow_context import context_flanks
from narrow_profiles import _cross_gain, _fit, _refine_frequency, _spectral_state
from narrow_support import _runs, build_support


def _frequencies(rows, cfg):
    """Original tangent peaks and refinement order; only fitting diagnostics are omitted."""
    count, length = rows.shape
    if length < cfg.minimum_tangent_length:
        return
    energy, normalized, bins = _spectral_state(rows, cfg)
    if energy.max() < 1e-16:
        return
    k = min(cfg.pooling_rows, count)
    pooled = np.mean(np.partition(normalized, count-k, axis=0)[count-k:], axis=0)
    low = max(cfg.min_frequency, cfg.minimum_cycles/(length-1))
    peaks = np.flatnonzero((bins >= low) & (bins <= cfg.max_frequency)
                           & (pooled == maximum_filter1d(pooled, 5, mode="nearest")))
    chosen = []
    for index in sorted(peaks, key=lambda i: (-pooled[i], i)):
        initial = float(bins[index])
        if any(abs(initial-f) < 1.5/length for f in chosen):
            continue
        frequency, _ = _refine_frequency(rows, initial, cfg, normalized, bins)
        if any(abs(frequency-f) < 1.5/length for f in chosen):
            continue
        # A rejected frequency still consumes its old candidate slot.
        chosen.append(frequency)
        yield frequency
        if len(chosen) >= cfg.max_candidates:
            return


def _possible_rows(amplitude, cfg):
    """Necessary conditions only. A candidate satisfying them still passes build_support."""
    s = cfg.support
    above_floor = amplitude >= s.min_amplitude
    if not cfg.compact_support:
        return above_floor
    possible = np.zeros_like(above_floor)
    for start, stop in _runs(above_floor):
        if stop - start < s.min_width:
            continue
        if s.contrast_context == "amplitude":
            if s.limit_context_width and stop - start > s.max_width:
                continue
            flanks = [v for v in context_flanks(amplitude, start, stop, s.flank_width) if v is not None]
            # Every eventual qualified fragment uses these same flanks. Its median
            # cannot exceed the maximum of the enclosing amplitude interval.
            if not flanks or max(flanks) / max(amplitude[start:stop].max(), 1e-15) > s.max_flank_ratio:
                continue
        possible[start:stop] = True
    return possible


def _supported_fit(rows, positions, frequency, cfg):
    length = len(positions)
    coefficients, _ = _fit(rows, frequency, positions, length)
    amplitude = np.linalg.norm(coefficients[:, :2], axis=1)
    possible = _possible_rows(amplitude, cfg)
    if not possible.any():
        return None

    if cfg.use_gate:
        left, right = positions[:length//2], positions[length//2:]
        first, _ = _fit(rows, frequency, left, length)
        second, _ = _fit(rows, frequency, right, length)
        mass = np.linalg.norm(first[:, :2], axis=1) + np.linalg.norm(second[:, :2], axis=1)
        difference = np.linalg.norm(first[:, :2]-second[:, :2], axis=1)
        stability = np.clip(1-difference/np.maximum(mass, 1e-15), 0, 1)
        stability[mass < 1e-8] = 0.
        width = cfg.support.min_width if cfg.compact_support else 1
        if not any(end-start >= width for start, end in _runs(possible & (stability >= cfg.support.min_stability))):
            return None
        gain = np.minimum(_cross_gain(rows, frequency, left, second[:, :2], length),
                          _cross_gain(rows, frequency, right, first[:, :2], length))
    else:
        stability = gain = np.zeros_like(amplitude)
    support = build_support(dict(amplitude=amplitude, stability=stability, validation_gain=gain),
                            cfg.support, use_gate=cfg.use_gate, compact_support=cfg.compact_support)
    if not support["mask"].any():
        return None
    return coefficients, amplitude, support


def narrow_residual(observed, cfg, *, return_regions=False):
    """Return the unscaled, unclipped noise fit, its support, and optional accepted intervals."""
    observed = np.asarray(observed, dtype=np.float64)
    residual = None
    row_scores, column_scores = (np.zeros(size) for size in observed.shape)
    regions = []
    for axis, rows in enumerate((observed, observed.T)):
        positions = np.arange(rows.shape[1])
        scores = row_scores if axis == 0 else column_scores
        for frequency in _frequencies(rows, cfg.profiles):
            fit = _supported_fit(rows, positions, frequency, cfg)
            if fit is None:
                continue
            coefficients, amplitude, support = fit
            # Priority is constant along each full row/column. Track each axis
            # in 1-D and resolve crossings only on winning columns (row wins a tie).
            priority = amplitude**2
            active = np.flatnonzero(support["mask"] & (priority > scores))
            if active.size:
                if residual is None:
                    residual = np.zeros_like(observed)
                scores[active] = priority[active]
                phase = 2*np.pi*frequency*positions
                wave = coefficients[active, 0, None]*np.sin(phase) + coefficients[active, 1, None]*np.cos(phase)
                if axis == 0:
                    residual[active] = wave
                else:
                    beats_row = priority[active, None] > row_scores
                    residual.T[active] = np.where(beats_row, wave, residual.T[active])
            if return_regions:
                for r in support["intervals"]:
                    if r["accepted"]:
                        regions.append(dict(normal_axis=axis, start=r["start"], stop=r["stop"],
                                            context_start=r["context_start"], context_stop=r["context_stop"],
                                            tangent_frequency=frequency))
    if residual is None:
        return None, None, []
    support_mask = (row_scores[:, None] > 0) | (column_scores > 0)
    return residual, support_mask, regions
