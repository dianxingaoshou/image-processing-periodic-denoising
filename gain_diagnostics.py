"""Reference-only diagnostics for candidate subtraction, outside the denoiser.

The reference is accepted by this evaluation module only.  Each tile is examined
in isolation, before clipping and overlap-add; its oracle gain is not an oracle
for the reconstructed image.  Overlapping tiles are correlated observations and
must not be treated as independent images when reporting uncertainty.
"""
from __future__ import annotations

import math

import numpy as np

from denoiser_v3 import effective_gains
from local_estimation_v3 import _geometry


def _finite_image(value, name):
    image = np.asarray(value, dtype=np.float64)
    if image.ndim != 2 or not image.size or not np.isfinite(image).all():
        raise ValueError(f'{name} must be a nonempty finite 2-D image')
    if image.min() < 0 or image.max() > 1:
        raise ValueError(f'{name} must contain pixels in [0,1]')
    return image


def _scalar_features(decision):
    """Keep audit features without retaining image-sized arrays or NaN values."""
    result = {}
    for key, value in decision.items():
        if value is None or isinstance(value, (str, bool)):
            result[key] = value
        elif isinstance(value, (int, float, np.number)):
            numeric = float(value)
            result[key] = numeric if math.isfinite(numeric) else None
    return result


def candidate_gain_rows(observed, clean, prepared, *, strength=None, combination=None):
    """Return reference diagnostics, one row per tile, without changing a model.

    ``oracle_gain`` minimizes ``sum(W*(N-g*n_hat)**2)`` on ``[0,1]``, where
    ``N = observed-clean`` includes any clipping during degradation.  ``W`` is
    exactly the fitting/synthesis Hann window with positive endpoints.  A zero
    candidate has no identifiable gain and is recorded as ``None``, not zero.
    Positive SSE/MSE delta denotes harm; negative delta denotes improvement.
    """
    observed = _finite_image(observed, 'observed')
    clean = _finite_image(clean, 'clean reference')
    source = np.asarray(prepared.estimation.image, dtype=np.float64)
    if observed.shape != clean.shape or observed.shape != source.shape:
        raise ValueError('Observed, clean and prepared image shapes must match')
    if not np.array_equal(observed, source):
        raise ValueError('Prepared candidates must belong to the supplied observed image')
    tiles = prepared.estimation.tiles
    alpha = prepared.config.strength if strength is None else strength
    mode = prepared.config.combination if combination is None else combination
    gains = effective_gains(prepared.fit_weights, prepared.safety_weights, alpha, mode)
    if len(gains) != len(tiles) or len(prepared.decisions) != len(tiles):
        raise ValueError('Prepared tile, gain and decision counts must match')
    rows = []
    for index, (tile, decision, gain) in enumerate(zip(tiles, prepared.decisions, gains)):
        y, x, h, w = (int(tile[key]) for key in ('y', 'x', 'h', 'w'))
        if min(y, x) < 0 or min(h, w) <= 0 or y+h > observed.shape[0] or x+w > observed.shape[1]:
            raise ValueError('Tile lies outside the supplied image')
        estimate = np.asarray(tile['residual'], dtype=np.float64)
        if estimate.shape != (h, w) or not np.isfinite(estimate).all():
            raise ValueError('Candidate residual must be finite and match its tile')
        weight = _geometry((h, w))[4]
        mass = float(weight.sum())
        true_noise = (observed-clean)[y:y+h, x:x+w]
        energy = float(np.sum(weight*estimate*estimate))
        projection = float(np.sum(weight*estimate*true_noise))
        baseline = float(np.sum(weight*true_noise*true_noise))
        nonzero = bool(energy > 1e-24*mass)
        has_candidate = tile.get('selected') is not None
        # Reconstruction forces zero gain for an absent candidate, irrespective
        # of any hand-built Prepared object's numerical weight array.
        actual_gain = float(gain) if has_candidate else 0.
        oracle = float(np.clip(projection/energy, 0., 1.)) if nonzero else None
        actual_delta = actual_gain*actual_gain*energy-2*actual_gain*projection
        oracle_delta = oracle*oracle*energy-2*oracle*projection if nonzero else None
        fit_weight = float(prepared.fit_weights[index])
        eligible = bool(has_candidate and fit_weight > 0 and decision.get('candidate_gate', True))
        row = dict(tile_index=index, y=y, x=x, h=h, w=w,
            has_candidate=has_candidate, nonzero_candidate=nonzero, eligible=eligible,
            fit_weight=fit_weight, safety_weight=float(prepared.safety_weights[index]),
            strength=float(alpha), combination=mode, actual_gain=actual_gain,
            oracle_gain=oracle, candidate_energy=energy, candidate_mean_energy=energy/mass,
            noise_projection=projection, baseline_weighted_sse=baseline,
            actual_weighted_sse_delta=float(actual_delta),
            oracle_weighted_sse_delta=None if oracle_delta is None else float(oracle_delta),
            actual_local_mse_delta=float(actual_delta/mass),
            oracle_local_mse_delta=None if oracle_delta is None else float(oracle_delta/mass),
            avoidable_harm=bool(nonzero and actual_gain > 0 and actual_delta/mass > 1e-14),
            over_subtraction=None if oracle is None else float(max(actual_gain-oracle, 0.)),
            under_subtraction=None if oracle is None else float(max(oracle-actual_gain, 0.)),
            texture_risk=float(decision.get('texture_risk', 0.)),
            global_ambiguity_risk=float(decision.get('global_ambiguity_risk', 0.)),
            risk_features=_scalar_features(decision))
        rows.append(row)
    return rows


def _mean(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def _tail_mean(rows, key):
    values = sorted(float(row[key]) for row in rows if row.get(key) is not None)
    return float(np.mean(values[-max(1, math.ceil(.2*len(values))):])) if values else None


def _correlation(rows, key1, key2):
    values = [(float(row[key1]), float(row[key2])) for row in rows
              if row.get(key1) is not None and row.get(key2) is not None]
    if len(values) < 2:
        return None
    first, second = np.asarray(values).T
    if np.std(first) <= 1e-14 or np.std(second) <= 1e-14:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def summarize_gain_rows(rows, bins=(0., .2, .4, .6, .8, 1.)):
    """Bin eligible nonzero candidates by safety weight, preserving exclusions.

    Tail damage is the mean of the largest ceil(20%*n) local MSE deltas, not a
    per-image statistic.  Empty bins and constant-score correlations are null.
    Confidence intervals are intentionally absent: neighboring tiles overlap.
    """
    rows = list(rows)
    bins = np.asarray(bins, dtype=float)
    if (bins.ndim != 1 or len(bins) < 2 or not np.isfinite(bins).all()
            or bins[0] != 0 or bins[-1] != 1 or np.any(np.diff(bins) <= 0)):
        raise ValueError('Safety bins must increase strictly from 0 to 1')
    valid = [row for row in rows if row['eligible'] and row['nonzero_candidate']]
    if any(not 0 <= row['safety_weight'] <= 1 for row in valid):
        raise ValueError('Safety weights must be in [0,1]')

    def statistics(subset):
        return dict(count=len(subset), mean_safety_weight=_mean(subset, 'safety_weight'),
            mean_oracle_gain=_mean(subset, 'oracle_gain'), mean_actual_gain=_mean(subset, 'actual_gain'),
            harmful_fraction=_mean(subset, 'avoidable_harm'),
            mean_local_mse_delta=_mean(subset, 'actual_local_mse_delta'),
            worst20_local_mse_delta=_tail_mean(subset, 'actual_local_mse_delta'),
            mean_oracle_local_mse_delta=_mean(subset, 'oracle_local_mse_delta'),
            mean_over_subtraction=_mean(subset, 'over_subtraction'),
            mean_under_subtraction=_mean(subset, 'under_subtraction'),
            mean_candidate_energy=_mean(subset, 'candidate_mean_energy'))

    grouped = []
    for index, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        subset = [row for row in valid if low <= row['safety_weight']
                  and (row['safety_weight'] <= high if index == len(bins)-2 else row['safety_weight'] < high)]
        grouped.append(dict(safety_low=float(low), safety_high=float(high),
                            right_closed=index == len(bins)-2, **statistics(subset)))
    return dict(total_tiles=len(rows), candidate_tiles=sum(row['has_candidate'] for row in rows),
        nonzero_tiles=sum(row['nonzero_candidate'] for row in rows),
        eligible_nonzero_tiles=len(valid), excluded_tiles=len(rows)-len(valid),
        safety_oracle_gain_correlation=_correlation(valid, 'safety_weight', 'oracle_gain'),
        fit_oracle_gain_correlation=_correlation(valid, 'fit_weight', 'oracle_gain'),
        overall=statistics(valid), safety_bins=grouped,
        interpretation='reference-only isolated-tile diagnostics before clipping/overlap-add; '
                       'not independent image samples or a full-image oracle')


__all__ = ['candidate_gain_rows', 'summarize_gain_rows']
