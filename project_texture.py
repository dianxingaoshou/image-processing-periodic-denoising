"""Screen broad, phase-consistent components using the existing per-tile texture evidence."""
from __future__ import annotations

import numpy as np

from project_pipeline import _agree_on_overlap


# The existing harmonic score normalizes harmonic excess power by 0.20 in
# the project configuration. A score of 0.25 then corresponds to 5% power.
HARMONIC_SCORE_LIMIT = 0.25
STRUCTURE_LOSS_LIMIT = 0.8
AMBIGUOUS_FRACTION_LIMIT = 0.5


def _corroborated_components(tiles, gains):
    """Return spatially broad components using the original general-route geometry."""
    grid = {(tile["gy"], tile["gx"]): i for i, tile in enumerate(tiles) if gains[i] > 0}
    seen = set()
    for first in grid.values():
        if first in seen:
            continue
        stack = [first]
        seen.add(first)
        members = []
        while stack:
            i = stack.pop()
            members.append(i)
            tile = tiles[i]
            y, x = tile["gy"], tile["gx"]
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                j = grid.get(neighbor)
                if j is not None and j not in seen and _agree_on_overlap(tile, tiles[j]):
                    stack.append(j)
                    seen.add(j)
        span_y = np.ptp([tiles[i]["y"] for i in members])
        span_x = np.ptp([tiles[i]["x"] for i in members])
        if span_y >= tiles[first]["h"] and span_x >= tiles[first]["w"]:
            yield members


def screen_tiles(observed, prepared, gains):
    """Return allowed general-route tiles and decisions per connected component.

    Waveform agreement establishes periodicity. The additional tests look for
    repeatable evidence of harmonic structure or near-pure periodic content
    whose subtraction would eliminate much of the smoothed gradient energy.
    All three features have already been measured on the input and proposed
    residual by the general estimator. No reference or new fit is required.
    """
    if np.shape(observed) != prepared.estimation.image.shape:
        raise ValueError("Observation must match the prepared image geometry")
    tiles = prepared.estimation.tiles
    gains = np.asarray(gains, dtype=float)
    if gains.shape != (len(tiles),) or not np.isfinite(gains).all() or np.any((gains < 0) | (gains > 1)):
        raise ValueError("Expected one finite gain in [0, 1] per tile")
    allowed = np.zeros(len(tiles), dtype=bool)
    records = []
    eligible_count = 0
    concentration_limit = prepared.config.guard.local_ambiguity_high
    for members in _corroborated_components(tiles, gains):
        eligible_count += len(members)
        decisions = [prepared.decisions[i] for i in members]
        harmonic = float(np.median([d["harmonic_support"] for d in decisions]))
        ambiguous_fraction = float(np.mean([
            d["local_periodic_fraction"] >= concentration_limit and d["structure_loss"] >= STRUCTURE_LOSS_LIMIT
            for d in decisions
        ]))
        if harmonic >= HARMONIC_SCORE_LIMIT:
            reason = "periodic_harmonics"
        elif ambiguous_fraction >= AMBIGUOUS_FRACTION_LIMIT:
            reason = "near_pure_structure_loss"
        else:
            reason = "accepted"
            allowed[members] = True
        records.append(dict(tile_indices=sorted(members), reason=reason,
                            median_harmonic_score=harmonic, near_pure_loss_fraction=ambiguous_fraction))
    return allowed, dict(spatially_eligible_tiles=eligible_count, texture_vetoed_tiles=eligible_count-int(allowed.sum()),
                         harmonic_score_limit=HARMONIC_SCORE_LIMIT, periodic_fraction_limit=concentration_limit,
                         structure_loss_limit=STRUCTURE_LOSS_LIMIT, components=records)
