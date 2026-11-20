"""Conservative project routing with input-only ambiguity checks on both paths."""
from __future__ import annotations

import numpy as np

import denoiser_v3 as general
from local_estimation_v3 import _geometry
from local_spectral import _checked
from project_directional import narrow_residual
from project_pipeline import MIN_OVERLAP_AGREEMENT, load_config
from project_texture import AMBIGUOUS_FRACTION_LIMIT, HARMONIC_SCORE_LIMIT, STRUCTURE_LOSS_LIMIT, screen_tiles
from texture_guard_v3 import assess_global


# One 8-bit intensity level, on the common [0, 1] working scale.
PLANE_RMS_LIMIT = 1 / 255


def routing_parameters():
    return dict(version="texture_screen_v1", minimum_overlap_agreement=MIN_OVERLAP_AGREEMENT,
                narrow_plane_rms_limit=PLANE_RMS_LIMIT, harmonic_score_limit=HARMONIC_SCORE_LIMIT,
                structure_loss_limit=STRUCTURE_LOSS_LIMIT, minimum_ambiguous_fraction=AMBIGUOUS_FRACTION_LIMIT)


def _plane_errors(candidate, support):
    """Fit one affine brightness plane, score proposed band and surroundings separately."""
    if support.all():
        return None
    h, w = candidate.shape
    y = np.arange(h, dtype=float) - (h - 1) / 2
    x = np.arange(w, dtype=float) - (w - 1) / 2
    # On a rectangular grid, the intercept and the two centered axes are orthogonal.
    slope_y = np.dot(y, candidate.mean(axis=1)) / np.dot(y, y)
    slope_x = np.dot(x, candidate.mean(axis=0)) / np.dot(x, x)
    error = candidate - candidate.mean() - slope_y * y[:, None] - slope_x * x
    squared = error * error
    return dict(inside=float(np.sqrt(squared[support].mean())),
                outside=float(np.sqrt(squared[~support].mean())))


def _restore_general(observed, tiles, gains, allowed, diagnostics):
    """Reconstruct only the requested output; every window still enters the denominator."""
    numerator = np.zeros_like(observed)
    denominator = np.zeros_like(observed)
    support = np.zeros(observed.shape, dtype=bool) if diagnostics else None
    regions = []
    for index, t in enumerate(tiles):
        h, w, y, x = t["h"], t["w"], t["y"], t["x"]
        where = np.s_[y:y+h, x:x+w]
        window = _geometry((h, w))[4]
        denominator[where] += window
        if allowed[index]:
            numerator[where] += window * gains[index] * t["residual"]
            if diagnostics:
                support[where] = True
                c = t["selected"]
                regions.append(dict(y=y, x=x, height=h, width=w, scale=c["scale"],
                                    frequency_y=c["fy"], frequency_x=c["fx"], gain=float(gains[index])))
    if np.any(denominator <= 0):
        raise RuntimeError("Overlap weights do not cover the image")
    result = np.clip(observed - numerator / denominator, 0, 1)
    return result, support, regions


def denoise(image, config=None, *, return_diagnostics=False):
    """Apply the refined route to a single observed grayscale image."""
    observed = _checked(image)
    cfg = load_config() if config is None else config
    evidence = dict(narrow_supported_pixels=0, general_candidate_tiles=None, general_corroborated_tiles=None,
                    minimum_overlap_agreement=MIN_OVERLAP_AGREEMENT)

    def finish(route, reason, result=None, support=None, regions=None):
        if result is None:
            result = observed.copy()
        if not return_diagnostics:
            return result
        if support is None:
            support = np.zeros(observed.shape, dtype=bool)
        return result, dict(route=route, reason=reason, global_assessment=assessment, evidence=evidence,
                            accepted_regions=regions or [], residual=observed-result, support_mask=support)

    if observed.min() == observed.max():
        assessment = assess_global(observed, cfg.general.guard) if return_diagnostics else None
        return finish("unchanged", "constant_image")

    noise, support, regions = narrow_residual(observed, cfg.narrow, return_regions=return_diagnostics)
    if noise is not None:
        evidence["narrow_supported_pixels"] = int(support.sum())
        assessment = assess_global(observed, cfg.general.guard)
        if assessment["abstain"]:
            return finish("unchanged", "global_periodic_ambiguity")
        # The ambiguity concerns the proposed component, independently of how
        # much of it the caller ultimately subtracts. Clip each proposed output
        # only after applying its own strength, including for saturated pixels.
        unit_result = np.clip(observed - noise, 0, 1)
        errors = _plane_errors(unit_result, support)
        evidence["narrow_plane_rmse"] = errors
        if errors is None:
            return finish("unchanged", "no_narrow_background")
        if max(errors.values()) <= PLANE_RMS_LIMIT:
            return finish("unchanged", "narrow_affine_background_ambiguity")
        if cfg.narrow.strength == 0:
            return finish("unchanged", "narrow_strength_zero")
        result = unit_result if cfg.narrow.strength == 1 else np.clip(observed - cfg.narrow.strength * noise, 0, 1)
        return finish("narrow", "compact_directional_support", result, support, regions)

    # A component cannot span a full tile displacement on both axes in this geometry.
    if min(observed.shape) < 2 * cfg.general.estimation.synthesis_size:
        assessment = assess_global(observed, cfg.general.guard) if return_diagnostics else None
        return finish("unchanged", "insufficient_image_extent")

    prepared = general.prepare(observed, cfg.general)
    assessment = prepared.global_assessment
    if assessment["abstain"]:
        return finish("unchanged", "global_periodic_ambiguity")
    gains = general.effective_gains(prepared.fit_weights, prepared.safety_weights,
                                    cfg.general.strength, cfg.general.combination)
    allowed, screening = screen_tiles(observed, prepared, gains)
    evidence.update(general_candidate_tiles=int(np.count_nonzero(gains)),
                    general_corroborated_tiles=screening["spatially_eligible_tiles"],
                    general_accepted_tiles=int(allowed.sum()), texture_screen=screening)
    if not allowed.any():
        reason = "texture_component_veto" if screening["texture_vetoed_tiles"] else "insufficient_spatial_support"
        return finish("unchanged", reason)
    result, support, regions = _restore_general(observed, prepared.estimation.tiles, gains, allowed, return_diagnostics)
    return finish("general", "spatially_corroborated_tiles", result, support, regions)
