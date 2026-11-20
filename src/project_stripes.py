"""Check whether a compact directional fit selected a harmonic of recurring texture."""
import numpy as np

import denoiser_v3 as general
import narrow_periodicity
from project_pipeline import load_config
from project_refined import _restore_general
from project_texture import screen_tiles
from project_weave import denoise as previous, routing_parameters as previous_parameters
from weave_guard import _frequency_distance, analyze_lattice


SEPARATION_BINS = 1.


def routing_parameters():
    return dict(version="narrow_harmonic_review_v1", previous=previous_parameters(),
                narrow_periodicity=narrow_periodicity.routing_parameters(),
                broad_separation_tile_bins=SEPARATION_BINS, broad_reconsideration_model="sinusoid")


def _independent_components(observed, tiles, allowed, components, rejected):
    """Keep a whole corroborated component only if its stationary carrier is distinct."""
    indices = np.flatnonzero(allowed)
    fits = [tiles[i]["selected"] for i in indices]
    frequencies = np.array([[fit["fy"], fit["fx"]] for fit in fits])
    sizes = np.array([[tiles[i]["h"], tiles[i]["w"]] for i in indices])
    regions = [dict(frequency_y=y, frequency_x=x, height=h, width=w)
               for (y, x), (h, w) in zip(frequencies, sizes)]
    peaks, _, lattice = analyze_lattice(observed, regions)
    # Even a dominant measured lattice peak stays protected during this retry.
    eligible = np.array([fit["model"] == "sinusoid" for fit in fits])
    if len(peaks):
        eligible &= _frequency_distance(frequencies, peaks, sizes[:, None]).min(axis=1) > SEPARATION_BINS
    for region in rejected:
        axis = 1 - region["normal_axis"]
        for protected in (region["tangent_frequency"], *region["confirmed_companions"]):
            distance = abs(abs(frequencies[:, axis]) - protected)
            eligible &= distance * sizes[:, axis] > SEPARATION_BINS

    # Do not cut a bridge from a connected component and then treat the
    # disconnected remnants as independent evidence of a broad disturbance.
    keep = np.zeros(len(tiles), dtype=bool)
    for component in components:
        if component["reason"] != "accepted":
            continue
        members = component["tile_indices"]
        positions = np.searchsorted(indices, members)
        members_f = frequencies[positions]
        locked = _frequency_distance(members_f, members_f[:1], sizes[positions, None])[:, 0]
        if eligible[positions].all() and np.all(locked <= SEPARATION_BINS):
            keep[members] = True
    return keep, lattice


def _broad_retry(observed, cfg, rejected, evidence):
    """Examine a different fixed carrier in the original observation after a full narrow veto."""
    if min(observed.shape) < 2 * cfg.estimation.synthesis_size or cfg.strength == 0:
        return None
    prepared = general.prepare(observed, cfg)
    if prepared.global_assessment["abstain"]:
        return None
    gains = general.effective_gains(prepared.fit_weights, prepared.safety_weights, cfg.strength, cfg.combination)
    allowed, screening = screen_tiles(observed, prepared, gains)
    evidence.update(general_candidate_tiles=int(np.count_nonzero(gains)),
                    general_corroborated_tiles=screening["spatially_eligible_tiles"],
                    general_accepted_tiles=0, texture_screen=screening)
    if not allowed.any():
        return None
    tiles = prepared.estimation.tiles
    keep, lattice = _independent_components(observed, tiles, allowed, screening["components"], rejected)
    evidence.update(lattice_screen=lattice, general_accepted_tiles=int(keep.sum()))
    if keep.any():
        return _restore_general(observed, tiles, gains, keep, True)
    return None


def _review(observed, result, diag, cfg):
    regions = diag["accepted_regions"]
    support_cfg = cfg.narrow.support
    reviews = narrow_periodicity.screen_narrow(observed, diag, flank_width=support_cfg.flank_width,
                                                max_band_width=support_cfg.max_width)
    rejected = [r | {"confirmed_companions": review["confirmed_companions"]}
                for r, review in zip(regions, reviews, strict=True) if review["veto"]]
    evidence = diag["evidence"]
    evidence["narrow_periodicity"] = reviews
    if not rejected:
        return result
    evidence["previous_narrow_regions"] = regions
    evidence["previous_narrow_supported_pixels"] = evidence["narrow_supported_pixels"]
    evidence["narrow_supported_pixels"] = 0
    if len(rejected) < len(regions):
        # Where two intervals cross or compete along the same axis, the old
        # winning wave can belong to either interval. Preserve the intersection.
        support = diag["support_mask"].copy()
        for region in rejected:
            lines = support if region["normal_axis"] == 0 else support.T
            lines[region["start"]:region["stop"]] = False
        if support.any():
            result = np.where(support, result, observed)
            surviving = []
            for region, review in zip(regions, reviews):
                lines = support if region["normal_axis"] == 0 else support.T
                if not review["veto"] and lines[region["start"]:region["stop"]].any():
                    surviving.append(region)
            diag.update(reason="compact_support_after_texture_review", support_mask=support,
                        accepted_regions=surviving, residual=observed-result)
            evidence["narrow_supported_pixels"] = int(support.sum())
            return result
    else:
        broad = _broad_retry(observed, cfg.general, rejected, evidence)
        if broad is not None:
            result, support, active = broad
            diag.update(route="general", reason="distinct_corroborated_stationary_carrier", support_mask=support,
                        accepted_regions=active, residual=observed-result)
            return result
    diag.update(route="unchanged", reason="recurring_narrow_texture", accepted_regions=[],
                support_mask=np.zeros(observed.shape, bool), residual=np.zeros_like(observed))
    return observed.copy()


def denoise(image, config=None, *, return_diagnostics=False):
    """Restore a single grayscale observation and optionally return the actual decision and maps."""
    cfg = load_config() if config is None else config
    result, diag = previous(image, cfg, return_diagnostics=True)
    if diag["route"] == "narrow":
        observed = np.asarray(image, dtype=np.float64)
        result = _review(observed, result, diag, cfg)
    return (result, diag) if return_diagnostics else result
