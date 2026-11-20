"""Preserve established texture frequencies while seeking an independent broad carrier."""
import numpy as np

from local_estimation_v3 import prepare as estimate_tiles
from local_spectral import _checked, _starts
from project_pipeline import load_config
from project_refined import _restore_general, denoise as previous, routing_parameters as previous_parameters
import weave_guard
from weave_carrier import (FREQUENCY_ERROR_BINS, MAX_COEFFICIENT_ERROR, MIN_AMPLITUDE,
                           MIN_AGREEING_FRACTION, MIN_CLEAR_PATCHES,
                           PEAK_CLEARANCE_BINS, independent_carrier)


def routing_parameters():
    return dict(previous=previous_parameters(), version="lattice_screen_v1",
                lattice_tolerance_bins=weave_guard.LATTICE_TOLERANCE_BINS, min_lattice_peaks=weave_guard.MIN_PEAKS,
                max_order=weave_guard.MAX_ORDER, min_relative_amplitude=weave_guard.MIN_RELATIVE_AMPLITUDE,
                min_lattice_power_fraction=weave_guard.MIN_LATTICE_POWER_FRACTION,
                dominance_ratio=weave_guard.DOMINANCE_RATIO, min_locked_fraction=weave_guard.MIN_LOCKED_FRACTION,
                peak_clearance_bins=PEAK_CLEARANCE_BINS, frequency_error_bins=FREQUENCY_ERROR_BINS,
                max_coefficient_error=MAX_COEFFICIENT_ERROR, minimum_carrier_amplitude=MIN_AMPLITUDE,
                minimum_clear_patches=MIN_CLEAR_PATCHES, minimum_agreeing_fraction=MIN_AGREEING_FRACTION)


def _reconstruct(observed, cfg, regions, matches, wave, carrier, diagnostics):
    """Keep original gains and full window normalization, even for removed candidates."""
    h, w = observed.shape
    size = cfg.general.estimation.synthesis_size
    sh, sw = min(h, size), min(w, size)
    stride = cfg.general.estimation.stride
    by_location = {(r["y"], r["x"]): (r, matched) for r, matched in zip(regions, matches)}
    original = {}
    if not matches.all():
        fitted = estimate_tiles(observed, cfg.general.estimation)
        original = {(t["y"], t["x"]): t for t in fitted.tiles}
    tiles, gains = [], []
    for y in _starts(h, sh, stride):
        for x in _starts(w, sw, stride):
            tile = dict(y=y, x=x, h=sh, w=sw)
            gain = 0.
            if (y, x) in by_location:
                region, matched = by_location[y, x]
                if not matched:
                    tile = original[y, x]
                    gain = region["gain"]
                elif wave is not None:
                    fy, fx = carrier["center_frequency"]
                    yy, yx, xx = carrier["hessian"]
                    dy, dx = y+(sh-1)/2-(h-1)/2, x+(sw-1)/2-(w-1)/2
                    tile.update(residual=wave[y:y+sh, x:x+sw],
                                selected=dict(fy=fy+yy*dy+yx*dx, fx=fx+yx*dy+xx*dx,
                                              scale=carrier["patch_size"]))
                    gain = region["gain"]
            tiles.append(tile)
            gains.append(gain)
    gains = np.asarray(gains)
    return _restore_general(observed, tiles, gains, gains > 0, diagnostics)


def denoise(image, config=None, *, return_diagnostics=False):
    """Run the previous router, then examine only its authorized broad corrections."""
    cfg = load_config() if config is None else config
    observed = _checked(image)
    result, diag = previous(observed, cfg, return_diagnostics=True)
    if diag["route"] == "general":
        regions = diag["accepted_regions"]
        peaks, matches, lattice = weave_guard.analyze_lattice(observed, regions)
        diag["evidence"]["lattice_screen"] = lattice
        if matches.any():
            lattice["protected_regions"] = [r for r, matched in zip(regions, matches) if matched]
            bounds = cfg.general.estimation.min_frequency, cfg.general.estimation.max_frequency
            wave, carrier = independent_carrier(observed, peaks, np.array(lattice["basis_cycles_per_pixel"]),
                                               bounds, enable_chirp=cfg.general.estimation.enable_chirp)
            diag["evidence"]["alternative_carrier"] = carrier
            if matches.all() and wave is None:
                result = observed.copy()
                diag.update(route="unchanged", reason="protected_lattice_texture", accepted_regions=[])
                if return_diagnostics:
                    diag["support_mask"] = np.zeros(observed.shape, bool)
            else:
                result, support, active = _reconstruct(observed, cfg, regions, matches, wave, carrier,
                                                       return_diagnostics)
                diag.update(reason=("verified_non_lattice_carrier" if wave is not None else "protected_lattice_tiles"),
                            accepted_regions=active)
                if return_diagnostics:
                    diag["support_mask"] = support
            diag["evidence"]["previous_accepted_tiles"] = len(regions)
            if return_diagnostics:
                diag["evidence"]["general_accepted_tiles"] = len(diag["accepted_regions"])
                diag["residual"] = observed-result
    return (result, diag) if return_diagnostics else result
