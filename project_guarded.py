"""Review an abstained lattice-texture image using independent carrier and clipping evidence."""
import numpy as np

import guarded_carrier
from project_pipeline import load_config
from project_weak import denoise as previous, routing_parameters as previous_parameters


def routing_parameters():
    return dict(version="clip_supported_carrier_v1", previous=previous_parameters(),
                carrier=guarded_carrier.routing_parameters(), gain="min(general.strength, 1)",
                clipped_reconstruction="opposite-phase empirical intensity tails with one midpoint observation")


def _eligible(diag):
    if diag["route"] != "unchanged" or diag["global_assessment"]["abstain"]:
        return False
    evidence = diag["evidence"]
    if diag["reason"] == "protected_lattice_texture":
        # An ambiguous choice among several previously verified waves stays protected.
        return (evidence.get("lattice_screen", {}).get("detected", False)
                and evidence.get("alternative_carrier", {}).get("verified_models") == 0)
    return evidence.get("weak_carrier", {}).get("reason") == "no_verified_full_field_carrier"


def _restore(observed, wave, gain):
    """Estimate censored endpoints from visible intensities at the opposite carrier phase."""
    raw = observed-wave
    restored = np.clip(raw, 0., 1.)
    visible = (observed > 0) & (observed < 1) & (raw >= 0) & (raw <= 1)
    for endpoint, outward in ((0., -wave), (1., wave)):
        censored = (observed == endpoint) & (outward > 0)
        if not censored.any():
            continue
        # At the opposite phase this endpoint is shifted inwards. Its intensity
        # distribution can be sampled without clipping off the same tail again.
        distances = np.sort(abs(raw[visible & (outward < 0)] - endpoint))
        radius = np.minimum(outward[censored], 1.)
        count = np.searchsorted(distances, radius, side="right")
        total = np.r_[0., np.cumsum(distances)]
        # A single uniform-interval midpoint supplies a bounded fallback for
        # sparse or empty tails. Every resulting value lies in the feasible range.
        mean_distance = (total[count] + .5*radius)/(count+1)
        restored[censored] = mean_distance if endpoint == 0 else 1-mean_distance
    return restored if gain == 1. else observed + gain*(restored-observed)


def denoise(image, config=None, *, return_diagnostics=False):
    """Try a new wave only when the previous integrated workflow left the input untouched."""
    cfg = load_config() if config is None else config
    result, diag = previous(image, cfg, return_diagnostics=True)
    if cfg.general.strength > 0 and _eligible(diag):
        observed = result
        est = cfg.general.estimation
        wave, record = guarded_carrier.find_carrier(observed, (est.min_frequency, est.max_frequency),
                                                   enable_chirp=est.enable_chirp)
        evidence = diag["evidence"]
        evidence["guarded_carrier"] = record
        if wave is not None:
            gain = min(cfg.general.strength, 1.)
            result = _restore(observed, wave, gain)
            if np.any(result != observed):
                support = wave != 0
                h, w = observed.shape
                fy, fx = record["center_frequency"]
                region = dict(y=0, x=0, height=h, width=w, scale=record["patch_size"],
                              frequency_y=fy, frequency_x=fx, hessian=record["hessian"],
                              model=record["model"], gain=gain, support_pixels=int(support.sum()))
                evidence["prior_reason"] = diag["reason"]
                diag.update(route="general", reason="clip_supported_independent_carrier",
                            accepted_regions=[region], support_mask=support, residual=observed-result)
                if return_diagnostics:
                    diag["carrier_wave"] = wave
    return (result, diag) if return_diagnostics else result
