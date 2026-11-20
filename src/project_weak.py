"""Recheck an unexplained textile observation for a spatially verified independent carrier."""
import numpy as np

from project_pipeline import load_config
from project_stripes import denoise as previous, routing_parameters as previous_parameters
import weak_carrier


RETRY_REASONS = ("insufficient_spatial_support", "recurring_narrow_texture")


def routing_parameters():
    return dict(version="verified_weak_carrier_v1", previous=previous_parameters(),
                retry_reasons=list(RETRY_REASONS), carrier=weak_carrier.routing_parameters(),
                saturated_pixel_estimate="midpoint of feasible clean intensity interval",
                carrier_gain="min(general.strength, 1)")


def _subtract(observed, wave, gain):
    """At clipped input endpoints, the fitted wave constrains an interval of clean intensities."""
    unit_result = np.clip(observed-wave, 0., 1.)
    saturated = (observed == 0.) | (observed == 1.)
    unit_result[saturated] = (observed[saturated] + unit_result[saturated]) * .5
    if gain == 1.:
        return unit_result
    return observed + gain*(unit_result-observed)


def denoise(image, config=None, *, return_diagnostics=False):
    """Return the established result, or subtract one verified carrier on its measured support."""
    cfg = load_config() if config is None else config
    result, diag = previous(image, cfg, return_diagnostics=True)
    if (diag["route"] == "unchanged" and diag["reason"] in RETRY_REASONS
            and cfg.general.strength > 0 and not diag["global_assessment"]["abstain"]
            and min(result.shape) >= 2*cfg.general.estimation.synthesis_size):
        # Every eligible prior decision returns an exact copy of the input.
        observed = result
        estimation = cfg.general.estimation
        bounds = (estimation.min_frequency, estimation.max_frequency)
        wave, carrier = weak_carrier.find_carrier(observed, bounds, enable_chirp=estimation.enable_chirp)
        evidence = diag["evidence"]
        evidence["weak_carrier"] = carrier
        if wave is not None:
            gain = min(cfg.general.strength, 1.)
            result = _subtract(observed, wave, gain)
            support = wave != 0
            if np.any(result != observed):
                fy, fx = carrier["center_frequency"]
                h, w = observed.shape
                region = dict(y=0, x=0, height=h, width=w, scale=carrier["patch_size"],
                              frequency_y=fy, frequency_x=fx, hessian=carrier["hessian"], gain=gain,
                              model="verified_carrier", support_pixels=int(support.sum()),
                              saturated_support_pixels=int(np.count_nonzero(support & ((observed==0) | (observed==1)))))
                evidence["prior_reason"] = diag["reason"]
                diag.update(route="general", reason="independent_carrier_after_abstention",
                            accepted_regions=[region], residual=observed-result, support_mask=support)
    return (result, diag) if return_diagnostics else result
