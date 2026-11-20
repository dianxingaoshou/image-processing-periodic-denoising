"""Integrated single-image denoising and reconstruction of protected chirps."""
import numpy as np

import collision_review
import peak_guard
import phase_refit
from project_collision import routing_parameters as previous_parameters
from project_guarded import denoise as guarded
from project_pipeline import load_config


def routing_parameters():
    return dict(version="protected_chirp_phase_refit_v1", previous=previous_parameters(),
                phase_refit=phase_refit.routing_parameters())


def _region_model(model):
    fy, fx = model["center_frequency"]
    return dict(frequency_y=fy, frequency_x=fx, hessian=model["hessian"],
                amplitude=model["amplitude"], phase_radians=model["phase_radians"])


def _review_abstention(base, diag, cfg, diagnostics):
    """Only a rejected recurring strip with a completed broad search can be reconsidered."""
    est = cfg.general.estimation
    evidence = diag["evidence"]
    if (cfg.general.strength <= 0 or not est.enable_chirp
            or evidence.get("guarded_carrier", {}).get("reason") != "no_independent_verified_carrier"):
        return (base, diag) if diagnostics else base
    gain = min(cfg.general.strength, 1.)
    out, wave, review = phase_refit.review_abstention(
        base, diag, bounds=(est.min_frequency, est.max_frequency), gain=gain)
    if not diagnostics:
        return base if out is None else out
    evidence["phase_refit"] = review
    if out is None:
        return base, diag
    support = wave != 0
    h, w = base.shape
    evidence["prior_reason"] = diag["reason"]
    region = dict(y=0, x=0, height=h, width=w, scale=min(h,w)//4, model="quadratic", gain=gain,
                  support_pixels=int(support.sum()), **_region_model(review["applied_model"]))
    diag.update(route="general", reason="phase_refit_after_texture_abstention", carrier_wave=wave,
                support_mask=support, residual=base-out, accepted_regions=[region])
    return out, diag


def denoise(image, config=None, *, return_diagnostics=False):
    cfg = load_config() if config is None else config
    base, diag = guarded(image, cfg, return_diagnostics=True)
    if diag["reason"] == "recurring_narrow_texture" and diag["route"] == "unchanged":
        return _review_abstention(base, diag, cfg, return_diagnostics)
    if diag["reason"] != "clip_supported_independent_carrier":
        return (base, diag) if return_diagnostics else base
    carrier = diag["evidence"]["guarded_carrier"]
    protection = carrier["protection"]
    if carrier["model"] != "quadratic" or not (protection["tapered_pixels"] or protection["protected_pixels"]):
        return (base, diag) if return_diagnostics else base

    obs = np.asarray(image, dtype=float)
    estimation = cfg.general.estimation
    bounds = (estimation.min_frequency, estimation.max_frequency)
    restored, wave, details = collision_review.review(obs, diag, bounds=bounds)
    diag["evidence"]["collision_review"] = details
    reason = "locally_verified_chirp_release"
    if restored is None and details["reason"] in (
            "inconclusive_exterior_phase_fit", "exterior_model_failed_local_confirmation"):
        restored, wave, details = phase_refit.refine(obs, diag, details, bounds=bounds)
        diag["evidence"]["phase_refit"] = details
        reason = "phase_refit_verified_chirp"

    if restored is None:
        restored, support, peaks = peak_guard.refine(obs, base, diag)
        if not return_diagnostics:
            return restored
        diag["evidence"]["texture_peak_review"] = peaks
        if restored is base or np.array_equal(restored, base):
            return base, diag
        reason = "texture_reviewed_chirp"
    else:
        if not return_diagnostics:
            return restored
        support = diag["support_mask"] | (wave != 0)
        diag["prior_carrier_wave"], diag["carrier_wave"] = diag["carrier_wave"], wave
        diag["accepted_regions"][0].update(_region_model(details["applied_model"]))

    diag.update(reason=reason, support_mask=support, residual=obs-restored)
    diag["accepted_regions"][0]["support_pixels"] = int(support.sum())
    return restored, diag
