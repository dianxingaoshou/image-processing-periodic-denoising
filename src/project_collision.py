"""Review protected chirp regions before the previous distant-peak correction."""
import numpy as np

import collision_review
import peak_guard
from project_guarded import denoise as guarded
from project_peak import routing_parameters as previous_parameters
from project_pipeline import load_config


def routing_parameters():
    return dict(version="local_chirp_collision_review_v1", previous=previous_parameters(),
                collision_review=collision_review.routing_parameters())


def denoise(image, config=None, *, return_diagnostics=False):
    # Both alternatives use the very same guarded fit. Running project_peak first
    # would pay for an iterative reconstruction that the accepted review replaces.
    cfg = load_config() if config is None else config
    result, diag = guarded(image, cfg, return_diagnostics=True)
    if diag["reason"] == "clip_supported_independent_carrier":
        fit = diag["evidence"]["guarded_carrier"]
        protection = fit["protection"]
        if fit["model"] == "quadratic" and (protection["tapered_pixels"] or protection["protected_pixels"]):
            observed = np.asarray(image, dtype=float)
            estimation = cfg.general.estimation
            reviewed, wave, evidence = collision_review.review(
                observed, diag, bounds=(estimation.min_frequency, estimation.max_frequency))
            diag["evidence"]["collision_review"] = evidence
            if reviewed is None:
                # Exact previous reconstruction, without repeating candidate discovery.
                candidate, support, peaks = peak_guard.refine(observed, result, diag)
                if not return_diagnostics:
                    return candidate
                diag["evidence"]["texture_peak_review"] = peaks
                if candidate is not result and not np.array_equal(candidate, result):
                    result = candidate
                    diag.update(reason="texture_reviewed_chirp", support_mask=support, residual=observed-result)
                    diag["accepted_regions"][0]["support_pixels"] = int(support.sum())
            else:
                if not return_diagnostics:
                    return reviewed
                result = reviewed
                support = diag["support_mask"] | (wave != 0)
                diag["prior_carrier_wave"] = diag["carrier_wave"]
                diag["carrier_wave"] = wave
                diag.update(reason="locally_verified_chirp_release", support_mask=support, residual=observed-result)
                model = evidence["applied_model"]
                fy, fx = model["center_frequency"]
                diag["accepted_regions"][0].update(support_pixels=int(support.sum()),
                    frequency_y=fy, frequency_x=fx, hessian=model["hessian"], amplitude=model["amplitude"],
                    phase_radians=model["phase_radians"])
    return (result, diag) if return_diagnostics else result
