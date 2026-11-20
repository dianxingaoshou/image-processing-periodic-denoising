"""Refine the reconstruction of an already verified, spatially tapered chirp."""
import numpy as np

import peak_guard
from project_guarded import denoise as previous, routing_parameters as previous_parameters


def routing_parameters():
    return dict(version="tapered_chirp_texture_review_v1", previous=previous_parameters(),
                texture_review=peak_guard.routing_parameters())


def denoise(image, config=None, *, return_diagnostics=False):
    result, diag = previous(image, config, return_diagnostics=True)
    if diag["reason"] == "clip_supported_independent_carrier":
        carrier = diag["evidence"]["guarded_carrier"]
        protection = carrier["protection"]
        if carrier["model"] == "quadratic" and (protection["tapered_pixels"] or protection["protected_pixels"]):
            observed = np.asarray(image, dtype=float)
            candidate, support, evidence = peak_guard.refine(observed, result, diag)
            diag["evidence"]["texture_peak_review"] = evidence
            if candidate is not result and not np.array_equal(candidate, result):
                result = candidate
                diag["reason"] = "texture_reviewed_chirp"
                diag["support_mask"] = support
                diag["residual"] = observed-result
                diag["accepted_regions"][0]["support_pixels"] = int(support.sum())
    return (result, diag) if return_diagnostics else result
