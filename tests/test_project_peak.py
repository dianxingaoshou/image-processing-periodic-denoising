"""Check version routing and the consistency of the public return values."""
import numpy as np
import pytest

import project_peak


@pytest.mark.parametrize(("reason", "model", "protected", "tapered"), [
    ("compact_directional_support", None, 0, 0),
    ("global_periodic_ambiguity", None, 0, 0),
    ("clip_supported_independent_carrier", "stationary", 0, 0),
    ("clip_supported_independent_carrier", "quadratic", 0, 0),
])
def test_existing_routes_skip_peak_review(monkeypatch, reason, model, protected, tapered):
    image = np.full((32, 40), .4)
    evidence = {}
    if model:
        protection = {} if model == "stationary" else dict(protected_pixels=protected, tapered_pixels=tapered)
        evidence["guarded_carrier"] = dict(model=model, protection=protection)
    diag = dict(reason=reason, evidence=evidence)
    monkeypatch.setattr(project_peak, "previous", lambda *a, **k: (image.copy(), diag))
    def unexpected(*args):
        pytest.fail("Reviewed a path without a tapered, independently verified wave")
    monkeypatch.setattr(project_peak.peak_guard, "refine", unexpected)
    out, actual = project_peak.denoise(image, return_diagnostics=True)
    np.testing.assert_array_equal(out, image)
    assert actual is diag and "texture_peak_review" not in actual["evidence"]


@pytest.mark.parametrize("adjust", [True, False])
def test_peak_review_records_only_effective_reconstruction(monkeypatch, adjust):
    image = np.full((37, 35), .41, dtype=np.float32).T
    support = np.zeros(image.shape, dtype=bool)
    support[:9, :] = True
    observed = image.astype(float)
    old = observed-.05*support
    reason = "clip_supported_independent_carrier"
    def previous(*args, **kwargs):
        assert kwargs["return_diagnostics"]
        return old.copy(), dict(reason=reason, accepted_regions=[{"support_pixels": int(support.sum())}],
                                evidence={"guarded_carrier": dict(model="quadratic",
                                          protection=dict(protected_pixels=10, tapered_pixels=3))},
                                residual=observed-old, support_mask=support)
    def review(source, current, diag):
        np.testing.assert_array_equal(source, observed)
        return current+.001*support if adjust else current.copy(), support, {"adjusted": adjust}
    monkeypatch.setattr(project_peak, "previous", previous)
    monkeypatch.setattr(project_peak.peak_guard, "refine", review)
    out, diag = project_peak.denoise(image, return_diagnostics=True)
    np.testing.assert_array_equal(project_peak.denoise(image), out)
    np.testing.assert_array_equal(out[~support], observed[~support])
    np.testing.assert_array_equal(observed-out, diag["residual"])
    assert diag["reason"] == ("texture_reviewed_chirp" if adjust else reason)
    assert diag["evidence"]["texture_peak_review"]["adjusted"] == adjust
