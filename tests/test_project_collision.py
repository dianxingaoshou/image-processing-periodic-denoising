"""An accepted local review and its fallback share exactly one original fit."""
import numpy as np
import pytest

import project_collision as project


@pytest.mark.parametrize("release", [True, False])
def test_release_or_previous_peak_refinement_runs_once(monkeypatch, release):
    image = np.full((35, 40), .4, dtype=np.float32)
    observed = image.astype(float)
    prior = np.full_like(observed, .02)
    prior[:, 10:20] = 0.
    full = np.full_like(prior, .02)
    calls = []
    def guarded(array, cfg, *, return_diagnostics):
        calls.append("fit")
        return observed-prior, dict(reason="clip_supported_independent_carrier",
                                    evidence={"guarded_carrier": {
                                        "model":"quadratic","protection":{"tapered_pixels":8,"protected_pixels":3}}},
                                    accepted_regions=[{}], carrier_wave=prior, support_mask=prior!=0)
    def review(array, diag, *, bounds):
        calls.append("local")
        assert len(bounds) == 2 and 0 < bounds[0] < bounds[1] <= .5
        model = dict(center_frequency=[.12, .08], hessian=[0,0,-.0001], amplitude=.02, phase_radians=.4)
        return (observed-full, full, {"applied_model":model}) if release else (None, None, {})
    def peaks(array, result, diag):
        calls.append("peaks")
        return result+.01*(prior!=0), prior!=0, {}
    monkeypatch.setattr(project, "guarded", guarded)
    monkeypatch.setattr(project.collision_review, "review", review)
    monkeypatch.setattr(project.peak_guard, "refine", peaks)
    restored, diag = project.denoise(image, return_diagnostics=True)
    assert calls == (["fit","local"] if release else ["fit","local","peaks"])
    np.testing.assert_array_equal(diag["residual"], observed-restored)
    np.testing.assert_array_equal(diag["support_mask"], (full if release else prior)!=0)
    assert diag["accepted_regions"][0]["support_pixels"] == diag["support_mask"].sum()
    if release:
        np.testing.assert_array_equal(diag["prior_carrier_wave"], prior)
        np.testing.assert_array_equal(diag["carrier_wave"], full)
        assert diag["accepted_regions"][0]["frequency_y"] == .12
        assert diag["accepted_regions"][0]["phase_radians"] == .4
    np.testing.assert_array_equal(project.denoise(image), restored)


def test_narrow_path_never_enters_collision_review(monkeypatch):
    image = np.full((33, 35), .3)
    diag = dict(route="narrow", reason="compact_directional_support")
    monkeypatch.setattr(project, "guarded", lambda *a, **k: (image, diag))
    monkeypatch.setattr(project.collision_review, "review", lambda *a: pytest.fail("Reviewed a narrow strip"))
    output, info = project.denoise(image, return_diagnostics=True)
    assert info is diag and output is image
