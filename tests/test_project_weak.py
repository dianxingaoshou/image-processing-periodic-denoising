"""Single-image routing, clipping, strength and diagnostic map contracts for the new retry."""
from dataclasses import replace

import numpy as np
import pytest

import project_weak as project


def prior_decision(image, route, reason, abstain=False):
    def run(observed, cfg, *, return_diagnostics):
        assert return_diagnostics
        return image.copy(), dict(route=route, reason=reason, accepted_regions=[], evidence={},
                                  global_assessment={"abstain": abstain}, residual=np.zeros_like(image),
                                  support_mask=np.zeros(image.shape, bool))
    return run


@pytest.mark.parametrize("route,reason,abstain,strength,height", [
    ("narrow", "compact_directional_support", False, 2., 160),
    ("general", "spatially_corroborated_tiles", False, 2., 160),
    ("unchanged", "recurring_narrow_texture", True, 2., 160),
    ("unchanged", "texture_component_veto", False, 2., 160),
    ("unchanged", "insufficient_spatial_support", False, 0., 160),
    ("unchanged", "insufficient_spatial_support", False, 2., 120),
])
def test_no_retry_after_accepted_or_protected_decisions(monkeypatch, route, reason, abstain, strength, height):
    image = np.full((height, 180), .5)
    cfg = project.load_config()
    cfg = replace(cfg, general=replace(cfg.general, strength=strength))
    monkeypatch.setattr(project, "previous", prior_decision(image, route, reason, abstain))
    monkeypatch.setattr(project.weak_carrier, "find_carrier", lambda *a, **kw: pytest.fail("Unexpected retry"))
    result, diag = project.denoise(image, cfg, return_diagnostics=True)
    np.testing.assert_array_equal(result, image)
    assert diag["route"] == route and "weak_carrier" not in diag["evidence"]


@pytest.mark.parametrize("reason", project.RETRY_REASONS)
def test_negative_verification_keeps_old_maps_and_reason(monkeypatch, reason):
    image = np.full((144, 166), .55)
    monkeypatch.setattr(project, "previous", prior_decision(image, "unchanged", reason))
    monkeypatch.setattr(project.weak_carrier, "find_carrier", lambda *a, **kw: (None, dict(reason="no_lattice")))
    result, diag = project.denoise(image, return_diagnostics=True)
    np.testing.assert_array_equal(result, image)
    assert not diag["support_mask"].any() and not diag["residual"].any()
    assert diag["reason"] == reason and diag["evidence"]["weak_carrier"]["reason"] == "no_lattice"


def test_verified_wave_respects_signed_support_and_fractional_strength(monkeypatch):
    image = np.full((160, 128), .5)
    image[20, 30:34] = [0, 1, .98, .01]
    wave = np.zeros_like(image)
    wave[20, 30:34] = [-.08, .12, -.08, -.05]
    before = image.copy()
    cfg = project.load_config()
    cfg = replace(cfg, general=replace(cfg.general, strength=.5,
                                       estimation=replace(cfg.general.estimation, enable_chirp=False)))
    monkeypatch.setattr(project, "previous", prior_decision(image, "unchanged", "insufficient_spatial_support"))
    def verify(observed, bounds, *, enable_chirp):
        assert not enable_chirp and tuple(bounds) == (.025, .45)
        return wave, dict(center_frequency=[.1, .12], hessian=[0., 0., 0.], patch_size=32)
    monkeypatch.setattr(project.weak_carrier, "find_carrier", verify)
    output, diag = project.denoise(image, cfg, return_diagnostics=True)
    np.testing.assert_allclose(output[20, 30:34], [.02, .97, .99, .035], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(output[wave == 0], image[wave == 0])
    np.testing.assert_array_equal(diag["support_mask"], wave != 0)
    np.testing.assert_array_equal(diag["residual"], image-output)
    assert diag["route"] == "general" and diag["accepted_regions"][0]["gain"] == .5
    np.testing.assert_array_equal(output, project.denoise(image, cfg))
    np.testing.assert_array_equal(image, before)
