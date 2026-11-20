from dataclasses import replace

import numpy as np
import pytest

import project_guarded as project


@pytest.mark.parametrize("route,reason,abstain,evidence,eligible", [
    ("narrow", "compact_directional_support", False, {}, False),
    ("general", "spatially_corroborated_tiles", False, {}, False),
    ("unchanged", "recurring_narrow_texture", True, {"weak_carrier": {"reason": "no_verified_full_field_carrier"}}, False),
    ("unchanged", "protected_lattice_texture", False, {"lattice_screen": {"detected": True},
                                                      "alternative_carrier": {"verified_models": 2}}, False),
    ("unchanged", "protected_lattice_texture", False, {"lattice_screen": {"detected": True},
                                                      "alternative_carrier": {"verified_models": 0}}, True),
    ("unchanged", "insufficient_spatial_support", False, {"weak_carrier": {"reason": "no_verified_full_field_carrier"}}, True),
])
def test_existing_routes_and_ambiguity_vetoes(route, reason, abstain, evidence, eligible):
    diag = dict(route=route, reason=reason, global_assessment=dict(abstain=abstain), evidence=evidence)
    assert project._eligible(diag) == eligible


def test_censored_tail_uses_visible_values_from_opposite_phase_and_half_gain():
    # At the upper endpoint the feasible clean interval is [.8, 1].
    # Opposite-phase interior pixels give two clean estimates .82 and .84.
    obs = np.array([[1., .81, .83, .04, .95, .5]])
    wave = np.array([[.2, -.01, -.01, -.15, .30, 0.]])
    restored = project._restore(obs, wave, 1.)
    assert restored[0, 0] == pytest.approx((.82+.84+.9)/3)
    np.testing.assert_allclose(restored[0,1:], [.82,.84,.19,.65,.5])
    np.testing.assert_allclose(project._restore(1-obs, -wave, 1.), 1-restored)
    np.testing.assert_allclose(project._restore(obs, wave, .5), (restored+obs)/2)


def test_empty_opposite_tail_uses_the_feasible_midpoint_and_preserves_unaffected_pixels():
    obs = np.array([[0., 1., 0., .75]])
    wave = np.array([[-.2, .12, .1, 0.]])
    np.testing.assert_allclose(project._restore(obs, wave, 1.), [[.1,.94,0.,.75]])


def test_zero_strength_never_starts_a_new_search(monkeypatch):
    cfg = project.load_config()
    cfg = replace(cfg, general=replace(cfg.general, strength=0.))
    obs = np.full((128, 130), .45)
    monkeypatch.setattr(project, "previous", lambda *a, **k: (obs.copy(), {}))
    monkeypatch.setattr(project.guarded_carrier, "find_carrier", lambda *a, **k: pytest.fail("Unexpected search"))
    np.testing.assert_array_equal(project.denoise(obs, cfg), obs)


def test_output_maps_include_only_the_authorized_part_of_a_sweep(monkeypatch):
    obs = np.full((128, 140), .5)
    wave = np.zeros_like(obs)
    wave[:, 43:61] = .04
    def old(*args, **kw):
        return obs.copy(), dict(route="unchanged", reason="insufficient_spatial_support",
                                global_assessment=dict(abstain=False),
                                evidence=dict(weak_carrier=dict(reason="no_verified_full_field_carrier")))
    def verify(*args, **kw):
        return wave, dict(model="quadratic", center_frequency=[.1,.12], hessian=[0.,.0001,0.], patch_size=32)
    monkeypatch.setattr(project, "previous", old)
    monkeypatch.setattr(project.guarded_carrier, "find_carrier", verify)
    out, d = project.denoise(obs, return_diagnostics=True)
    assert d["reason"] == "clip_supported_independent_carrier" and d["route"] == "general"
    np.testing.assert_array_equal(d["carrier_wave"], wave)
    np.testing.assert_array_equal(d["support_mask"], wave != 0)
    np.testing.assert_array_equal(out, obs-wave)
    np.testing.assert_array_equal(d["residual"], obs-out)
    np.testing.assert_array_equal(project.denoise(obs), out)
