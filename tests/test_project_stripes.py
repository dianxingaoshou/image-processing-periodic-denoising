"""Routing after review of a previously accepted narrow proposal."""
import numpy as np
import pytest

import project_stripes as project


@pytest.mark.parametrize("conflict", ["none", "rejected", "companion", "lattice", "drift", "chirp"])
def test_broad_retry_keeps_only_whole_independent_components(monkeypatch, conflict):
    first = dict(fy=.11, fx=.06, model="sinusoid")
    second = dict(fy=-.11, fx=-.06, model="sinusoid")
    # Both signs represent the same real sinusoid. A conflict in just one
    # member must protect the entire component, including the other member.
    first["fy"] = {"rejected": -.075, "companion": .037, "drift": .14}.get(conflict, first["fy"])
    if conflict == "chirp":
        first["model"] = "chirp"
    peaks = np.array([[.11, .06]]) if conflict == "lattice" else np.empty((0, 2))
    monkeypatch.setattr(project, "analyze_lattice", lambda obs, regions: (peaks, np.zeros(3, bool), {}))
    fits = [first, None, second, dict(fy=.18, fx=.024, model="sinusoid")]
    tiles = [dict(selected=fit, h=64, w=80) for fit in fits]
    keep, _ = project._independent_components(np.zeros((128,160)), tiles, np.array([1,0,1,1], bool),
                [dict(reason="accepted", tile_indices=[0,2]), dict(reason="accepted", tile_indices=[3])],
                [dict(normal_axis=1, tangent_frequency=.074, confirmed_companions=[.037])])
    np.testing.assert_array_equal(keep, [conflict == "none", False, conflict == "none", True])


def test_horizontal_texture_checks_x_frequency_in_a_rectangular_tile(monkeypatch):
    monkeypatch.setattr(project, "analyze_lattice", lambda *a: (np.empty((0, 2)), None, {}))
    fits = [dict(fy=.09, fx=x, model="sinusoid") for x in (-.075, -.0375, .15)]
    fits.append(dict(fy=.075, fx=.13, model="sinusoid"))
    keep, _ = project._independent_components(np.zeros((128, 192)), [dict(h=64, w=96, selected=f) for f in fits],
                np.ones(4, bool), [dict(reason="accepted", tile_indices=[i]) for i in range(4)],
                [dict(normal_axis=0, tangent_frequency=.075, confirmed_companions=[.0375, .15])])
    np.testing.assert_array_equal(keep, [False, False, False, True])


def test_mixed_narrow_review_preserves_crossings_and_never_switches_to_broad(monkeypatch):
    image = np.full((40, 52), .5)
    rows = dict(normal_axis=0, start=2, stop=6, tangent_frequency=.13)
    columns = dict(normal_axis=1, start=11, stop=15, tangent_frequency=.09)
    original_mask = np.zeros(image.shape, bool)
    original_mask[2:6] = True
    original_mask[:, 11:15] = True
    original_output = image - .2*original_mask

    def prior(observed, cfg, *, return_diagnostics):
        return original_output.copy(), dict(route="narrow", accepted_regions=[rows, columns],
                evidence=dict(narrow_supported_pixels=int(original_mask.sum())), support_mask=original_mask.copy())

    monkeypatch.setattr(project, "previous", prior)
    monkeypatch.setattr(project.narrow_periodicity, "screen_narrow", lambda *a, **kw:
                        [dict(veto=False), dict(veto=True, confirmed_companions=[.045])])
    monkeypatch.setattr(project, "_broad_retry", lambda *a: pytest.fail("Surviving narrow interval"))
    out, diag = project.denoise(image, return_diagnostics=True)
    expected = original_mask.copy()
    expected[:, 11:15] = False
    assert diag["accepted_regions"] == [rows] and diag["route"] == "narrow"
    assert diag["evidence"]["narrow_supported_pixels"] == expected.sum()
    np.testing.assert_array_equal(diag["support_mask"], expected)
    np.testing.assert_array_equal(out, np.where(expected, original_output, image))
    np.testing.assert_array_equal(diag["residual"], image-out)
    np.testing.assert_array_equal(out, project.denoise(image))
