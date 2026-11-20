"""Routing, pixel support and overlap normalization of the lattice revision."""
from types import SimpleNamespace

import numpy as np
import pytest

import project_weave as weave
from project_pipeline import load_config


def starts(length, size, stride):
    # The last block is anchored to the edge even when stride does not divide the span.
    return sorted(set(range(0, length-size+1, stride)) | {length-size})


@pytest.mark.parametrize("alternative", [False, True])
def test_partial_replacement_uses_old_gains_and_all_overlap_windows(monkeypatch, alternative):
    config = load_config()
    h, w = 150, 164
    size = config.general.estimation.synthesis_size
    stride = config.general.estimation.stride
    y, x = np.indices((h, w))
    observed = .06+.5*x/(w-1)
    replacement = .30*np.cos(2*np.pi*(.07*y+.14*x)) if alternative else None
    hessian = (.0001, -.0002, .0003)
    carrier = dict(center_frequency=(.07, .14), hessian=hessian, patch_size=32)
    locs = [(yy, xx) for yy in starts(h, size, stride) for xx in starts(w, size, stride)]
    tested = {(0, 0): (True, .35), (stride, stride): (False, .65), (h-size, w-size): (True, .85)}
    regions = [dict(y=a, x=b, gain=g) for (a, b), (_, g) in tested.items()]
    matched = np.array([flag for flag, _ in tested.values()])
    original = .28*np.sin(2*np.pi*(.03*y+.11*x))
    fitted = []
    expected_support = np.zeros((h, w), bool)
    numerator = np.zeros((h, w))
    denominator = np.zeros((h, w))
    window = np.outer(np.hanning(size+2)[1:-1], np.hanning(size+2)[1:-1])
    for yy, xx in locs:
        sl = np.s_[yy:yy+size, xx:xx+size]
        fitted.append(dict(y=yy, x=xx, h=size, w=size, residual=original[sl],
                           selected=dict(fy=.03, fx=.11, scale=128)))
        denominator[sl] += window
        if (yy, xx) in tested:
            match, gain = tested[(yy, xx)]
            if not match or alternative:
                candidate = replacement if match else original
                expected_support[sl] = True
                numerator[sl] += gain*window*candidate[sl]
    monkeypatch.setattr(weave, "estimate_tiles", lambda *args: SimpleNamespace(tiles=fitted))
    result, mask, active = weave._reconstruct(observed, config, regions, matched, replacement, carrier, True)
    np.testing.assert_allclose(result, np.clip(observed-numerator/denominator, 0, 1), rtol=0, atol=2e-15)
    np.testing.assert_array_equal(mask, expected_support)
    np.testing.assert_array_equal(result[~mask], observed[~mask])
    assert {(r["y"], r["x"]) for r in active} == {k for k, (flag, _) in tested.items() if alternative or not flag}
    for r in active:
        match, gain = tested[r["y"], r["x"]]
        assert r["gain"] == gain
        if match:
            dy = r["y"]+(size-h)/2
            dx = r["x"]+(size-w)/2
            np.testing.assert_allclose([r["frequency_y"], r["frequency_x"]],
                                       [.07+hessian[0]*dy+hessian[1]*dx, .14+hessian[1]*dy+hessian[2]*dx])
        else:
            assert (r["frequency_y"], r["frequency_x"], r["scale"]) == (.03, .11, 128)


def fake_previous(monkeypatch, route):
    def run(a, cfg, *, return_diagnostics):
        old = a.copy()
        old[:32] *= .9
        return old, dict(route=route, reason="original decision", residual=a-old, support_mask=a!=old,
                         accepted_regions=[dict(y=0, x=0, gain=.4)], evidence={})
    monkeypatch.setattr(weave, "previous", run)


@pytest.mark.parametrize("route", ["narrow", "unchanged", "general"])
def test_unmatched_or_non_general_route_preserves_previous_output(monkeypatch, route):
    a = np.full((80, 96), .4)
    fake_previous(monkeypatch, route)
    def examine(*args):
        assert route == "general"
        return np.empty((0, 2)), np.array([False]), dict(detected=False)
    monkeypatch.setattr(weave.weave_guard, "analyze_lattice", examine)
    def unexpected(*args, **kwargs):
        pytest.fail("Unmatched regions must not trigger an alternative fit")
    monkeypatch.setattr(weave, "independent_carrier", unexpected)
    old, _ = weave.previous(a, load_config(), return_diagnostics=True)
    result, diag = weave.denoise(a, return_diagnostics=True)
    assert diag["route"] == route
    np.testing.assert_array_equal(result, old)
    np.testing.assert_array_equal(weave.denoise(a), result)


def test_whole_lattice_veto_clears_old_support_and_residual(monkeypatch):
    a = np.full((128, 128), .4)
    fake_previous(monkeypatch, "general")
    basis = np.eye(2)*.07
    monkeypatch.setattr(weave.weave_guard, "analyze_lattice",
                        lambda *args: (basis, np.array([True]), dict(basis_cycles_per_pixel=basis)))
    monkeypatch.setattr(weave, "independent_carrier", lambda *args, **kwargs: (None, dict(reason="none")))
    def unexpected(*args):
        pytest.fail("Full veto has no surviving tiles to refit")
    monkeypatch.setattr(weave, "estimate_tiles", unexpected)
    result, diag = weave.denoise(a, return_diagnostics=True)
    assert diag["route"] == "unchanged" and diag["reason"] == "protected_lattice_texture"
    assert diag["accepted_regions"] == [] and diag["evidence"]["general_accepted_tiles"] == 0
    assert not diag["support_mask"].any() and not diag["residual"].any()
    np.testing.assert_array_equal(result, a)
    np.testing.assert_array_equal(weave.denoise(a), result)
