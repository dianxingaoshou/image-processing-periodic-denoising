"""Check evidence needed to change any pixel in the old collision protection."""
from copy import deepcopy

import numpy as np
import pytest

import collision_review as review
from peak_guard import _carrier_geometry
from project_guarded import _restore


@pytest.fixture
def scene():
    shape = 180, 210
    y, x = np.indices(shape)
    fit = dict(model="quadratic", amplitude=.085, center_frequency=[.13,-.11], hessian=[0.,0.,.0003],
               final_phase_radians=.38, protection=dict(protected_pixels=1))
    wave, _ = _carrier_geometry(shape, fit)
    clean = .5 + .48*np.tanh(1.9*(np.cos(2*np.pi*(11*y/shape[0]+4*x/shape[1]))
                                   + .65*np.cos(2*np.pi*(6*y/shape[0]+11*x/shape[1]))))
    core = ((x>=43) & (x<98)) | ((x>=142) & (x<204))
    centers = np.array([[11,4],[6,11],[-5,7],[17,15]]) / shape
    diag = dict(reason="clip_supported_independent_carrier", carrier_wave=wave*(~core),
                evidence=dict(guarded_carrier=fit, weak_carrier=dict(measured_lattice_frequencies=centers.tolist())),
                accepted_regions=[dict(gain=1.)])
    return clean, wave, diag


@pytest.mark.parametrize("gain", [1., .5])
def test_accepted_bands_restore_full_verified_wave(scene, gain):
    clean, wave, diag = scene
    diag["accepted_regions"][0]["gain"] = gain
    observed = np.clip(clean+wave, 0., 1.)
    restored, applied, record = review.review(observed, diag)
    assert record["reason"] == "all_inferred_bands_verified"
    assert [(r["start"], r["stop"]) for r in record["bands"]] == [(43,98),(142,204)]
    assert np.linalg.norm(applied-wave) < .2*np.linalg.norm(wave)
    model = record["applied_model"]
    replay, _ = _carrier_geometry(clean.shape, dict(model, final_phase_radians=model["phase_radians"]))
    np.testing.assert_allclose(applied, replay, atol=2e-14)
    np.testing.assert_allclose(restored, _restore(observed, applied, gain), atol=2e-14)


def test_transposed_evidence_accepts_row_spanning_guard_bands(scene):
    clean, wave, diag = scene
    t = deepcopy(diag)
    t["carrier_wave"] = t["carrier_wave"].T
    fit = t["evidence"]["guarded_carrier"]
    fit["hessian"] = fit["hessian"][::-1]
    fit["center_frequency"] = fit["center_frequency"][::-1]
    t["evidence"]["weak_carrier"]["measured_lattice_frequencies"] = [
        pair[::-1] for pair in t["evidence"]["weak_carrier"]["measured_lattice_frequencies"]]
    obs = np.clip(clean+wave, 0., 1.)
    out, _, rec = review.review(obs.T,t)
    assert rec["sweep_axis"] == 0
    baseline, _, _ = review.review(obs, diag)
    np.testing.assert_allclose(out.T, baseline, atol=1e-12)


def test_entire_exterior_fit_ignores_guarded_strip_brightness(scene):
    clean, wave, diag = scene
    guarded = np.all(diag["carrier_wave"] == 0., axis=0)
    baseline = clean + wave
    contaminated = baseline.copy()
    contaminated[:, guarded] = np.random.default_rng(7).normal(size=contaminated[:, guarded].shape)
    fit = diag["evidence"]["guarded_carrier"]
    new_wave, _, model = review._exterior_fit(contaminated, guarded, 1, fit)
    old_wave, _, original = review._exterior_fit(baseline, guarded, 1, fit)
    np.testing.assert_array_equal(old_wave, new_wave)
    assert original == model
    # A tiny exterior window cannot identify curvature over the whole frame.
    narrow = np.ones(clean.shape[1], bool)
    narrow[60:78] = False
    assert review._exterior_fit(baseline,narrow,1,fit) is None


@pytest.mark.parametrize("state", ["absent", "inverted", "independent_texture", "no_endpoints"])
def test_one_unconfirmed_band_preserves_all_old_protection(scene, state):
    clean, wave, diag = scene
    y, x = np.indices(clean.shape)
    observed = clean+wave
    right = (x>=142) & (x<204)
    if state in ("absent","inverted"):
        observed[right] -= wave[right]*(1 if state=="absent" else 2)
        expected = "local_brightness_disagrees"
    elif state == "independent_texture":
        # A separate stationary texture close to the fitted sweep, only in the second band.
        phase = (.13+1/180)*(y-89.5)+(-.11+.0003*(173-104.5))*(x-104.5)
        observed[right] += .65*.085*np.cos(2*np.pi*phase[right]+.4)
        expected = "persistent_local_periodic_texture"
    else:
        observed = .5 + .3*(clean-.5) + wave
        expected = "insufficient_local_held_endpoints"
    output, applied, detail = review.review(np.clip(observed, 0, 1), diag)
    assert output is applied is None and detail["reason"] == expected


def test_measured_lattice_collision_keeps_veto(scene):
    clean, wave, diag = scene
    diag["evidence"]["weak_carrier"]["measured_lattice_frequencies"].append([.13,-.12])
    out, _, detail = review.review(np.clip(clean+wave,0,1), diag)
    assert out is None and detail["reason"] == "collision_with_measured_peak"


def test_refitted_frequency_segment_respects_bounds_and_measured_peaks():
    shape = np.array([150,181])
    # Both edges are in range, but the sweep's midpoint approaches DC.
    model = dict(center_frequency=[.024,0.], hessian=[0.,0.,.001])
    far = np.array([[55.,40.],[-55.,-40.]])
    assert not review._admissible_trajectory(model,shape,1,far,(.025,.45))
    assert review._admissible_trajectory(model,shape,1,far,(.020,.45))
    assert not review._admissible_trajectory(model,shape,1,far,(.020,.090))
    assert not review._admissible_trajectory(model,shape,1,np.array([[3.4,2.7]]),(.020,.45))


def test_alternate_lattice_record_matches_recomputed_spectrum(scene, monkeypatch):
    clean, wave, diag = scene
    obs = np.clip(clean+wave,0,1)
    caches = diag["evidence"].pop("weak_carrier")["measured_lattice_frequencies"]
    centers = np.array(caches)*obs.shape
    monkeypatch.setattr(review, "_spectrum", lambda x: (centers, None, np.ones(len(centers))))
    monkeypatch.setattr(review, "_fit_lattice", lambda p, power, shape: (np.ones(len(p), bool), np.eye(2)))
    reference, _, old = review.review(obs, diag)
    def unexpected(*args):
        pytest.fail("Recomputed an available measured lattice")
    monkeypatch.setattr(review, "_spectrum", unexpected)
    diag["evidence"]["lattice_screen"] = {"measured_lattice_peaks":caches}
    cached, _, record = review.review(obs, diag)
    np.testing.assert_array_equal(cached, reference)
    assert record == old


@pytest.mark.parametrize("shape,start,end", [
    ((96,128),(6.,62.),(7.,65.)),
    ((95,127),(-6.,-62.),(-5.,-65.)),
    ((95,128),(47.,-2.),(48.,1.)),
])
def test_windowed_local_periodogram_matches_complex_fft_across_nyquist(shape,start,end):
    raw = np.random.default_rng(7).normal(size=(77,57))
    start, end = np.asarray(start), np.asarray(end)
    radius = 1.4
    actual = review._residual_tone(raw, start, end, shape, radius)
    win = np.hanning(raw.shape[0])[:, None]*np.hanning(raw.shape[1])
    freq = np.fft.fft2((raw-np.sum(win*raw)/win.sum())*win, s=shape, axes=(0,1))
    selected = []
    for y in range(int(min(start[0],end[0]))-2, int(max(start[0],end[0]))+3):
        for x in range(int(min(start[1],end[1]))-2, int(max(start[1],end[1]))+3):
            d = end-start
            z = np.array([y,x])-start
            if np.linalg.norm(z-d*np.clip(z.dot(d)/d.dot(d),0,1)) <= radius:
                selected.append(abs(freq[y % shape[0], x % shape[1]]))
    assert actual == pytest.approx(2*max(selected)/win.sum(), rel=1e-12)
