"""Check the Fourier correction's geometry, range, support and gain constraints."""
import numpy as np
import pytest

from peak_guard import (_adjust, _carrier_geometry, _project, _texture_band, MAX_GAIN_DRIFT,
                        MAX_LOCAL_CHANGE, refine)
from project_guarded import _restore


def test_carrier_phase_and_gradient_on_odd_rectangle():
    shape = (139, 165)
    fit = dict(center_frequency=[.086, -.124], hessian=[.0007, -.0002, .0001],
               amplitude=.075, final_phase_radians=.36)
    wave, field = _carrier_geometry(shape, fit)
    y, x = (shape[0]-1)//2, (shape[1]-1)//2
    assert wave[y,x] == pytest.approx(.075*np.cos(.36))
    np.testing.assert_allclose(field[y,x], np.array(shape)*fit["center_frequency"])
    np.testing.assert_allclose(field[y+2,x-1]-field[y,x], np.array(shape)*[.0016, -.0005])


@pytest.fixture
def synthetic_chirp():
    shape = (160, 144)
    y, x = np.indices(shape)
    fit = dict(center_frequency=[.12, -.097], hessian=[0., .00004, .0004],
               amplitude=.09, final_phase_radians=.27, model="quadratic",
               protection=dict(tapered_pixels=1))
    wave, field = _carrier_geometry(shape, fit)
    base = .50 + .17*np.cos(2*np.pi*(7*y/shape[0]+8*x/shape[1]))
    base += .32*np.cos(2*np.pi*(3*y/shape[0]-5*x/shape[1]))
    observed = np.clip(base+wave, 0, 1)
    weight = np.clip((x-28)/33, 0., 1.)
    applied = wave*weight
    old = _restore(observed, applied, 1.)
    diag = dict(reason="clip_supported_independent_carrier", carrier_wave=applied, support_mask=weight>0,
                evidence=dict(guarded_carrier=fit), accepted_regions=[dict(gain=1.)])
    return base, observed, fit, wave, field, weight, old, diag


def test_observed_spectral_peaks_exclude_chirp_and_include_their_conjugates(synthetic_chirp):
    _, observed, _, wave, field, _, _, _ = synthetic_chirp
    band, bins = _texture_band(observed, wave, field)
    h, w = observed.shape
    assert [7, 8] in bins and [3, -5] in bins
    assert band[7,8] and band[-7,-8]
    np.testing.assert_array_equal(band, band[np.ix_((-np.arange(h)) % h, (-np.arange(w)) % w)])
    for i, j in np.rint(field[[0,-1],[0,-1]]).astype(int):
        assert not band[i,j]
    empty, bins = _texture_band(np.zeros_like(observed), wave, field)
    assert not empty.any() and not bins


@pytest.mark.parametrize("gain", [.5, 1.])
def test_refinement_reduces_predicted_leakage_within_existing_support_and_range(synthetic_chirp, gain):
    _, observed, fit, wave, field, weight, _, diag = synthetic_chirp
    diag["accepted_regions"][0]["gain"] = gain
    old = _restore(observed, diag["carrier_wave"], gain)
    reviewed, support, evidence = refine(observed, old, diag)
    assert evidence["iterations"] > 0
    np.testing.assert_array_equal(support, diag["support_mask"])
    np.testing.assert_array_equal(reviewed[~support], observed[~support])
    assert np.max(np.abs(reviewed-old)-MAX_LOCAL_CHANGE*gain*fit["amplitude"]*weight) <= 1e-12
    assert np.min(reviewed) >= 0 and np.max(reviewed) <= 1
    target = _restore(observed, wave, gain)
    assert np.mean((reviewed-target)**2) <= np.mean((old-target)**2)
    band, _ = _texture_band(observed, wave, field)
    assert np.linalg.norm(_project(reviewed-target, band)) < np.linalg.norm(_project(old-target, band))
    model = observed-target
    assert abs(np.sum((reviewed-old)*model)) <= MAX_GAIN_DRIFT*np.sum(model**2)
    assert np.all(reviewed[observed==1.] >= np.minimum(old, 1-gain*np.maximum(wave,0))[observed==1.])
    assert np.all(reviewed[observed==0.] <= np.maximum(old, -gain*np.minimum(wave,0))[observed==0.])


@pytest.mark.parametrize("reason,model,protected,tapered", [
    ("no_verified_full_field_carrier","quadratic",4,8),
    ("clip_supported_independent_carrier","stationary",4,8),
    ("clip_supported_independent_carrier","quadratic",0,0),
])
def test_existing_routes_and_uncollided_models_retain_their_output(reason,model,protected,tapered):
    image = np.full((144,160), .5)
    diag = dict(reason=reason, support_mask=np.ones_like(image, bool), evidence={
        "guarded_carrier": dict(model=model, protection=dict(protected_pixels=protected,tapered_pixels=tapered))})
    restored, support, record = refine(image, image, diag)
    assert restored is image and support is diag["support_mask"]
    assert record["reason"] == "no_spatially_tapered_verified_chirp"


def test_inactive_support_has_no_numerical_correction():
    flat = np.ones((19,21))*.42
    result, iterations, ratio = _adjust(flat, flat-.1, np.full_like(flat,.1), np.zeros_like(flat),
                                          np.ones_like(flat,bool), flat-.1, flat+.1)
    assert result is flat and iterations == 0 and ratio == 1.
