"""Endpoint ambiguity, genuine lattice harmonics, and local protection on observed signals."""
from itertools import product

import numpy as np
import pytest
from scipy.spatial import cKDTree

import guarded_carrier as guarded


SHAPE = (167, 199)
BASIS = np.array([[8., 13.], [12., -8.]])
ORDERS = np.array([[1,0],[0,1],[1,1],[1,-1],[2,1],[1,-2],[2,0]])
BOUNDS = (.025, .45)


@pytest.fixture(scope="module")
def scene():
    y, x = np.indices(SHAPE, dtype=float)
    cloth = np.zeros(SHAPE)
    for i, (dy,dx) in enumerate(ORDERS@BASIS):
        cloth += np.cos(2*np.pi*(dy*y/SHAPE[0]+dx*x/SHAPE[1])+.31*i)
    clean = .5+.48*np.tanh(1.3*cloth/cloth.std())
    carrier = .09*np.cos(2*np.pi*(-.108*(y-83)+.171*(x-99)+.08))
    return clean, carrier


def test_endpoint_and_brightness_support_independent_wave_on_odd_rectangle(scene):
    clean, carrier = scene
    image = np.clip(clean+carrier, 0, 1)
    wave, diag = guarded.find_carrier(image, BOUNDS, enable_chirp=False)
    assert diag["model"] == "stationary" and min(diag["held_quadrant_counts"]) >= 8
    assert diag["brightness_check"]["positive_gain_checking_cells"] == 8
    assert diag["protection"]["other_lattice_peaks"] >= 6
    assert np.mean((wave-carrier)**2) < 8e-5
    assert np.allclose(diag["center_frequency"], [.108,-.171], atol=6e-4)
    if diag["phase_refined"]:
        assert diag["final_calibration"]["endpoints"]["mean_signed_phase"] >= guarded.MIN_ENDPOINT_ALIGNMENT


def test_exposure_clipping_without_added_noise_preserves_periodic_texture(scene):
    clean, _ = scene
    clipped = np.clip(1.4*(clean-.5)+.5, 0, 1)
    assert np.count_nonzero((clipped==0)|(clipped==1)) > .1*clipped.size
    wave, _ = guarded.find_carrier(clipped, BOUNDS)
    assert wave is None


def test_coherent_saturated_existing_lattice_harmonic_stays_protected(scene):
    clean, _ = scene
    y, x = np.indices(SHAPE)
    dy, dx = BASIS[0]
    # Strengthen a legitimate thread-frequency peak that otherwise looks just
    # like a globally coherent, independently verified sinusoidal wave.
    dye = .09*np.cos(2*np.pi*(dy*(y-83)/SHAPE[0]+dx*(x-99)/SHAPE[1])+.1)
    image = np.clip(clean+dye, 0, 1)
    wave, _ = guarded.find_carrier(image, BOUNDS)
    assert wave is None


@pytest.mark.parametrize("side", ["left","corner"])
def test_local_clipped_wave_is_not_applied_to_untouched_side(scene, side):
    clean, carrier = scene
    y, x = np.indices(SHAPE)
    mask = x < SHAPE[1]//2
    if side == "corner":
        mask &= y < SHAPE[0]//2
    wave, diag = guarded.find_carrier(np.clip(clean+carrier*mask,0,1), BOUNDS)
    assert wave is None and diag["reason"] == "insufficient_spatial_clipping_evidence"


def test_no_clipping_retains_prior_and_reports_absence_of_endpoint_evidence(scene):
    clean, carrier = scene
    frame = .15+.7*clean+.25*carrier
    wave, info = guarded.find_carrier(frame, BOUNDS)
    assert wave is None and info["reason"] == "insufficient_clipping_evidence"


def test_chirp_support_tapers_around_real_and_conjugate_nodes():
    bins = np.arange(-4.,4.01,.25)
    field = np.stack((bins,np.zeros_like(bins)),axis=-1)[None]
    mask, info = guarded._collision_taper(field, cKDTree([[0., 0.]]), field.shape[:2])
    assert info["protected_pixels"] == np.count_nonzero(abs(bins)<=1.25)
    np.testing.assert_array_equal(mask[0, abs(bins)<=1.25], 0.)
    assert np.all((mask[0,abs(bins)==1.75]>0)&(mask[0,abs(bins)==1.75]<1))
    np.testing.assert_array_equal(mask[0,abs(bins)>=2.25], 1.)
    assert guarded._collision_taper(field*0, cKDTree([[0., 0.]]), field.shape[:2]) == (None, None)


def test_quadratic_gradient_on_asymmetric_coordinates():
    params = np.array([7., 12., 4., -3., 8.])
    points = np.array(list(product([-.35,.42],[-.17,.31])))
    actual = guarded._field(params, points)
    np.testing.assert_allclose(actual[:,0], 7+4*points[:,0]-3*points[:,1])
    np.testing.assert_allclose(actual[:,1], 12-3*points[:,0]+8*points[:,1])


def test_local_phase_relation_distinguishes_texture_sum_from_free_carrier():
    h,w = 192,208
    y,x = np.indices((h,w))
    coords = np.stack(((y-(h-1)/2)/h,(x-(w-1)/2)/w),axis=-1)
    a,b = np.array([[14.,5.],[-3.,17.]])
    rng = np.random.default_rng(42)
    cells = (4*(y//48)+x//52).astype(int)
    p = 2*np.pi*(coords@a)+rng.uniform(-np.pi,np.pi,16)[cells]
    q = 2*np.pi*(coords@b)+rng.uniform(-np.pi,np.pi,16)[cells]
    scores = []
    for phase in (p+q, 2*np.pi*(coords@(a+b))):
        image = .5+.1*np.cos(p)+.1*np.cos(q)+.08*np.cos(phase+.45)
        basis, labels, signal, weights, _ = guarded._measurements(image)
        scores.append(guarded._phase_lock(a+b,np.stack((a,b)),basis[...,:2],labels,signal,weights))
    assert scores[0] > guarded.LOCKED_PHASE_RESULTANT > scores[1]
