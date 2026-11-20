"""Numerical contracts of the independent broad-carrier search."""
import numpy as np
import pytest

from weave_carrier import independent_carrier


BASIS = np.array([[.067, 0.], [0., .064]])
BOUNDS = (.025, .45)


def centered_grid(shape):
    y, x = np.indices(shape, dtype=float)
    return y - (shape[0]-1)/2, x - (shape[1]-1)/2


def test_constant_and_out_of_band_signal_do_not_supply_a_candidate():
    y, _ = centered_grid((128, 128))
    for observed in (np.full(y.shape, .5), .5 + .1*np.cos(2*np.pi*.469*y)):
        wave, diag = independent_carrier(observed, BASIS, BASIS, BOUNDS)
        assert wave is None and diag["proposed_models"] == 0


def test_insufficient_extent_or_missing_texture_evidence_abstains():
    for image, protected in ((np.full((127, 200), .5), BASIS),
                             (np.full((160, 160), .5), np.empty((0, 2)))):
        wave, diag = independent_carrier(image, protected, BASIS, BOUNDS)
        assert wave is None and diag["reason"] == "insufficient_patch_extent"


@pytest.mark.parametrize("flip", [False, True])
def test_quadratic_phase_units_and_intensity_on_rectangular_image(flip):
    y, x = centered_grid((128, 192))
    fy, fx = .114, .156
    yy, yx, xx = .00008, .00003, -.00009
    true_wave = .10*np.cos(2*np.pi*(fy*y+fx*x+.5*yy*y*y+yx*y*x+.5*xx*x*x+.065))
    image = .5+true_wave+.05*np.cos(2*np.pi*(.067*y+.15))+.06*np.cos(2*np.pi*(.064*x+.3))
    if flip:
        image, true_wave = np.fliplr(image), np.fliplr(true_wave)
        fx, yx = -fx, -yx
    output, diag = independent_carrier(image, BASIS, BASIS, BOUNDS)
    assert diag["reason"] == "verified_non_lattice_carrier"
    assert np.mean((output - true_wave)**2) < 1e-7
    # (f,H) and (-f,-H) describe the same real wave after conjugation.
    sign = np.sign(diag["center_frequency"][0])
    np.testing.assert_allclose(diag["center_frequency"], sign*np.array([fy, fx]), atol=5e-5)
    np.testing.assert_allclose(diag["hessian"], sign*np.array([yy, yx, xx]), atol=5e-6)
    assert diag["amplitude"] == pytest.approx(.10, abs=.001)


def test_matching_patch_frequency_without_agreeing_check_phase_is_rejected():
    y, x = np.indices((128, 128))
    carrier = 2*np.pi*(.114*y+.156*x)
    coherent = .5+.1*np.cos(carrier)
    control, _ = independent_carrier(coherent, BASIS, BASIS, BOUNDS)
    assert control is not None
    # For this shape the 16 disjoint patches tile the complete image.
    check = (y//32+x//32) % 2 == 1
    incoherent = .5+.1*np.cos(carrier+np.pi*check)
    np.testing.assert_array_equal(incoherent[~check], coherent[~check])
    wave, diag = independent_carrier(incoherent, BASIS, BASIS, BOUNDS)
    assert diag["proposed_models"] > 0 and diag["verified_models"] == 0 and wave is None


def test_weak_unlisted_higher_order_lattice_peak_is_protected():
    y, x = centered_grid((128, 128))
    f = 2*BASIS[0]+3*BASIS[1]
    signal = .5+.032*np.cos(2*np.pi*(f[0]*y+f[1]*x))
    wave, diag = independent_carrier(signal, BASIS, BASIS, BOUNDS)
    assert wave is None and diag["lattice_models_rejected"] >= 1


def test_two_verified_non_lattice_frequencies_leave_the_image_unchanged():
    y, x = centered_grid((128, 128))
    image = .5+.12*np.sin(2*np.pi*(.114*y+.156*x))+.11*np.sin(2*np.pi*(.203*y-.115*x))
    wave, diag = independent_carrier(image, BASIS, BASIS, BOUNDS)
    assert wave is None and diag["verified_models"] >= 2


def test_frequency_bound_applies_over_the_whole_reconstruction_extent():
    y, x = centered_grid((192, 192))
    signal = .5+.09*np.cos(2*np.pi*(.12*y+.115*x+.5*.0008*x*x))
    # Central frequency passes, while the right edge has radius above 0.18.
    wave, diag = independent_carrier(signal, BASIS, BASIS, (.025, .18))
    assert wave is None and diag["reason"] == "carrier_outside_frequency_bounds"


def test_disabling_chirp_fits_stationary_carriers_and_preserves_zero_curvature():
    y, x = centered_grid((192, 192))
    phase = .12*y+.115*x
    stationary = .09*np.cos(2*np.pi*phase)
    wave, diag = independent_carrier(.5+stationary, BASIS, BASIS, BOUNDS, enable_chirp=False)
    np.testing.assert_allclose(wave, stationary, atol=1e-6)
    assert diag["hessian"] == [0., 0., 0.]
    chirp = .5+.09*np.cos(2*np.pi*(phase+.0004*x*x))
    wave, diag = independent_carrier(chirp, BASIS, BASIS, BOUNDS, enable_chirp=False)
    assert wave is None
