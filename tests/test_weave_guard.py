"""Reciprocal-lattice detection on observed patterns, independently of natural images."""
import numpy as np
import pytest

from weave_guard import analyze_lattice


SHAPE = (167, 199)
BASIS = np.array([[8, 13], [12, -8]])
ORDERS = np.array([[1, 0], [0, 1], [1, 1], [1, -1], [2, 1], [1, -2], [2, 0]])


def pattern(peaks, amplitudes, shape=SHAPE):
    y, x = np.indices(shape)
    image = np.full(shape, .5)
    for i, ((fy, fx), amp) in enumerate(zip(peaks, amplitudes)):
        image += amp*np.cos(2*np.pi*(fy*y/shape[0] + fx*x/shape[1]) + .31*i)
    return image


def region(peak, shape=SHAPE):
    return dict(height=64, width=64, frequency_y=peak[0]/shape[0], frequency_x=peak[1]/shape[1])


def test_recognizes_signed_lattice_nodes_and_excludes_isolated_carrier():
    lattice_peaks = ORDERS @ BASIS
    isolated = np.array([29, 5])
    image = pattern([*lattice_peaks, isolated], [.04]*len(lattice_peaks) + [.035])
    peaks, matches, diag = analyze_lattice(image, [region(lattice_peaks[0]), region(-lattice_peaks[3]), region(isolated)])
    assert diag["detected"] and diag["lattice_power_fraction"] > .5
    assert matches.tolist() == [True, True, False]
    assert len(peaks) >= len(lattice_peaks)


@pytest.mark.parametrize("peaks", [
    [[6, 8]], [[6, 8], [9, -5]], [[2*i, 4*i] for i in range(1, 9)],
])
def test_few_peaks_or_one_dimensional_harmonics_do_not_define_a_lattice(peaks):
    image = pattern(peaks, [.04]*len(peaks))
    lattice, matches, diag = analyze_lattice(image, [region(peaks[0])])
    assert not diag["detected"]
    assert lattice.shape == (0, 2) and not matches.any()


def test_dominant_locked_carrier_can_still_be_subtracted():
    peaks = ORDERS @ BASIS
    image = pattern(peaks, [.20] + [.03]*(len(peaks)-1))
    _, matches, diag = analyze_lattice(image, [region(peaks[0]), region(-peaks[0])])
    assert diag["detected"] and diag["dominant_carrier_retained"]
    assert not matches.any()
    # Dominance alone does not overrule inconsistent local estimates.
    _, matches, diag = analyze_lattice(image, [region(peaks[0]), region(peaks[1])])
    assert not diag["dominant_carrier_retained"] and matches.all()
