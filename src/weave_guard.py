"""Find observed Fourier peaks belonging to a common two-dimensional periodic texture."""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.ndimage import maximum_filter

from local_spectral import _detrend


# Fit integer combinations in full-image FFT bins. Local fitted frequencies
# have coarser resolution; matching them uses one bin of their synthesis tile.
LATTICE_TOLERANCE_BINS = 1.25
MIN_PEAKS = 6
MAX_ORDER = 6
MIN_RELATIVE_AMPLITUDE = .08
MIN_LATTICE_POWER_FRACTION = .20
DOMINANCE_RATIO = 2.
MIN_LOCKED_FRACTION = .90


def _spectrum(image):
    h, w = image.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    amplitude = np.abs(np.fft.fftshift(np.fft.fft2(_detrend(image) * window)))
    row = np.arange(h)[:, None] - h // 2
    col = np.arange(w)[None, :] - w // 2
    radius = np.hypot(row / h, col / w)
    band = (radius >= .025) & (radius <= .45)
    maxima = (amplitude == maximum_filter(amplitude, 7, mode="wrap")) & band
    maxima &= (row > 0) | ((row == 0) & (col > 0))
    centers = np.argwhere(maxima)
    heights = amplitude[maxima]
    if not heights.size or heights.max() <= 0:
        return np.empty((0, 2)), np.empty(0), np.empty(0)
    order = np.argsort(-heights, kind="stable")
    order = order[heights[order] >= heights.max()*MIN_RELATIVE_AMPLITUDE][:24]
    centers = centers[order]
    peak_bins = centers.astype(float) - [h // 2, w // 2]
    powers = []
    total = np.sum(amplitude[band] ** 2)
    offsets = np.arange(-2, 3)
    disk = offsets[:, None]**2 + offsets**2 <= 4
    dy, dx = np.nonzero(disk)
    for i, (y, x) in enumerate(centers):
        for axis, profile in enumerate((amplitude[y-1:y+2, x], amplitude[y, x-1:x+2])):
            left, middle, right = np.log(np.maximum(profile, 1e-30))
            curvature = left - 2*middle + right
            if curvature < 0:
                peak_bins[i, axis] += np.clip(.5 * (left-right) / curvature, -.5, .5)
        disk_y, disk_x = (y + dy - 2) % h, (x + dx - 2) % w
        disk_power = amplitude[disk_y, disk_x]**2 * band[disk_y, disk_x]
        powers.append(2 * np.sum(disk_power) / max(total, 1e-30))
    return peak_bins, heights[order], np.asarray(powers)


def _fit_lattice(peaks, power, shape):
    """Use two nonparallel peaks to propose a reciprocal lattice; confirm other peaks."""
    best_score, best_members, best_basis = None, None, None
    for first, second in combinations(peaks, 2):
        basis = np.stack((first, second))
        physical = basis / shape
        if abs(np.linalg.det(physical)) < .35 * np.linalg.norm(physical[0]) * np.linalg.norm(physical[1]):
            continue
        orders = np.rint(peaks @ np.linalg.inv(basis))
        valid_order = (np.max(abs(orders), axis=1) <= MAX_ORDER) & np.any(orders != 0, axis=1)
        distances = np.linalg.norm(peaks - orders @ basis, axis=1)
        members = valid_order & (distances <= LATTICE_TOLERANCE_BINS)
        if members.sum() < MIN_PEAKS:
            continue
        basis = np.linalg.lstsq(orders[members], peaks[members], rcond=None)[0]
        distances = np.linalg.norm(peaks - orders @ basis, axis=1)
        members = valid_order & (distances <= LATTICE_TOLERANCE_BINS)
        count = int(members.sum())
        if count < MIN_PEAKS or np.count_nonzero(np.all(orders[members] != 0, axis=1)) < 2:
            continue
        share = float(power[members].sum())
        if share < MIN_LATTICE_POWER_FRACTION:
            continue
        # Inlier count rewards a repeatedly observed geometry. The secondary
        # terms resolve ties without one strong isolated carrier defining it.
        score = (count, share, -float(distances[members].mean()))
        if best_score is None or score > best_score:
            best_score, best_members, best_basis = score, members, basis
    return best_members, best_basis


def _frequency_distance(frequencies, peaks, bin_scale):
    delta = (frequencies[:, None] - peaks) * bin_scale
    flipped = (frequencies[:, None] + peaks) * bin_scale
    return np.minimum(np.linalg.norm(delta, axis=2), np.linalg.norm(flipped, axis=2))


def analyze_lattice(observed, accepted_regions):
    """Return observed lattice peaks (fy,fx), per-region matches, and audit evidence.

    Opposite frequencies represent the same real wave. The output peak list
    contains measured peaks only, never unobserved integer combinations.
    """
    shape = np.asarray(observed.shape)
    flags = np.zeros(len(accepted_regions), dtype=bool)
    empty = np.empty((0, 2))
    peaks, heights, power = _spectrum(observed)
    evidence = dict(detected=False, candidate_peaks=len(peaks), lattice_tolerance_bins=LATTICE_TOLERANCE_BINS,
                    region_match_tolerance_tile_bins=1., dominant_carrier_retained=False)
    if not accepted_regions or len(peaks) < MIN_PEAKS:
        return empty, flags, evidence
    members, basis = _fit_lattice(peaks, power, shape)
    if members is None:
        return empty, flags, evidence
    frequencies = np.array([[r["frequency_y"], r["frequency_x"]] for r in accepted_regions])
    sizes = np.array([[r["height"], r["width"]] for r in accepted_regions])
    protected = peaks[members] / shape
    flags = np.min(_frequency_distance(frequencies, protected, sizes[:, None]), axis=1) <= 1.
    distances = _frequency_distance(frequencies, peaks[:1]/shape, shape)[:, 0]
    locked = distances <= 1.
    ratio = float(heights[0] / heights[1])
    if ratio >= DOMINANCE_RATIO and np.mean(locked) >= MIN_LOCKED_FRACTION:
        flags[locked] = False
        evidence["dominant_carrier_retained"] = True
    evidence.update(detected=True, measured_lattice_peaks=protected.tolist(),
                    basis_cycles_per_pixel=(basis/shape).tolist(), lattice_power_fraction=float(power[members].sum()),
                    matched_regions=int(flags.sum()), dominant_amplitude_ratio=ratio,
                    fraction_locked_to_dominant=float(np.mean(locked)))
    return protected, flags, evidence
