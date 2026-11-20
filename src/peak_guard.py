"""Correct leakage from a verified, spatially tapered chirp into distant texture peaks."""
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

from local_spectral import _detrend
from project_guarded import _restore


PEAK_RADIUS = 2
MAX_PEAKS = 24
MIN_RELATIVE_HEIGHT = .12
MAX_STEPS = 16
MAX_LOCAL_CHANGE = .25
MAX_GAIN_DRIFT = .001


def routing_parameters():
    return dict(version="bounded_peak_projection_v1", peak_radius_bins=PEAK_RADIUS,
                peak_search_cycles_per_pixel=[.025, .45], maximum_observed_peaks=MAX_PEAKS,
                minimum_relative_height=MIN_RELATIVE_HEIGHT, minimum_frequency_clearance_bins=5.,
                maximum_model_band_density="mean full-wave power per FFT bin",
                maximum_iterations=MAX_STEPS, maximum_adjustment_fraction=MAX_LOCAL_CHANGE,
                maximum_model_gain_drift=MAX_GAIN_DRIFT)


def _carrier_geometry(shape, fit):
    """Reconstruct the untapered wave and its local frequency from the verified fit."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    y = y - (shape[0]-1)/2
    x = x - (shape[1]-1)/2
    fy, fx = fit["center_frequency"]
    yy, yx, xx = fit["hessian"]
    cycles = fy*y + fx*x + .5*yy*y*y + yx*y*x + .5*xx*x*x
    wave = fit["amplitude"] * np.cos((2*np.pi)*cycles + fit["final_phase_radians"])
    field = np.stack(((fy + yy*y + yx*x)*shape[0], (fx + yx*y + xx*x)*shape[1]), axis=-1)
    return wave, field


def _texture_band(observed, wave, field):
    """Use strong measured peaks outside both the frequency field and its Fourier sidelobes."""
    h, w = observed.shape
    by = np.rint(np.fft.fftfreq(h)*h).astype(int)[:, None]
    bx = np.rint(np.fft.fftfreq(w)*w).astype(int)[None, :]
    radius = np.hypot(by/h, bx/w)
    magnitude = abs(np.fft.fft2(_detrend(observed) * np.hanning(h)[:, None] * np.hanning(w)))
    valid = (radius >= .025) & (radius <= .45) & ((by > 0) | ((by == 0) & (bx > 0)))
    maxima = valid & (magnitude == maximum_filter(magnitude, 3, mode="wrap"))
    bins = np.flatnonzero(maxima)
    band = np.zeros(observed.shape, bool)
    if not len(bins):
        return band, []
    heights = magnitude.ravel()[bins]
    order = np.argsort(-heights, kind="stable")
    centers = []
    for j in order:
        if heights[j] <= 0 or heights[j] < MIN_RELATIVE_HEIGHT*heights[order[0]] or len(centers) == MAX_PEAKS:
            break
        i, k = np.unravel_index(bins[j], (h, w))
        point = np.array([by[i, 0], bx[0, k]])
        if all(min(np.linalg.norm(point-c), np.linalg.norm(point+c)) >= 2*PEAK_RADIUS for c in centers):
            centers.append(point)
    if not centers:
        return band, []
    centers = np.asarray(centers)
    trajectory = cKDTree(field.reshape(-1, 2))
    gaps = np.minimum(trajectory.query(centers)[0], trajectory.query(-centers)[0])
    model_power = abs(np.fft.fft2(wave))**2
    mean_power = model_power.mean()
    selected = []
    for (fy, fx), gap in zip(centers, gaps):
        if gap <= 2*PEAK_RADIUS+1:
            continue
        disk = ((by-fy)**2 + (bx-fx)**2 <= PEAK_RADIUS**2) | ((by+fy)**2 + (bx+fx)**2 <= PEAK_RADIUS**2)
        if model_power[disk].mean() < mean_power:
            band |= disk
            selected.append([int(fy), int(fx)])
    return band, selected


def _project(image, band):
    return np.fft.ifft2(np.fft.fft2(image)*band).real


def _adjust(previous, target, model, weight, band, lower, upper):
    """Fit selected Fourier coefficients within the old support and bound any change in gain."""
    weighted_model = weight*model
    mass = np.sum(weighted_model*model)
    total = np.sum(model**2)
    if mass <= np.finfo(float).eps or total == 0:
        return previous, 0, 1.

    def neutral(direction):
        # Subtract only the component needed to keep the total model projection constant.
        return weight * (direction-model*(np.sum(weighted_model*direction)/mass))

    current = previous
    difference = target-current
    band_error = _project(difference, band)
    band_energy = np.sum(band_error**2)
    original_band_energy = band_energy
    count = 0
    for _ in range(MAX_STEPS):
        step = neutral(band_error)
        step_energy = np.sum(step**2)
        if step_energy <= np.finfo(float).eps * max(original_band_energy, 1e-15):
            break
        # Decreasing the error inside a few peaks alone can increase the error elsewhere.
        # Allow a gain-neutral move toward the full fit to supply descent in image space.
        needed = step_energy - np.sum(step*difference)
        if needed > 0:
            toward = neutral(difference)
            available = np.sum(toward*difference)
            if available <= 0:
                break
            step += (needed/available)*toward
        candidate = np.clip(current+step, lower, upper)
        delta = candidate-current
        # Recheck after clipping, which may change the direction and its projection.
        if (abs(np.sum((candidate-previous)*model)) > MAX_GAIN_DRIFT*total
                or 2*np.sum(difference*delta) < np.sum(delta**2)):
            break
        next_difference = target-candidate
        next_projection = _project(next_difference, band)
        next_energy = np.sum(next_projection**2)
        if next_energy >= band_energy:
            break
        current, difference = candidate, next_difference
        band_error, band_energy = next_projection, next_energy
        count += 1
    ratio = float(band_energy/original_band_energy) if original_band_energy > 0 else 1.
    return current, count, ratio


def refine(observed, previous_output, diag):
    """Return output, unchanged support and evidence; only an already accepted chirp is eligible."""
    support = diag["support_mask"]
    fit = diag.get("evidence", {}).get("guarded_carrier", {})
    protection = fit.get("protection", {})
    if (diag.get("reason") != "clip_supported_independent_carrier" or fit.get("model") != "quadratic"
            or not (protection.get("tapered_pixels") or protection.get("protected_pixels"))):
        return previous_output, support, dict(reason="no_spatially_tapered_verified_chirp")

    wave, field = _carrier_geometry(observed.shape, fit)
    band, peaks = _texture_band(observed, wave, field)
    if not peaks:
        return previous_output, support, dict(reason="no_distant_observed_texture_peaks")
    # Near carrier zero-crossings the ratio is numerically indeterminate. Leave these
    # pixels fixed, as well as every collision pixel the previous pass protected.
    weight = np.zeros_like(wave)
    np.divide(diag["carrier_wave"], wave, out=weight,
              where=support & (abs(wave) > np.sqrt(np.finfo(float).eps)*fit["amplitude"]))
    np.clip(weight, 0., 1., out=weight)
    gain = diag["accepted_regions"][0]["gain"]
    target = _restore(observed, wave, gain)
    model = observed-target
    limit = MAX_LOCAL_CHANGE*gain*fit["amplitude"]*weight
    lower = np.maximum(0., previous_output-limit)
    upper = np.minimum(1., previous_output+limit)
    # A censored pixel also has a one-sided feasible intensity interval.
    high, low = observed == 1., observed == 0.
    lower[high] = np.maximum(lower[high], np.minimum(previous_output[high], 1-gain*np.maximum(wave[high], 0.)))
    upper[low] = np.minimum(upper[low], np.maximum(previous_output[low], -gain*np.minimum(wave[low], 0.)))
    result, count, ratio = _adjust(previous_output, target, model, weight, band, lower, upper)
    evidence = dict(reason="bounded_texture_band_correction" if count else "no_feasible_texture_correction",
                    observed_peak_bins=peaks, fourier_bins=int(band.sum()), iterations=count,
                    fraction_initial_model_band_error=ratio)
    model_energy = np.sum(model**2)
    if model_energy > 0:
        evidence["model_gain_before"] = float(np.sum((observed-previous_output)*model)/model_energy)
        evidence["model_gain_after"] = float(np.sum((observed-result)*model)/model_energy)
    return result, support, evidence
