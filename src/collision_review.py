"""Recheck inferred lattice collisions of an already verified single-axis stripe."""
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d
from scipy.optimize import minimize

from guarded_carrier import MIN_ENDPOINT_ALIGNMENT, MIN_QUADRANT_EVENTS, TAPER_BINS
from project_guarded import _restore
from weave_carrier import MAX_COEFFICIENT_ERROR
from weave_guard import LATTICE_TOLERANCE_BINS, _fit_lattice, _spectrum


MAX_TRANSVERSE_DRIFT = .5  # Full-image Fourier bins across the transverse axis.
MIN_WIDTH = 12
MIN_CLEAR_FRACTION = .20


def routing_parameters():
    return dict(version="exterior_single_axis_collision_review_v2", minimum_band_width=MIN_WIDTH,
                maximum_transverse_drift_bins=MAX_TRANSVERSE_DRIFT, minimum_clear_fraction=MIN_CLEAR_FRACTION,
                measured_peak_veto_bins=LATTICE_TOLERANCE_BINS, minimum_held_events=MIN_QUADRANT_EVENTS,
                minimum_signed_alignment=MIN_ENDPOINT_ALIGNMENT, coefficient_tolerance=MAX_COEFFICIENT_ERROR,
                local_stationary_residual_limit=MAX_COEFFICIENT_ERROR, brightness_highpass_sigma=5.,
                exterior_buffer_periods=1, exterior_taper_periods=2, minimum_exterior_span_fraction=.5,
                minimum_exterior_weight_fraction=.125, maximum_refit_corner_drift_bins=.5,
                amplitude_correction="one standard error of four transverse band estimates")


def _segment_distances(points, start, end):
    delta = end-start
    length = np.dot(delta, delta)
    t = np.clip((points-start) @ delta / length, 0., 1.) if length else 0.
    nearest = start + np.asarray(t)[..., None]*delta
    return np.linalg.norm(points-nearest, axis=-1)


def _residual_tone(raw, start, end, full_shape, radius):
    """Stationary amplitude in a windowed section, normalized to sinusoid amplitude."""
    win = np.hanning(raw.shape[0])[:, None]*np.hanning(raw.shape[1])
    avg = np.sum(win*raw)/win.sum()
    spectrum = np.fft.rfft2((raw-avg)*win, s=full_shape, axes=(0, 1))
    lower = np.floor(np.minimum(start, end)-radius).astype(int)
    upper = np.ceil(np.maximum(start, end)+radius).astype(int)
    iy, ix = np.mgrid[lower[0]:upper[0]+1, lower[1]:upper[1]+1]
    grid = np.column_stack((iy.ravel(), ix.ravel()))
    bins = grid[_segment_distances(grid, start, end) <= radius]
    # Wrap both Nyquist edges before mapping to the real FFT's stored half plane.
    ky, kx = bins.T
    kx = kx % full_shape[1]
    reflected = kx > full_shape[1]//2
    values = np.abs(spectrum[np.where(reflected, -ky, ky) % full_shape[0],
                             np.where(reflected, full_shape[1]-kx, kx)])
    return float(2*values.max(initial=0.)/win.sum())


def _exterior_fit(observed, guarded, axis, fit):
    """Estimate a single-axis chirp from clear strips, excluding a one-period buffer."""
    data = observed.T if axis == 0 else observed
    cross_size, sweep_size = data.shape
    frequency = np.asarray(fit["center_frequency"])
    period = int(np.ceil(1/np.linalg.norm(frequency)))
    distance = distance_transform_edt(~guarded)
    fraction = np.clip((distance-period)/(2*period), 0., 1.)
    weights = .5-.5*np.cos(np.pi*fraction)
    used = weights > 0
    if weights.sum() < sweep_size/8 or np.ptp(np.flatnonzero(used)) < sweep_size/2:
        return None
    cross = (np.arange(cross_size)-(cross_size-1)/2)/cross_size
    sweep = (np.arange(sweep_size)-(sweep_size-1)/2)/sweep_size
    columns = sweep[used]
    square = .5*columns**2
    # Detrend only the observed exterior pixels. A full-image blur would
    # otherwise let brightness from inside the guarded band leak into the fit.
    data = data[:, used]
    observations = np.ascontiguousarray((data-gaussian_filter1d(data,5.,axis=0))*weights[used])
    scale = 2/(cross_size*weights.sum())
    # With a single changing axis the quadratic phase separates. Each optimizer
    # step needs only two one-dimensional exponentials, even on a large image.
    def objective(params):
        py = np.exp((-2j*np.pi)*cross*params[0])
        px = np.exp((-2j*np.pi)*(columns*params[1]+square*params[2]))
        trace = np.einsum("i,ij->j", py, observations)
        total = np.dot(trace, px)*scale
        along_cross = np.einsum("i,ij->j", cross*py, observations)
        grad = np.array([np.dot(along_cross,px), np.dot(trace,px*columns), np.dot(trace,px*square)])
        grad *= (-2j*np.pi)*scale
        return -abs(total)**2, -2*(total.conjugate()*grad).real

    curvature = np.asarray(fit["hessian"])[2*axis]
    start = np.array([frequency[1-axis]*cross_size, frequency[axis]*sweep_size, curvature*sweep_size**2])
    width = np.array([.8,.8,4.])
    solved = minimize(objective, start, jac=True, method="L-BFGS-B",
                      bounds=list(zip(start-width, start+width)), options=dict(maxiter=30, ftol=1e-14, gtol=1e-11))
    if not solved.success:
        return None
    p = solved.x
    shape = np.asarray(observed.shape)
    freq = np.empty(2)
    freq[[1-axis,axis]] = p[:2]/[cross_size,sweep_size]
    hessian = np.zeros(3)
    hessian[2*axis] = p[2]/sweep_size**2
    yy,yx,xx = hessian-np.asarray(fit["hessian"])
    gradient = np.array([[yy,yx],[yx,xx]])
    corners = .5*(shape-1)*np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
    drift = np.linalg.norm((freq-frequency+corners@gradient)*shape, axis=1).max()
    if drift > .5:
        return None

    py = np.exp((-2j*np.pi)*cross*p[0])
    px = np.exp((-2j*np.pi)*(sweep*p[1]+.5*sweep**2*p[2]))
    demodulated = observations * py[:,None]*px[None,used]
    coefficient = scale*demodulated.sum()
    if abs(coefficient) == 0:
        return None
    phase = coefficient/abs(coefficient)
    by_quarter = [2*(part.sum()*phase.conjugate()).real/(len(part)*weights.sum())
                  for part in np.array_split(demodulated, 4, axis=0)]
    amplitude = abs(coefficient)-np.std(by_quarter, ddof=1)/2
    if amplitude <= 0:
        return None
    phasor = (py[:,None]*px)*phase.conjugate()
    if axis == 0:
        phasor = phasor.T
    model = dict(center_frequency=freq.tolist(), hessian=hessian.tolist(), amplitude=float(amplitude),
                 phase_radians=float(np.angle(phase)), exterior_weight_columns=float(weights.sum()),
                 maximum_corner_drift_bins=float(drift), transverse_amplitudes=by_quarter)
    return amplitude*phasor.real, phasor, model


def _admissible_trajectory(model, shape, axis, measured, bounds):
    """Check the continuous segment after refitting rather than just its central frequency."""
    delta = np.zeros(2)
    delta[axis] = .5*(shape[axis]-1)*model["hessian"][2*axis]
    start, end = np.asarray(model["center_frequency"])-delta, np.asarray(model["center_frequency"])+delta
    nearest_dc = float(_segment_distances(np.zeros((1,2)), start, end)[0])
    radius_max = max(np.linalg.norm(start), np.linalg.norm(end))
    clearance = _segment_distances(measured, start*shape, end*shape).min()
    return nearest_dc >= bounds[0] and radius_max <= bounds[1] and clearance > LATTICE_TOLERANCE_BINS


def review(observed, diag, *, bounds=(.025, .45)):
    """Return (restored, applied wave, evidence), or (None, None, evidence) on abstention.

    Geographically disjoint clear strips estimate a complex coefficient. Both halves
    of every guarded strip must agree with it, and the old fit's held tiles must
    corroborate the signed clipping phase. Any failed strip keeps the old solution.
    """
    record = dict(bands=[])
    def reject(reason):
        return None, None, dict(record, reason=reason)

    fit = diag.get("evidence", {}).get("guarded_carrier", {})
    if (diag.get("reason") != "clip_supported_independent_carrier" or fit.get("model") != "quadratic"
            or not fit["protection"].get("protected_pixels")):
        return reject("no_verified_chirp_with_protected_core")
    shape = np.asarray(observed.shape)
    h, w = observed.shape
    yy, yx, xx = fit["hessian"]
    hessian = np.array([[yy, yx], [yx, xx]])
    gradient_per_pixel = hessian*shape[:, None]
    excursion = np.linalg.norm(gradient_per_pixel*(shape-1), axis=0)
    axis = int(np.argmax(excursion))
    transverse = 1-axis
    if excursion[transverse] > MAX_TRANSVERSE_DRIFT or excursion[axis] <= 2*(LATTICE_TOLERANCE_BINS+TAPER_BINS):
        return reject("not_a_resolved_single_axis_sweep")
    record["sweep_axis"] = axis

    y, x = np.ogrid[:h, :w]
    y = y-(h-1)/2
    x = x-(w-1)/2
    fy, fx = fit["center_frequency"]
    phase = 2*np.pi*(fy*y+fx*x+.5*yy*y*y+yx*y*x+.5*xx*x*x)+fit["final_phase_radians"]
    phasor = np.exp(-1j*phase)
    amplitude = fit["amplitude"]
    full_wave = amplitude*phasor.real
    # The old wave equals the full wave times a real number in [0,1]. Avoid division at zero crossings.
    different = np.abs(diag["carrier_wave"]-full_wave) > amplitude*1e-7
    guarded = np.any(different, axis=transverse)
    if np.mean(~guarded) < MIN_CLEAR_FRACTION:
        return reject("insufficient_unguarded_reference")
    runs = np.flatnonzero(np.diff(np.r_[False, guarded, False])).reshape(-1, 2)
    if len(runs) == 0 or np.any(np.diff(runs, axis=1) < MIN_WIDTH):
        return reject("unresolved_guarded_band")

    # Both previous routes fit the same lattice to this observation. Reuse its
    # measured peaks before considering a fresh fit for an older diagnostic record.
    ev = diag["evidence"]
    cached = ev.get("weak_carrier", {}).get("measured_lattice_frequencies")
    if cached is None:
        cached = ev.get("lattice_screen", {}).get("measured_lattice_peaks")
    if cached is None:
        centers, _, power = _spectrum(observed)
        members, _ = _fit_lattice(centers, power, shape)
        if members is None:
            return reject("no_measurable_texture_peaks")
        centers = centers[members]
    else:
        centers = np.asarray(cached)*shape if len(cached) else np.empty((0, 2))
        if not len(centers):
            return reject("no_measurable_texture_peaks")
    # The unfiltered spectrum also contains broad maxima of the candidate chirp.
    # A measured member of the original protecting lattice keeps its veto. A
    # persistent off-lattice tone is tested after subtraction within each strip.
    measured = np.vstack((centers, -centers))
    center_bins = np.asarray(fit["center_frequency"])*shape
    mid = (shape[axis]-1)/2
    uncertainty = excursion[transverse]/2
    segments = []
    for lo, hi in runs:
        start = center_bins + (lo-mid)*gradient_per_pixel[:, axis]
        end = center_bins + (hi-1-mid)*gradient_per_pixel[:, axis]
        gap = float(_segment_distances(measured, start, end).min()-uncertainty)
        local = dict(start=int(lo), stop=int(hi), minimum_measured_clearance_bins=gap)
        record["bands"].append(local)
        if gap <= LATTICE_TOLERANCE_BINS:
            return reject("collision_with_measured_peak")
        segments.append((start, end))

    signal = observed-gaussian_filter(observed, 5.)
    axes = [np.searchsorted(np.linspace(0,n,5,dtype=int)[1:-1], np.arange(n), side="right") for n in shape]
    training = (axes[0][:, None]+axes[1])%2 == 0
    clear = np.broadcast_to((~guarded)[:, None] if axis == 0 else ~guarded, observed.shape)
    exterior = clear & training
    if exterior.sum() < observed.size*MIN_CLEAR_FRACTION/4:
        return reject("insufficient_exterior_training_area")
    normalized = (2/amplitude)*signal*phasor
    anchor = normalized[exterior].mean()
    record["exterior_relative_coefficient"] = [float(anchor.real), float(anchor.imag)]
    if abs(anchor-1) > MAX_COEFFICIENT_ERROR:
        return reject("unguarded_brightness_disagrees")
    events = (observed==1.).astype(np.int8) - (observed==0.)
    signed_phase = events*phasor.real
    for (lo, hi), (start, end), detail in zip(runs, segments, record["bands"]):
        strip = (slice(lo,hi), slice(None)) if axis == 0 else (slice(None), slice(lo,hi))
        coeffs = np.array([half.mean() for half in np.array_split(normalized[strip], 2, axis=transverse)])
        detail["transverse_relative_coefficients"] = [[float(c.real),float(c.imag)] for c in coeffs]
        if (np.max(abs(coeffs-anchor)) > MAX_COEFFICIENT_ERROR*abs(anchor)
                or np.max(abs(coeffs-1)) > MAX_COEFFICIENT_ERROR):
            return reject("local_brightness_disagrees")
        held = (events[strip]!=0) & ~training[strip]
        count = int(held.sum())
        detail["held_endpoint_count"] = count
        if count < MIN_QUADRANT_EVENTS:
            return reject("insufficient_local_held_endpoints")
        alignment = float(signed_phase[strip][held].mean())
        detail["held_endpoint_alignment"] = alignment
        if alignment < MIN_ENDPOINT_ALIGNMENT:
            return reject("local_endpoint_phase_disagrees")
        raw = observed[strip]-full_wave[strip]
        ratio = _residual_tone(raw, start, end, (h,w), LATTICE_TOLERANCE_BINS+uncertainty)/amplitude
        detail["local_stationary_residual_ratio"] = ratio
        if ratio >= MAX_COEFFICIENT_ERROR:
            return reject("persistent_local_periodic_texture")
    # Texture confined to a collision band can pull an all-image phase fit
    # towards itself. Only the geographically disjoint fit is used for removal.
    exterior_fit = _exterior_fit(observed, guarded, axis, fit)
    if exterior_fit is None:
        return reject("inconclusive_exterior_phase_fit")
    wave, independent_phase, applied_model = exterior_fit
    if not _admissible_trajectory(applied_model, shape, axis, measured, bounds):
        return reject("exterior_model_frequency_outside_safe_band")
    if np.sum((wave-full_wave)**2) > MAX_COEFFICIENT_ERROR**2*np.sum(full_wave**2):
        return reject("exterior_model_disagrees_with_verified_fit")
    independent_brightness = 2*signal*independent_phase/applied_model["amplitude"]
    signed_phase = events*independent_phase.real
    for (lo,hi), detail in zip(runs,record["bands"]):
        strip = (slice(lo,hi), slice(None)) if axis == 0 else (slice(None), slice(lo,hi))
        coeffs = [half.mean() for half in np.array_split(independent_brightness[strip], 2, axis=transverse)]
        held = (events[strip]!=0) & ~training[strip]
        alignment = float(signed_phase[strip][held].mean())
        detail.update(exterior_fit_half_coefficients=[[float(c.real),float(c.imag)] for c in coeffs],
                      exterior_fit_endpoint_alignment=alignment)
        if max(abs(c-1) for c in coeffs) > MAX_COEFFICIENT_ERROR or alignment < MIN_ENDPOINT_ALIGNMENT:
            return reject("exterior_model_failed_local_confirmation")
    gain = diag["accepted_regions"][0]["gain"]
    return _restore(observed, wave, gain), wave, dict(record, applied_model=applied_model,
                                                      reason="all_inferred_bands_verified")
