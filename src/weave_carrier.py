"""Estimate a broad non-lattice carrier and verify its phase in separate patches."""
from itertools import combinations, product

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize

from local_spectral import _delta, _detrend
from weave_guard import LATTICE_TOLERANCE_BINS, MAX_ORDER


PEAKS_PER_PATCH = 2
PEAK_CLEARANCE_BINS = 1.5
FREQUENCY_ERROR_BINS = .45
MAX_COEFFICIENT_ERROR = .45
MIN_AMPLITUDE = .02
MIN_CLEAR_PATCHES = 4
MIN_AGREEING_FRACTION = .75


def _patch_peaks(patch, window, protected, bounds):
    size = len(patch)
    power = np.abs(np.fft.fft2(patch * window, s=(2*size, 2*size)))
    bins = np.fft.fftfreq(2*size)
    fy, fx = np.meshgrid(bins, bins, indexing="ij")
    radius = np.hypot(fy, fx)
    maxima = (power == maximum_filter(power, size=5, mode="wrap")) & (radius >= bounds[0]) & (radius <= bounds[1])
    locations = np.argwhere(maxima & ((fy > 0) | ((fy == 0) & (fx > 0))))
    ranked = sorted(locations, key=lambda p: -power[tuple(p)])
    logmag = np.log(np.maximum(power, 1e-20))
    peaks = []
    for y, x in ranked:
        if 2*power[y, x]/window.sum() < MIN_AMPLITUDE:
            break
        f = np.array([bins[y] + _delta(logmag, y, x, 0)/(2*size),
                      bins[x] + _delta(logmag, y, x, 1)/(2*size)])
        distance = np.linalg.norm(protected-f, axis=1)
        opposite = np.linalg.norm(protected+f, axis=1)
        if np.minimum(distance, opposite).min() * size >= PEAK_CLEARANCE_BINS:
            peaks.append(f)
            if len(peaks) == PEAKS_PER_PATCH:
                break
    return peaks


def _frequency_design(centers):
    """Gradient of a shared quadratic phase with a symmetric Hessian."""
    design = np.zeros((len(centers), 2, 5))
    y, x = centers.T
    design[:, 0, 0] = design[:, 1, 1] = 1
    design[:, 0, 2] = design[:, 1, 3] = y
    design[:, 0, 3] = design[:, 1, 4] = x
    return design


def _proposals(design, peaks, train, size):
    """Use proposals from three spatially separated fitting patches, then score all fitting patches."""
    hypotheses = []
    for triple in combinations(np.flatnonzero(train), 3):
        indices = list(triple)
        a = design[indices].reshape(6, 5)
        if np.linalg.matrix_rank(a) < 5:
            continue
        for subset in product(*(peaks[i] for i in indices)):
            f = np.array(subset)
            f[(f @ f[0]) < 0] *= -1
            hypotheses.append(np.linalg.lstsq(a, f.ravel(), rcond=None)[0])
    if not hypotheses:
        return []
    candidates = np.array(hypotheses)
    predicted = np.einsum("pdj,sj->spd", design[train], candidates)
    options = np.full((train.sum(), PEAKS_PER_PATCH, 2), np.inf)
    for row, i in enumerate(np.flatnonzero(train)):
        if peaks[i]:
            options[row, :len(peaks[i])] = peaks[i]
    direct = np.linalg.norm(predicted[:, :, None]-options, axis=-1)
    flipped = np.linalg.norm(predicted[:, :, None]+options, axis=-1)
    distances = np.minimum(direct, flipped).min(axis=2) * size
    hits = (distances <= FREQUENCY_ERROR_BINS).sum(axis=1)
    ranks = np.lexsort((np.minimum(distances, 1.).sum(axis=1), -hits))
    best = []
    for i in ranks:
        if hits[i] < MIN_CLEAR_PATCHES or len(best) == 3:
            break
        similar = any(np.minimum(np.linalg.norm(predicted[i]-predicted[j], axis=1),
                                 np.linalg.norm(predicted[i]+predicted[j], axis=1)).mean() * size < .35 for j in best)
        if not similar:
            best.append(i)
    return candidates[best]


def _phase_basis(y, x):
    return np.stack(np.broadcast_arrays(y, x, .5*y*y, y*x, .5*x*x), axis=-1)


def _clear_patches(model, design, protected, size):
    predicted = design @ model
    distance = np.minimum(np.linalg.norm(predicted[:, None]-protected, axis=2),
                          np.linalg.norm(predicted[:, None]+protected, axis=2)).min(axis=1)
    return distance * size >= PEAK_CLEARANCE_BINS


def _fit_and_verify(model, patches, basis, window, train, design, protected, size, length, *, enable_chirp=True):
    # Exclude lattice collisions from the phase fit and from the verification.
    if not enable_chirp:
        model = model.copy()
        model[2:] = 0
    clear = _clear_patches(model, design, protected, size)
    fitting, checking = clear & train, clear & ~train
    if min(fitting.sum(), checking.sum()) < MIN_CLEAR_PATCHES:
        return None
    coordinates = basis[fitting].reshape(-1, 5)
    signal = (patches[fitting] * window).reshape(-1) * (2/(window.sum()*fitting.sum()))
    # Fit phase on alternating disjoint patches. The free complex coefficient
    # absorbs phase offset. Optimize cycles across the image, for stable units.
    def objective(parameters):
        demodulated = signal * np.exp((-2j*np.pi) * (coordinates @ parameters))
        total = demodulated.sum()
        gradient = -2*np.real(np.conj(total)*(-2j*np.pi)*(demodulated @ coordinates))
        return -abs(total)**2, gradient

    start = model * length
    parameter_bounds = [(v-2, v+2) for v in start]
    if not enable_chirp:
        parameter_bounds[2:] = [(0, 0)] * 3
    opt = minimize(objective, start, jac=True, method="L-BFGS-B",
                   bounds=parameter_bounds, options=dict(maxiter=40, gtol=1e-8, ftol=1e-12))
    params = opt.x
    checking &= _clear_patches(params / length, design, protected, size)
    if checking.sum() < MIN_CLEAR_PATCHES:
        return None
    angles = basis @ params
    coefficients = (patches*window*np.exp(-2j*np.pi*angles)).sum(axis=(1, 2)) * (2/window.sum())
    coefficient = coefficients[fitting].mean()
    amplitude = float(abs(coefficient))
    disagreement = abs(coefficients[checking]-coefficient)/max(amplitude, 1e-12)
    agreeing = disagreement <= MAX_COEFFICIENT_ERROR
    if amplitude < MIN_AMPLITUDE or agreeing.mean() < MIN_AGREEING_FRACTION:
        return None
    return params, coefficient, dict(amplitude=amplitude, fitting_patches=int(fitting.sum()),
                                    checking_patches=int(checking.sum()), agreeing_patches=int(agreeing.sum()),
                                    median_relative_coefficient_error=float(np.median(disagreement)))


def _follows_lattice(field, lattice_basis, shape):
    # A coherent but weak harmonic can be absent from the list of measured
    # strong peaks. It still belongs to the original reciprocal lattice.
    indices = np.arange(-MAX_ORDER, MAX_ORDER+1)
    orders = np.stack(np.meshgrid(indices, indices), axis=-1).reshape(-1, 2)
    nodes = orders[np.any(orders != 0, axis=1)] @ lattice_basis
    distances = np.linalg.norm((field[:, None]-nodes)*shape, axis=2)
    return np.any((distances <= LATTICE_TOLERANCE_BINS).mean(axis=0) >= .5)


def independent_carrier(observed, protected, lattice_basis, bounds, *, enable_chirp=True):
    """Return a verified wave in intensity units, or None when the texture is ambiguous."""
    h, w = observed.shape
    size = min(80, min(h, w)//4)
    if size < 32 or len(protected) == 0:
        return None, dict(reason="insufficient_patch_extent")
    length = max(h, w)
    window = np.outer(np.hanning(size), np.hanning(size))
    origins = product(np.linspace(0, h-size, 4, dtype=int), np.linspace(0, w-size, 4, dtype=int))
    patches, peaks, basis, centers = [], [], [], []
    for y, x in origins:
        patch = _detrend(observed[y:y+size, x:x+size])
        patches.append(patch)
        peaks.append(_patch_peaks(patch, window, protected, bounds))
        dy = (y+np.arange(size)[:, None]-(h-1)/2)/length
        dx = (x+np.arange(size)[None, :]-(w-1)/2)/length
        basis.append(_phase_basis(dy, dx))
        centers.append((dy.mean(), dx.mean()))
    design = _frequency_design(np.array(centers))
    train = np.indices((4, 4)).sum(axis=0).ravel() % 2 == 0
    hypotheses = _proposals(design, peaks, train, size)
    confirmed = []
    patches, basis = np.asarray(patches), np.asarray(basis)
    harmonic_models = 0
    for model in hypotheses:
        fit = _fit_and_verify(model, patches, basis, window, train, design,
                              protected, size, length, enable_chirp=enable_chirp)
        if fit is not None:
            field = design @ (fit[0] / length)
            if _follows_lattice(field, lattice_basis, observed.shape):
                harmonic_models += 1
            else:
                # Separate coarse seeds can converge to the same frequency
                # field. Count it once within one full-image FFT bin.
                for old in confirmed:
                    other = design @ (old[0] / length)
                    separation = np.minimum(np.linalg.norm((field-other)*observed.shape, axis=1),
                                            np.linalg.norm((field+other)*observed.shape, axis=1))
                    if separation.max() <= 1:
                        break
                else:
                    confirmed.append(fit)
    details = dict(proposed_models=len(hypotheses), verified_models=len(confirmed),
                   lattice_models_rejected=harmonic_models, patch_size=size)
    if len(confirmed) != 1:
        return None, dict(**details, reason="no_unique_verified_carrier")
    params, coefficient, record = confirmed[0]
    dy = (np.arange(h)[:, None]-(h-1)/2)/length
    dx = (np.arange(w)[None, :]-(w-1)/2)/length
    fy, fx, yy, yx, xx = params / length
    radius = np.hypot(fy+yy*dy+yx*dx, fx+yx*dy+xx*dx)
    if radius.min() < bounds[0] or radius.max() > bounds[1]:
        return None, dict(**details, reason="carrier_outside_frequency_bounds")
    cycles = length*(fy*dy+fx*dx+.5*yy*dy**2+yx*dy*dx+.5*xx*dx**2)
    wave = np.real(coefficient*np.exp(2j*np.pi*cycles))
    return wave, dict(**details, **record, reason="verified_non_lattice_carrier",
                      center_frequency=[float(fy), float(fx)], hessian=(np.array([yy, yx, xx])/length).tolist())
