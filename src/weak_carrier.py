"""Find a broad, phase-coherent carrier in an observed two-dimensional periodic texture."""
from itertools import product

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize
from scipy.signal.windows import tukey

from local_spectral import _detrend
from weave_carrier import MAX_COEFFICIENT_ERROR, MIN_AGREEING_FRACTION, MIN_AMPLITUDE
from weave_guard import LATTICE_TOLERANCE_BINS, MAX_ORDER, _fit_lattice, _spectrum


MIN_SIDE, MAX_SIDE = 128, 512
SWEEP_STEP, MAX_SWEEP = 4, 24  # Fourier bins traversed across the changing axis.
SEEDS_PER_SPECTRUM, MAX_FITS = 16, 12


def routing_parameters():
    return dict(min_side=MIN_SIDE, max_side=MAX_SIDE, sweep_step_bins=SWEEP_STEP,
                maximum_sweep_bins=MAX_SWEEP, peaks_per_spectrum=SEEDS_PER_SPECTRUM,
                maximum_fits_per_stage=MAX_FITS, grid=[4, 4], validation_window_tukey_alpha=.25,
                max_relative_coefficient_error=MAX_COEFFICIENT_ERROR,
                minimum_agreeing_fraction_per_axis=MIN_AGREEING_FRACTION,
                minimum_amplitude=MIN_AMPLITUDE, clearance_full_image_bins=LATTICE_TOLERANCE_BINS,
                maximum_final_frequency_change_bins=.5, final_amplitude_cap="training amplitude")


def _measurements(image):
    """Tile the entire rectangle, including odd sizes, into disjoint fitting and checking cells."""
    h, w = image.shape
    y, x = np.indices(image.shape, dtype=float)
    y = (y-(h-1)/2)/h
    x = (x-(w-1)/2)/w
    basis = np.stack((y, x, .5*y*y, .5*x*x), axis=-1)
    cells = np.empty(image.shape, dtype=np.int8)
    signal = np.empty_like(image)
    weights = np.empty_like(image)
    discovery = np.zeros_like(image)
    edges_y, edges_x = np.linspace(0,h,5,dtype=int), np.linspace(0,w,5,dtype=int)
    for row, col in product(range(4), repeat=2):
        box = np.s_[edges_y[row]:edges_y[row+1], edges_x[col]:edges_x[col+1]]
        patch = _detrend(image[box])
        sh, sw = patch.shape
        cells[box] = 4*row+col
        signal[box] = patch
        # A broad flat taper integrates adjacent texture frequencies with greater
        # spatial aperture. Use the smoother Hann exclusively for FFT proposals.
        weights[box] = np.outer(tukey(sh,.25), tukey(sw,.25))
        if (row+col)%2 == 0:
            discovery[box] = patch*np.outer(np.hanning(sh), np.hanning(sw))
    return basis, cells, signal, weights, discovery


def _field(parameters, shape):
    """All distinct instantaneous frequencies for a stationary or single-axis quadratic wave."""
    shape = np.asarray(shape)
    axis = int(abs(parameters[3]) > abs(parameters[2]))
    positions = (np.arange(shape[axis])-(shape[axis]-1)/2)/shape[axis]
    field = np.broadcast_to(parameters[:2]/shape, (len(positions),2)).copy()
    field[:,axis] += parameters[2+axis]*positions/shape[axis]
    return field


def _clear(parameters, shape, protected_bins, bounds):
    field = _field(parameters, shape)
    radius = np.linalg.norm(field,axis=1)
    if radius.min() < bounds[0] or radius.max() > bounds[1]:
        return False
    # Exact point-to-segment distance also protects collisions between pixel
    # centers. Both signs of every measured peak and lattice node are supplied.
    ends = field[[0,-1]]*shape
    direction = ends[1]-ends[0]
    relative = protected_bins-ends[0]
    fraction = np.clip(relative@direction / max(direction@direction,1e-20),0,1)
    return bool(np.min(np.linalg.norm(relative-fraction[:,None]*direction,axis=1)) > LATTICE_TOLERANCE_BINS)


def _seeds(discovery, basis, protected, bounds, curved):
    shape = np.array(discovery.shape)
    fy, fx = np.meshgrid(*(np.fft.fftfreq(n) for n in shape),indexing="ij")
    band = (np.hypot(fy,fx) >= bounds[0]) & (np.hypot(fy,fx) <= bounds[1])
    rates = [(0,0)] if not curved else [(axis,sweep) for axis in (0,1)
             for sweep in range(SWEEP_STEP,MAX_SWEEP+1,SWEEP_STEP)]
    seeds = []
    for axis, sweep in rates:
        # Searching both signs of frequency suffices for signed sweeps of a real signal.
        dechirped = discovery*np.exp((-2j*np.pi*sweep)*basis[:,:,2+axis]) if sweep else discovery
        power = abs(np.fft.fft2(dechirped))
        peaks = band & (power == maximum_filter(power,7,mode="wrap"))
        if not curved:
            peaks &= (fy>0)|((fy==0)&(fx>0))
        locations = np.argwhere(peaks)
        order = np.argsort(-power[peaks],kind="stable")[:SEEDS_PER_SPECTRUM]
        for iy, ix in locations[order]:
            parameters = np.array([fy[iy,ix]*shape[0],fx[iy,ix]*shape[1],0.,0.])
            parameters[2+axis] = sweep
            if parameters[0]<0 or (parameters[0]==0 and parameters[1]<0):
                parameters *= -1
            # Only screen exact locked stationary seeds here. A coarse chirp can
            # cross a node which the refined carrier resolves, or vice versa.
            if not curved and not _clear(parameters, shape, protected, bounds):
                continue
            seeds.append((float(power[iy,ix]), parameters, axis))
    seeds.sort(key=lambda item:-item[0])
    return seeds[:MAX_FITS]


def _optimize(seed, axis, coords, weighted_signal):
    active = [0,1,2+axis] if seed[2+axis] else [0,1]
    coords = np.ascontiguousarray(coords[:,active])
    def objective(params):
        demod = weighted_signal*np.exp((-2j*np.pi)*(coords@params))
        total = demod.sum()
        grad = -2*np.real(np.conj(total)*(-2j*np.pi)*(demod@coords))
        return -abs(total)**2, grad
    center = seed[active]
    width = np.array([1.5,1.5,SWEEP_STEP,SWEEP_STEP])[active]
    lower, upper = center-width, center+width
    if len(active)==3:
        lower[-1], upper[-1] = max(lower[-1],-MAX_SWEEP), min(upper[-1],MAX_SWEEP)
    fit = minimize(objective,center,method="L-BFGS-B",jac=True,bounds=list(zip(lower,upper)),
                   options=dict(maxiter=40,gtol=1e-8,ftol=1e-11))
    params = seed.copy()
    params[active] = fit.x
    return params


def _verify(params, basis, cells, signal, weights, training, *, coefficient=None):
    angle = basis@params
    phasor = np.exp((-2j*np.pi)*angle)
    samples = (2*weights*signal*phasor).ravel()
    cell = cells.ravel()
    coefficients = (np.bincount(cell,weights=samples.real)+1j*np.bincount(cell,weights=samples.imag))
    coefficients /= np.bincount(cell,weights=weights.ravel())
    fitted = coefficients[training].mean() if coefficient is None else coefficient
    if abs(fitted) < MIN_AMPLITUDE:
        return None
    relative = coefficients/fitted
    # The fitted sine must reduce weighted squared error on *each* withheld
    # cell, even where the background has a similar tangent frequency.
    prediction = np.real(fitted*phasor.conj())
    gain = np.bincount(cell,weights=(weights*prediction*(2*signal-prediction)).ravel())
    if np.any(gain[~training] <= 0) or np.any(relative[training].real <= .25):
        return None
    # Within each row (and separately each column), pool its two disjoint
    # held-out cells. Their longer aperture resolves close one-axis frequencies.
    # These are two views of the same eight held-out observations.
    checking = np.where(training, np.nan+0j, relative).reshape(4,4)
    errors = [abs(np.nanmean(checking,axis=axis)-1) for axis in (1,0)]
    if any(np.mean(e <= MAX_COEFFICIENT_ERROR) < MIN_AGREEING_FRACTION for e in errors):
        return None
    return dict(amplitude=float(abs(fitted)), fitting_cells=8, checking_cells=8,
                positive_gain_checking_cells=int(np.sum(gain[~training]>0)),
                checking_row_errors=errors[0].tolist(), checking_column_errors=errors[1].tolist()), fitted


def _final_wave(params, coefficient, basis, weights, signal, cells, training, shape, protected, bounds):
    """Calibrate a unique, already validated model within its frequency-resolution cell."""
    axis = int(abs(params[3]) > abs(params[2]))
    fit = _optimize(params,axis,basis.reshape(-1,4),2*(signal*weights).ravel()/weights.sum())
    corners = np.array(list(product((-.5,.5),repeat=2)))*(shape-1)/shape
    shift = np.linalg.norm((fit[:2]-params[:2])+corners*(fit[2:]-params[2:]),axis=1).max()
    record = dict(training_amplitude=float(abs(coefficient)), phase_refined=False,
                  final_frequency_change_bins=0., final_check=None)
    if shift<=.5 and _clear(fit,shape,protected,bounds):
        carrier = np.exp((2j*np.pi)*(basis@fit))
        measured = (weights*signal*carrier.conj()).sum()
        if abs(measured):
            phase = measured/abs(measured)
            unit = np.real(phase*carrier)
            # Project onto a fixed phase on all pixels. Validation cannot
            # authorize larger amplitude than the original training estimate.
            amplitude = np.clip(np.sum(weights*signal*unit)/np.sum(weights*unit**2),0.,abs(coefficient))
            checked = _verify(fit,basis,cells,signal,weights,training,coefficient=amplitude*phase)
            if checked is not None:
                record.update(amplitude=float(amplitude), phase_refined=True, final_frequency_change_bins=float(shift),
                              final_check=checked[0])
                return fit, amplitude*unit, record
    record["amplitude"] = float(abs(coefficient))
    return params, np.real(coefficient*np.exp((2j*np.pi)*(basis@params))), record


def find_carrier(observed, bounds, *, enable_chirp=True):
    """Return a full-field verified sinusoid in luminance units, or None with a reason."""
    image = np.asarray(observed,dtype=float)
    h,w = image.shape
    if min(h,w)<MIN_SIDE or max(h,w)>MAX_SIDE:
        return None, dict(reason="outside_full_field_search_geometry")
    shape = np.array(image.shape)
    peaks, _, power = _spectrum(image)
    members, lattice = _fit_lattice(peaks,power,shape)
    if lattice is None:
        return None, dict(reason="no_resolved_texture_lattice")
    orders = np.arange(-MAX_ORDER,MAX_ORDER+1)
    pairs = np.stack(np.meshgrid(orders,orders),axis=-1).reshape(-1,2)
    nodes = pairs[np.any(pairs,axis=1)]@lattice
    protected = np.vstack((peaks[members],-peaks[members],nodes))
    basis,cells,signal,weights,discovery = _measurements(image)
    training = np.indices((4,4)).sum(axis=0).ravel()%2 == 0
    train = training[cells]
    coords = basis[train]
    train_signal = 2*(signal*weights)[train]/weights[train].sum()
    details = dict(patch_size=min(h,w)//4, patch_grid=[4,4], lattice_peaks=int(members.sum()),
                   measured_lattice_frequencies=(peaks[members]/shape).tolist(), stages=[])
    for curved in ([False,True] if enable_chirp else [False]):
        seeds = _seeds(discovery,basis,protected,bounds,curved)
        fits = []
        for _, initial, axis in seeds:
            params = _optimize(initial,axis,coords,train_signal)
            if curved and abs(params[2+axis])<SWEEP_STEP:
                continue
            if not _clear(params,shape,protected,bounds):
                continue
            verified = _verify(params,basis,cells,signal,weights,training)
            if verified is None:
                continue
            # Different coarse sweeps can converge to the same phase. Compare
            # endpoints at all image corners, in Fourier bins, up to conjugation.
            corners = np.array([[-.5,-.5],[-.5,.5],[.5,-.5],[.5,.5]])
            field = params[:2]+corners*params[2:]
            if any(min(np.linalg.norm(field-sign*other,axis=1).max() for sign in (-1,1))<=1
                   for other, _, _ in fits):
                continue
            fits.append((field,params,verified))
        details["stages"].append(dict(model="axial_chirp" if curved else "stationary", seeds=len(seeds),
                                      verified_models=len(fits)))
        if len(fits)>1:
            return None, dict(**details,reason="ambiguous_verified_carriers")
        if fits:
            _,params,(record,coefficient) = fits[0]
            fit,wave,calibration = _final_wave(params,coefficient,basis,weights,signal,cells,training,
                                               shape,protected,bounds)
            record.update(calibration)
            return wave, dict(**details,**record,reason="verified_full_field_carrier",
                      model="axial_chirp" if curved else "stationary",
                      training_center_frequency=(params[:2]/shape).tolist(),
                      training_hessian=[float(params[2]/h**2),0.,float(params[3]/w**2)],
                      center_frequency=(fit[:2]/shape).tolist(),
                      hessian=[float(fit[2]/h**2),0.,float(fit[3]/w**2)],
                      support_policy="whole image, all eight checking cells have positive prediction gain; "
                                     "continuous frequency path clear of protected lattice nodes")
    return None, dict(**details,reason="no_verified_full_field_carrier")
