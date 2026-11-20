"""Verify a single-axis chirp in the presence of local, stationary weave tones."""
import numpy as np
from scipy.fft import fft2
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from collision_review import MIN_WIDTH, _admissible_trajectory, _segment_distances
from guarded_carrier import (MIN_ENDPOINT_ALIGNMENT, MIN_EVENTS, MIN_QUADRANT_ALIGNMENT,
                             MIN_QUADRANT_EVENTS, TAPER_BINS, _node_bins)
from nine_chirp import CURVATURE_STEP, MAX_CURVATURE
from project_guarded import _restore
from weak_carrier import MAX_SIDE, MIN_SIDE
from weave_carrier import MAX_COEFFICIENT_ERROR
from weave_guard import LATTICE_TOLERANCE_BINS, _fit_lattice, _spectrum


RETRY_REASONS = ("inconclusive_exterior_phase_fit", "exterior_model_failed_local_confirmation")


def routing_parameters():
    return dict(version="conditional_exterior_chirp_v1", curvature_step_bins=CURVATURE_STEP,
                endpoint_alignment=MIN_ENDPOINT_ALIGNMENT, quadrant_alignment=MIN_QUADRANT_ALIGNMENT,
                collision_radius_bins=LATTICE_TOLERANCE_BINS, taper_bins=TAPER_BINS,
                maximum_frequency_shift_bins=.5, minimum_resolved_rms_fraction=.25,
                coefficient_tolerance=MAX_COEFFICIENT_ERROR, minimum_exterior_fraction=.125,
                minimum_distinct_clear_strips=2)


def _axis_seed(events, train, coordinates, bounds):
    """Only the two axial families are proposed. Refinement uses sparse training endpoints."""
    shape = np.array(events.shape)
    positions = np.nonzero(train & (events != 0))
    signs = events[positions]
    if len(signs) < MIN_EVENTS//2:
        return None
    coords = np.column_stack([v[i] for v,i in zip(coordinates, positions)])
    frequencies = [np.fft.fftfreq(n) for n in shape]
    radius = np.hypot(frequencies[0][:,None], frequencies[1])
    allowed = (radius >= bounds[0]) & (radius <= bounds[1]) & (frequencies[0][:,None] >= 0)
    work = (events*train).astype(np.float32)
    candidates = []
    for axis in (0,1):
        for sweep in range(-MAX_CURVATURE, MAX_CURVATURE+1, CURVATURE_STEP):
            if sweep == 0:
                continue
            rot = np.exp((-1j*np.pi*sweep)*coordinates[axis]**2).astype(np.complex64)
            spectrum = abs(fft2(work*(rot if axis else rot[:,None])))
            spectrum[~allowed] = 0
            i,j = np.unravel_index(np.argmax(spectrum), events.shape)
            candidates.append((spectrum[i,j], axis, frequencies[0][i]*shape[0],
                               frequencies[1][j]*shape[1], sweep))
    # Prefer distinct coarse peaks before optimizing endpoint alignment. A single
    # narrow stationary tone can otherwise crowd out a broad, changing carrier.
    selected = []
    for candidate in sorted(candidates, reverse=True):
        _,axis,fy,fx,_ = candidate
        if not any(axis==other[1] and np.hypot(fy-other[2],fx-other[3])<2 for other in selected):
            selected.append(candidate)
            if len(selected)==4:
                break
    best = None
    for _,axis,fy,fx,sweep in selected:
        basis = np.column_stack((coords,.5*coords[:,axis]**2))
        def loss(p):
            demod = signs*np.exp((-2j*np.pi)*(basis@p))/len(signs)
            total = demod.sum()
            return -abs(total)**2, -2*np.real(total.conjugate()*(-2j*np.pi)*(demod@basis))
        initial = np.array([fy,fx,sweep])
        width = np.array([1.5,1.5,CURVATURE_STEP])
        solution = minimize(loss,initial,method="L-BFGS-B",jac=True,
                            bounds=list(zip(initial-width,initial+width)))
        if solution.success and (best is None or solution.fun < best[0]):
            best = solution.fun, axis, solution.x
    if best is None or -best[0] < MIN_ENDPOINT_ALIGNMENT**2:
        return None
    return best[1], best[2]


def _external_wave(observed, axis, coordinates, weights, center, sweep):
    data = observed.T if axis == 0 else observed
    cross, along = coordinates[1-axis], coordinates[axis]
    keep = weights>0
    u = along[keep]
    data = data[:,keep]
    weighted = (data-gaussian_filter1d(data,5.,axis=0))*weights[keep]
    curved = np.ascontiguousarray(weighted*np.exp((-1j*np.pi*sweep)*u**2))
    scale = 2/(len(cross)*weights.sum())
    def loss(p):
        y = np.exp((-2j*np.pi*p[0])*cross)
        x = np.exp((-2j*np.pi*p[1])*u)
        trace = np.einsum("i,ij->j",y,curved)
        total = scale*(trace@x)
        derivative = (-2j*np.pi)*scale*np.array([
            np.einsum("i,ij->j",y*cross,curved)@x, trace@(x*u)])
        return -abs(total)**2, -2*(total.conjugate()*derivative).real
    initial = center[[1-axis,axis]]
    fit = minimize(loss,initial,jac=True,method="L-BFGS-B",bounds=list(zip(initial-1,initial+1)),
                   options=dict(ftol=1e-14,gtol=1e-10))
    if not fit.success:
        return None
    py,px = fit.x
    z = np.exp((-2j*np.pi*py)*cross[:,None])*np.exp((-2j*np.pi)*(px*along+.5*sweep*along**2))
    samples = z[:,keep]*weighted
    coefficient = scale*samples.sum()
    if abs(coefficient)==0:
        return None
    phase = coefficient/abs(coefficient)
    quarters = [2*(q.sum()*phase.conjugate()).real/(len(q)*weights.sum()) for q in np.array_split(samples,4)]
    amplitude = abs(coefficient)-np.std(quarters,ddof=1)/2
    if amplitude<=0:
        return None
    shape = np.asarray(observed.shape)
    frequency = np.empty(2)
    frequency[[1-axis,axis]] = fit.x/shape[[1-axis,axis]]
    hessian = np.zeros(3)
    hessian[2*axis] = sweep/shape[axis]**2
    model = dict(center_frequency=frequency.tolist(), hessian=hessian.tolist(), amplitude=float(amplitude),
                 phase_radians=float(np.angle(phase)))
    unit = (z*phase.conjugate()).conj()
    return (unit.T if axis==0 else unit),model


def _conditional_band(signal, unit, atom, events, held, axis, amplitude):
    """Check the chirp after projecting out a stationary sinusoid at the inferred weave node.

    The two hypotheses must be distinguishable. In each transverse half, compare
    the remaining observed carrier with its exterior prediction. A nuisance
    coefficient learned on the other half predicts clipping signs on held pixels.
    Only the exterior carrier is ever subtracted from the photograph.
    """
    B = np.stack((atom.real,atom.imag),axis=-1)
    S = np.stack((unit.real,unit.imag),axis=-1)
    brightness = signal/amplitude
    residual = brightness-unit.real
    parts = np.array_split(np.arange(signal.shape[1-axis]),2)
    predicted = np.zeros(signal.shape)
    details = []
    for half,other in ((parts[0],parts[1]),(parts[1],parts[0])):
        def rows(a,indices):
            return np.take(a,indices,axis=1-axis)
        b = rows(B,half).reshape(-1,2)
        s = rows(S,half).reshape(-1,2)
        y = rows(brightness,half).ravel()
        independent = s-b@np.linalg.lstsq(b,s,rcond=None)[0]
        gram = independent.T@independent
        if np.linalg.eigvalsh(gram)[0] < .25**2*.5*len(y):
            return None
        coef = np.linalg.solve(gram, independent.T@y)
        error = independent@(coef-[1.,0.])
        relative_error = np.linalg.norm(error)/np.linalg.norm(s[:,0])
        if relative_error>MAX_COEFFICIENT_ERROR or coef[0]<.25 or coef[0]<MIN_ENDPOINT_ALIGNMENT*np.linalg.norm(coef):
            return None
        b_other = rows(B,other).reshape(-1,2)
        nuisance = np.linalg.lstsq(b_other,rows(residual,other).ravel(),rcond=None)[0]
        analytic = rows(unit,half) + complex(nuisance[0],-nuisance[1])*rows(atom,half)
        signed_prediction = analytic.real/np.maximum(abs(analytic),1.e-10)
        if axis==1:
            predicted[half,:] = signed_prediction
        else:
            predicted[:,half] = signed_prediction
        details.append(dict(carrier_coefficient=coef.tolist(), relative_error=float(relative_error),
                            identifiable_rms=float(np.sqrt(np.linalg.eigvalsh(gram)[0]/(.5*len(y))))))
    count = np.count_nonzero(held)
    if count<MIN_QUADRANT_EVENTS:
        return None
    alignment = float(np.mean(events[held]*predicted[held]))
    # A half-strength trial must improve both halves and the full-dose signal
    # must, on average, predict better than preserving the input.
    if alignment<MIN_QUADRANT_ALIGNMENT or np.mean([r["carrier_coefficient"][0] for r in details])<=.5:
        return None
    return predicted, dict(halves=details, held_signed_agreement=alignment, held_count=int(count))


def _run(observed, bounds, gain, seed=None):
    def reject(reason):
        return None,None,dict(reason=reason)
    shape = np.asarray(observed.shape)
    if np.min(shape)<MIN_SIDE or np.max(shape)>MAX_SIDE:
        return reject("unsupported_image_size")
    events = (observed==1).astype(np.int8)-(observed==0)
    if np.count_nonzero(events)<MIN_EVENTS:
        return reject("insufficient_clip_events")
    groups = [np.searchsorted(np.linspace(0,n,5,dtype=int)[1:-1],np.arange(n),side="right") for n in shape]
    train = (groups[0][:,None]+groups[1])%2==0
    held = (events!=0)&~train
    quadrants = 2*(groups[0][:,None]//2)+groups[1]//2
    counts = np.bincount(quadrants[held],minlength=4)
    if counts.min()<MIN_QUADRANT_EVENTS:
        return reject("insufficient_held_endpoints")
    coords = [(np.arange(n)-(n-1)/2)/n for n in shape]
    if seed is None:
        seed = _axis_seed(events,train,coords,bounds)
    if seed is None:
        return reject("no_axial_endpoint_seed")
    axis, params = seed
    sweep = CURVATURE_STEP*np.round(params[2]/CURVATURE_STEP)
    center = np.rint(params[:2])
    if abs(sweep)<=CURVATURE_STEP or abs(sweep)>MAX_CURVATURE or np.linalg.norm(center/shape)==0:
        return reject("unresolved_axial_sweep")
    field = np.broadcast_to(center,(shape[axis],2)).copy()
    field[:,axis] += sweep*coords[axis]
    peaks,_,powers = _spectrum(observed)
    # Withhold the whole coarse trajectory before fitting the 2-D weave basis.
    # A legitimate local tone within the trajectory must not redefine that basis.
    edge = LATTICE_TOLERANCE_BINS+TAPER_BINS
    outside = np.minimum(_segment_distances(peaks,*field[[0,-1]]),
                         _segment_distances(-peaks,*field[[0,-1]]))>edge
    peaks = peaks[outside]
    members,lattice = _fit_lattice(peaks,powers[outside],shape)
    if lattice is None:
        return reject("no_independently_measured_lattice")
    measured = np.vstack((peaks[members],-peaks[members]))
    tree = cKDTree(np.vstack((measured,_node_bins(lattice))))
    distance, nearest = tree.query(field)
    guard = distance<edge
    runs = np.flatnonzero(np.diff(np.r_[False,guard,False])).reshape(-1,2)
    if not len(runs) or np.any(np.diff(runs,axis=1)<MIN_WIDTH):
        return reject("unresolved_collision_bands")
    period = int(np.ceil(1/np.linalg.norm(center/shape)))
    weights = .5-.5*np.cos(np.pi*np.clip((distance_transform_edt(~guard)-period)/(2*period),0.,1.))
    ext = np.flatnonzero(weights>0)
    if weights.sum()<shape[axis]/8 or np.ptp(ext)<shape[axis]/2 or not np.any(np.diff(ext)>1):
        return reject("insufficient_disjoint_exterior")
    fit = _external_wave(observed,axis,coords,weights,center,sweep)
    if fit is None:
        return reject("unresolved_exterior_fit")
    unit,model = fit
    actual = np.array(model["center_frequency"])*shape
    changes = np.broadcast_to(actual-params[:2],(2,2)).copy()
    changes[:,axis] += (sweep-params[2])*coords[axis][[0,-1]]
    if np.linalg.norm(changes,axis=1).max()>.5 or not _admissible_trajectory(model,shape,axis,measured,bounds):
        return reject("frequency_incompatible_with_endpoint_proposal")
    amp = model["amplitude"]
    signal = observed-gaussian_filter(observed,5.)
    predicted = unit.real.copy()
    # The fit must describe each geographically disjoint exterior support.
    # This also excludes partial-field waves which end in an unguarded region.
    demodulated = (signal*unit.conj()).mean(axis=1-axis)
    clear_runs = np.flatnonzero(np.diff(np.r_[False,weights>0,False])).reshape(-1,2)
    exterior_coefficients = []
    for lo,hi in clear_runs:
        local = weights[lo:hi]
        coefficient = 2*np.average(demodulated[lo:hi],weights=local)/amp
        if abs(coefficient-1)>MAX_COEFFICIENT_ERROR:
            return reject("exterior_strips_disagree")
        exterior_coefficients.append([float(coefficient.real),float(coefficient.imag)])
    segments = []
    for lo,hi in runs:
        sl = (slice(lo,hi),slice(None)) if axis==0 else (slice(None),slice(lo,hi))
        node_ids = np.unique(nearest[lo:hi])
        if len(node_ids)!=1 or node_ids[0]<len(measured):
            return reject("overlapping_or_measured_nodes")
        node = tree.data[node_ids[0]]
        atom = np.exp(2j*np.pi*(node[0]*coords[0][sl[0],None]+node[1]*coords[1][sl[1]]))
        result = _conditional_band(signal[sl],unit[sl],atom,events[sl],held[sl],axis,amp)
        if result is None:
            return reject("local_stripe_not_separable_from_stationary_tone")
        predicted[sl],detail = result
        segments.append(dict(start=int(lo),stop=int(hi),stationary_node=node.tolist(),**detail))
    signed = predicted[held]*events[held]
    by_quadrant = np.bincount(quadrants[held],weights=signed,minlength=4)/counts
    if signed.mean()<MIN_ENDPOINT_ALIGNMENT or by_quadrant.min()<MIN_QUADRANT_ALIGNMENT:
        return reject("held_phase_disagrees")
    wave = amp*unit.real
    return _restore(observed,wave,gain),wave,dict(reason="conditionally_verified_exterior_wave",
                sweep_axis=int(axis), applied_model=model,bands=segments,exterior_coefficients=exterior_coefficients,
                held_signed_agreement=float(signed.mean()),held_quadrant_agreement=by_quadrant.tolist())


def refine(observed, diag, base_evidence, bounds=(.025,.45)):
    if base_evidence.get("reason") not in RETRY_REASONS:
        return None,None,dict(reason="not_an_exterior_fit_failure")
    fit = diag["evidence"]["guarded_carrier"]
    axis = base_evidence["sweep_axis"]
    params = np.r_[fit["training_parameters_bins"][:2],fit["training_parameters_bins"][2+2*axis]]
    return _run(observed,bounds,diag["accepted_regions"][0]["gain"],(axis,params))


def review_abstention(observed, diag, *, bounds=(.025,.45), gain=1.):
    if (diag.get("reason")!="recurring_narrow_texture"
            or diag.get("evidence",{}).get("guarded_carrier",{}).get("reason")!="no_independent_verified_carrier"):
        return None,None,dict(reason="not_an_unresolved_texture")
    return _run(observed,bounds,gain)
