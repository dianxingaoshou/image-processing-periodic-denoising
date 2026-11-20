"""Use spatially replicated clipping phase and brightness to disambiguate periodic texture."""
from itertools import combinations, product

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import minimize
from scipy.spatial import cKDTree

import nine_chirp
from weak_carrier import MAX_SIDE, MAX_SWEEP, MIN_SIDE, _measurements, _verify
from weave_guard import LATTICE_TOLERANCE_BINS, MAX_ORDER, _fit_lattice, _spectrum


MIN_EVENTS = 64
MIN_QUADRANT_EVENTS = 8
MIN_ENDPOINT_ALIGNMENT = .65
MIN_QUADRANT_ALIGNMENT = .5
LOCKED_PHASE_RESULTANT = .75
COARSE_PEAKS = 8
TAPER_BINS = 1.


def routing_parameters():
    return dict(version="endpoint_phase_lattice_review_v1", image_side_range=[MIN_SIDE,MAX_SIDE],
                min_endpoint_events=MIN_EVENTS, min_held_quadrant_events=MIN_QUADRANT_EVENTS,
                minimum_held_alignment=MIN_ENDPOINT_ALIGNMENT, minimum_quadrant_alignment=MIN_QUADRANT_ALIGNMENT,
                minimum_lattice_phase_resultant=LOCKED_PHASE_RESULTANT, exact_lattice_bins=.5,
                collision_radius_bins=LATTICE_TOLERANCE_BINS, collision_taper_bins=TAPER_BINS,
                minimum_clear_fraction=.5, fixed_seeds=COARSE_PEAKS, quadratic_proposals=nine_chirp.routing_parameters(),
                maximum_final_frequency_change_bins=.5, candidate_exclusion_radius_bins=2.,
                calibration="all brightness pixels, amplitude capped by independently checked training fit")


def _stationary_seeds(events, train, bounds):
    h,w=events.shape
    y,x=np.meshgrid(np.fft.fftfreq(h),np.fft.fftfreq(w),indexing="ij")
    radius=np.hypot(y,x)
    allowed=(radius>=bounds[0]) & (radius<=bounds[1]) & ((y>0)|((y==0)&(x>0)))
    power=abs(np.fft.fft2(events*train))
    flat=np.flatnonzero(allowed & (power==maximum_filter(power,5,mode="wrap")))
    best=flat[np.argsort(-power.ravel()[flat],kind="stable")[:COARSE_PEAKS]]
    return [np.array([y.flat[i]*h,x.flat[i]*w,0.,0.,0.]) for i in best]


def _fit_phase(seed, coordinates, signal, curved, *, local=False):
    """Maximum complex demodulation, in full-image bins with a symmetric phase Hessian."""
    count=5 if curved else 2
    coords=np.ascontiguousarray(coordinates[:,:count])
    def objective(p):
        demod=signal*np.exp((-2j*np.pi)*(coords@p))
        total=demod.sum()
        return -abs(total)**2, -2*np.real(np.conj(total)*(-2j*np.pi)*(demod@coords))
    width = np.array([1.5,1.5,5.,5.,5.])[:count] if not local else np.full(count,.5)
    lo,hi=seed[:count]-width,seed[:count]+width
    lo[2:]=np.maximum(lo[2:],-MAX_SWEEP)
    hi[2:]=np.minimum(hi[2:],MAX_SWEEP)
    opt=minimize(objective,np.clip(seed[:count],lo,hi),method="L-BFGS-B",jac=True,
                 bounds=list(zip(lo,hi)),options=dict(maxiter=35,gtol=1e-8,ftol=1e-12))
    result=np.zeros(5)
    result[:count]=opt.x
    return result


def _field(params, coordinates):
    gradient=np.array([[params[2],params[3]],[params[3],params[4]]])
    return params[:2]+coordinates@gradient


def _endpoint_evidence(params, coefficient, event_coords, signs, quadrants, testing):
    if not np.any(testing):
        return None
    phases = (2*np.pi)*(event_coords[testing]@params)+np.angle(coefficient)
    aligned = signs[testing]*np.cos(phases)
    counts = np.bincount(quadrants[testing],minlength=4)
    if np.any(counts<MIN_QUADRANT_EVENTS):
        return None
    by_quadrant=np.bincount(quadrants[testing],weights=aligned,minlength=4)/counts
    if aligned.mean()<MIN_ENDPOINT_ALIGNMENT or by_quadrant.min()<MIN_QUADRANT_ALIGNMENT:
        return None
    return dict(held_events=int(testing.sum()), held_quadrant_counts=counts.tolist(),
                mean_signed_phase=float(aligned.mean()), quadrant_signed_phase=by_quadrant.tolist())


def _node_bins(lattice):
    orders = np.array([order for order in product(range(-MAX_ORDER,MAX_ORDER+1),repeat=2) if any(order)])
    return orders@lattice


def _phase_lock(frequency, anchors, coordinates, cells, signal, weights):
    """Audit combinations of *other* observed peaks using spatial phase variation across cells."""
    flat=cells.ravel()
    mass=np.bincount(flat,weights=weights.ravel())
    centers=np.array([np.bincount(flat,weights=axis.ravel())/np.bincount(flat)
                      for axis in np.moveaxis(coordinates,-1,0)]).T
    def coefficient(f):
        z=(signal*weights*np.exp((-2j*np.pi)*(coordinates@f))).ravel()
        return 2*(np.bincount(flat,weights=z.real)+1j*np.bincount(flat,weights=z.imag))/mass
    carrier=coefficient(frequency)
    anchors_coef=[coefficient(p) for p in anchors]
    scores=[]
    for i,j in combinations(range(len(anchors)),2):
        pair=anchors[[i,j]]
        if abs(np.linalg.det(pair))<.35*np.prod(np.linalg.norm(pair,axis=1)):
            continue
        orders=np.rint(frequency@np.linalg.inv(pair))
        delta=orders@pair-frequency
        if np.max(abs(orders))>MAX_ORDER or np.linalg.norm(delta)>LATTICE_TOLERANCE_BINS:
            continue
        valid=abs(carrier)>=.02
        phase=-np.angle(carrier)+(2*np.pi)*(centers@delta)
        for a,n in zip((i,j),orders):
            if n:
                valid &= abs(anchors_coef[a])>=.02
                phase+=n*np.angle(anchors_coef[a])
        if valid.sum()>=8:
            scores.append(float(abs(np.exp(1j*phase[valid]).mean())))
    return max(scores,default=None)


def _stationary_protection(params, peaks, powers, shape, coordinates, cells, signal, weights):
    frequency=params[:2]
    distance=np.minimum(np.linalg.norm(peaks-frequency,axis=1),np.linalg.norm(peaks+frequency,axis=1))
    # This candidate must never provide a datum for predicting its own lattice site.
    other=distance>2.
    members,lattice=_fit_lattice(peaks[other],powers[other],shape)
    if lattice is None:
        return None
    gap=float(np.min(np.linalg.norm(_node_bins(lattice)-frequency,axis=1)))
    if gap<=.5:
        return None
    score=_phase_lock(frequency,peaks[other][members],coordinates,cells,signal,weights)
    if score is not None and score>=LOCKED_PHASE_RESULTANT:
        return None
    if gap<=LATTICE_TOLERANCE_BINS and score is None:
        return None
    return dict(other_lattice_peaks=int(members.sum()), withheld_candidate_peaks=int((~other).sum()),
                nearest_predicted_node_bins=gap, maximum_harmonic_phase_resultant=score)


def _calibrate(params, coefficient, basis, cells, signal, weights, training, corners,
               event_coords, signs, quadrants, testing, curved):
    """An all-pixel brightness refit may polish, but cannot promote, a checked model."""
    calibrated = _fit_phase(params, basis.reshape(-1,5), 2*(signal*weights).ravel()/weights.sum(),
                            curved, local=True)
    drift = np.linalg.norm(_field(params-calibrated, corners), axis=1).max()
    if drift > .5:
        return params, coefficient, None
    carrier = np.exp((2j*np.pi)*(basis@calibrated))
    measured = (signal*weights*carrier.conj()).sum()
    if abs(measured) == 0:
        return params, coefficient, None
    phase = measured/abs(measured)
    unit = np.real(phase*carrier)
    amplitude = np.clip((weights*signal*unit).sum()/(weights*unit**2).sum(), 0, abs(coefficient))
    adjusted = amplitude * phase
    brightness = _verify(calibrated,basis,cells,signal,weights,training,coefficient=adjusted)
    endpoints = _endpoint_evidence(calibrated,adjusted,event_coords,signs,quadrants,testing)
    if brightness is None or endpoints is None:
        return params, coefficient, None
    return calibrated, adjusted, dict(maximum_corner_shift_bins=float(drift), brightness=brightness[0],
                                      endpoints=endpoints)


def _collision_taper(field, tree, shape):
    """Preserve pixels where the local chirp overlaps a measured or inferred texture node."""
    distance = tree.query(field.reshape(-1, 2))[0].reshape(shape)
    if np.mean(distance > LATTICE_TOLERANCE_BINS) < .5:
        return None, None
    f = np.clip((distance-LATTICE_TOLERANCE_BINS)/TAPER_BINS, 0, 1)
    taper = f*f*(3-2*f)
    return taper, dict(protected_pixels=int(np.count_nonzero(f == 0)),
                       tapered_pixels=int(np.count_nonzero((f > 0) & (f < 1))))


def find_carrier(observed, bounds, *, enable_chirp=True):
    """Check one observed image for a unique signed clipping and brightness phase pattern.

    Exact clipped endpoints propose a wave and provide a sign check on held-out
    points. The unmodified luminance determines its phase and subtraction
    amplitude. With insufficient clipping only the preceding pipeline applies.
    """
    image=np.asarray(observed,dtype=float)
    if image.ndim!=2 or min(image.shape)<MIN_SIDE or max(image.shape)>MAX_SIDE:
        return None,dict(reason="outside_full_field_search_geometry")
    if not np.isfinite(image).all() or image.min()<0 or image.max()>1:
        raise ValueError("Expected finite grayscale intensities in [0,1]")
    events=(image==1).astype(float)-(image==0)
    locations=events!=0
    event_count=int(locations.sum())
    details=dict(endpoint_count=event_count,stages=[])
    if event_count<MIN_EVENTS:
        return None,dict(**details,reason="insufficient_clipping_evidence")
    shape=np.array(image.shape)
    peaks,_,powers=_spectrum(image)
    members,lattice=_fit_lattice(peaks,powers,shape)
    if lattice is None:
        return None,dict(**details,reason="no_resolved_texture_lattice")
    old_basis,cells,signal,weights,_=_measurements(image)
    coords=old_basis[...,:2]
    basis=np.stack((coords[...,0],coords[...,1],old_basis[...,2],coords[...,0]*coords[...,1],old_basis[...,3]),axis=-1)
    training=np.indices((4,4)).sum(axis=0).ravel()%2==0
    train=training[cells]
    signs=events[locations]
    event_coords=basis[locations]
    fit_locations=train[locations]
    quadrants=(2*((cells//4)//2)+(cells%4)//2)[locations]
    counts=np.bincount(quadrants[~fit_locations],minlength=4)
    if np.any(counts<MIN_QUADRANT_EVENTS) or fit_locations.sum()<MIN_EVENTS//2:
        return None,dict(**details,reason="insufficient_spatial_clipping_evidence")
    event_fit=signs[fit_locations]/fit_locations.sum()
    corners=np.array(list(product((-.5,.5),repeat=2)))*(shape-1)/shape
    nodes=np.vstack((peaks[members],-peaks[members],_node_bins(lattice)))
    tree=cKDTree(nodes)
    for curved in ([False,True] if enable_chirp else [False]):
        seeds=(nine_chirp.propose(events,bounds,train) if curved else _stationary_seeds(events,train,bounds))
        candidates=[]
        for seed in seeds:
            params=_fit_phase(seed,event_coords[fit_locations],event_fit,curved)
            if curved and np.max(abs(params[2:]))<2:
                continue
            # Endpoint coherence is measured on training only before expensive brightness verification.
            phasor=np.exp((-2j*np.pi)*(event_coords[fit_locations]@params))
            if abs(np.dot(phasor,event_fit))<MIN_ENDPOINT_ALIGNMENT:
                continue
            initial=_verify(params,basis,cells,signal,weights,training)
            if initial is None:
                continue
            evidence,coefficient=initial
            checked=_endpoint_evidence(params,coefficient,event_coords,signs,quadrants,~fit_locations)
            if checked is None:
                continue
            # Limited final calibration uses all brightness pixels only after the held-out phase test.
            trained = params
            training_phase = float(np.angle(coefficient))
            params, coefficient, calibration = _calibrate(
                params, coefficient, basis, cells, signal, weights, training, corners,
                event_coords, signs, quadrants, ~fit_locations, curved)
            field=_field(params,coords)
            radius=np.linalg.norm(field/shape,axis=2)
            if radius.min()<bounds[0] or radius.max()>bounds[1]:
                continue
            if curved:
                taper, protection = _collision_taper(field, tree, image.shape)
                if taper is None:
                    continue
            else:
                protection=_stationary_protection(params,peaks,powers,shape,coords,cells,signal,weights)
                if protection is None:
                    continue
                taper=1.
            endpoints=_field(params,corners)
            if any(min(np.max(np.linalg.norm(endpoints-sign*entry[0],axis=1)) for sign in (-1,1))<=1 for entry in candidates):
                continue
            wave=taper*np.real(coefficient*np.exp((2j*np.pi)*(basis@params)))
            record=dict(**checked,brightness_check=evidence,protection=protection,amplitude=float(abs(coefficient)),
                        model="quadratic" if curved else "stationary",patch_size=int(min(shape)//4),
                        training_parameters_bins=trained.tolist(), training_phase_radians=training_phase,
                        phase_refined=calibration is not None, final_calibration=calibration,
                        final_phase_radians=float(np.angle(coefficient)),
                        center_frequency=(params[:2]/shape).tolist(),
                        hessian=(params[2:]/[shape[0]**2,shape.prod(),shape[1]**2]).tolist())
            candidates.append((endpoints,wave,record))
        details["stages"].append(dict(model="quadratic" if curved else "stationary",seeds=len(seeds),verified=len(candidates)))
        if len(candidates)>1:
            return None,dict(**details,reason="ambiguous_verified_carriers")
        if candidates:
            _,wave,record=candidates[0]
            return wave,dict(**details,**record,reason="endpoint_and_luminance_phase_confirmed")
    return None,dict(**details,reason="no_independent_verified_carrier")
