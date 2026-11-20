"""Input-only one-dimensional proposals for long, narrow periodic bands.

The tangent frequency is proposed from the strongest few normal profiles rather
than averaging their powers over the whole image. A component must span at least
three tangent cycles. Opposite tangent halves use the same absolute phase origin
to audit whether their sinusoidal coefficients agree. None of these measures
identifies the component semantically as noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class ProfileConfig:
    min_frequency: float = .005
    max_frequency: float = .45
    minimum_cycles: float = 3.
    minimum_tangent_length: int = 64
    max_candidates: int = 5
    pooling_rows: int = 4
    fft_oversampling: int = 4

    def __post_init__(self):
        if (not np.isfinite([self.min_frequency,self.max_frequency,self.minimum_cycles]).all()
                or not 0 <= self.min_frequency < self.max_frequency < .5 or self.minimum_cycles <= 0):
            raise ValueError('Invalid profile frequency range or cycle count')
        if (not isinstance(self.minimum_tangent_length,int) or self.minimum_tangent_length < 32
                or not isinstance(self.max_candidates,int) or not 1 <= self.max_candidates <= 8
                or not isinstance(self.pooling_rows,int) or not 1 <= self.pooling_rows <= 16
                or self.fft_oversampling not in (2,4,8)):
            raise ValueError('Invalid profile geometry or candidate settings')


def _rows(image,axis):
    a=np.asarray(image,dtype=np.float64)
    if (a.ndim!=2 or min(a.shape)<16 or not np.isfinite(a).all() or a.min()<0 or a.max()>1):
        raise ValueError('Expected a finite [0,1] two-dimensional image, dimensions >=16')
    if axis not in (0,1):raise ValueError('axis must be0(normalrows) or1(normalcolumns)')
    return a if axis==0 else a.T


def _fit(rows,frequency,positions,full_length):
    """Fit all normal profiles jointly in one small linear solve."""
    t=np.asarray(positions,dtype=float)
    phase=2*np.pi*frequency*t
    z=(t-(full_length-1)/2)/full_length
    design=np.stack((np.sin(phase),np.cos(phase),np.ones_like(t),z),axis=-1)
    root=np.sqrt(np.hanning(len(t)+2)[1:-1])
    coefficients=np.linalg.lstsq(design*root[:,None],rows[:,positions].T*root[:,None],rcond=None)[0].T
    prediction=coefficients@design.T
    error=np.mean((rows[:,positions]-prediction)**2,axis=1)
    return coefficients,error


def _linear_residual(rows,positions,full_length):
    t=np.asarray(positions,dtype=float)
    design=np.stack((np.ones_like(t),(t-(full_length-1)/2)/full_length),axis=-1)
    coefficients=np.linalg.lstsq(design,rows[:,positions].T,rcond=None)[0].T
    return rows[:,positions]-coefficients@design.T


def _cross_gain(rows,frequency,positions,coefficients,full_length):
    t=np.asarray(positions,dtype=float)
    oscillation=coefficients[:,0,None]*np.sin(2*np.pi*frequency*t)+coefficients[:,1,None]*np.cos(2*np.pi*frequency*t)
    baseline=_linear_residual(rows,positions,full_length)
    # Only nuisance intercept/slope is fitted on validation pixels. The
    # periodic coefficients come unchanged from the opposite tangent half.
    remainder=rows.copy()
    remainder[:,positions]-=oscillation
    residual=_linear_residual(remainder,positions,full_length)
    energy=np.mean(baseline**2,axis=1)
    gain=np.clip(1-np.mean(residual**2,axis=1)/np.maximum(energy,1e-20),0,1)
    gain[energy<1e-16]=0.
    return gain


def _make_profile(rows,axis,frequency,**metadata):
    """Same fitting/validation for either proposal source."""
    _,length=rows.shape;positions=np.arange(length)
    full,fit_mse=_fit(rows,frequency,positions,length)
    split=length//2;halves=(positions[:split],positions[split:])
    first,_=_fit(rows,frequency,halves[0],length)
    second,_=_fit(rows,frequency,halves[1],length)
    split_coefficients=np.stack((first[:,:2],second[:,:2]))
    mass=np.linalg.norm(first[:,:2],axis=1)+np.linalg.norm(second[:,:2],axis=1)
    stability=np.clip(1-np.linalg.norm(first[:,:2]-second[:,:2],axis=1)/np.maximum(mass,1e-15),0,1)
    stability[mass<1e-8]=0.
    validation_gain=np.minimum(_cross_gain(rows,frequency,halves[0],second[:,:2],length),
                               _cross_gain(rows,frequency,halves[1],first[:,:2],length))
    return dict(axis=int(axis),frequency=float(frequency),tangent_length=length,
        coefficients=full[:,:2],split_coefficients=split_coefficients,
        amplitude=np.linalg.norm(full[:,:2],axis=1),stability=stability,
        validation_gain=validation_gain,
        row_background_energy=np.mean(_linear_residual(rows,positions,length)**2,axis=1),
        cycles=float(frequency*(length-1)),fit_mse=fit_mse,**metadata)


def _spectral_state(rows,cfg):
    length=rows.shape[1];positions=np.arange(length)
    detrended=_linear_residual(rows,positions,length)
    energy=np.mean(detrended**2,axis=1)
    nfft=cfg.fft_oversampling*length
    power=np.abs(np.fft.rfft(detrended*np.hanning(length),n=nfft,axis=1))**2
    normalized=power/np.maximum(power.sum(axis=1,keepdims=True),1e-20)
    normalized[energy<1e-16]=0.
    return energy,normalized,np.fft.rfftfreq(nfft)


def _refine_frequency(rows,initial,cfg,normalized,frequencies):
    """Same top-profile selection and bounded frequency solve for both pools."""
    length=rows.shape[1];positions=np.arange(length)
    index=int(np.argmin(np.abs(frequencies-initial)))
    k=min(cfg.pooling_rows,len(rows))
    chosen=np.argsort(normalized[:,index],kind='stable')[-k:]
    if float(np.max(normalized[:,index]))<=0:return initial,chosen
    sample=rows[chosen]

    def objective(f):return float(np.mean(_fit(sample,f,positions,length)[1]))

    lower=max(cfg.min_frequency,cfg.minimum_cycles/(length-1))
    low=max(lower,initial-1/length);high=min(cfg.max_frequency,initial+1/length)
    fit=minimize_scalar(objective,bounds=(low,high),method='bounded',options={'xatol':1e-10}) if high>low else None
    frequency=float(fit.x) if fit is not None and fit.success else initial
    return frequency,chosen


def profiles_from_frequencies(image,axis,frequencies,config:ProfileConfig|None=None,*,refine=False):
    """Fit externally proposed tangent frequencies, optionally refining them.

    This supports a controlled detector ablation: the caller extracts coarse
    components from an input-only legacy candidate pool, then uses exactly the
    same profile fit and validation as the new detector. No reference, mask or
    noise-truth object is accepted. Supplying frequencies does not label them
    as noise. Negative conjugates are canonicalized to positive frequencies.
    ``refine=True`` uses exactly the same normal-row selection, bounds and
    one-dimensional objective as :func:`propose_profiles`; the default keeps
    externally supplied frequencies unchanged.
    """
    cfg=config or ProfileConfig();rows=_rows(image,axis);length=rows.shape[1]
    values=np.asarray(frequencies,dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all():raise ValueError('frequencies must be a finite one-dimensional sequence')
    if length<cfg.minimum_tangent_length:return []
    lower=max(cfg.min_frequency,cfg.minimum_cycles/(length-1))
    spectrum=_spectral_state(rows,cfg) if refine else None
    pool=[]
    for value in values:
        initial=float(abs(value))
        if not lower<=initial<=cfg.max_frequency:continue
        frequency,chosen=(_refine_frequency(rows,initial,cfg,spectrum[1],spectrum[2]) if refine else (initial,[]))
        if any(abs(frequency-c['frequency'])<1.5/length for c in pool):continue
        pool.append(_make_profile(rows,axis,frequency,spectral_score=0.,peak_fraction=0.,
            initial_frequency=initial,pooling_rows=np.asarray(chosen,dtype=int).tolist(),proposal_source='external_frequency'))
        if len(pool)>=cfg.max_candidates:break
    return pool


def propose_profiles(image,axis,config:ProfileConfig|None=None):
    """Return a small deterministic pool of tangent-frequency profile models.

    ``axis=0``: normal rows, tangent columns. ``axis=1``: normal columns,
    tangent rows. ``coefficients[n]`` is the sin/cos pair at absolute tangent
    coordinate0; reconstruct with ``c_sin*sin(2*pi*f*t)+c_cos*cos(...)``.

    Arrays in each returned candidate are normal-profile diagnostics, not
    image/reference truth. Their sizes are N, Nx2, or2xNx2. Callers should use
    compact-support and reliability checks before subtracting a candidate.
    ``row_background_energy`` is observed detrended row variance, including
    periodic content; it is not an estimate from a noise-free background.
    """
    cfg=config or ProfileConfig();rows=_rows(image,axis);n,length=rows.shape
    if length<cfg.minimum_tangent_length:return []
    background_energy,normalized,frequencies=_spectral_state(rows,cfg)
    if float(background_energy.max())<1e-16:return []
    k=min(cfg.pooling_rows,n)
    pooled=np.mean(np.partition(normalized,n-k,axis=0)[n-k:],axis=0)
    lower=max(cfg.min_frequency,cfg.minimum_cycles/(length-1))
    valid=(frequencies>=lower)&(frequencies<=cfg.max_frequency)
    if not valid.any():return []
    background=ndi.gaussian_filter1d(pooled,4.,mode='nearest')
    contrast=pooled-background
    median=float(np.median(contrast[valid]));mad=1.4826*float(np.median(np.abs(contrast[valid]-median)))+1e-12
    score=(contrast-median)/mad
    peaks=np.flatnonzero(valid&(pooled==ndi.maximum_filter1d(pooled,5,mode='nearest')))
    order=sorted(peaks.tolist(),key=lambda i:(-float(pooled[i]),i))
    candidates=[]
    for index in order:
        initial=float(frequencies[index])
        if any(abs(initial-c['frequency'])<1.5/length for c in candidates):continue
        frequency,chosen=_refine_frequency(rows,initial,cfg,normalized,frequencies)
        if any(abs(frequency-c['frequency'])<1.5/length for c in candidates):continue
        candidates.append(_make_profile(rows,axis,frequency,
            spectral_score=float(score[index]),peak_fraction=float(pooled[index]),
            initial_frequency=initial,pooling_rows=chosen.astype(int).tolist(),proposal_source='tangent_fft'))
        if len(candidates)>=cfg.max_candidates:break
    return candidates


__all__=['ProfileConfig','propose_profiles','profiles_from_frequencies']
