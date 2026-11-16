"""V3 input-only local sinusoidal/chirp residual estimation.

This module estimates *candidate* periodic components.  It does not determine
whether those components are image texture or interference; a caller should
provide conservative weights to :func:`restore`.  The legacy implementation
remains unchanged in local_spectral.py.

The default is a numerical-correctness revision of V2: every fitting stage uses
all pixels and effective analysis windows are deduplicated. Optional two-candidate
reranking uses a fixed spatially stratified holdout, followed by full-block refit.
An optional one-pass overlap check compares candidate residuals on shared pixels.
Neither validation nor agreement proves that a periodic component is noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from local_spectral import _checked, _starts, _detrend, _canonical, _delta, _frequency_distance


@dataclass(frozen=True)
class EstimationConfig:
    scales: tuple[int, ...] = (64, 128, 256)
    stride: int = 32
    synthesis_size: int = 64
    force_scale: int | None = None
    max_candidates: int = 3
    peak_threshold: float = 4.5
    min_frequency: float = .025
    max_frequency: float = .45
    min_amplitude: float = .005
    neighbor_tolerance: float = .014
    enable_chirp: bool = True
    enable_amplitude_model: bool = True
    chirp_iterations: int = 5
    chirp_gain_threshold: float = .004
    amplitude_gain_threshold: float = .003
    model_complexity_penalty: float = .08
    max_phase_curvature: float = 4.
    refine_candidates: int = 1
    validation_fraction: float = .25
    validation_seed: int = 2026111603
    rerank_complexity_penalty: float = .02
    use_overlap_reranking: bool = False
    overlap_penalty: float = .15
    frequency_merge_tolerance: float = .0078125

    def __post_init__(self):
        if not self.scales or any(s < 32 for s in self.scales):
            raise ValueError('scales must contain sizes >=32')
        if self.force_scale is not None and self.force_scale not in self.scales:
            raise ValueError('force_scale must be one of scales')
        if not 0 < self.stride <= self.synthesis_size // 2 or self.synthesis_size < 32:
            raise ValueError('Require synthesis_size>=32 and stride<=synthesis_size/2')
        if not 1 <= self.max_candidates <= 8 or not 0 < self.min_frequency < self.max_frequency < .5:
            raise ValueError('Invalid candidate count or frequency interval')
        if self.peak_threshold <= 0 or self.min_amplitude <= 0 or self.chirp_iterations < 1:
            raise ValueError('Thresholds and iteration count must be positive')
        if self.refine_candidates not in (1,2):
            raise ValueError('refine_candidates must be 1 or 2')
        if not 0 < self.validation_fraction < .5 or self.rerank_complexity_penalty < 0:
            raise ValueError('Invalid candidate validation configuration')
        if self.use_overlap_reranking and self.refine_candidates < 2:
            raise ValueError('Overlap reranking requires two candidate refinement')
        if self.overlap_penalty < 0 or self.frequency_merge_tolerance <= 0:
            raise ValueError('Invalid overlap or frequency merge settings')


@dataclass
class PreparedV3:
    image: np.ndarray
    config: EstimationConfig
    tiles: list[dict[str, Any]]
    analysis_count: int


@lru_cache(maxsize=32)
def _geometry(shape):
    h,w=shape
    yy,xx=np.indices(shape,dtype=float)
    u=(yy-(h-1)/2)/h;v=(xx-(w-1)/2)/w
    weight=np.outer(np.hanning(h+2)[1:-1],np.hanning(w+2)[1:-1])
    plane=np.stack([np.ones(shape),u,v],axis=-1)
    phase_basis=np.stack([u,v,.5*u*u,u*v,.5*v*v],axis=-1)
    return yy,xx,u,v,weight,plane,phase_basis


def spectral_candidates(a,cfg):
    """Several conjugate-merged local maxima; frequency units are cycles/pixel."""
    h,w=a.shape
    window=np.outer(np.hanning(h),np.hanning(w))
    spectrum=np.fft.fft2(_detrend(a)*window,s=(2*h,2*w))
    power=np.abs(spectrum)**2
    logmag=.5*np.log(power+1e-18)
    residual=logmag-ndi.gaussian_filter(logmag,5.,mode='wrap')
    fy=np.fft.fftfreq(2*h)[:,None];fx=np.fft.fftfreq(2*w)[None,:]
    radius=np.hypot(fy,fx)
    valid=(radius>=cfg.min_frequency)&(radius<=cfg.max_frequency)
    values=residual[valid]
    med=float(np.median(values));mad=1.4826*float(np.median(np.abs(values-med)))+1e-9
    score=(residual-med)/mad
    fraction=power/max(float(power.sum()),1e-15)
    # A nearly pure off-grid sinusoid can broaden the robust background-score
    # distribution and obtain a surprisingly modest z score. A concentrated
    # energy maximum is therefore also a *candidate*, never a noise label.
    peaks=valid&((score>=cfg.peak_threshold)|(fraction>=.008))&(logmag==ndi.maximum_filter(logmag,size=5,mode='wrap'))
    positions=np.argwhere(peaks)
    positions=sorted(positions.tolist(),key=lambda p:(-max(score[p[0],p[1]],0)*np.sqrt(power[p[0],p[1]]),p[0],p[1]))
    out=[]
    offsets=np.arange(-8,9)
    outer=(np.abs(offsets[:,None])>=6)|(np.abs(offsets[None,:])>=6)
    for iy,ix in positions:
        fyy=float(np.fft.fftfreq(2*h)[iy]+_delta(logmag,iy,ix,0)/(2*h))
        fxx=float(np.fft.fftfreq(2*w)[ix]+_delta(logmag,iy,ix,1)/(2*w))
        fyy,fxx=_canonical(fyy,fxx)
        if not cfg.min_frequency <= np.hypot(fyy,fxx) <= cfg.max_frequency:continue
        if any(_frequency_distance(fyy,fxx,c['fy'],c['fx'])<2/max(h,w) for c in out):continue
        local=power[np.ix_((iy+offsets)%(2*h),(ix+offsets)%(2*w))]
        excess=np.maximum(local-np.median(local[outer]),0)
        mass=float(excess.sum())+1e-15
        wy=float(np.sqrt(np.sum(excess*offsets[:,None]**2)/mass)/2)
        wx=float(np.sqrt(np.sum(excess*offsets[None,:]**2)/mass)/2)
        width=float(np.hypot(wy,wx))
        amplitude=float(2*np.abs(spectrum[iy,ix])/max(float(window.sum()),1e-15))
        if amplitude<.35*cfg.min_amplitude:continue
        prominence=float(max(np.clip((score[iy,ix]-cfg.peak_threshold+1)/3,0,1),
                             np.clip((fraction[iy,ix]-.004)/.025,0,1)))
        compactness=1/(1+max(width-1.2,0)**2/4)
        evidence=float(prominence*compactness*np.clip(amplitude/(2*cfg.min_amplitude),0,1))
        out.append(dict(fy=fyy,fx=fxx,score=float(score[iy,ix]),amplitude=amplitude,
                        width_bins=width,evidence=evidence,peak_energy_fraction=float(fraction[iy,ix]),
                        scale=int(min(a.shape))))
        if len(out)>=cfg.max_candidates:break
    return out


def _fit_weights(shape,fit_mask=None):
    weight=_geometry(shape)[4]
    if fit_mask is None:return weight
    mask=np.asarray(fit_mask,dtype=bool)
    if mask.shape!=shape or np.count_nonzero(mask)<10:
        raise ValueError('fit_mask must match the patch and select >=10 pixels')
    return weight*mask


def _fit_constant(a,phase,subsample=1,*,fit_mask=None):
    """Weighted fit on all eligible pixels, including all sampling parities.

    ``subsample`` remains for V2 call compatibility but never selects a fixed
    checkerboard. Optional holdout fitting uses an explicit spatial mask.
    """
    if not isinstance(subsample,(int,np.integer)) or subsample<1:
        raise ValueError('subsample must be a positive integer')
    _,_,u,v,weight,plane,_=_geometry(a.shape)
    weight=_fit_weights(a.shape,fit_mask)
    sn,cs=np.sin(phase),np.cos(phase)
    design=np.concatenate([sn[...,None],cs[...,None],plane],axis=-1)
    x=design.reshape(-1,5);z=a.ravel();root=np.sqrt(weight.ravel())
    coef=np.linalg.lstsq(x*root[:,None],z*root,rcond=None)[0]
    fitted=design@coef
    sse=float(np.sum(weight*(a-fitted)**2))
    return coef,fitted,sse


def _refine_phase(a,fy,fx,allow_chirp,cfg,fit_mask=None):
    """Small analytic Gauss-Newton steps; no clean reference or true gradient."""
    yy,xx,u,v,weight,plane,basis=_geometry(a.shape)
    initial=2*np.pi*(fy*yy+fx*xx)
    parameters=np.zeros(5)
    n_parameters=5 if allow_chirp else 2
    weight=_fit_weights(a.shape,fit_mask)
    root=np.sqrt(weight.ravel())
    iterations=cfg.chirp_iterations if allow_chirp else min(3,cfg.chirp_iterations)
    for _ in range(iterations):
        phase=initial+2*np.pi*(basis@parameters)
        coef,fitted,sse=_fit_constant(a,phase,fit_mask=fit_mask)
        derivative=coef[0]*np.cos(phase)-coef[1]*np.sin(phase)
        jac=np.concatenate([np.sin(phase)[...,None],np.cos(phase)[...,None],plane,
                            derivative[...,None]*2*np.pi*basis[...,:n_parameters]],axis=-1)
        step=np.linalg.lstsq(jac.reshape(-1,5+n_parameters)*root[:,None],
                             (a-fitted).ravel()*root,rcond=None)[0][5:]
        step[:2]=np.clip(step[:2],-.45,.45)
        if allow_chirp:step[2:]=np.clip(step[2:],-.9,.9)
        accepted=False
        for fraction in (1.,.5,.25):
            trial=parameters.copy();trial[:n_parameters]+=fraction*step
            trial[:2]=np.clip(trial[:2],-1.5,1.5)
            trial[2:]=np.clip(trial[2:],-cfg.max_phase_curvature,cfg.max_phase_curvature)
            _,_,new_sse=_fit_constant(a,initial+2*np.pi*(basis@trial),fit_mask=fit_mask)
            if new_sse<sse-1e-12:
                parameters=trial;accepted=True;break
        if not accepted or np.linalg.norm(step)<1e-5:break
    phase=initial+2*np.pi*(basis@parameters)
    coef,fitted,sse=_fit_constant(a,phase,fit_mask=fit_mask)
    return dict(phase=phase,parameters=parameters,coef=coef,fitted=fitted,sse=sse,
                fy=float(fy+parameters[0]/a.shape[0]),fx=float(fx+parameters[1]/a.shape[1]))


def fit_candidate(a,candidate,cfg,*,fit_mask=None,force_model=None):
    """Fit stationary, quadratic-phase and affine-envelope alternatives."""
    yy,xx,u,v,weight,plane,basis=_geometry(a.shape)
    weight=_fit_weights(a.shape,fit_mask)
    stationary=_refine_phase(a,candidate['fy'],candidate['fx'],False,cfg,fit_mask)
    # A compact peak from a much larger window has more frequency precision
    # than a free re-fit on this small synthesis tile. Keep that estimate;
    # otherwise background details can pull an already precise global frequency.
    if candidate['scale'] >= 2*min(a.shape) and candidate['width_bins'] < 1.05:
        phase=2*np.pi*(candidate['fy']*yy+candidate['fx']*xx)
        coef,fitted,sse=_fit_constant(a,phase,fit_mask=fit_mask)
        stationary=dict(phase=phase,parameters=np.zeros(5),coef=coef,fitted=fitted,sse=sse,
                        fy=float(candidate['fy']),fx=float(candidate['fx']))
    best=stationary;model='sinusoid';chirp_gain=0.
    baseline_coef=np.linalg.lstsq(plane.reshape(-1,3)*np.sqrt(weight.ravel())[:,None],
                                a.ravel()*np.sqrt(weight.ravel()),rcond=None)[0]
    baseline_sse=float(np.sum(weight*(a-plane@baseline_coef)**2))
    explained=max(baseline_sse-stationary['sse'],1e-15)
    stationary_ratio=float(np.clip(explained/max(baseline_sse,1e-15),0,1))
    complexity_penalty=cfg.model_complexity_penalty*(1-stationary_ratio)**2
    compact_large_window=(candidate['scale']>=2*min(a.shape) and candidate['width_bins']<1.05)
    force_chirp=force_model is not None and 'chirp' in force_model
    allow_chirp=(cfg.enable_chirp and not compact_large_window) if force_model is None else force_chirp
    if allow_chirp:
        curved=_refine_phase(a,stationary['fy'],stationary['fx'],True,cfg,fit_mask)
        chirp_gain=float((stationary['sse']-curved['sse'])/explained)
        if force_chirp or (chirp_gain>cfg.chirp_gain_threshold+complexity_penalty and np.linalg.norm(curved['parameters'][2:])>.06):
            best=curved;model='chirp'
    coef=best['coef'];phase=best['phase']
    amplitude=float(np.hypot(coef[0],coef[1]))
    carrier=(coef[0]*np.sin(phase)+coef[1]*np.cos(phase))/max(amplitude,1e-15)
    noise=amplitude*carrier;amplitude_gain=0.;envelope=np.full(a.shape,amplitude)
    prediction=best['fitted']
    force_amplitude=force_model is not None and 'affine_amplitude' in force_model
    allow_amplitude=cfg.enable_amplitude_model if force_model is None else force_amplitude
    if allow_amplitude and amplitude>1e-8:
        amp_basis=np.stack([carrier,carrier*u,carrier*v],axis=-1)
        design=np.concatenate([amp_basis,plane],axis=-1)
        root=np.sqrt(weight.ravel())
        amp_coef=np.linalg.lstsq(design.reshape(-1,6)*root[:,None],a.ravel()*root,rcond=None)[0]
        # Nonnegative envelope avoids extending a local stripe with inverted phase.
        field=np.clip(amp_coef[0]+amp_coef[1]*u+amp_coef[2]*v,0,2.5*amplitude)
        trial_noise=field*carrier
        nuisance=np.linalg.lstsq(plane.reshape(-1,3)*root[:,None],
                                 (a-trial_noise).ravel()*root,rcond=None)[0]
        trial_sse=float(np.sum(weight*(a-trial_noise-plane@nuisance)**2))
        amplitude_gain=float((best['sse']-trial_sse)/max(baseline_sse-best['sse'],1e-15))
        if force_amplitude or amplitude_gain>cfg.amplitude_gain_threshold+complexity_penalty:
            noise=trial_noise;envelope=field;model+='_affine_amplitude';final_sse=trial_sse
            prediction=trial_noise+plane@nuisance
        else:final_sse=best['sse']
    else:final_sse=best['sse']
    fit_ratio=float(np.clip(np.sum(weight*noise**2)/max(baseline_sse,1e-15),0,1))
    improvement=float(np.clip((baseline_sse-final_sse)/max(baseline_sse,1e-15),0,1))
    params=best['parameters']
    # Curvature has normalized coordinates; convert Hessian to cycles/pixel^2.
    h,w=a.shape
    # The norm of an affine frequency field can attain its minimum inside
    # the tile even when every corner is in band. Check every actual pixel,
    # using the same coordinates as the phase fit (not an enlarged +/- .5
    # rectangle). This is a discrete-pixel constraint, not a proof about all
    # points in the continuous window.
    frequency_y=best['fy']+(params[2]*u+params[3]*v)/h
    frequency_x=best['fx']+(params[3]*u+params[4]*v)/w
    frequency_radius=np.hypot(frequency_y,frequency_x)
    frequency_min=float(frequency_radius.min());frequency_max=float(frequency_radius.max())
    if np.any((frequency_radius<cfg.min_frequency)|(frequency_radius>cfg.max_frequency)):
        # The diagnostic prediction and SSE must describe the same fallback.
        noise=np.zeros_like(a);fit_ratio=0.;improvement=0.;amplitude=0.;envelope[:]=0.
        model='rejected_frequency_range'
        prediction=plane@baseline_coef
        final_sse=baseline_sse
    return dict(residual=noise,prediction=prediction,amplitude=float(np.sqrt(np.sum(weight*envelope**2)/weight.sum())),
                fit_ratio=fit_ratio,fit_improvement=improvement,
                fy=best['fy'],fx=best['fx'],model=model,
                chirp_gain=chirp_gain,amplitude_gain=amplitude_gain,
                hessian_yy=float(params[2]/h**2),hessian_yx=float(params[3]/(h*w)),hessian_xx=float(params[4]/w**2),
                final_sse=float(final_sse),baseline_sse=baseline_sse,
                frequency_min=frequency_min,frequency_max=frequency_max,
                frequency_range_scope='all discrete pixel centres')


@lru_cache(maxsize=32)
def _holdout_mask(shape,fraction,seed):
    """Fixed random positions within every 4x4 spatial stratum, not one parity."""
    h,w=shape;rng=np.random.default_rng(seed);mask=np.zeros(shape,dtype=bool)
    for y in range(0,h,4):
        for x in range(0,w,4):
            hh,ww=min(4,h-y),min(4,w-x);n=hh*ww
            if n<=1:continue
            count=max(1,min(n-1,int(round(n*fraction))))
            chosen=rng.choice(n,count,replace=False)
            mask[y+chosen//ww,x+chosen%ww]=True
    mask.setflags(write=False)
    return mask


def _model_complexity(model):
    return 7+3*int('chirp' in model)+2*int('affine_amplitude' in model)


def _model_alternatives(patch,pool,cfg):
    """Refine a selected coarse candidate, or validate <=2 distinct frequencies."""
    ranked=sorted(pool,key=lambda c:(-c['selection_score'],c['analysis_window_id']))
    if cfg.refine_candidates==1:
        c=ranked[0];result=fit_candidate(patch,c,cfg);result.pop('prediction')
        return [dict(candidate=c,fit=result,validation_mse=None,validation_score=None,
                     validation_baseline_mse=None,complexity=_model_complexity(result['model']))]
    distinct=[]
    for candidate in ranked:
        if any(_frequency_distance(candidate['fy'],candidate['fx'],c['fy'],c['fx'])<cfg.frequency_merge_tolerance
               for c in distinct):continue
        distinct.append(candidate)
        if len(distinct)>=cfg.refine_candidates:break
    validation=_holdout_mask(patch.shape,cfg.validation_fraction,cfg.validation_seed)
    train=~validation;weight=_geometry(patch.shape)[4]
    plane=_geometry(patch.shape)[5];root=np.sqrt((weight*train).ravel())
    baseline_coef=np.linalg.lstsq(plane.reshape(-1,3)*root[:,None],patch.ravel()*root,rcond=None)[0]
    baseline_mse=float(np.sum(weight[validation]*(patch-plane@baseline_coef)[validation]**2)/weight[validation].sum())
    alternatives=[]
    for c in distinct:
        fitted=fit_candidate(patch,c,cfg,fit_mask=train)
        prediction=fitted.pop('prediction')
        validation_mse=float(np.sum(weight[validation]*(patch-prediction)[validation]**2)/weight[validation].sum())
        complexity=_model_complexity(fitted['model'])
        score=validation_mse/max(baseline_mse,1e-12)+cfg.rerank_complexity_penalty*max(complexity-7,0)/5
        if fitted['model']=='rejected_frequency_range':score=1e12
        else:
            # Freeze the discrete model family selected on training pixels,
            # then refit its parameters using every pixel of the block.
            fitted=fit_candidate(patch,c,cfg,force_model=fitted['model']);fitted.pop('prediction')
        alternatives.append(dict(candidate=c,fit=fitted,validation_mse=validation_mse,
            validation_baseline_mse=baseline_mse,validation_score=float(score),complexity=complexity))
    return sorted(alternatives,key=lambda m:(m['validation_score'],-m['candidate']['selection_score']))


def _install_alternative(tile,alternative):
    c=alternative['candidate'];fitted=alternative['fit']
    tile['selected']=dict(c,fy=fitted['fy'],fx=fitted['fx'],model=fitted['model'])
    tile.update(fitted)
    tile['validation_mse']=alternative['validation_mse'];tile['validation_score']=alternative['validation_score']


def _overlap_discrepancy(tile,noise,neighbor,neighbor_noise):
    y0=max(tile['y'],neighbor['y']);x0=max(tile['x'],neighbor['x'])
    y1=min(tile['y']+tile['h'],neighbor['y']+neighbor['h'])
    x1=min(tile['x']+tile['w'],neighbor['x']+neighbor['w'])
    if y1<=y0 or x1<=x0:return None
    a=noise[y0-tile['y']:y1-tile['y'],x0-tile['x']:x1-tile['x']]
    b=neighbor_noise[y0-neighbor['y']:y1-neighbor['y'],x0-neighbor['x']:x1-neighbor['x']]
    ea=float(np.mean(a*a));eb=float(np.mean(b*b))
    # Small/absent residual support at a local interference boundary is not
    # forced to match the amplitude of an adjacent contaminated block.
    presence=2*np.sqrt(ea*eb)/(ea+eb+1e-15)
    return float(np.clip(np.mean((a-b)**2)/(ea+eb+1e-15)*presence,0,2))


def _rerank_overlap(tiles,cfg):
    by_grid={(t['gy'],t['gx']):t for t in tiles}
    # One deterministic Jacobi pass: every comparison uses initial choices.
    initial={(t['gy'],t['gx']):t['residual'] for t in tiles}
    choices=[]
    for t in tiles:
        alternatives=t.get('_alternatives',[])
        for m in alternatives:
            differences=[]
            for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
                q=by_grid.get((t['gy']+dy,t['gx']+dx))
                if q is None or q['selected'] is None:continue
                value=_overlap_discrepancy(t,m['fit']['residual'],q,initial[(q['gy'],q['gx'])])
                if value is not None:differences.append(value)
            discrepancy=float(np.mean(differences)) if differences else 0.
            m['overlap_discrepancy']=discrepancy
            m['rerank_score']=float(m['validation_score']+cfg.overlap_penalty*discrepancy)
        choices.append(min(alternatives,key=lambda m:(m['rerank_score'],-m['candidate']['selection_score'])) if alternatives else None)
    for t,choice in zip(tiles,choices):
        if choice is None:continue
        before=t['selected'];_install_alternative(t,choice)
        t['overlap_discrepancy']=choice['overlap_discrepancy']
        t['overlap_reranked']=bool(_frequency_distance(before['fy'],before['fx'],t['selected']['fy'],t['selected']['fx'])>1e-8)


def prepare(image,config: EstimationConfig | None = None):
    cfg=config or EstimationConfig();a=_checked(image).copy();h,w=a.shape
    sy,sx=min(h,cfg.synthesis_size),min(w,cfg.synthesis_size)
    ys,xs=_starts(h,sy,cfg.stride),_starts(w,sx,cfg.stride)
    cache={};tiles=[]
    for gy,y in enumerate(ys):
        for gx,x in enumerate(xs):
            candidates=[]
            windows={}
            for scale in sorted(set(cfg.scales)):
                ah,aw=min(scale,h),min(scale,w)
                ay=min(max(round(y+sy/2-ah/2),0),h-ah);ax=min(max(round(x+sx/2-aw/2),0),w-aw)
                key=(ay,ax,ah,aw)
                windows.setdefault(key,[]).append(int(scale))
            for key,requested_scales in windows.items():
                ay,ax,ah,aw=key
                if key not in cache:cache[key]=spectral_candidates(a[ay:ay+ah,ax:ax+aw],cfg)
                requested=(cfg.force_scale if cfg.force_scale in requested_scales else min(requested_scales))
                candidates.extend(dict(c,requested_scale=requested,requested_scales=list(requested_scales),
                                       analysis_window_id=key) for c in cache[key])
            t=dict(y=y,x=x,h=sy,w=sx,gy=gy,gx=gx,candidates=candidates,selected=None,
                   residual=np.zeros((sy,sx)),amplitude=0.,fit_ratio=0.,fit_improvement=0.,
                   cross_scale=0.,cross_window=0.,different_window_support=False,
                   neighbor=0.,confidence=0.,candidate_models=[],overlap_discrepancy=0.,overlap_reranked=False,
                   effective_window_count=len(windows),validation_mse=None,validation_score=None)
            pool=[c for c in candidates if cfg.force_scale is None or cfg.force_scale in c['requested_scales']]
            patch=a[y:y+sy,x:x+sx]
            if pool:
                # Cheap local fits score all scales/candidates on the same pixels.
                yy,xx,_,_,weight,plane,_=_geometry(patch.shape)
                for c in pool:
                    phase=2*np.pi*(c['fy']*yy+c['fx']*xx)
                    coef,fitted,sse=_fit_constant(patch,phase)
                    amp=float(np.hypot(coef[0],coef[1]))
                    # The local captured energy rejects peaks belonging mainly
                    # to a distant part of a large analysis window.
                    score=amp**2*max(c['evidence'],.05)*(1+.04*np.log2(c['scale']/min(cfg.scales)))
                    c['local_amplitude']=amp;c['selection_score']=float(score)
                alternatives=_model_alternatives(patch,pool,cfg)
                t['_alternatives']=alternatives
                _install_alternative(t,alternatives[0])
            tiles.append(t)
    if cfg.use_overlap_reranking:_rerank_overlap(tiles,cfg)
    by_grid={(t['gy'],t['gx']):t for t in tiles}
    for t in tiles:
        c=t['selected']
        if c is None:continue
        # Different requested scales may produce the exact same clipped window.
        # Only distinct effective windows contribute additional evidence; these
        # overlapping windows are not claimed to be statistically independent.
        matches=[]
        for other in t['candidates']:
            if other['analysis_window_id']==c['analysis_window_id']:continue
            tol=max(1.5/other['scale'],1.5/c['scale'])
            matches.append(float(np.exp(-.5*(_frequency_distance(c['fy'],c['fx'],other['fy'],other['fx'])/tol)**2)))
        t['cross_scale']=t['cross_window']=max(matches,default=0.)
        t['different_window_support']=bool(matches)
        t['candidate_models']=[dict(frequency_y=m['fit']['fy'],frequency_x=m['fit']['fx'],model=m['fit']['model'],
            analysis_window_id=m['candidate']['analysis_window_id'],validation_mse=m['validation_mse'],
            validation_score=m['validation_score'],validation_baseline_mse=m['validation_baseline_mse'],
            complexity=m['complexity'],overlap_discrepancy=m.get('overlap_discrepancy'),
            rerank_score=m.get('rerank_score'),chirp_gain=m['fit']['chirp_gain'],amplitude_gain=m['fit']['amplitude_gain'])
            for m in t.get('_alternatives',[])]
        matches=[]
        for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
            q=by_grid.get((t['gy']+dy,t['gx']+dx))
            if not q or q['selected'] is None:continue
            # Predict neighbor centre frequency under the selected chirp model.
            cy=(q['y']+q['h']/2)-(t['y']+t['h']/2);cx=(q['x']+q['w']/2)-(t['x']+t['w']/2)
            py=c['fy']+t['hessian_yy']*cy+t['hessian_yx']*cx
            px=c['fx']+t['hessian_yx']*cy+t['hessian_xx']*cx
            distance=_frequency_distance(py,px,q['selected']['fy'],q['selected']['fx'])
            matches.append(float(np.exp(-.5*(distance/cfg.neighbor_tolerance)**2)))
        t['neighbor']=float(np.mean(matches)) if matches else 0.
        t['confidence']=float(c['evidence']*(.25+.75*t['neighbor'])*(.5+.5*t['cross_scale']))
    for t in tiles:t.pop('_alternatives',None)
    return PreparedV3(a,cfg,tiles,len(cache))


def restore(prepared,weights=None,strength=1.,return_diagnostics=False):
    """Normalized overlap subtraction with caller-supplied per-tile weights."""
    p=prepared
    if not np.isfinite(strength) or not 0<=strength<=2:raise ValueError('strength must be in [0,2]')
    if weights is None:weights=[t['confidence'] for t in p.tiles]
    weights=np.asarray(weights,float)
    if weights.shape!=(len(p.tiles),) or not np.isfinite(weights).all() or np.any((weights<0)|(weights>1)):
        raise ValueError('weights must be finite [0,1], one per tile')
    numerator=np.zeros(p.image.shape);denominator=np.zeros_like(numerator)
    maps={k:np.zeros_like(numerator) for k in ('frequency_y','frequency_x','selected_scale','confidence','suppression','model_chirp','model_affine_amplitude')}
    records=[]
    for t,confidence in zip(p.tiles,weights):
        sl=np.s_[t['y']:t['y']+t['h'],t['x']:t['x']+t['w']]
        weight=_geometry((t['h'],t['w']))[4];c=t['selected']
        gain=min(float(strength*confidence),1.) if c else 0.
        numerator[sl]+=weight*gain*t['residual'];denominator[sl]+=weight
        values=dict(frequency_y=c['fy'] if c else 0.,frequency_x=c['fx'] if c else 0.,
                    selected_scale=c['scale'] if c else 0.,confidence=float(confidence),suppression=gain,
                    model_chirp=float(c is not None and 'chirp' in c['model']),
                    model_affine_amplitude=float(c is not None and 'affine_amplitude' in c['model']))
        for k,value in values.items():maps[k][sl]+=weight*value
        records.append({k:v for k,v in t.items() if k!='residual'}|values)
    if np.any(denominator<=0):raise RuntimeError('Overlap weights do not cover image')
    raw=numerator/denominator;output=np.clip(p.image-raw,0,1)
    if not return_diagnostics:return output
    for k in maps:maps[k]/=denominator
    maps.update(residual=p.image-output,raw_residual=raw,tiles=records,weight_sum=denominator,analysis_count=p.analysis_count)
    return output,maps


def denoise(image,config=None,weights=None,strength=1.,return_diagnostics=False):
    return restore(prepare(image,config),weights,strength,return_diagnostics)


__all__=['EstimationConfig','PreparedV3','prepare','restore','denoise','spectral_candidates','fit_candidate']
