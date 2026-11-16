"""Local multi-scale sinusoidal interference estimation, without a clean reference.

The scores below are heuristic reliability scores, not probabilities.  A single
image cannot identify a periodic component as signal or interference in general.
This implementation deliberately abstains on almost pure periodic inputs.

Only the analysis copy is Hann-windowed.  Reconstruction subtracts a locally
fitted sinusoid from the original pixels and uses normalized overlap-add.
Frequency values always use cycles/pixel, never FFT-bin coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class LocalConfig:
    scales: tuple[int, ...] = (64, 128, 256)
    stride: int = 32
    synthesis_size: int = 64
    fixed_scale: int | None = None
    selection_scale: int | None = None
    conservative: bool = True
    strength: float = 1.0
    peak_threshold: float = 5.0
    min_frequency: float = 0.025
    max_frequency: float = 0.45
    min_amplitude: float = 0.012
    neighbor_tolerance: float = 0.014
    pure_texture_threshold: float = 0.97

    def __post_init__(self):
        if not self.scales or any(s < 32 for s in self.scales):
            raise ValueError('scales must contain sizes >=32')
        if self.fixed_scale is not None and self.fixed_scale < 32:
            raise ValueError('fixed_scale must be >=32')
        if self.selection_scale is not None and self.selection_scale not in self.scales:
            raise ValueError('selection_scale must be one of scales')
        if self.fixed_scale is not None and self.selection_scale is not None:
            raise ValueError('fixed_scale and selection_scale are mutually exclusive')
        if self.synthesis_size < 32 or not 0 < self.stride <= self.synthesis_size // 2:
            raise ValueError('Require synthesis_size>=32 and 0<stride<=synthesis_size/2')
        if not 0 <= self.strength <= 2:
            raise ValueError('strength must be in [0,2]')
        if not 0 < self.min_frequency < self.max_frequency < .5:
            raise ValueError('frequency limits must satisfy 0<min<max<.5')
        if self.peak_threshold <= 0 or self.min_amplitude <= 0:
            raise ValueError('thresholds must be positive')


@dataclass
class Prepared:
    image: np.ndarray
    config: LocalConfig
    tiles: list[dict[str, Any]]
    pure_texture_score: float
    analysis_count: int


def _checked(image):
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 32:
        raise ValueError('Expected a grayscale image with dimensions >=32')
    if not np.isfinite(a).all() or np.min(a) < 0 or np.max(a) > 1:
        raise ValueError('Expected finite pixels in [0,1]')
    return a


def _starts(length, size, stride):
    if length <= size:
        return [0]
    p = list(range(0, length - size + 1, stride))
    if p[-1] != length - size:
        p.append(length - size)
    return p


def _detrend(a):
    """Remove a least-squares plane for detection, preserving the input."""
    h, w = a.shape
    y = np.arange(h) - (h - 1) / 2
    x = np.arange(w) - (w - 1) / 2
    out = a - a.mean()
    out = out - (out.mean(axis=1) @ y / max(float(y @ y), 1)) * y[:, None]
    out = out - (out.mean(axis=0) @ x / max(float(x @ x), 1)) * x[None, :]
    return out


def _canonical(fy, fx):
    return (-fy, -fx) if fy < -1e-12 or (abs(fy) < 1e-12 and fx < 0) else (fy, fx)


def _frequency_distance(fy, fx, gy, gx):
    """A real sinusoid's f and -f are the same conjugate frequency pair."""
    return min(float(np.hypot(fy-gy,fx-gx)),float(np.hypot(fy+gy,fx+gx)))


def _delta(logmag, iy, ix, axis):
    s = logmag.shape[axis]
    if axis == 0:
        lo, mid, hi = logmag[(iy - 1) % s, ix], logmag[iy, ix], logmag[(iy + 1) % s, ix]
    else:
        lo, mid, hi = logmag[iy, (ix - 1) % s], logmag[iy, ix], logmag[iy, (ix + 1) % s]
    den = lo - 2 * mid + hi
    return float(np.clip(.5 * (lo - hi) / den, -.5, .5)) if den < -1e-10 else 0.


def _analyze(a, cfg):
    h, w = a.shape
    centered = _detrend(a)
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    # Twofold padding improves continuous-frequency estimation, not resolution.
    sh = (2 * h, 2 * w)
    spectrum = np.fft.fft2(centered * win, s=sh)
    power = np.abs(spectrum) ** 2
    logmag = .5 * np.log(power + 1e-18)
    background = ndi.gaussian_filter(logmag, 5., mode='wrap')
    residual = logmag - background
    fy = np.fft.fftfreq(sh[0])[:, None]
    fx = np.fft.fftfreq(sh[1])[None, :]
    radius = np.hypot(fy, fx)
    valid = (radius >= cfg.min_frequency) & (radius <= cfg.max_frequency)
    vals = residual[valid]
    med = np.median(vals)
    mad = 1.4826 * np.median(np.abs(vals - med)) + 1e-9
    score = (residual - med) / mad
    score[~valid] = -np.inf
    iy, ix = np.unravel_index(np.argmax(score), score.shape)
    z = float(score[iy, ix])
    if z < cfg.peak_threshold:
        return None
    fyy = float(np.fft.fftfreq(sh[0])[iy] + _delta(logmag, iy, ix, 0) / sh[0])
    fxx = float(np.fft.fftfreq(sh[1])[ix] + _delta(logmag, iy, ix, 1) / sh[1])
    fyy, fxx = _canonical(fyy, fxx)
    # Width uses a fixed neighborhood expressed in original FFT bins.
    t = np.arange(-8, 9)
    local = power[np.ix_((iy + t) % sh[0], (ix + t) % sh[1])]
    outer = (np.abs(t[:, None]) >= 6) | (np.abs(t[None, :]) >= 6)
    excess = np.maximum(local - np.median(local[outer]), 0)
    mass = float(excess.sum()) + 1e-15
    wy = np.sqrt(float(np.sum(excess * t[:, None] ** 2)) / mass) / 2
    wx = np.sqrt(float(np.sum(excess * t[None, :] ** 2)) / mass) / 2
    width = float(np.hypot(wy, wx))
    # Energy in the conjugate peak neighborhoods relative to detrended energy.
    concentration = min(1., float(2 * mass / max(float(power.sum()), 1e-15)))
    amplitude = float(2 * np.abs(spectrum[iy, ix]) / max(float(win.sum()), 1e-15))
    prominence = float(np.clip((z - cfg.peak_threshold + 1.) / 3., 0., 1.))
    compactness = float(np.exp(-max(width - .95, 0) ** 2 / 1.5))
    evidence = float(prominence * compactness * np.clip(amplitude / (2 * cfg.min_amplitude), 0, 1))
    return dict(fy=fyy, fx=fxx, score=z, amplitude=amplitude, width_bins=width,
                concentration=concentration, evidence=evidence, scale=int(min(a.shape)))


def _fit(a, fy, fx):
    """Weighted sine fit plus nuisance plane, all fitted using the input alone."""
    h, w = a.shape
    yy, xx = np.indices(a.shape)
    phase = 2 * np.pi * (fy * yy + fx * xx)
    sn, cs = np.sin(phase), np.cos(phase)
    # A positive endpoint prevents a singular weighting scheme on tiny images.
    weight = np.outer(np.hanning(h + 2)[1:-1], np.hanning(w + 2)[1:-1])
    design = np.stack([sn, cs, np.ones_like(a), (yy - h/2) / h, (xx - w/2) / w], axis=-1).reshape(-1, 5)
    root = np.sqrt(weight.ravel())
    fit, *_ = np.linalg.lstsq(design * root[:, None], a.ravel() * root, rcond=None)
    n = fit[0] * sn + fit[1] * cs
    amp = float(np.hypot(fit[0], fit[1]))
    detrended = _detrend(a)
    ratio = float(np.sum(weight * n ** 2) / max(float(np.sum(weight * detrended ** 2)), 1e-15))
    return n, amp, min(ratio, 1.)


def prepare(image, config: LocalConfig | None = None):
    """Estimate all residuals once; reuse with restore for ablations/strength sweeps."""
    cfg = config or LocalConfig()
    a = _checked(image).copy()
    h, w = a.shape
    sy, sx = min(cfg.synthesis_size, h), min(cfg.synthesis_size, w)
    ys, xs = _starts(h, sy, cfg.stride), _starts(w, sx, cfg.stride)
    cache = {}
    scales = (cfg.fixed_scale,) if cfg.fixed_scale else tuple(sorted(set(cfg.scales)))
    tiles = []
    for gy, y in enumerate(ys):
        for gx, x in enumerate(xs):
            candidates = []
            for scale in scales:
                ah, aw = min(scale, h), min(scale, w)
                ay = min(max(int(round(y + sy / 2 - ah / 2)), 0), h - ah)
                ax = min(max(int(round(x + sx / 2 - aw / 2)), 0), w - aw)
                key = (ay, ax, ah, aw)
                if key not in cache:
                    cache[key] = _analyze(a[ay:ay+ah, ax:ax+aw], cfg)
                c = cache[key]
                if c is not None:
                    candidates.append(dict(c,requested_scale=int(scale)))
            tile = dict(y=y, x=x, h=sy, w=sx, gy=gy, gx=gx, candidates=candidates,
                        selected=None, residual=np.zeros((sy,sx)), amplitude=0., fit_ratio=0.,
                        cross_scale=0., neighbor=0., texture_risk=0., confidence=0.)
            selection_pool = ([c for c in candidates if c['requested_scale']==cfg.selection_scale]
                              if cfg.selection_scale is not None else candidates)
            if selection_pool:
                # Compact spectra receive a size preference. Large spread peaks
                # lose this preference, so smaller windows track varying stripes.
                for c in candidates:
                    c['selection_score'] = c['evidence'] * (1 + .12 * np.log2(c['scale'] / min(scales)))
                selected = max(selection_pool, key=lambda c: (c['selection_score'], c['scale']))
                # Weak evidence falls back to the most reliable larger scale;
                # absence at all scales leaves the tile unchanged.
                fy, fx = selected['fy'], selected['fx']
                cross = []
                for c in candidates:
                    if c is selected:
                        continue
                    tol = max(1.5 / c['scale'], 1.5 / selected['scale'])
                    distance = _frequency_distance(fy,fx,c['fy'],c['fx'])
                    cross.append(float(np.exp(-.5*(distance/tol)**2)))
                cross_scale = max(cross, default=.5 if len(scales)>1 else 1.)
                noise, amplitude, ratio = _fit(a[y:y+sy,x:x+sx], fy, fx)
                tile.update(selected=selected, residual=noise, amplitude=amplitude,
                            fit_ratio=ratio, cross_scale=cross_scale)
            tiles.append(tile)
    # A global near-perfect sinusoidal explanation is explicitly ambiguous; it
    # covers the supplied legitimate-periodic-texture failure example.
    global_candidate = _analyze(a, cfg)
    pure_score = 0.
    if global_candidate is not None:
        _, _, pure_score = _fit(a, global_candidate['fy'], global_candidate['fx'])
    by_grid = {(t['gy'], t['gx']): t for t in tiles}
    for t in tiles:
        c = t['selected']
        if c is None:
            continue
        matches = []
        for dy, dx in ((-1,0),(1,0),(0,-1),(0,1)):
            q = by_grid.get((t['gy']+dy,t['gx']+dx))
            if q is None or q['selected'] is None:
                continue
            qc = q['selected']
            distance = _frequency_distance(c['fy'],c['fx'],qc['fy'],qc['fx'])
            # Smooth local frequency changes are accepted within tolerance.
            matches.append(float(np.exp(-.5 * (distance/cfg.neighbor_tolerance)**2)))
        neighbor = float(np.mean(matches)) if matches else 0.
        # Nearly pure periodic images trigger abstention. Tile-only purity has
        # a small penalty, since smooth areas with additive stripes look alike.
        global_risk = float(np.clip((pure_score-cfg.pure_texture_threshold) / max(1-cfg.pure_texture_threshold,1e-6),0,1))
        local_risk = .2 * float(np.clip((t['fit_ratio']-.92)/.08,0,1))
        texture_risk = max(global_risk, local_risk)
        amp_gate = float(np.clip((t['amplitude']-cfg.min_amplitude)/(2*cfg.min_amplitude),0,1))
        confidence = c['evidence'] * (.25+.75*neighbor) * (.5+.5*t['cross_scale']) * amp_gate * (1-texture_risk)
        t.update(neighbor=neighbor, texture_risk=texture_risk, confidence=float(np.clip(confidence,0,1)))
    return Prepared(a, cfg, tiles, pure_score, len(cache)+1)


def restore(prepared: Prepared, strength: float | None = None,
            conservative: bool | None = None, return_diagnostics: bool = False):
    """Subtract a scored residual. Changing strength does not redo detection."""
    p = prepared
    cfg = p.config
    gain = cfg.strength if strength is None else float(strength)
    if not 0 <= gain <= 2:
        raise ValueError('strength must be in [0,2]')
    use_conservative = cfg.conservative if conservative is None else bool(conservative)
    h,w = p.image.shape
    total = np.zeros((h,w)); denominator=np.zeros((h,w))
    maps = {name:np.zeros((h,w)) for name in ('frequency_y','frequency_x','selected_scale','confidence','suppression')}
    records=[]
    for t in p.tiles:
        y,x,sy,sx = t['y'],t['x'],t['h'],t['w']
        sl = np.s_[y:y+sy,x:x+sx]
        weight=np.outer(np.hanning(sy+2)[1:-1],np.hanning(sx+2)[1:-1])
        c=t['selected']
        coefficient = min(gain * (t['confidence'] if use_conservative else (1. if c else 0.)),1.)
        total[sl] += weight * coefficient * t['residual']
        denominator[sl] += weight
        values=dict(frequency_y=c['fy'] if c else 0.,frequency_x=c['fx'] if c else 0.,
                    selected_scale=c['scale'] if c else 0.,confidence=t['confidence'],suppression=coefficient)
        for name,value in values.items():
            maps[name][sl] += weight*value
        records.append(dict(y=y,x=x,h=sy,w=sx,**values,amplitude=t['amplitude'],fit_ratio=t['fit_ratio'],
                            neighbor=t['neighbor'],cross_scale=t['cross_scale'],texture_risk=t['texture_risk'],
                            candidates=t['candidates']))
    if np.any(denominator<=0):
        raise RuntimeError('overlap-add weights failed to cover the image')
    residual=total/denominator
    result=np.clip(p.image-residual,0,1)
    if not return_diagnostics:
        return result
    for name in maps:
        maps[name] /= denominator
    maps.update(residual=p.image-result,raw_residual=residual,tiles=records,
                pure_texture_score=p.pure_texture_score,analysis_count=p.analysis_count,
                weight_sum=denominator)
    return result,maps


def denoise(image, config: LocalConfig | None = None, return_diagnostics=False):
    return restore(prepare(image,config),return_diagnostics=return_diagnostics)


__all__=['LocalConfig','Prepared','prepare','restore','denoise']
