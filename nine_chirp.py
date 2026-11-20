"""Discover two-dimensional quadratic phase seeds using the fitting cells only."""
from itertools import product

import numpy as np
from scipy.fft import fft2, fftfreq


# At a centered image corner, half a grid step of cross curvature is half
# a phase cycle. A coarser grid can favor an alias of the fitting checkerboard.
CURVATURE_STEP = 4
MAX_CURVATURE = 24
PEAKS_PER_SWEEP = 4
MAX_SEEDS = 12


def routing_parameters():
    return dict(curvature_step_bins=CURVATURE_STEP, maximum_curvature_bins=MAX_CURVATURE,
                peaks_per_sweep=PEAKS_PER_SWEEP, maximum_seeds=MAX_SEEDS,
                minimum_center_separation_bins=3)


def propose(feature_image, bounds, train):
    """Return [fy*h, fx*w, Hyy*h², Hyx*h*w, Hxx*w²] in centered image coordinates.

    Each trial cancels a coarse curvature and searches both frequency signs.
    Conjugacy of real images lets one representative of each ±H pair suffice.
    These seeds carry no permission to subtract a wave.
    """
    image = np.asarray(feature_image, dtype=float)
    train = np.asarray(train, dtype=bool)
    if image.ndim != 2 or image.shape != train.shape:
        raise ValueError("Discovery needs an image and a fitting mask of the same shape")
    if not train.any() or np.ptp(image[train]) == 0:
        return []
    h, w = image.shape
    fy, fx = fftfreq(h), fftfreq(w)
    radius = np.hypot(fy[:, None], fx[None, :])
    band = (radius >= bounds[0]) & (radius <= bounds[1])
    if not band.any():
        return []
    y = (np.arange(h)[:, None]-(h-1)/2)/h
    x = (np.arange(w)[None, :]-(w-1)/2)/w
    values = range(-MAX_CURVATURE, MAX_CURVATURE+1, CURVATURE_STEP)
    row_phase, cross_phase, col_phase = (
        {v: np.exp(-2j*np.pi*v*term).astype(np.complex64) for v in values}
        for term in (.5*y*y, y*x, .5*x*x)
    )
    work = np.zeros(image.shape, dtype=np.float32)
    work[train] = image[train] - image[train].mean()
    offsets = np.arange(-2, 3)
    choices = []
    for yy, yx, xx in product(values, repeat=3):
        if (yy, yx, xx) <= (0, 0, 0):
            continue
        dechirped = work * row_phase[yy] * col_phase[xx] * cross_phase[yx]
        amplitude = abs(fft2(dechirped))
        amplitude[~band] = 0
        for _ in range(PEAKS_PER_SWEEP):
            i, j = np.unravel_index(np.argmax(amplitude), image.shape)
            score = amplitude[i, j]
            if score <= 0:
                break
            seed = np.array([fy[i]*h, fx[j]*w, yy, yx, xx])
            if seed[0] < 0 or (seed[0] == 0 and seed[1] < 0):
                seed *= -1
            choices.append((float(score), seed))
            amplitude[np.ix_((i+offsets)%h, (j+offsets)%w)] = 0
    # Adjacent curvature trials give many nearly identical carrier centers.
    # Continuous phase fitting downstream resolves the curvature near each seed.
    kept = []
    for _, seed in sorted(choices, key=lambda item: -item[0]):
        if all(min(np.linalg.norm(seed[:2]-s[:2]), np.linalg.norm(seed[:2]+s[:2])) > 3 for s in kept):
            kept.append(seed)
            if len(kept) == MAX_SEEDS:
                break
    return kept
