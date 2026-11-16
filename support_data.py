"""Paired narrow-support degradations, strictly outside inference.

Pixel widths are fixed before experiments. Edge strips extend from an image
boundary through ``width`` pixels; interior strips have total support width
``width``. None of the support masks are accepted by the denoiser.
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

from data_pipeline import _compact_window
from image_io_v3 import structure_seed
from validation_data_v3 import make_sample as legacy_sample

KINDS = ('clean', 'edge8', 'edge16', 'edge32', 'interior16', 'patch32', 'global')


def make_sample(clean, image_id, kind, amplitude=.12, variant=0):
    image = np.asarray(clean, dtype=float)
    if (image.ndim != 2 or min(image.shape) < 32 or not np.isfinite(image).all()
            or image.min() < 0 or image.max() > 1):
        raise ValueError('Expected a finite [0,1] image, dimensions >=32')
    if kind not in KINDS or not np.isfinite(amplitude) or not 0 <= amplitude <= .5:
        raise ValueError('Invalid support degradation kind or amplitude')
    if kind in ('clean', 'global'):
        observed, metadata, masks = legacy_sample(image, image_id, kind, amplitude)
    else:
        # Width changes do not change frequency, phase, orientation or amplitude.
        seed = structure_seed(str(image_id), 'local_support', seed=2026111605)
        rng = np.random.default_rng(seed)
        h, w = image.shape; yy, xx = np.indices(image.shape, dtype=float)
        frequency = float(rng.uniform(.09, .18)); angle = float(rng.uniform(.15, 1.4))
        fy, fx = frequency*np.sin(angle), frequency*np.cos(angle)
        if rng.integers(2): fy = -fy
        offset = float(rng.uniform(-np.pi, np.pi))
        side = int(variant) % 4
        if kind.startswith('edge'):
            width = int(kind.removeprefix('edge'))
            coordinate = (xx if side < 2 else yy)
            length = w if side < 2 else h
            center = 0. if side % 2 == 0 else float(length-1)
            envelope = _compact_window(coordinate, center, width)
        elif kind == 'interior16':
            width = 16
            coordinate = xx if side < 2 else yy
            length = w if side < 2 else h
            center = (.35 if side % 2 == 0 else .65)*(length-1)
            envelope = _compact_window(coordinate, center, width/2)
        else:
            width = 32
            cx = (.35 if side % 2 == 0 else .65)*(w-1)
            cy = (.35 if side < 2 else .65)*(h-1)
            envelope = _compact_window(xx, cx, width/2)*_compact_window(yy, cy, width/2)
        envelope[envelope < 1e-6] = 0.
        envelope /= max(float(envelope.max()), 1e-15)
        noise = amplitude*envelope*np.sin(2*np.pi*(fy*yy+fx*xx)+offset)
        raw = image+noise; observed = np.clip(raw, 0., 1.)
        masks = dict(contaminated=envelope > .01, clean=envelope == 0.,
            transition=(envelope > 0.) & (envelope <= .01), envelope=envelope,
            true_noise=observed-image, true_fy=np.full(image.shape, fy), true_fx=np.full(image.shape, fx))
        metadata = dict(kind=kind, seed=seed, amplitude=amplitude, variant=side,
            width_pixels=width, edge_support='width pixels inwards from boundary' if kind.startswith('edge') else None,
            interior_support='total width in pixels' if not kind.startswith('edge') else None,
            fy=float(fy), fx=float(fx), frequency=frequency, phase_offset=offset,
            clipping_fraction=float(np.mean((raw < 0) | (raw > 1))),
            contaminated_fraction=float(masks['contaminated'].mean()))
    clean_mask = np.asarray(masks['clean'], dtype=bool)
    masks['adjacent_clean'] = clean_mask & (distance_transform_edt(clean_mask) <= 16) if kind != 'clean' else np.zeros(image.shape, bool)
    return observed, metadata, masks
