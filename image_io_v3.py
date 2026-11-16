"""Shared intensity preprocessing and paired-degradation seeds for v3.

RGB values use the same floating luminance weights as the frozen experiments.
There is no contrast stretching or implicit per-image intensity normalization.
RGBA is composited against an explicit background (white by default) before
luminance conversion. Both CLI and dataset preprocessing call this conversion.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

LUMINANCE_WEIGHTS = (0.2125, 0.7154, 0.0721)
DEFAULT_STRUCTURE_SEED = 20261116


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _unit_float(array: np.ndarray) -> np.ndarray:
    source = np.asarray(array)
    if source.dtype.kind == 'b':
        result = source.astype(np.float64)
    elif source.dtype.kind == 'u':
        result = source.astype(np.float64) / np.iinfo(source.dtype).max
    elif source.dtype.kind == 'f':
        result = source.astype(np.float64, copy=False)
    else:
        raise ValueError('Image values must be unsigned integers, booleans, or floats in [0,1]')
    if result.size == 0 or not np.isfinite(result).all() or result.min() < 0 or result.max() > 1:
        raise ValueError('Expected finite image values in [0,1] after dtype scaling')
    return result


def to_float_gray(array: np.ndarray, *, rgba_background=1.0) -> np.ndarray:
    """Convert grayscale/RGB/RGBA to float64 luminance, retaining spatial size.

    Unsigned integers are scaled by their dtype maximum; floats must already be
    in [0,1]. Signed integer inputs are rejected instead of guessing their range.
    Alpha composition uses the supplied scalar or three-channel [0,1] background.
    """
    a = _unit_float(array)
    if a.ndim == 2:
        return a.copy()
    if a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError('Expected a 2-D grayscale, RGB, or RGBA image')
    rgb = a[..., :3]
    if a.shape[2] == 4:
        background = np.asarray(rgba_background, dtype=np.float64)
        if background.ndim == 0:
            background = np.full(3, float(background))
        if background.shape != (3,) or not np.isfinite(background).all() or np.any((background < 0) | (background > 1)):
            raise ValueError('RGBA background must be a scalar or RGB triple in [0,1]')
        alpha = a[..., 3:4]
        rgb = alpha * rgb + (1 - alpha) * background
    return rgb @ np.asarray(LUMINANCE_WEIGHTS, dtype=np.float64)


def _supported_pil(image: Image.Image) -> Image.Image:
    # Palette and LA transparency must not silently disappear. RGB conversion is
    # explicit for other color spaces; common grayscale dtypes remain intact.
    if image.mode in ('P', 'LA'):
        return image.convert('RGBA')
    if image.mode in ('RGB', 'RGBA', 'L', '1', 'F', 'I;16', 'I;16L', 'I;16B'):
        return image.copy()
    if image.mode == 'I':
        raise ValueError('Signed integer image mode I has ambiguous intensity range; convert explicitly')
    return image.convert('RGB')


def _metadata(path, image, output, *, geometry, rgba_background):
    background = np.asarray(rgba_background, dtype=float)
    return dict(input_path=str(Path(path).resolve()), input_sha256=sha256_file(path),
                input_mode=image.mode, input_shape=[image.height, image.width],
                output_shape=list(output.shape), output_dtype='float64', intensity_range=[0, 1],
                grayscale_rule='0.2125R+0.7154G+0.0721B',
                grayscale_weights=list(LUMINANCE_WEIGHTS), rgba_background=background.tolist(),
                alpha_policy='composite before luminance', contrast_normalization=False,
                geometry=geometry)


def load_gray(path: str | Path, *, rgba_background=1.0, return_metadata=False):
    """Load a user image without resizing, using the shared float conversion."""
    path = Path(path)
    with Image.open(path) as original:
        image = _supported_pil(original)
    output = to_float_gray(np.asarray(image), rgba_background=rgba_background)
    metadata = _metadata(path, image, output, geometry='preserve original spatial dimensions',
                         rgba_background=rgba_background)
    return (output, metadata) if return_metadata else output


def preprocess_image(path: str | Path, size: int = 256, *, rgba_background=1.0):
    """Short-side Lanczos resize and center crop, then shared float luminance.

    Returns (gray, metadata). For RGB image files this reproduces the frozen
    data_pipeline._preprocess numeric path exactly, including resize rounding.
    """
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError('size must be a positive integer')
    path = Path(path)
    with Image.open(path) as original:
        image = _supported_pil(original)
    factor = size / min(image.width, image.height)
    target = (round(image.width * factor), round(image.height * factor))
    resized = image.resize(target, Image.Resampling.LANCZOS)
    left, top = (resized.width - size) // 2, (resized.height - size) // 2
    cropped = resized.crop((left, top, left + size, top + size))
    output = to_float_gray(np.asarray(cropped), rgba_background=rgba_background)
    metadata = _metadata(path, image, output,
        geometry=dict(resize='short-side Lanczos', resized_wh=list(target), center_crop_xywh=[left, top, size, size]),
        rgba_background=rgba_background)
    return output, metadata


def structure_seed(image_id: str, kind: str, seed: int = DEFAULT_STRUCTURE_SEED) -> int:
    """Seed only the interference structure; amplitude is deliberately absent.

    This preserves the actually executed v1/v2 make_sample seed convention.
    The unused historical data_pipeline.sample_seed interface remains frozen.
    """
    token = f'{int(seed)}:{image_id}:{kind}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(token).digest()[:4], 'little')


__all__ = ['LUMINANCE_WEIGHTS', 'to_float_gray', 'load_gray', 'preprocess_image',
           'structure_seed', 'sha256_file']
