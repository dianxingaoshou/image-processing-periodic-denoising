"""ARPENOS frequency-statistics baseline, Python port of the authors' reg=0 path.

Source: Frederic Sur and Michel Grediac, ARPENOS v1.2 (August 2016),
https://members.loria.fr/FSur/software/ARPENOS/ARPENOS.m
Paper: Automated removal of quasiperiodic noise using frequency domain
statistics, Journal of Electronic Imaging 24(1), 013003 (2015).

The authors restrict their code to academic research. This derivative retains
that academic-use restriction. Original MATLAB files and terms are included in
third_party/arpenos. The implementation preserves the default, non-TV pipeline;
its bisquare IRLS/MAD fit replaces MATLAB's proprietary robustfit statistics.
It is therefore a source-guided port, not a claim of bitwise equivalence or an
independent reproduction of the paper's published numbers. See PORT_NOTES.txt.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve2d


def _bisquare_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Intercept/slope bisquare IRLS with leverage-adjusted residuals.

    Unlike MATLAB robustfit's stats.s, the reported scale is the MAD of adjusted
    residuals divided by 0.6744897501960817. This explicit substitution is the
    main numerical deviation from the author program.
    """
    design = np.column_stack((np.ones_like(x), x))
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    q, _ = np.linalg.qr(design, mode="reduced")
    adjustment = 1 / np.sqrt(np.maximum(1 - np.sum(q * q, axis=1), 1e-6))
    scale_floor = max(float(np.std(y)) * 1e-8, 1e-10)
    for _ in range(100):
        residual = (y - design @ coef) * adjustment
        scale = max(float(np.median(np.abs(residual - np.median(residual))))
                    / 0.6744897501960817, scale_floor)
        u = residual / (4.685 * scale)
        weights = np.square(1 - np.square(u)) * (np.abs(u) < 1)
        root = np.sqrt(weights)
        updated = np.linalg.lstsq(design * root[:, None], y * root, rcond=None)[0]
        if np.linalg.norm(updated - coef) <= 1e-8 * (1 + np.linalg.norm(coef)):
            coef = updated
            break
        coef = updated
    residual = (y - design @ coef) * adjustment
    scale = max(float(np.median(np.abs(residual - np.median(residual))))
                / 0.6744897501960817, scale_floor)
    return coef, scale


def arpenos(image: np.ndarray, patch_size: int = 128, *,
            return_diagnostics: bool = False):
    """Remove globally recurring quasiperiodic components using ARPENOS reg=0.

    Parameters
    ----------
    image : finite 2-D float array
        Intensities may be in [0, 1] or [0, 255]. No clipping or rescaling is
        performed. No clean reference, injected frequencies or region labels
        are accepted by this method.
    patch_size : even integer >= 32 and smaller than both image dimensions
        L in the original program; original example scripts use L=128.
    return_diagnostics : bool
        Return (restored, dict) instead of restored when true.

    Notes
    -----
    Retains the original 1/8-patch stride, geometric mean power spectrum,
    f0=2/L, f1=0.2 regression band, f2=8/L protected low frequencies,
    3-sigma threshold, linear interpolation and 12x12 sigma=2 Gaussian mask.
    Original MATLAB patch endpoints (excluding the last possible start) are
    retained. Odd image dimensions are supported using true FFT frequencies.
    """
    im = np.asarray(image, dtype=np.float64)
    if im.ndim != 2 or not np.isfinite(im).all():
        raise ValueError("ARPENOS requires a finite 2-D image")
    if (not isinstance(patch_size, (int, np.integer)) or patch_size < 32
            or patch_size % 2 or patch_size >= min(im.shape)):
        raise ValueError("patch_size must be even, >=32, and < both image dimensions")
    length = int(patch_size)
    step = max(1, int(np.floor(length / 8 + 0.5)))
    starts_y = range(0, im.shape[0] - length, step)
    starts_x = range(0, im.shape[1] - length, step)
    count = len(starts_y) * len(starts_x)
    if np.ptp(im) <= 1e-12:
        restored = im.copy()
        diag = {"method": "arpenos_python_reg0", "patch_size": length,
                "patch_count": count, "outlier_fraction": 0.0,
                "notch_mask": np.zeros_like(im), "constant_input": True}
        return (restored, diag) if return_diagnostics else restored
    hann = np.outer(np.hanning(length), np.hanning(length))
    log_power_mean = np.zeros((length, length), dtype=np.float64)
    tiny = np.finfo(np.float64).tiny
    for y0 in starts_y:
        for x0 in starts_x:
            patch = im[y0:y0 + length, x0:x0 + length]
            amplitude = np.abs(np.fft.fft2(hann * patch))
            log_power_mean += (2 / count) * np.log(np.maximum(amplitude, tiny))
    log_power_mean = np.fft.fftshift(log_power_mean)
    axis = np.fft.fftshift(np.fft.fftfreq(length))
    fy, fx = np.meshgrid(axis, axis, indexing="ij")
    frequency = np.hypot(fx, fy)
    band = (frequency > 2 / length) & (frequency < 0.2)
    coef, scale = _bisquare_fit(np.log(frequency[band]), log_power_mean[band])
    model = coef[0] + coef[1] * np.log(np.maximum(frequency, tiny))
    outliers = ((log_power_mean - model) / scale > 3) & (frequency >= 8 / length)
    # Scaling the power avoids overflow and cancels under the subsequent >0.
    peaks = np.exp(log_power_mean - np.max(log_power_mean)) * outliers
    padded = np.pad(peaks, 1, mode="constant")
    padded_axis = np.arange(-length // 2 - 1, length // 2 + 1) / length
    interpolate = RegularGridInterpolator((padded_axis, padded_axis), padded,
                                          method="linear", bounds_error=False,
                                          fill_value=0.0)
    image_y = np.fft.fftshift(np.fft.fftfreq(im.shape[0]))
    image_x = np.fft.fftshift(np.fft.fftfreq(im.shape[1]))
    yy, xx = np.meshgrid(image_y, image_x, indexing="ij")
    mask_binary = interpolate(np.stack((yy, xx), axis=-1)) > 0
    gaussian_axis = np.arange(12) - 5.5
    gy, gx = np.meshgrid(gaussian_axis, gaussian_axis, indexing="ij")
    gaussian = np.exp(-(gx * gx + gy * gy) / (2 * 2.0 ** 2))
    gaussian /= gaussian.sum()
    # sym_conv2 pads by length(kernel) and then MATLAB conv2(..., 'same').
    # scipy's 'same' even-kernel origin differs by one pixel from MATLAB's.
    mask_padded = np.pad(mask_binary.astype(float), 12, mode="symmetric")
    full = convolve2d(mask_padded, gaussian, mode="full")
    same_matlab = full[6:6 + mask_padded.shape[0], 6:6 + mask_padded.shape[1]]
    notch = same_matlab[12:-12, 12:-12]
    notch = np.fft.fftshift(np.fft.fft2(np.real(
        np.fft.ifft2(np.fft.ifftshift(notch))))).real
    if notch.max() > 0:
        notch /= notch.max()
    spectrum = np.fft.fft2(im)
    residual = np.fft.ifft2(spectrum * np.fft.ifftshift(notch)).real
    restored = im - residual
    diagnostics = {
        "method": "arpenos_python_reg0", "patch_size": length,
        "patch_count": count, "outlier_fraction": float(outliers.mean()),
        "power_law_intercept": float(coef[0]),
        "power_law_slope": float(coef[1]), "fit_mad_scale": float(scale),
        "notch_mask": notch, "normalized_spectrum": (log_power_mean - model) / scale,
    }
    return (restored, diagnostics) if return_diagnostics else restored


def denoise(image: np.ndarray, *, return_diagnostics: bool = False):
    """Runner API, using the authors' example setting L=128 and reg=0.

    For images whose smaller dimension is <=128, use the largest even L below
    that dimension (minimum 32). This explicit small-image fallback is not used
    in the 256x256 benchmark and is recorded in diagnostics.
    """
    minimum = min(np.shape(image))
    patch_size = min(128, 2 * ((minimum - 1) // 2))
    return arpenos(image, patch_size, return_diagnostics=return_diagnostics)


__all__ = ["arpenos", "denoise"]
