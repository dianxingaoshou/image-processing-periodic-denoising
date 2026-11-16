"""Reproducible exploratory demo: automatic periodic-noise notch filtering.

Run: python demo.py --out results
Own image: python demo.py --input noisy.png --out own_result
No neural models; no clean reference or noise parameters are passed to the detector.
This is a course-project starting point, NOT a reproduction of the published AGNF.
"""
from __future__ import annotations
import argparse
import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
import skimage
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy import ndimage as ndi
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


@dataclass(frozen=True)
class Config:
    threshold: float = 6.5
    median_window: int = 15
    dc_radius: float = 8.0
    min_distance: float = 5.0
    max_pairs: int = 8
    bandwidth_patch_radius: int = 7
    bandwidth_scale: float = 3.0
    sigma_min: float = 0.6
    sigma_max: float = 4.0
    fixed_sigma: float = 2.0
    strength: float = 1.0

    def __post_init__(self) -> None:
        if self.threshold <= 0 or self.median_window < 3 or self.median_window % 2 != 1:
            raise ValueError('threshold must be positive and median_window odd and >= 3')
        if not 0 < self.strength <= 1:
            raise ValueError('strength must be in (0, 1]')
        if self.max_pairs < 1 or self.min_distance <= 0 or self.dc_radius < 0:
            raise ValueError('invalid peak detection configuration')
        if self.sigma_min <= 0 or self.sigma_max < self.sigma_min or self.fixed_sigma <= 0:
            raise ValueError('invalid bandwidth configuration')


def checked_image(image: np.ndarray) -> np.ndarray:
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2 or min(a.shape) < 32:
        raise ValueError('Expected a grayscale image with both dimensions >= 32.')
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError('Expected finite pixel values in [0, 1].')
    return a


def conjugate_position(y: int, x: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    # Correct for even AND odd array sizes in fftshift coordinates.
    return ((2 * (h // 2) - y) % h, (2 * (w // 2) - x) % w)


def detect_peaks(image: np.ndarray, cfg: Config):
    """Return original complex spectrum, detected peak pairs and detection log spectrum."""
    a = checked_image(image)
    h, w = a.shape
    spectrum = np.fft.fftshift(np.fft.fft2(a))
    # Hann is ONLY used for detection. Reconstruction uses the unwindowed spectrum.
    window = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    det = np.fft.fftshift(np.fft.fft2((a - a.mean()) * window))
    logmag = np.log(np.abs(det) + 1e-8)
    background = ndi.median_filter(logmag, size=cfg.median_window, mode='wrap')
    residual = logmag - background
    yy, xx = np.indices(a.shape)
    valid = np.hypot(yy - h // 2, xx - w // 2) > cfg.dc_radius
    samples = residual[valid]
    if not samples.size:
        return spectrum, [], logmag
    median = np.median(samples)
    scale = 1.4826 * np.median(np.abs(samples - median)) + 1e-8
    score = (residual - median) / scale
    candidate_mask = (valid & (score > cfg.threshold) & (residual > np.log(4.0))
                      & (score == ndi.maximum_filter(score, size=5, mode='wrap')))
    candidates = np.argwhere(candidate_mask)
    # Deterministic tie-breaking; the conjugate of a peak is not a second pair.
    candidates = sorted(candidates.tolist(), key=lambda p: (-score[p[0], p[1]], p[0], p[1]))
    peaks: list[dict] = []
    selected: list[tuple[int, int]] = []
    for py, px in candidates:
        if any(np.hypot(min(abs(py-qy), h-abs(py-qy)),
                        min(abs(px-qx), w-abs(px-qx))) < cfg.min_distance
               for qy, qx in selected):
            continue
        qy, qx = conjugate_position(py, px, a.shape)
        if min(score[py, px], score[qy, qx]) < cfg.threshold:
            continue
        peaks.append({'y': py, 'x': px, 'score': float(score[py, px])})
        selected.extend([(py, px), (qy, qx)])
        if len(peaks) >= cfg.max_pairs:
            break
    return spectrum, peaks, logmag


def estimate_bandwidth(spectrum: np.ndarray, py: int, px: int, cfg: Config):
    """Heuristic bandwidth from local excess-power second moments (not ground truth)."""
    h, w = spectrum.shape
    r = cfg.bandwidth_patch_radius
    t = np.arange(-r, r + 1)
    dy, dx = np.meshgrid(t, t, indexing='ij')
    power = np.abs(spectrum[np.ix_((py+t) % h, (px+t) % w)]) ** 2
    outer = (np.abs(dy) >= max(1, r-2)) | (np.abs(dx) >= max(1, r-2))
    background = np.median(power[outer])
    excess = np.maximum(power-background, 0)
    excess[excess < excess.max() * 0.005] = 0
    mass = excess.sum() + 1e-12
    vy = float(np.sum(excess * dy**2) / mass)
    vx = float(np.sum(excess * dx**2) / mass)
    sy = float(np.clip(cfg.bandwidth_scale*np.sqrt(vy), cfg.sigma_min, cfg.sigma_max))
    sx = float(np.clip(cfg.bandwidth_scale*np.sqrt(vx), cfg.sigma_min, cfg.sigma_max))
    return sy, sx


def apply_notches(image: np.ndarray, spectrum: np.ndarray, peaks: list[dict],
                  cfg: Config, adaptive: bool = True):
    h, w = image.shape
    yy, xx = np.indices(image.shape)
    mask = np.ones_like(image, dtype=np.float64)
    details = []
    for peak in peaks:
        py, px = peak['y'], peak['x']
        sy, sx = (estimate_bandwidth(spectrum, py, px, cfg) if adaptive
                  else (cfg.fixed_sigma, cfg.fixed_sigma))
        details.append({**peak, 'fy_bins': py-h//2, 'fx_bins': px-w//2,
                        'sigma_y_bins': sy, 'sigma_x_bins': sx})
        for cy, cx in {(py, px), conjugate_position(py, px, image.shape)}:
            dy = np.abs(yy-cy); dy = np.minimum(dy, h-dy)
            dx = np.abs(xx-cx); dx = np.minimum(dx, w-dx)
            mask *= 1 - cfg.strength * np.exp(-0.5*((dy/sy)**2 + (dx/sx)**2))
    mask[h//2, w//2] = 1.0  # Keep DC exactly: do not change mean before clipping.
    if not peaks:
        return image.copy(), mask, details, 0.0
    result_complex = np.fft.ifft2(np.fft.ifftshift(spectrum * mask))
    imaginary_max = float(np.abs(result_complex.imag).max())
    if imaginary_max > 1e-9:
        raise RuntimeError('Notch mask lost conjugate symmetry.')
    # Do NOT take abs(ifft2(...)); do NOT min-max normalize the output.
    return np.clip(result_complex.real, 0, 1), mask, details, imaginary_max


def denoise(image: np.ndarray, cfg: Config = Config(), adaptive: bool = True):
    a = checked_image(image)
    spectrum, peaks, _ = detect_peaks(a, cfg)
    restored, mask, details, _ = apply_notches(a, spectrum, peaks, cfg, adaptive)
    return restored, details, mask


def synthesize(clean: np.ndarray, components: list[tuple[float, float, float, float]]):
    """Each component is (fy cycles/image, fx cycles/image, amplitude, phase)."""
    clean = checked_image(clean)
    h, w = clean.shape
    y, x = np.indices(clean.shape)
    noise = np.zeros_like(clean)
    for fy, fx, amplitude, phase in components:
        noise += amplitude*np.sin(2*np.pi*(fy*y/h + fx*x/w) + phase)
    raw = clean + noise
    clipping_fraction = float(np.mean((raw < 0) | (raw > 1)))
    return np.clip(raw, 0, 1), clipping_fraction


def metrics(clean: np.ndarray, output: np.ndarray):
    mse = float(np.mean((clean-output)**2))
    return {'psnr_db': float('inf') if mse == 0 else float(peak_signal_noise_ratio(clean, output, data_range=1)),
            'ssim': float(structural_similarity(clean, output, data_range=1)),
            'mae': float(np.mean(np.abs(clean-output)))}


def save_gray(path: Path, image: np.ndarray):
    Image.fromarray(np.rint(np.clip(image, 0, 1)*255).astype(np.uint8)).save(path)


def comparison_image(path: Path, clean: np.ndarray, noisy: np.ndarray, output: np.ndarray,
                     title: str):
    # Image contact sheet, not a generated illustration. All panels use the SAME [0, 1] mapping.
    arrays = [clean, noisy, output]
    names = ['Reference', 'Noisy', 'Adaptive notch']
    cell_w, cell_h, header = 360, 360, 42
    sheet = Image.new('RGB', (3*cell_w, header+cell_h+50), 'white')
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('DejaVuSans.ttf', 17)
    except OSError:
        font = ImageFont.load_default()
    draw.text((12, 10), title, fill='black', font=font)
    for i, (arr, name) in enumerate(zip(arrays, names)):
        tile = Image.fromarray(np.rint(np.clip(arr, 0, 1)*255).astype(np.uint8)).convert('RGB')
        tile.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
        sheet.paste(tile, (i*cell_w+(cell_w-tile.width)//2, header+(cell_h-tile.height)//2))
        if i:
            mm = metrics(clean, arr)
            name += f"  {mm['psnr_db']:.2f} dB"
        draw.text((i*cell_w+10, header+cell_h+12), name, fill='black', font=font)
    sheet.save(path)


def process_case(clean, noisy, outdir, cfg, title, reference_available=True):
    outdir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    spectrum, peaks, logmag = detect_peaks(noisy, cfg)
    detection_ms = (time.perf_counter()-start)*1000
    outputs = {'noisy': noisy}
    times = {'noisy': 0.0}
    for name, fn in [('gaussian_sigma1', lambda: ndi.gaussian_filter(noisy, 1, mode='reflect')),
                     ('median_3x3', lambda: ndi.median_filter(noisy, 3, mode='reflect'))]:
        start = time.perf_counter(); outputs[name] = fn()
        times[name] = (time.perf_counter()-start)*1000
    details = []
    for name, adaptive in [('fixed_notch_sigma2', False), ('adaptive_notch', True)]:
        start = time.perf_counter()
        result, mask, dd, imaginary_max = apply_notches(noisy, spectrum, peaks, cfg, adaptive)
        times[name] = (time.perf_counter()-start)*1000 + detection_ms
        outputs[name] = result
        if adaptive:
            details = dd
            save_gray(outdir/'notch_mask.png', mask)
    # Spectrum visualizations use the SAME display limits; this does not affect metrics.
    log_before = np.log1p(np.abs(spectrum))
    log_after = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(outputs['adaptive_notch']))))
    low, high = np.percentile(log_before, [1.0, 99.9])
    denom = max(float(high-low), 1e-12)
    save_gray(outdir/'spectrum_before.png', (log_before-low)/denom)
    save_gray(outdir/'spectrum_after.png', (log_after-low)/denom)
    save_gray(outdir/'noisy.png', noisy)
    save_gray(outdir/'restored.png', outputs['adaptive_notch'])
    # Save floats: PNG images are only for display and are quantized to 8 bits.
    float_arrays = dict(outputs)
    if reference_available:
        float_arrays['reference'] = clean
        save_gray(outdir/'reference.png', clean)
        comparison_image(outdir/'comparison.png', clean, noisy, outputs['adaptive_notch'], title)
    np.savez_compressed(outdir/'arrays.npz', **float_arrays)
    (outdir/'peaks.json').write_text(json.dumps({'peaks': details, 'config': asdict(cfg)},
                                              indent=2, ensure_ascii=False), encoding='utf-8')
    rows = []
    for name, result in outputs.items():
        row = {'method': name, 'runtime_ms': times[name], 'detected_peak_pairs': len(peaks)}
        if reference_available:
            row.update(metrics(clean, result))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--input', type=Path, help='Optional image to denoise (converted to 8-bit gray).')
    parser.add_argument('--threshold', type=float, default=6.5)
    parser.add_argument('--strength', type=float, default=1.0)
    parser.add_argument('--names', nargs='+', choices=['camera', 'coins', 'page', 'brick', 'grass'],
                        default=['camera', 'coins', 'page', 'brick', 'grass'])
    args = parser.parse_args()
    cfg = Config(threshold=args.threshold, strength=args.strength)
    args.out.mkdir(parents=True, exist_ok=True)
    environment = {'python': platform.python_version(), 'numpy': np.__version__,
                   'scipy': scipy.__version__, 'scikit_image': skimage.__version__,
                   'config': asdict(cfg), 'evaluation_note': 'Exploratory, NOT a held-out benchmark.'}
    (args.out/'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    if args.input:
        if not args.input.is_file():
            parser.error(f'Input file not found: {args.input}')
        with Image.open(args.input) as im:
            noisy = np.asarray(ImageOps.exif_transpose(im).convert('L'), dtype=np.float64)/255
        process_case(noisy, noisy, args.out/'own_image', cfg, args.input.name, False)
        print('Saved restored.png, notch_mask.png, peaks.json and arrays.npz in', args.out/'own_image')
        return
    scenarios = {
        'integer_single': [(23., 37., .12, .7)],
        'offgrid_single': [(23.35, 37.65, .12, .7)],
        'integer_double': [(23., 37., .12, .7), (-41., 13., .12, .7)],
        'no_noise': []}
    (args.out/'noise_config.json').write_text(json.dumps(scenarios, indent=2), encoding='utf-8')
    rows = []
    for name in args.names:
        clean = img_as_float(getattr(data, name)())
        for case, components in scenarios.items():
            noisy, fraction = synthesize(clean, components)
            rr = process_case(clean, noisy, args.out/f'{name}_{case}', cfg, f'{name}: {case}')
            for row in rr:
                rows.append({'image': name, 'case': case, 'clipping_fraction': fraction, **row})
            result = next(r for r in rr if r['method'] == 'adaptive_notch')
            print(f"{name:>6} {case:>15}: PSNR={result['psnr_db']:.2f}, SSIM={result['ssim']:.4f}", flush=True)
    # Deliberate counterexample: a CLEAN sinusoidal texture is not necessarily noise.
    y, x = np.indices((256, 256))
    clean_texture = 0.5 + 0.15*np.sin(2*np.pi*(23*y/256 + 37*x/256)+.7)
    rr = process_case(clean_texture, clean_texture, args.out/'clean_periodic_texture', cfg,
                      'Failure case: legitimate periodic texture (NO noise added)')
    for row in rr:
        rows.append({'image': 'synthetic_periodic_texture', 'case': 'no_noise',
                     'clipping_fraction': 0., **row})
    with (args.out/'metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print('All results:', args.out.resolve())


if __name__ == '__main__':
    main()
