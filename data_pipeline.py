"""Reproducible BSDS500 image split and synthetic periodic degradations.

The synthetic truth returned by :func:`synthesize` is evaluation-only data.  A
denoiser must receive only ``noisy``.  No segmentation annotations are extracted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

SOURCE_URL = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
SOURCE_PAGE = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html"
KINDS = ("global", "local", "chirp", "orientation")
AMPLITUDES = (0.06, 0.12, 0.18)
DEFAULT_SEED = 20261116


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(path: Path) -> None:
    """Stream to a temporary file; an interrupted download is never accepted."""
    import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".part")
    with requests.get(SOURCE_URL, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                if block:
                    stream.write(block)
    temporary.replace(path)


def _extract_images(archive: Path, destination: Path) -> list[Path]:
    """Copy regular JPEG members only, after validating every member path.

    Does not use ``extractall`` and does not extract symlinks or MATLAB files.
    All output paths are constructed from an allowed split and filename.
    """
    destination.mkdir(parents=True, exist_ok=True)
    extracted = []
    with tarfile.open(archive, "r:gz") as package:
        for member in package:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive member: {member.name}")
            parts = relative.parts
            if (not member.isfile() or len(parts) < 3
                    or parts[-3] != "images" or parts[-2] not in {"train", "val", "test"}
                    or relative.suffix.lower() != ".jpg"):
                continue
            output = destination / parts[-2] / parts[-1]
            output.parent.mkdir(parents=True, exist_ok=True)
            resolved = output.resolve()
            if not resolved.is_relative_to(destination.resolve()):
                raise ValueError("Image output escaped the dataset root")
            stream = package.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable image member: {member.name}")
            output.write_bytes(stream.read())
            extracted.append(output)
    if len(extracted) != 500:
        raise ValueError(f"Expected 500 BSDS500 images, found {len(extracted)}")
    return sorted(extracted)


def _preprocess(path: Path, size: int) -> np.ndarray:
    """Center crop after Lanczos resize of the short side; preserve aspect ratio."""
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        factor = size / min(width, height)
        rgb = rgb.resize((round(width * factor), round(height * factor)), Image.Resampling.LANCZOS)
        left = (rgb.width - size) // 2
        top = (rgb.height - size) // 2
        a = np.asarray(rgb.crop((left, top, left + size, top + size)), dtype=np.float64) / 255.0
    # Same luminance weights as skimage.color.rgb2gray; no per-image rescaling.
    return a @ np.array([0.2125, 0.7154, 0.0721])


def build_dataset(root: str | Path = "data", seed: int = DEFAULT_SEED,
                  n_dev: int = 30, n_test: int = 60, size: int = 256,
                  download: bool = True) -> dict:
    """Create one frozen image-level split and reproducible float references."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = load_manifest(root)
        config = existing["selection"]
        requested = dict(seed=seed, n_dev=n_dev, n_test=n_test, size=size)
        if config != requested:
            raise ValueError(f"Existing manifest has {config}; use another root for {requested}")
        if all((root / item["clean_npy"]).exists() for item in existing["images"]):
            return existing
    if n_dev < 1 or n_test < 1 or n_dev > 300 or n_test > 200 or size < 32:
        raise ValueError("Invalid image counts or image size")
    archive = root / "source" / "BSR_bsds500.tgz"
    if not archive.exists():
        if not download:
            raise FileNotFoundError(archive)
        _download(archive)
    images_root = root / "source" / "images"
    paths = _extract_images(archive, images_root)
    rng = np.random.default_rng(seed)
    dev_pool = [p for p in paths if p.parent.name in {"train", "val"}]
    test_pool = [p for p in paths if p.parent.name == "test"]
    selected = [("dev", p) for p in sorted(rng.choice(dev_pool, n_dev, replace=False))]
    selected += [("test", p) for p in sorted(rng.choice(test_pool, n_test, replace=False))]
    records = []
    for split, image_path in selected:
        image_id = image_path.stem
        clean = _preprocess(image_path, size)
        relative_npy = Path("clean") / split / f"{image_id}.npy"
        relative_png = relative_npy.with_suffix(".png")
        (root / relative_npy).parent.mkdir(parents=True, exist_ok=True)
        np.save(root / relative_npy, clean)
        Image.fromarray(np.rint(clean * 255).astype(np.uint8)).save(root / relative_png)
        records.append(dict(id=image_id, image_id=image_id, split=split, official_split=image_path.parent.name,
                            source_jpg=image_path.relative_to(root).as_posix(),
                            source_sha256=sha256(image_path), npy_path=relative_npy.as_posix(), clean_npy=relative_npy.as_posix(),
                            clean_png=relative_png.as_posix(), clean_sha256=sha256(root / relative_npy)))
    if len({r["image_id"] for r in records}) != len(records):
        raise ValueError("Image-level split is not disjoint")
    manifest = dict(schema_version=1, dataset="BSDS500 original images only", source_url=SOURCE_URL,
                    source_page=SOURCE_PAGE, archive_sha256=sha256(archive),
                    selection=dict(seed=seed, n_dev=n_dev, n_test=n_test, size=size),
                    preprocessing="RGB short-side Lanczos resize; centered square crop; float luminance "
                                  "0.2125R+0.7154G+0.0721B; no contrast normalization",
                    evaluation_note="Original image split respected. Ground-truth segmentations are neither "
                                    "extracted nor used. All synthetic variants inherit their image split.",
                    citation="P. Arbelaez, M. Maire, C. Fowlkes, J. Malik. Contour Detection and Hierarchical "
                             "Image Segmentation. IEEE TPAMI 33(5):898-916, 2011.", images=records)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_manifest(root: str | Path = "data") -> dict:
    path = Path(root)
    if path.is_dir() or path.suffix != ".json":
        path = path / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def sample_seed(image_id: str, kind: str, amplitude: float, seed: int = DEFAULT_SEED) -> int:
    """Stable across Python processes and independent of randomized hash()."""
    token = f"{seed}|{image_id}|{kind}|{amplitude:.8f}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def _compact_window(coordinate: np.ndarray, center: float, halfwidth: float) -> np.ndarray:
    distance = np.abs((coordinate - center) / halfwidth)
    result = np.zeros_like(distance, dtype=np.float64)
    inside = distance < 1
    result[inside] = 0.5 * (1 + np.cos(np.pi * distance[inside]))
    return result


def synthesize(clean: np.ndarray, kind: str, amplitude: float, seed: int):
    """Return (noisy, JSON metadata, evaluation masks/truth).

    ``amplitude`` is the maximum sinusoidal amplitude before clipping.  Local
    conditions intentionally have less total noise energy than full-image ones;
    methods are compared within condition, not by raw PSNR across conditions.
    A phase gradient defines every changing frequency field, avoiding an
    inconsistent angle field that does not integrate to a scalar phase.
    """
    image = np.asarray(clean, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 32:
        raise ValueError("Expected a grayscale image at least 32 by 32")
    if not np.isfinite(image).all() or np.min(image) < 0 or np.max(image) > 1:
        raise ValueError("Image must be finite and in [0, 1]")
    if kind not in KINDS or not np.isfinite(amplitude) or not 0 < amplitude <= 0.5:
        raise ValueError("Invalid degradation kind or amplitude")
    rng = np.random.default_rng(seed)
    h, w = image.shape
    y, x = np.indices(image.shape, dtype=np.float64)
    # Frequencies are deliberately non-integer FFT bins in most realizations.
    frequency = float(rng.uniform(0.065, 0.145))
    angle = float(rng.uniform(0.15, 1.40))
    sign = -1.0 if rng.integers(2) else 1.0
    fx = frequency * np.cos(angle)
    fy = sign * frequency * np.sin(angle)
    phase_offset = float(rng.uniform(-np.pi, np.pi))
    envelope = np.ones_like(image)
    true_fx = np.full_like(image, fx)
    true_fy = np.full_like(image, fy)
    phase_cycles = fx * x + fy * y
    parameters = dict(base_frequency_cycles_per_pixel=frequency, angle_radians=angle,
                      fx_cycles_per_pixel=float(fx), fy_cycles_per_pixel=float(fy),
                      phase_offset_radians=phase_offset)
    if kind == "local":
        cx = float(rng.uniform(0.40, 0.60) * (w - 1))
        cy = float(rng.uniform(0.40, 0.60) * (h - 1))
        wx = float(rng.uniform(0.28, 0.39) * w)
        wy = float(rng.uniform(0.28, 0.39) * h)
        envelope = _compact_window(x, cx, wx) * _compact_window(y, cy, wy)
        # Normalize to make the requested amplitude comparable across seeds.
        envelope /= max(float(envelope.max()), 1e-12)
        parameters.update(center_xy_pixels=[cx, cy], halfwidth_xy_pixels=[wx, wy],
                          envelope="product of compact raised-cosine windows")
    elif kind == "chirp":
        delta_fx = float(rng.uniform(0.035, 0.075)) * sign
        centered_x = x - (w - 1) / 2
        phase_cycles += 0.5 * delta_fx * centered_x ** 2 / max(w - 1, 1)
        true_fx += delta_fx * centered_x / max(w - 1, 1)
        envelope = 0.65 + 0.35 * (0.5 + 0.5 * np.cos(2 * np.pi * y / max(h - 1, 1)))
        parameters.update(delta_fx_cycles_per_pixel=delta_fx,
                          envelope="row-dependent 0.65 to 1 cosine amplitude")
    elif kind == "orientation":
        # Hessian off-diagonal term rotates the phase gradient across the image.
        cross = float(rng.uniform(0.065, 0.105))
        centered_x = x - (w - 1) / 2
        centered_y = y - (h - 1) / 2
        length = max(h - 1, w - 1)
        phase_cycles += cross * centered_x * centered_y / length
        true_fx += cross * centered_y / length
        true_fy += cross * centered_x / length
        envelope = 0.75 + 0.25 * (x / max(w - 1, 1))
        parameters.update(cross_phase_cycles_per_pixel=cross,
                          envelope="column-dependent 0.75 to 1 linear amplitude")
    # Explicitly zero numerically negligible taper values.  Consequently every
    # pixel labelled clean is exactly unchanged before and after clipping.
    envelope[envelope < 1e-6] = 0.0
    added_noise = amplitude * envelope * np.sin(2 * np.pi * phase_cycles + phase_offset)
    raw = image + added_noise
    noisy = np.clip(raw, 0.0, 1.0)
    contaminated = envelope > 0.01
    clean_mask = envelope < 1e-6
    transition = ~(contaminated | clean_mask)
    metadata = dict(kind=kind, amplitude=float(amplitude), seed=int(seed), shape=[h, w],
                    frequency_unit="cycles/pixel", parameters=parameters,
                    clipping_fraction=float(np.mean((raw < 0) | (raw > 1))),
                    contaminated_fraction=float(contaminated.mean()),
                    clean_fraction=float(clean_mask.mean()),
                    transition_fraction=float(transition.mean()),
                    preclip_noise_rms=float(np.sqrt(np.mean(added_noise ** 2))),
                    postclip_noise_rms=float(np.sqrt(np.mean((noisy-image) ** 2))),
                    mask_definition="contaminated: envelope>0.01; clean: envelope<1e-6; "
                                    "transition: neither; all masks evaluation only")
    masks = dict(contaminated=contaminated, clean=clean_mask, transition=transition,
                 envelope=envelope, true_noise=noisy-image, preclip_noise=added_noise,
                 true_fy=true_fy, true_fx=true_fx)
    return noisy, metadata, masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dev", type=int, default=30)
    parser.add_argument("--test", type=int, default=60)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    manifest = build_dataset(args.root, args.seed, args.dev, args.test, args.size, not args.offline)
    print(f"Prepared {len(manifest['images'])} images; archive SHA256 {manifest['archive_sha256']}")
