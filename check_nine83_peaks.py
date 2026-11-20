"""Post-evaluation breakdown of the frozen six-peak textile MSE on the 9+16 conditions."""
import argparse
import csv
from pathlib import Path

import numpy as np

from evaluate_nine83 import DIRECTIONS, old_focus, samples, snapshot
from evaluate_weak83 import texture_band
from evaluate_weave import read_json
from image_io_v3 import sha256_file
from project_comparison import ROOT, array_hash
from project_guarded import denoise as current
from project_pipeline import load_config
from project_weak import denoise as previous


def peak_masks(clean, recorded_peaks):
    union, peaks = texture_band(clean)
    if peaks != recorded_peaks:
        raise ValueError("Reference peak selection differs from the frozen experiment")
    h, w = clean.shape
    y = np.rint(np.fft.fftfreq(h)*h).astype(int)[:, None]
    x = np.rint(np.fft.fftfreq(w)*w).astype(int)
    disks = [((y-fy)**2+(x-fx)**2 <= 4) | ((y+fy)**2+(x+fx)**2 <= 4) for fy, fx in peaks]
    if not np.array_equal(np.sum(disks, axis=0), union.astype(int)):
        raise ValueError("Individual peak pairs must cover the published union exactly once")
    return peaks, disks


def run(folder, output):
    manifest = read_json(folder/"run.json")
    frozen = read_json(folder/"freeze.json")
    if snapshot() != frozen or any(sha256_file(folder/name) != manifest["artifact_sha256"][name]
                                   for name in ("freeze.json", "metrics.csv")):
        raise ValueError("Source or recorded metrics differ from the completed run")
    with (folder/"metrics.csv").open(encoding="utf8", newline="") as handle:
        archive = {(r["cohort"], r["image_id"], r["kind"], r["variant"], r["method"]): r
                   for r in csv.DictReader(handle)}
    plan = frozen["protocol"]
    targets = old_focus(plan)
    cfg = load_config()
    rows = []
    for case, amplitude in samples(plan, frozen["endpoint_controls"]):
        cohort, name, kind, variant, clean, observed, _ = case
        key = cohort, name, kind, str(variant)
        if key not in targets and cohort != DIRECTIONS:
            continue
        reference_sha = array_hash(clean)
        observed_sha = array_hash(observed)
        peaks, disks = peak_masks(clean, manifest["reference_texture_peaks_in_fft_bins"][reference_sha])
        values = {}
        for method, restored in (("input", observed), ("previous", previous(observed, cfg)),
                                 ("current", current(observed, cfg))):
            official = archive[*key, method]
            if (reference_sha, observed_sha, array_hash(restored)) != tuple(
                    official[field] for field in ("reference_sha256", "observed_sha256", "output_sha256")):
                raise ValueError(f"Reference, observation or output changed for {key} / {method}")
            energy = abs(np.fft.fft2(restored-clean, norm="ortho"))**2 / clean.size
            values[method] = [float(energy[disk].sum()) for disk in disks]
            if not np.isclose(sum(values[method]), float(official["texture_band_mse"]), rtol=1e-12, atol=1e-22):
                raise ValueError(f"Peak energies do not sum to the published total at {key} / {method}")
        for i, (fy, fx) in enumerate(peaks):
            old, new = values["previous"][i], values["current"][i]
            rows.append(dict(group="known_nine" if key in targets else "new_sixteen", kind=kind, variant=variant,
                             amplitude=amplitude, peak=i+1, fy_fft_bin=fy, fx_fft_bin=fx,
                             fft_bins_including_conjugate=int(disks[i].sum()), input_mse=values["input"][i],
                             previous_mse=old, current_mse=new, delta_current_minus_previous=new-old))
    if len(rows) != 25*6 or snapshot() != frozen:
        raise ValueError("Incomplete post-evaluation breakdown or changed source")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"results/nine83")
    parser.add_argument("--out", type=Path, default=ROOT/"results/nine83/per_peak.csv")
    args = parser.parse_args()
    run(args.results, args.out)
