"""Plot six paired texture inputs after verifying the completed local evaluation."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

from evaluate_project_refinement import extended_cases, identity
from image_io_v3 import sha256_file
from project_comparison import ROOT, array_hash
from project_pipeline import load_config
from project_refined import denoise as previous
from project_weave import denoise as current


TEXTURES = ("dtd_knitted_0108", "dtd_woven_0094")
CONDITIONS = (("clean", "Clean texture"), ("global", "Added stripes"), ("chirp", "Added chirp"))
METHODS = ("input", "previous", "current")


def paired_panels(directory):
    run = json.loads((directory / "run.json").read_text(encoding="utf8"))
    path = directory / "metrics.csv"
    if sha256_file(path) != run["metrics_sha256"] or sha256_file(directory / "freeze.json") != run["freeze_sha256"]:
        raise ValueError("The metrics or freeze file does not match the completed run")
    cases = {identity(r): r for r in run["cases"]}
    with path.open(newline="", encoding="utf8") as handle:
        metrics = {(identity(r), r["method"]): r for r in csv.DictReader(handle)}
    panels = {}
    config = load_config()
    for cohort, name, kind, variant, clean, observed, _ in extended_cases():
        if name not in TEXTURES or kind not in ("clean", "global", "chirp"):
            continue
        key = cohort, name, kind, str(variant)
        if array_hash(observed) != cases[key]["observed_sha256"] or array_hash(clean) != cases[key]["reference_sha256"]:
            raise ValueError(f"Input or reference differs for {key}")
        outputs = observed, previous(observed, config), current(observed, config)
        for method, image in zip(METHODS, outputs):
            row = metrics[key, method]
            if array_hash(image) != row["output_sha256"]:
                raise ValueError(f"Recomputed result differs for {key} and {method}")
            metric = "mae" if kind == "clean" else "mse"
            panels[name, kind, method] = image, metric, float(row[metric])
    if len(panels) != len(TEXTURES)*len(CONDITIONS)*len(METHODS):
        raise ValueError("Incomplete paired texture records")
    return panels


def plot(directory, destination):
    panels = paired_panels(directory)
    fig = plt.figure(figsize=(12.7, 7.7))
    grid = fig.add_gridspec(3, 7, width_ratios=(1, 1, 1, .13, 1, 1, 1),
                            left=.097, right=.995, bottom=.1, top=.89, wspace=.05, hspace=.25)
    for group, name in enumerate(TEXTURES):
        first = grid[0, group*4].get_position(fig)
        last = grid[0, group*4+2].get_position(fig)
        fig.text((first.x0 + last.x1)/2, .97, name.split("_")[1].capitalize(),
                 ha="center", va="center", fontsize=16, weight="bold")
        for i, (kind, title) in enumerate(CONDITIONS):
            for j, method in enumerate(METHODS):
                image, metric, value = panels[name, kind, method]
                ax = fig.add_subplot(grid[i, group*4+j])
                ax.imshow(image[64:192, 64:192], cmap="gray", vmin=0., vmax=1., interpolation="nearest")
                if i == 0:
                    ax.set_title(method.capitalize(), fontsize=11)
                if group == 0 and j == 0:
                    ax.set_ylabel(title, rotation=0, ha="right", va="center", labelpad=8, fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(f"{metric.upper()} {value:.5f}", fontsize=9)
    fig.text(.10, .025, "Central 128 × 128 crop in every panel. Images share a 0–1 gray scale. "
             "Errors use the full 256 × 256 image and its original texture reference.", fontsize=9)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/weave")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/figures/weave-pairs.png")
    args = parser.parse_args()
    plot(args.results, args.out)
