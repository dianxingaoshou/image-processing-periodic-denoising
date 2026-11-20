"""Compare actual removed signals on the known woven and narrow-edge cases."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from diagnose_woven_narrow import inputs
from evaluate_weave import read_json
from image_io_v3 import sha256_file
from project_comparison import ROOT, array_hash
from project_pipeline import load_config
from project_stripes import denoise as current
from project_weave import denoise as previous


def panels(directory):
    """Recreate only the diagnosed inputs and require exact agreement with the completed study."""
    manifest = read_json(directory / "run.json")
    for name in ("freeze.json", "metrics.csv"):
        if sha256_file(directory / name) != manifest["artifact_sha256"][name]:
            raise ValueError(f"Completed study disagrees with {name}")
    with (directory / "metrics.csv").open(newline="", encoding="utf8") as stream:
        metrics = {(r["image_id"], r["kind"], r["method"]): r for r in csv.DictReader(stream)
                   if r["cohort"] in ("archived_photo", "regression")}
    results = []
    cfg = load_config()
    for name, kind, ref, observed, _ in inputs():
        if kind == "chirp":
            continue
        image_id = "dtd_" + name if name.startswith("woven") else name
        old, new = previous(observed, cfg), current(observed, cfg)
        for method, image in (("input", observed), ("previous", old), ("current", new)):
            row = metrics[image_id, kind, method]
            actual = array_hash(ref), array_hash(observed), array_hash(image)
            if actual != tuple(row[field] for field in ("reference_sha256", "observed_sha256", "output_sha256")):
                raise ValueError(f"Recomputed figure differs from study: {image_id}/{kind}/{method}")
        results.append((name, kind, (observed, observed-ref, observed-old, observed-new, new)))
    if len(results) != 4:
        raise ValueError("Expected the paired woven photo and both bottom strips on 69022")
    return results


def plot(directory, destination):
    rows = panels(directory)
    limit = max(np.max(np.abs(image)) for _, _, images in rows for image in images[1:4])
    limit = np.ceil(limit / .05) * .05
    fig = plt.figure(figsize=(12.2, 8.4))
    grid = fig.add_gridspec(4, 5, height_ratios=(1, 1, .5, .5), left=.151, right=.981,
                            bottom=.16, top=.845, wspace=.12, hspace=.22)
    fig.text(.027, .954, "What did the filter subtract?", fontsize=20, weight="bold", va="center")
    fig.text(.027, .91, "Same input for both methods. The middle three columns share one signed intensity scale.",
             fontsize=11, color="#374151")
    labels = ("Input", "Actual addition", "Previous removal", "Updated removal", "Updated output")
    conditions = {"clean":"Clean texture", "global":"Added global stripes", "edge8":"Bottom 8 px", "edge16":"Bottom 16 px"}
    for i, (name, kind, images) in enumerate(rows):
        is_woven = name.startswith("woven")
        y0, y1 = (96, 160) if is_woven else (224, 256)
        x0, x1 = 123, 187
        for j, image in enumerate(images):
            ax = fig.add_subplot(grid[i, j])
            signed = j in (1, 2, 3)
            im = ax.imshow(image[y0:y1, x0:x1], cmap="RdBu_r" if signed else "gray",
                           vmin=-limit if signed else 0, vmax=limit if signed else 1,
                           extent=(x0-.5, x1-.5, y1-.5, y0-.5), interpolation="nearest")
            if signed:
                scale = im
            if is_woven and j == 0:
                ax.add_patch(Rectangle((152.5, y0-.5), 4, y1-y0, fill=False,
                                       edgecolor="#e69517", linewidth=1.25))
            if i == 0:
                ax.set_title(labels[j], fontsize=12, fontweight="bold", pad=8)
            for spine in ax.spines.values():
                spine.set_linewidth(.5)
                spine.set_color("#89929d")
            ax.set_yticks([y0, y1-1] if j == 0 else [])
            ax.set_xticks([x0, (x0+x1)//2, x1-1] if i == len(rows)-1 else [])
            ax.tick_params(axis="both", labelsize=8, length=2, pad=2, colors="#374151")
            if j == 0:
                pos = ax.get_position()
                fig.text(.027, pos.y0+pos.height/2+.014, name, fontsize=12, weight="bold")
                fig.text(.027, pos.y0+pos.height/2-.014, conditions[kind], fontsize=9.5, color="#374151")
    fig.text(.027, .095, "Orange box: columns 153–156 on woven inputs.", fontsize=10, color="#553508")
    fig.text(.027, .066, "x = column, y = row (increases down). All views show original pixel coordinates.",
             fontsize=9, color="#374151")
    fig.text(.027, .04, "Grayscale images use [0, 1]. Addition = input minus clean. Removal = input minus output.",
             fontsize=9, color="#374151")
    bar_ax = fig.add_axes([.69, .073, .274, .015])
    colorbar = fig.colorbar(scale, cax=bar_ax, orientation="horizontal", ticks=(-limit, 0, limit),
                            format=FormatStrFormatter("%.2f"))
    colorbar.ax.tick_params(labelsize=8, length=2, pad=2)
    colorbar.ax.set_title("Signed intensity difference", fontsize=10, pad=6)
    colorbar.outline.set_linewidth(.45)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/narrow83")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/figures/woven-0083-narrow.png")
    args = parser.parse_args()
    plot(args.results, args.out)
