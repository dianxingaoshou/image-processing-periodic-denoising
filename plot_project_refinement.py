"""Plot three paired examples after matching their inputs and outputs to the saved run."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from project_comparison import ROOT, array_hash, cases
from project_pipeline import denoise as baseline, load_config
from project_refined import denoise as refined


EXAMPLES = (
    (("DTD", "dtd_banded_0105", "clean", ""), "DTD banded\nNo added noise", np.s_[:, :]),
    (("DTD", "dtd_woven_0094", "clean", ""), "DTD woven\nNo added noise", np.s_[:, :]),
    (("regression", "69022", "edge8", "3"), "BSDS 69022\n8 px edge band\n40 × 40 bottom crop", np.s_[-40:, 108:148]),
)


def case_key(row):
    return row["cohort"], row["image_id"], row["kind"], str(row["variant"])


def plot(directory, destination):
    run = json.loads((directory / "run.json").read_text(encoding="utf8"))
    desired = {key for key, _, _ in EXAMPLES}
    sources = {case_key(r): r for r in run["cases"] if case_key(r) in desired}
    with (directory / "metrics.csv").open(encoding="utf8", newline="") as f:
        metrics = {(case_key(r), r["method"]): r for r in csv.DictReader(f) if case_key(r) in desired}
    cfg = load_config()
    results = {}
    for cohort, name, kind, variant, clean, observed, _ in cases():
        key = cohort, name, kind, str(variant)
        if key not in desired:
            continue
        if (array_hash(observed) != sources[key]["observed_sha256"]
                or array_hash(clean) != sources[key]["reference_sha256"]):
            raise ValueError(f"Input or reference hash differs for {key}")
        outputs = {"input": observed, "baseline": baseline(observed, cfg), "refined": refined(observed, cfg)}
        for method, output in outputs.items():
            if array_hash(output) != metrics[key, method]["output_sha256"]:
                raise ValueError(f"Output hash differs for {key} / {method}")
        results[key] = outputs
        if results.keys() == desired:
            break
    if results.keys() != desired:
        raise ValueError("Missing source for one or more plot panels")

    fig, axes = plt.subplots(3, 3, figsize=(8.6, 8.1))
    fig.subplots_adjust(left=.21, right=.98, top=.93, bottom=.1, hspace=.23, wspace=.05)
    for ax, title in zip(axes[0], ("Input", "Previous", "Refined")):
        ax.set_title(title, fontsize=13)
    for row, (key, title, crop) in zip(axes, EXAMPLES):
        row[0].set_ylabel(title, rotation=0, ha="right", va="center", labelpad=10)
        for ax, method in zip(row, ("input", "baseline", "refined")):
            ax.imshow(results[key][method][crop], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel(f"MAE {float(metrics[key, method]['mae']):.5f}", fontsize=10)
    fig.text(.21, .025, "MAE uses the full image and the paired reference. Gray scale 0–1 throughout.", fontsize=9)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/project_refinement")
    parser.add_argument("--out", type=Path, help="Defaults to texture-comparison.png in the results directory.")
    args = parser.parse_args()
    plot(args.results, args.out or args.results / "texture-comparison.png")
