"""Render the three routes from the saved paired comparison, without refitting."""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(examples, destination):
    cases = [
        ("140055_global", "140055 · global stripes", np.s_[:, :]),
        ("69022_edge8", "69022 · 8 px edge band\n64 × 64 bottom crop", np.s_[-64:, 96:160]),
        ("140055_clean", "140055 · no added noise", np.s_[:, :]),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(11, 7.4))
    fig.subplots_adjust(left=.19, right=.99, top=.94, bottom=.08, wspace=.05, hspace=.06)
    for ax, label in zip(axes[0], ("Observed", "Restored", "Observed − restored", "Processing support")):
        ax.set_title(label, fontsize=12)
    for row, (stem, label, crop) in zip(axes, cases):
        with np.load(examples / f"{stem}.npz", allow_pickle=False) as data:
            row[0].set_ylabel(f"{label}\nRoute: {data['route'].item()}", fontsize=10,
                              rotation=0, ha="right", va="center", labelpad=9)
            for ax, key in zip(row, ("observed", "restored", "difference", "support_mask")):
                pixels = data[key][crop]
                if key == "difference":
                    residual = ax.imshow(pixels, cmap="RdBu_r", vmin=-.15, vmax=.15, interpolation="nearest")
                else:
                    ax.imshow(pixels, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
    last = axes[-1, 2].get_position()
    legend = fig.add_axes((last.x0, .048, last.width, .012))
    fig.colorbar(residual, cax=legend, orientation="horizontal", ticks=[-.15, 0, .15])
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=Path("results/project_reproduction/examples"))
    parser.add_argument("--out", type=Path, default=Path("results/project_reproduction/overview.png"))
    args = parser.parse_args()
    plot(args.examples, args.out)
