"""Plot the three paired textile cases using the same peak masks as the frozen study."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
import numpy as np

from check_nine83_peaks import peak_masks
from evaluate_nine83 import DIRECTIONS, samples
from evaluate_weave import projected_residual, read_json
from image_io_v3 import sha256_file
from project_comparison import ROOT, array_hash
from project_guarded import denoise as previous
from project_peak import denoise as current


def observations(folder):
    run = read_json(folder / "run.json")
    if sha256_file(folder / "metrics.csv") != run["artifact_sha256"]["metrics.csv"]:
        raise ValueError("Metrics differ from the completed run")
    with (folder / "metrics.csv").open(newline="", encoding="utf8") as f:
        metrics = {(r["variant"], r["method"]): r for r in csv.DictReader(f) if r["cohort"] == DIRECTIONS}
    prior = ROOT / "results/nine83"
    old = read_json(prior / "freeze.json")
    peaks = read_json(prior / "run.json")["reference_texture_peaks_in_fft_bins"]
    target = read_json(ROOT / "results/peak83/protocol.json")["known_damage"]
    selected = {}
    for case, _ in samples(old["protocol"], old["endpoint_controls"]):
        group, _, _, variant, clean, observed, _ = case
        if group != DIRECTIONS or variant not in target:
            continue
        baseline, candidate = previous(observed), current(observed)
        for method, output in (("input", observed), ("previous", baseline), ("current", candidate)):
            row = metrics[variant, method]
            if tuple(array_hash(a) for a in (clean, observed, output)) != tuple(
                row[k] for k in ("reference_sha256", "observed_sha256", "output_sha256")
            ):
                raise ValueError(f"Recomputed panel differs from frozen output for {variant}/{method}")
        noise = observed-clean
        gain = projected_residual(observed-candidate, noise)[0]
        old_gain = projected_residual(observed-baseline, noise)[0]
        matched = observed - (gain/old_gain)*(observed-baseline)
        _, disks = peak_masks(clean, peaks[array_hash(clean)])
        def energies(error):
            values = abs(np.fft.fft2(error, norm="ortho"))**2/error.size
            return np.array([values[d].sum() for d in disks])
        input_errors = energies(noise)
        curves = [energies(a-clean)/input_errors for a in (matched, candidate)]
        selected[variant] = ((clean, observed, baseline, candidate), curves, (old_gain, gain))
    return [(key, *selected[key]) for key in target]


def plot(folder, output):
    rows = observations(folder)
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    cjk = next((n for n in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei") if n in fonts), None)
    if cjk:
        plt.rcParams.update({"font.family": cjk, "axes.unicode_minus": False})
    title = "woven_0083 同一张织物的加噪前后对照" if cjk else "Paired stripe removal on woven_0083"
    labels = ("清晰原图", "同图加噪", "上版恢复", "本版恢复") if cjk else ("Clean", "Observed", "Previous", "Current")
    baseline_label, current_label = (("上版按本版去除比例缩放", "本版") if cjk else ("Previous at matched gain", "Current"))
    fig = plt.figure(figsize=(12, 7.2), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.03, 1], hspace=.07)
    fig.suptitle(title, fontsize=18, weight="bold")
    top.suptitle("axis_x_E   ·   0.045", fontsize=12)
    for ax, panel, label in zip(top.subplots(1, 4), rows[0][1], labels):
        ax.imshow(panel[80:176,80:176], cmap="gray", vmin=0., vmax=1., interpolation="nearest")
        ax.set(xticks=[], yticks=[], title=label)
    bottom.suptitle("六组织纹主峰的均方误差，以各峰的含噪输入为 1" if cjk
                    else "Error in six weave peaks relative to the noisy input (1.0)", fontsize=13)
    axes = bottom.subplots(1, 3, sharey=True)
    for ax, (variant, _, curves, gains) in zip(axes, rows):
        x = np.arange(1, 7)
        ax.axhline(1., color=".45", ls="--", lw=1)
        ax.scatter(x-.11, curves[0], label=baseline_label, color="#c36a22", marker="s", s=32)
        ax.scatter(x+.11, curves[1], label=current_label, color="#14777b", s=36)
        ax.set(xticks=x, yscale="log", xlabel="织纹主峰编号" if cjk else "Peak number")
        ax.set_title(variant.replace("@", "   ·   ") + f"\na = {gains[0]:.4f} → {gains[1]:.4f}", fontsize=11)
        ax.grid(axis="y", color=".9", lw=.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("误差比（对数刻度）" if cjk else "MSE ratio (log scale)")
    axes[0].legend(loc="lower left", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results/peak83")
    parser.add_argument("--out", type=Path, default=ROOT / "docs/figures/woven-0083-peak.png")
    args = parser.parse_args()
    plot(args.results, args.out)
