"""Render observed, removed and remaining stripes from verified peak83 outputs."""
import csv

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
import numpy as np

from evaluate_nine83 import DIRECTIONS, samples
from evaluate_weave import read_json
from image_io_v3 import sha256_file
from project_comparison import ROOT, array_hash
from project_peak import denoise


CASES = [("archived_photo", "clean", ""), ("weak83_carriers", "fixed", "fixed_C@0.09"),
         (DIRECTIONS, "axis_x", "axis_x_E@0.080")]


def load_panels():
    study = ROOT / "results/peak83"
    if sha256_file(study / "metrics.csv") != read_json(study / "run.json")["artifact_sha256"]["metrics.csv"]:
        raise ValueError("The metric table differs from the completed study")
    with (study / "metrics.csv").open(encoding="utf8", newline="") as stream:
        records = {(r["cohort"], r["kind"], r["variant"]): r for r in csv.DictReader(stream)
                   if r["image_id"] == "dtd_woven_0083" and r["method"] == "current"}
    freeze = read_json(ROOT / "results/nine83/freeze.json")
    selected = {}
    for case, _ in samples(freeze["protocol"], freeze["endpoint_controls"]):
        group, name, kind, variant, clean, observed, _ = case
        key = group, kind, variant
        if name != "dtd_woven_0083" or key not in CASES:
            continue
        restored = denoise(observed)
        row = records[key]
        if tuple(map(array_hash, (clean, observed, restored))) != tuple(
                row[k] for k in ("reference_sha256", "observed_sha256", "output_sha256")):
            raise ValueError(f"Recomputed pixels differ for {key}")
        selected[key] = ((observed, restored, observed-clean, observed-restored, restored-clean), row)
    return [selected[key] for key in CASES]


def plot():
    rows = load_panels()
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = next((name for name in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei") if name in fonts), None)
    if font:
        plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})
    title = "woven_0083 织纹保留与条纹去除" if font else "Preserving weave while removing stripes"
    headers = (("输入图像", "本版恢复", "实际加入的条纹", "实际减除量", "恢复误差") if font else
               ("Input", "Restored", "Injected stripes", "Actual removal", "Restored minus clean"))
    names = ("清晰原图", "定频 C", "横向渐变 E") if font else ("Clean control", "Fixed C", "Horizontal sweep E")
    limit = max(.01, np.ceil(100*max(abs(a).max() for panels, _ in rows for a in panels[2:]))/100)
    fig, axes = plt.subplots(3, 5, figsize=(12.5, 7.6))
    fig.subplots_adjust(left=.115, right=.985, bottom=.128, top=.865, wspace=.05, hspace=.11)
    fig.text(.024, .952, title, fontsize=19, weight="bold", va="center")
    for i, (panels, row) in enumerate(rows):
        for j, (ax, array) in enumerate(zip(axes[i], panels)):
            if j < 2:
                ax.imshow(array, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            else:
                image = ax.imshow(array, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
            ax.set(xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set(color=".7", linewidth=.4)
            if i == 0:
                ax.set_title(headers[j], fontsize=12, pad=7)
        box = axes[i, 0].get_position()
        middle = (box.y0+box.y1)/2
        fig.text(.024, middle+.014, names[i], fontsize=12, weight="bold")
        changed = row["changed_pixels"]
        detail = (f"改动 {changed} 像素" if font else f"{changed} changed pixels") if i == 0 else (
            ("幅值 " if font else "Amp. ") + f'{float(row["exact_amplitude"]):g}\n'
            + ("减除 " if font else "Removed ") + f'{float(row["noise_gain"]):.1%}')
        fig.text(.024, middle-.005, detail, fontsize=9, va="top", color=".25")
    caption = ("完整 256×256 图像。恢复误差 = 恢复图 − 清晰原图，白色表示差值为零。\n"
               "减除比例按整幅图的减除量在实际新增条纹上的投影计算。"
               if font else "Full 256×256 frames. Error = restored − clean. White denotes zero.\n"
               "Removal is the projection onto the actual injected stripe across the full image.")
    fig.text(.024, .058, caption, fontsize=10, va="center")
    bar = fig.colorbar(image, cax=fig.add_axes([.735, .064, .22, .012]), orientation="horizontal",
                       ticks=[-limit, 0, limit], format="%.2f")
    bar.ax.set_title("后三列共用亮度差色标" if font else "Intensity difference (last three columns)", fontsize=10)
    bar.ax.tick_params(labelsize=9)
    destination = ROOT / "docs/figures/woven-0083-restoration.png"
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plot()
