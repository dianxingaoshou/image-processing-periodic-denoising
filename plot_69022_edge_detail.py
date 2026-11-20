"""Recreate the bottom edge details and verify them against the paired restoration study."""
import csv

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
import numpy as np

from evaluate_weave import read_json
from image_io_v3 import sha256_file
from project_comparison import ROOT, array_hash, cases
from project_peak import denoise


def edge_examples(folder):
    manifest = read_json(folder / "run.json")
    if sha256_file(folder / "metrics.csv") != manifest["artifact_sha256"]["metrics.csv"]:
        raise ValueError("Metrics differ from the completed experiment")
    with (folder / "metrics.csv").open(encoding="utf8", newline="") as f:
        rows = {(r["kind"], r["method"]): r for r in csv.DictReader(f)
                if r["cohort"] == "regression" and r["image_id"] == "69022" and r["variant"] == "3"}
    examples = []
    for cohort, image_id, kind, variant, clean, observed, masks in cases():
        if (cohort, image_id, variant) != ("regression", "69022", 3) or kind not in ("edge8", "edge16"):
            continue
        restored = denoise(observed)
        for method, output in (("input", observed), ("current", restored)):
            row = rows[kind, method]
            actual = tuple(array_hash(a) for a in (clean, observed, output))
            if actual != tuple(row[f] for f in ("reference_sha256", "observed_sha256", "output_sha256")):
                raise ValueError(f"{kind}/{method} does not reproduce the saved arrays")
            for mask in ("contaminated", "adjacent_clean", "clean"):
                mse = np.mean((output[masks[mask]]-clean[masks[mask]])**2)
                if not np.isclose(mse, float(row[mask+"_mse"]), rtol=1e-12, atol=0):
                    raise ValueError(f"{kind}/{method} differs in {mask}")
        width = int(kind[4:])
        adjacent = np.flatnonzero(masks["adjacent_clean"][:, 0])
        loss = [float(rows[kind, method]["contaminated_mse"]) for method in ("input", "current")]
        examples.append((width, adjacent, loss, (clean, observed, restored, observed-clean, restored-clean)))
    if [e[0] for e in examples] != [8, 16]:
        raise ValueError("Expected both frozen bottom-edge conditions")
    return examples


def plot(folder, destination):
    examples = edge_examples(folder)
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    family = next((c for c in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei") if c in fonts), None)
    if family:
        plt.rcParams.update({"font.family": family, "axes.unicode_minus": False})
    def label(chinese, english):
        return chinese if family else english

    fig = plt.figure(figsize=(13.4, 5.8))
    fig.text(.025, .93, label("69022  底部窄条去除与相邻纹理保留", "69022  |  Bottom-edge stripe removal"),
             size=21, weight="bold")
    fig.text(.025, .86, label("各图均取列 96–159、行 224–255（从 0 计）。前三列共用灰度范围 [0,1]。",
             "Same crop in every panel: columns 96–159 and rows 224–255, zero-based. Grayscale range [0,1]."),
             size=11, color="#404c59")
    grid = fig.add_gridspec(2, 5, left=.156, right=.984, top=.76, bottom=.23, hspace=.3, wspace=.08)
    titles = [label("清晰参考", "Clean reference"), label("含噪输入", "Noisy input"),
              label("本版恢复", "Restored"), label("输入 − 参考", "Input − reference"),
              label("恢复 − 参考", "Restored − reference")]
    crop = np.s_[224:256, 96:160]
    max_error = max(np.max(np.abs(a[crop])) for *_, images in examples for a in images[-2:])
    limit = np.ceil(max_error * 100) / 100
    for i, (width, adjacent, loss, images) in enumerate(examples):
        for j, im in enumerate(images):
            ax = fig.add_subplot(grid[i, j])
            difference = j >= 3
            panel = ax.imshow(im[crop], cmap="RdBu_r" if difference else "gray",
                              vmin=-limit if difference else 0, vmax=limit if difference else 1,
                              extent=(95.5, 159.5, 255.5, 223.5), interpolation="nearest")
            ax.set(xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set(color="#666b70", linewidth=.5)
            if j == 0:
                ax.axhline(adjacent[-1]+.5, color="#ec9236", linestyle=(0,(3,2)), lw=1.2)
                pos = ax.get_position()
                mid = (pos.y0+pos.y1)/2
                fig.text(.025, mid+.042, label(f"{width} 像素窄条", f"{width} px strip"),
                         size=14, weight="bold")
                fig.text(.025, mid-.003, label("污染区 MSE", "Strip MSE"), size=10, color="#44515e")
                fig.text(.025, mid-.043, f"{loss[0]:.6f} →\n{loss[1]:.6f}", size=10,
                         va="center", linespacing=1.3)
            if i == 0:
                ax.set_title(titles[j], fontsize=12, pad=9)
    fig.text(.025, .166, label("左列橙线以下为窄条，线上方紧邻的 16 像素为干净区。两例的相邻干净区 MSE 均为 0。",
             "Below each orange line is the strip. In the 16 clean rows immediately above, MSE is 0 for both outputs."), size=10.6)
    fig.text(.025, .082, label("左侧 MSE 按全宽污染区域统计。16 像素窄条的外缘有一行渐入过渡，污染区评分不计此行。",
             "Printed MSE uses the full-width contaminated mask. It excludes one transition row in the 16 px case."),
             size=9.4, color="#44515e")
    cax = fig.add_axes([.748, .098, .215, .02])
    bar = fig.colorbar(panel, cax=cax, orientation="horizontal", ticks=[-limit, 0, limit], format="%.2f")
    bar.ax.set_title(label("两列误差共用亮度差色标", "Shared signed brightness difference"), fontsize=9.5, pad=5)
    bar.ax.tick_params(labelsize=9, length=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    plot(ROOT / "results/peak83", ROOT / "docs/figures/69022-edge-detail.png")
