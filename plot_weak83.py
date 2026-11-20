"""Render the four known woven targets after checking the official comparison hashes."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from diagnose_woven_weak import old_cases
from evaluate_weave import read_json
from image_io_v3 import sha256_file
from project_comparison import ROOT, array_hash
from project_pipeline import load_config
from project_stripes import denoise as previous


def panels(folder):
    from project_weak import denoise as current

    manifest = read_json(folder/"run.json")
    for name in ("freeze.json", "metrics.csv"):
        if sha256_file(folder/name) != manifest["artifact_sha256"][name]:
            raise ValueError(f"Study output changed: {name}")
    wanted = ["clean", "stationary_A", "stationary_B", "chirp"]
    with (folder/"metrics.csv").open(encoding="utf8", newline="") as f:
        metrics = {(r["kind"],r["variant"],r["method"]):r for r in csv.DictReader(f)
                   if r["cohort"] in ("archived_photo","new_woven") and r["image_id"] == "dtd_woven_0083"}
    cfg = load_config()
    by_kind = {}
    for kind, variant, clean, obs, _ in old_cases():
        if kind not in wanted:
            continue
        old, new = previous(obs, cfg), current(obs, cfg)
        for method, result in (("input",obs),("previous",old),("current",new)):
            key = (kind,variant,method)
            checks = [array_hash(a) for a in (clean,obs,result)]
            if checks != [metrics[key][f] for f in ("reference_sha256","observed_sha256","output_sha256")]:
                raise ValueError(f"Recomputed figure differs from {key}")
        by_kind[kind] = (obs, obs-clean, obs-old, obs-new, new, new-clean)
    return [by_kind[kind] for kind in wanted]


def plot(folder, out):
    rows = panels(folder)
    # Keep each texture and stripe at its native pixel sampling. The same
    # central field shows four-column weave detail and gradual phase changes.
    region = np.s_[64:192, 64:192]
    signed = {1,2,3,5}
    limit = max(abs(row[j][region]).max() for row in rows for j in signed)
    limit = max(.025, .025*np.ceil(limit/.025))
    names = {f.name for f in font_manager.fontManager.ttflist}
    cjk = next((name for name in ("Microsoft YaHei","Noto Sans CJK SC","SimHei") if name in names), None)
    if cjk:
        plt.rcParams.update({"font.family":cjk, "axes.unicode_minus":False})
        title = "woven_0083 同图加噪前后的条纹与织纹"
        labels = ("输入", "实际新增条纹", "上版减除", "本版减除", "本版恢复图", "恢复图 − 原图")
        conditions = ("干净原图", "弱定频 A", "弱定频 B", "渐变条纹")
        notes = ("未叠加", "幅值 0.07", "幅值 0.07", "生成幅值 0.12")
        foot = "各图均取同一中心区域，行、列 64–191（从 0 起）。输入与恢复图共用 [0, 1] 灰度。"
        scale = "有符号亮度差（四列共用）"
    else:
        title = "Removed stripes and remaining error on the same woven_0083 texture"
        labels = ("Input", "Actual addition", "Previous removal", "Updated removal", "Updated output", "Output − clean")
        conditions = ("Clean", "Weak fixed A", "Weak fixed B", "Chirp")
        notes = ("No addition", "Amplitude 0.07", "Amplitude 0.07", "Amplitude 0.12")
        foot = "All panels show rows and columns 64–191 (zero-based). Input and output share the [0, 1] grayscale."
        scale = "Signed intensity difference (four columns)"
    fig, axes = plt.subplots(4, 6, figsize=(13.6, 8.8))
    fig.subplots_adjust(top=.885, bottom=.12, left=.135, right=.986, wspace=.075, hspace=.15)
    fig.text(.025,.955,title,fontsize=19,weight="bold",va="center")
    for i, row in enumerate(rows):
        for j, arr in enumerate(row):
            ax = axes[i,j]
            if j in signed:
                diffs = ax.imshow(arr[region], cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
            else:
                ax.imshow(arr[region], cmap="gray", vmin=0., vmax=1., interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for side in ax.spines.values():
                side.set_linewidth(.45)
                side.set_color("#999")
            if i == 0:
                ax.set_title(labels[j], fontsize=12, pad=7)
        pos = axes[i,0].get_position()
        mid = (pos.y0+pos.y1)/2
        fig.text(.025,mid+.012,conditions[i],fontsize=13,weight="bold")
        fig.text(.025,mid-.016,notes[i],fontsize=10,color="#45515c")
    fig.text(.025,.055,foot,fontsize=10,color="#35424e",va="center")
    bar = fig.colorbar(diffs, cax=fig.add_axes([.727,.059,.23,.012]), orientation="horizontal",
                       ticks=[-limit,0,limit], format="%.3f")
    bar.ax.set_title(scale, fontsize=10, pad=5)
    bar.ax.tick_params(labelsize=9,length=2,pad=2)
    bar.outline.set_linewidth(.5)
    out.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=ROOT/"results/weak83")
    p.add_argument("--out", type=Path, default=ROOT/"docs/figures/woven-0083-weak.png")
    args = p.parse_args()
    plot(args.results, args.out)
