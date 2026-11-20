"""Show four historical same-photo pairs after checking the formal output hashes."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
import numpy as np

from diagnose_woven_nine import observations
from evaluate_weave import read_json
from image_io_v3 import sha256_file
from project_comparison import ROOT, array_hash
from project_pipeline import load_config


CASES = (("archived_photo", "clean", ""),
         ("weak83_carriers", "fixed", "fixed_C@0.09"),
         ("weak83_single_axis", "chirp", "axis_y@0.07"),
         ("weak83_carriers", "chirp", "sweep_D@0.05"))


def panels(folder):
    from project_guarded import denoise as current
    from project_weak import denoise as previous

    manifest = read_json(folder / "run.json")
    for name in ("freeze.json", "metrics.csv"):
        if sha256_file(folder/name) != manifest["artifact_sha256"][name]:
            raise ValueError(f"Study output changed: {name}")
    with (folder/"metrics.csv").open(encoding="utf8", newline="") as stream:
        records = {(r["cohort"],r["kind"],r["variant"],r["method"]): r for r in csv.DictReader(stream)
                   if r["image_id"]=="dtd_woven_0083"}
    samples = {(k,v):(x,y) for k,v,x,y,_ in observations()}
    cfg = load_config()
    arrays = []
    for group, kind, variant in CASES:
        clean, obs = samples[kind,variant]
        old, new = previous(obs,cfg), current(obs,cfg)
        for method, result in (("input",obs),("previous",old),("current",new)):
            row = records[group,kind,variant,method]
            if (array_hash(clean),array_hash(obs),array_hash(result)) != tuple(
                    row[k] for k in ("reference_sha256","observed_sha256","output_sha256")):
                raise ValueError(f"Recomputed {group}/{variant}/{method} differs from study")
        arrays.append((obs, obs-clean, obs-old, obs-new, new, new-clean))
    return arrays


def plot(folder, destination):
    rows = panels(folder)
    region = np.s_[64:192,64:192]
    differences = [1,2,3,5]
    maximum = max(np.max(abs(row[j][region])) for row in rows for j in differences)
    limit = max(.025, np.ceil(maximum/.025)*.025)
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    cjk = next((n for n in ("Microsoft YaHei","Noto Sans CJK SC","SimHei") if n in fonts), None)
    if cjk:
        plt.rcParams.update({"font.family":cjk, "axes.unicode_minus":False})
        title = "woven_0083 同一织物上的定频与渐变条纹"
        labels = ("输入", "实际新增条纹", "上版减除", "本版减除", "本版恢复", "恢复误差")
        cases = ("原图", "定频 C", "单轴 y 渐变", "双轴 D 渐变")
        notes = ("无新增条纹", "幅值 0.085", "幅值 0.075", "幅值 0.05")
        foot = "中心行列 64–191（从 0 起）。灰度范围 [0, 1]；恢复误差 = 恢复图 − 清晰原图。"
        bar_label = "有符号亮度差，四列共用"
    else:
        title = "woven_0083 under fixed-frequency and swept stripes"
        labels = ("Input", "Actual addition", "Previous removal", "Current removal", "Current output", "Output − clean")
        cases = ("Original", "Fixed C", "Single-axis y", "Two-axis D")
        notes = ("No added stripes", "Amplitude 0.085", "Amplitude 0.075", "Amplitude 0.05")
        foot = "All panels show rows and columns 64–191 (zero-based). Grayscale range [0, 1]."
        bar_label = "Signed intensity difference, four columns"
    fig, axes = plt.subplots(4,6,figsize=(13.6,8.8))
    fig.subplots_adjust(left=.135,right=.987,top=.886,bottom=.117,wspace=.08,hspace=.15)
    fig.text(.024,.956,title,fontsize=19,weight="bold",va="center")
    for i,row in enumerate(rows):
        for j,image in enumerate(row):
            ax=axes[i,j]
            if j in differences:
                mapped=ax.imshow(image[region],cmap="RdBu_r",vmin=-limit,vmax=limit,interpolation="nearest")
            else:
                ax.imshow(image[region],cmap="gray",vmin=0,vmax=1,interpolation="nearest")
            ax.set(xticks=[], yticks=[])
            if i==0:
                ax.set_title(labels[j],fontsize=12,pad=7)
            for spine in ax.spines.values():
                spine.set(color="#777777", linewidth=.4)
        box=axes[i,0].get_position()
        middle=(box.y0+box.y1)/2
        fig.text(.024,middle+.012,cases[i],fontsize=12,weight="bold")
        fig.text(.024,middle-.017,notes[i],fontsize=10,color="#414b57")
    fig.text(.024,.054,foot,fontsize=10,color="#35424e",va="center")
    bar=fig.colorbar(mapped,cax=fig.add_axes([.723,.058,.235,.013]),orientation="horizontal",
                     ticks=[-limit,0,limit],format="%.3f")
    bar.ax.set_title(bar_label,fontsize=10,pad=5)
    bar.ax.tick_params(labelsize=9,length=2,pad=2)
    destination.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(destination,dpi=150,facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=ROOT/"results/nine83")
    p.add_argument("--out",type=Path,default=ROOT/"docs/figures/woven-0083-nine.png")
    args=p.parse_args()
    plot(args.results,args.out)
