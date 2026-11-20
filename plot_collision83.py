"""Show the two protected strips at equal scale and the full-width error at unseen phases."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
import numpy as np

from evaluate_collision83 import PHASES, PLAN, samples
from evaluate_nine83 import DIRECTIONS
from evaluate_weave import read_json
from image_io_v3 import sha256_file
from project_collision import denoise as current
from project_comparison import ROOT, array_hash
from project_peak import denoise as previous


def panels(folder):
    run = read_json(folder / "run.json")
    if sha256_file(folder/"metrics.csv") != run["artifact_sha256"]["metrics.csv"]:
        raise ValueError("Plot input differs from the completed evaluation")
    with (folder/"metrics.csv").open(encoding="utf8", newline="") as stream:
        records = {(r["cohort"], r["variant"], r["method"]): r for r in csv.DictReader(stream)
                   if r["image_id"] == "dtd_woven_0083" and r["cohort"] in (DIRECTIONS, PHASES)}
    targets, phases = {}, {}
    for case, amplitude in samples(read_json(PLAN)):
        cohort, _, _, variant, clean, obs, _ = case
        if cohort not in (DIRECTIONS, PHASES) or not variant.startswith("axis_x_E"):
            continue
        old = previous(obs)
        new, diag = current(obs, return_diagnostics=True)
        for name, image in (("input", obs), ("previous", old), ("current", new)):
            r = records[cohort, variant, name]
            if (array_hash(clean), array_hash(obs), array_hash(image)) != tuple(
                    r[k] for k in ("reference_sha256", "observed_sha256", "output_sha256")):
                raise ValueError(f"Panel mismatch for {cohort}/{variant}/{name}")
        curves = np.stack([np.sqrt(np.mean((a-clean)**2, axis=0)) for a in (old, new)])
        if cohort == DIRECTIONS:
            targets[amplitude] = (clean, obs, old, new, curves)
        else:
            seed = int(variant.split("_s")[-1].split("@")[0])
            gains = [float(records[cohort, variant, name]["noise_gain"]) for name in ("previous", "current")]
            section = np.s_[96:152, 48:104]
            phases[amplitude, seed] = dict(curves=curves, gains=gains,
                errors=[(v-clean)[section] for v in (old, new)],
                decision=diag["evidence"]["collision_review"]["reason"])
    seeds = read_json(PLAN)["new_phase_seeds"]
    if len(targets) != 2 or any((a,s) not in phases for a in targets for s in seeds):
        raise ValueError("Incomplete amplitude/phase coverage")
    return targets, phases, seeds


def plot(folder, dest):
    targets, phases, seeds = panels(folder)
    names = {f.name for f in font_manager.fontManager.ttflist}
    font = next((n for n in ("Microsoft YaHei","Noto Sans CJK SC","SimHei") if n in names), None)
    if font:
        plt.rcParams.update({"font.family":font, "axes.unicode_minus":False})
    def label(chinese, english):
        return chinese if font else english
    dest.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 6, figsize=(13.4, 8.3))
    fig.subplots_adjust(left=.14, right=.99, top=.87, bottom=.11, wspace=.07, hspace=.15)
    fig.text(.02,.96,label("woven_0083  两处残留条纹的优化前后对照","woven_0083  |  Before and after in the two residual strips"),
             fontsize=18, weight="bold", va="center")
    fig.text(.02,.918,label("四行均截取第 96–151 行，显示同一块织纹。灰度图共用 [0, 1]；差值图共用色标。",
                            "All crops use rows 96–151. Grayscale: [0,1]. Both error columns share one color scale."),
             fontsize=10, color=".25")
    headers = (("清晰参考","含噪输入","优化前","优化后","优化前 − 参考","优化后 − 参考") if font else
               ("Clean","Observed","Previous","Current","Previous − clean","Current − clean"))
    target_error = max(abs(a-clean).max() for clean, _, old, new, _ in targets.values() for a in (old, new))
    phase_error = max(np.abs(p["errors"]).max() for p in phases.values())
    limit = max(.01, np.ceil(100*max(target_error, phase_error))/100)
    for i, amp in enumerate(sorted(targets)):
        clean, obs, old, new, _ = targets[amp]
        for k, start in enumerate((48, 200)):
            row = 2*i+k
            section = np.s_[96:152, start:start+56]
            for j, (ax, panel) in enumerate(zip(axes[row], (clean, obs, old, new, old-clean, new-clean))):
                if j < 4:
                    ax.imshow(panel[section], cmap="gray", vmin=0., vmax=1., interpolation="nearest")
                else:
                    difference = ax.imshow(panel[section], cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
                ax.set(xticks=[], yticks=[])
                for spine in ax.spines.values():
                    spine.set(color=".75", linewidth=.5)
                if row == 0:
                    ax.set_title(headers[j], fontsize=12)
            box = axes[row,0].get_position()
            mid = (box.y0+box.y1)/2
            fig.text(.024, mid+.019, label("幅值 ", "Amplitude ")+f"{amp:.3f}", weight="bold", fontsize=12)
            side = (label("左区", "Left"), label("右区","Right"))[k]
            fig.text(.024,mid-.008, side+label("  列 ","  x ")+f"{start}–{start+55}", fontsize=10)
    fig.text(.024,.045,label("坐标从 0 起算。误差接近白色表示恢复亮度接近同位置的清晰参考。",
                             "Zero-based coordinates. White indicates a close match to the clean reference."),fontsize=10)
    bar = fig.colorbar(difference,cax=fig.add_axes([.73,.061,.23,.012]),orientation="horizontal",
                       ticks=[-limit,0,limit],format="%.2f")
    bar.ax.set_title(label("有符号亮度差","Signed intensity error"),fontsize=9)
    fig.savefig(dest/"woven-0083-collision.png", dpi=155)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(9.5,5.4), layout="constrained")
    fig.suptitle(label("axis_x_E  逐列恢复误差与新相位对照","axis_x_E  |  Column-wise restoration error"), fontsize=16)
    colors = ("#ab5721","#147773")
    upper = 1.06*max(max(t[-1].max() for t in targets.values()), max(p["curves"].max() for p in phases.values()))
    for ax, amp in zip(axes, sorted(targets)):
        data = targets[amp][-1]
        fresh = np.stack([phases[amp,seed]["curves"] for seed in seeds])
        for left, right in read_json(PLAN)["fixed_column_ranges_inclusive"]:
            ax.axvspan(left-.5,right+.5,color="#e3e3e3",zorder=0)
        for i, title in enumerate((label("优化前","Previous"), label("优化后","Current"))):
            ax.fill_between(np.arange(data.shape[1]), fresh[:,i].min(axis=0), fresh[:,i].max(axis=0),
                            color=colors[i], alpha=.18, linewidth=0)
            ax.plot(data[i], color=colors[i], label=title, lw=1.45)
        ax.set(ylabel="RMSE", title=label("幅值 ","Amplitude ")+f"{amp:.3f}", ylim=(0,upper), xlim=(0,255))
        ax.spines[["top","right"]].set_visible(False)
    axes[0].legend(loc="upper center", ncols=2, fontsize=9)
    axes[-1].set_xlabel(label("列号（0–255）。实线为原相位，彩色带为三个新相位的最小至最大值，灰底为两处固定评价区。",
                              "Column. Lines: original phase. Color bands: min–max of three new phases. Gray: evaluation strips."))
    fig.savefig(dest/"woven-0083-collision-columns.png",dpi=155)
    plt.close(fig)

    fig, axes = plt.subplots(len(seeds), 4, figsize=(9.1, 7.3))
    fig.subplots_adjust(left=.17, right=.98, top=.85, bottom=.11, hspace=.37, wspace=.05)
    fig.suptitle(label("三个新相位下，同一块织纹上的剩余条纹","Residual stripes on the same weave under three new phases"),
                 fontsize=16, y=.97)
    for row, seed in enumerate(seeds):
        for pair, amp in enumerate(sorted(targets)):
            phase = phases[amp, seed]
            print(f"phase {seed}, amplitude {amp:.3f}: {phase['decision']}")
            for method, (error, gain) in enumerate(zip(phase["errors"], phase["gains"])):
                ax = axes[row, pair*2+method]
                bar_image = ax.imshow(error, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
                ax.set(xticks=[], yticks=[])
                if row == 0:
                    version = [label("前","before"), label("后","after")][method]
                    ax.set_title(f"{amp:.3f}   {version}", fontsize=12)
                ax.set_xlabel(label("整图去除 ", "Removed ")+f"{gain:.1%}", fontsize=10, labelpad=3)
            if not pair:
                axes[row,0].set_ylabel(label("相位种子", "Phase seed")+f"\n{seed}", rotation=0,
                                       ha="right", va="center", labelpad=14, fontsize=11)
    fig.text(.02,.042,label("第 96–151 行、第 48–103 列；颜色为恢复图减清晰图。","Rows 96–151, columns 48–103. Color shows restored minus clean."),fontsize=9)
    fig.colorbar(bar_image, cax=fig.add_axes([.69,.052,.23,.014]), orientation="horizontal",
                 ticks=[-limit,0,limit],format="%.2f")
    fig.savefig(dest/"woven-0083-collision-phases.png",dpi=155)
    plt.close(fig)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,default=ROOT/"results/collision83")
    p.add_argument("--out",type=Path,default=ROOT/"docs/figures")
    args=p.parse_args()
    plot(args.results,args.out)
