"""Plot the same left and right weave patches for every registered phase and amplitude."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
import numpy as np

from evaluate_weave import read_json
from image_io_v3 import sha256_file
from project_comparison import ROOT, array_hash


def read_cases(folder, plan):
    run = read_json(folder / "run.json")
    if run["artifact_sha256"]["metrics.csv"] != sha256_file(folder / "metrics.csv"):
        raise ValueError("Metrics do not match the completed experiment")
    with (folder / "metrics.csv").open(encoding="utf8", newline="") as stream:
        rows = {(r["variant"], r["method"]): r for r in csv.DictReader(stream)
                if r["image_id"] == plan["target"]}
    case_names = {f"axis_x_E_s{s}@{a:.3f}" for s in plan["target_seeds"] + plan["new_phase_seeds"]
                  for a in plan["amplitudes"]}
    examples = folder / "examples"
    missing = {name for name in case_names if not (examples / (name+".npz")).is_file()}
    rebuilt = {}
    if missing:
        # Public results store hashes and scores; source photographs and recovery
        # arrays can be rebuilt locally. Only the displayed inputs need inference.
        from evaluate_phase_refit import samples
        from project_collision import denoise as previous
        from project_phase import denoise as current
        from project_pipeline import load_config
        cfg = load_config()
        for case, _ in samples(plan):
            _, image_id, _, name, clean, obs, _ = case
            if image_id == plan["target"] and name in missing:
                rebuilt[name] = dict(clean=clean, input=obs, previous=previous(obs, cfg), current=current(obs, cfg))
                if rebuilt.keys() == missing:
                    break
        if rebuilt.keys() != missing:
            raise ValueError(f"Missing registered figure inputs: {sorted(missing - rebuilt.keys())}")
    cases = {}
    for seed in plan["target_seeds"] + plan["new_phase_seeds"]:
        for amp in plan["amplitudes"]:
            name = f"axis_x_E_s{seed}@{amp:.3f}"
            if name in rebuilt:
                images = rebuilt[name]
            else:
                with np.load(examples / (name+".npz"), allow_pickle=False) as data:
                    images = {key: data[key] for key in ("clean", "input", "previous", "current")}
            gains = []
            for method in ("input", "previous", "current"):
                entry = rows[name, method]
                actual = (array_hash(images["clean"]), array_hash(images["input"]), array_hash(images[method]))
                expected = tuple(entry[key] for key in ("reference_sha256", "observed_sha256", "output_sha256"))
                if actual != expected:
                    raise ValueError(f"Figure output differs from the measured array for {name}/{method}")
                if method != "input":
                    gains.append(float(entry["noise_gain"]))
            cases[seed, amp] = images, gains
    return cases


def plot(folder, dest):
    plan = read_json(ROOT / "results/phase_refit/protocol.json")
    cases = read_cases(folder, plan)
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei") if f in fonts), None)
    if font:
        plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})
    def tr(cn, en):
        return cn if font else en

    sections = (np.s_[96:152, 48:104], np.s_[96:152, 200:256])
    def crop(image):
        # A blank gutter separates two distant locations at the same magnification.
        return np.column_stack((image[sections[0]], np.full((56, 3), np.nan), image[sections[1]]))

    error_max = max(np.max(np.abs((images[m]-images["clean"])[s]))
                    for images, _ in cases.values() for m in ("previous", "current") for s in sections)
    limit = max(.01, np.ceil(100*error_max)/100)
    dest.mkdir(parents=True, exist_ok=True)
    for title, seeds, include_views, filename in (
        (tr("原先留有斜纹的四例","Four previously under-corrected cases"), plan["target_seeds"], True, "targets"),
        (tr("四个原目标的残留条纹","Residuals in the four original cases"), plan["target_seeds"], False, "public"),
        (tr("三个全新相位的六例","Six cases at three held-out phases"), plan["new_phase_seeds"], False, "holdout")):
        versions = ("clean", "input", "previous", "current") if include_views else ()
        columns = (*versions, "old_error", "new_error")
        headers = {"clean":tr("清晰参考","Clean"), "input":tr("含噪输入","Observed"),
                   "previous":tr("优化前","Previous"), "current":tr("优化后","Current"),
                   "old_error":tr("优化前误差","Previous error"), "new_error":tr("优化后误差","Current error")}
        rows = [(s,a) for s in seeds for a in plan["amplitudes"]]
        width = 15.6 if include_views else 8.5
        height = 1.05*len(rows)+2.2 if include_views else 1.19*len(rows)+2.0
        fig, axes = plt.subplots(len(rows), len(columns), figsize=(width,height))
        fig.subplots_adjust(left=.145 if include_views else .25, right=.98, top=.84 if include_views else .87,
                            bottom=.13 if include_views else .12, wspace=.08, hspace=.17)
        fig.suptitle(title, x=.02, ha="left", fontsize=17, weight="bold", y=.978)
        fig.text(.02,.914 if include_views else .922,
                 tr("每格并排展示同一行段的左右两处。白缝为分隔。", "Each panel pairs two spatially separated crops."),
                 fontsize=10)
        for row,(seed,amp) in enumerate(rows):
            images, gains = cases[seed, amp]
            for i, key in enumerate(columns):
                ax = axes[row,i]
                if key in ("old_error","new_error"):
                    residual = images["previous" if key=="old_error" else "current"]-images["clean"]
                    signed = ax.imshow(crop(residual), cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
                else:
                    ax.imshow(crop(images[key]), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(.5)
                    spine.set_edgecolor(".75")
                if not row:
                    ax.set_title(headers[key], fontsize=11)
            axes[row,0].set_ylabel(f"{seed}  ·  {amp:.3f}\n{gains[0]:.1%} → {gains[1]:.1%}", rotation=0,
                                   ha="right", va="center", labelpad=11, fontsize=10)
        fig.text(.02,.055, tr("行标为相位、幅值及整图去除比例。","Row labels: phase, amplitude and full-image removal."), fontsize=9)
        fig.text(.02,.026, tr("原图行 96–151，左列 48–103，右列 200–255。","Rows 96–151; columns 48–103 and 200–255."), fontsize=9)
        bar = fig.colorbar(signed, cax=fig.add_axes([.72,.059,.24,.011]), orientation="horizontal",
                           ticks=[-limit,0,limit], format="%.2f")
        bar.ax.set_title(tr("恢复图 − 清晰图","Restored minus clean"), fontsize=9)
        fig.savefig(dest / f"phase-refit-{filename}.png", dpi=155)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,default=ROOT/"results/phase_refit")
    parser.add_argument("--out",type=Path,default=ROOT/"docs/figures")
    args = parser.parse_args()
    plot(args.results,args.out)
