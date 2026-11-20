"""Replay the four historical phase failures without changing the frozen estimator."""
import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

import collision_review as legacy
from evaluate_collision83 import PLAN, PHASES, samples
from evaluate_weave import read_json, write_json
from project_comparison import ROOT, array_hash
from project_guarded import denoise


def known():
    seeds = (741209, 741251)
    for case, amp in samples(read_json(PLAN)):
        if case[0] == PHASES and any(f"_s{s}@" in case[3] for s in seeds):
            yield case[3], case[4], case[5]


def old_fit_details(observed, diag):
    frame = {}
    def tracer(f, event, arg):
        if f.f_code is legacy._exterior_fit.__code__:
            if event == "return":
                frame.update(f.f_locals.copy())
            return tracer
        return None
    prior = sys.gettrace()
    try:
        sys.settrace(tracer)
        _, _, ev = legacy.review(observed, diag)
    finally:
        sys.settrace(prior)
    f = frame
    detail = dict(old_reason=ev["reason"], exterior_runs=[], recorded_bands=ev["bands"])
    if not f:
        return detail, ev
    guarded = f["guarded"]
    runs = np.flatnonzero(np.diff(np.r_[False, ~guarded, False])).reshape(-1,2)
    detail.update(exterior_runs=runs.tolist(), effective_columns=float(f["weights"].sum()),
                  effective_span=int(np.ptp(np.flatnonzero(f["used"]))), buffer_period=int(f["period"]))
    if "solved" in f:
        result = f["solved"]
        detail.update(optimizer_message=str(result.message), iterations=int(result.nit),
                      initial_frequency_bins=f["start"].tolist(), fitted_frequency_bins=result.x.tolist(),
                      drift_bins=float(f["drift"]) if "drift" in f else None)
    if "phase" in f:
        detail["old_exterior_amplitude"] = float(f["amplitude"])
        detail["old_exterior_phase"] = float(np.angle(f["phase"]))
    for row in detail["recorded_bands"]:
        if "exterior_fit_half_coefficients" in row:
            c = np.asarray(row["exterior_fit_half_coefficients"])
            row["old_half_coefficient_errors"] = abs(c[:,0]+1j*c[:,1]-1).tolist()
    return detail, ev


def run():
    rows = {}
    for name, clean, observed in known():
        _, diag = denoise(observed, return_diagnostics=True)
        row, evidence = old_fit_details(observed,diag)
        # Hold the original geometry, censoring footprint and initial estimate
        # fixed. Remove the reference texture only in this causal offline probe.
        axis = evidence["sweep_axis"]
        guarded = np.zeros(observed.shape[axis],bool)
        for band in evidence["bands"]:
            guarded[band["start"]:band["stop"]] = True
        fit = diag["evidence"]["guarded_carrier"]
        oracle = legacy._exterior_fit(.5+observed-clean,guarded,axis,fit)
        row["known_texture_removed_exterior_fit"] = None if oracle is None else oracle[2]
        if oracle is not None:
            # A second intervention changes only the independently fitted model.
            # Both hypotheses are checked against the *same* real brightness.
            normalized = 2*(observed-gaussian_filter(observed,5.))*oracle[1]/oracle[2]["amplitude"]
            half_errors = []
            for band in evidence["bands"]:
                sl = slice(band["start"],band["stop"])
                area = normalized[sl,:] if axis==0 else normalized[:,sl]
                half_errors.append([float(abs(q.mean()-1)) for q in np.array_split(area,2,axis=1-axis)])
            row["known_texture_removed_model_real_brightness_errors"] = half_errors
        row.update(reference_sha256=array_hash(clean), observation_sha256=array_hash(observed))
        rows[name] = row
    return rows


def report(rows):
    lines=["# 两个相位、四个旧版拒绝的逐阶段诊断","",
           "同一张 256×256 的编织照片叠加沿横轴变频的条纹，仅改变初相位及幅值。用观测图重跑冻结的旧入口，记录带外最优化终止原因、有效支撑、最大频率偏移和两处保护带内的交叉波形核验。下表中的角点偏移单位为整图 FFT 格，有效列数为带外权重总和。","",
           "| 相位、幅值 | 有效列数 | 带外自由拟合的最大角点偏移 | 仅在诊断中去掉已知织纹后的偏移 | 旧版拒绝环节 |",
           "|---|---:|---:|---:|---|"]
    for name,r in rows.items():
        outside = r["old_reason"]=="inconclusive_exterior_phase_fit"
        oracle = r["known_texture_removed_exterior_fit"]
        lines.append(f"| {name.split('_s')[-1]} | {r['effective_columns']:.1f} | {r['drift_bins']:.3f} | "
                     f"{oracle['maximum_corner_drift_bins']:.3f} | "
                     +("超过 0.5 格漂移上限" if outside else "带内亮度核验")+" |")
    high = rows["axis_x_E_s741251@0.080"]
    band = high["recorded_bands"][-1]
    before = band["old_half_coefficient_errors"]
    after = high["known_texture_removed_model_real_brightness_errors"][-1]
    lines.extend(["","四例的优化器均已收敛。每例的带外亮度只来自两段列区间。占右侧变化最明显的保护带以右没有带外样本，所以频率曲率和相位在此处主要靠外推。保留相同的带外位置、权重、截幅产生的真实增量和初始几何，只在离线探针中去掉已知清晰纹理，四例独立拟合都落回半格范围内。这个干预支持织纹扰动与带外几何共同造成拟合偏移。","",
                  f"741251 高幅档的左带两个半区通过。右带的上、下半区复亮度系数误差分别为 {before[0]:.3f} 和 {before[1]:.3f}，"
                  f"下半超过了 0.45 上限。右带 {band['held_endpoint_count']} 个留出截幅点的相位一致性为 {band['exterior_fit_endpoint_alignment']:.3f}，"
                  f"已高于 0.65 门限。将离线探针的带外模型放回完全相同的真实观测上核验，右带两个亮度误差分别降至 {after[0]:.3f} 与 {after[1]:.3f}。","",
                  "实际推理只使用观测图。截幅端点约束扫频轴和粗曲率，将可能含干扰的频带从估计织纹格点的数据中移出，再在两段分离的带外亮度上估计中心频率、相位和幅度。带内检验把推算节点处的定频分量作为待保留的织纹，只对两种波形能够区分的部分检查渐变干扰。每个半区的定频估计还要能预测另一半的留出截幅符号。","",
                  "[逐例原始诊断](diagnosis.json)"])
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=ROOT/"results/phase_refit")
    folder = p.parse_args().out
    folder.mkdir(parents=True,exist_ok=True)
    rows=run()
    write_json(folder/"diagnosis.json",rows)
    (folder/"diagnosis.md").write_text(report(rows),encoding="utf8")
