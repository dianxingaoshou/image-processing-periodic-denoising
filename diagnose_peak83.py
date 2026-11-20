"""Replay the frozen guarded carrier and isolate the stages that alter textile peaks."""
import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.spatial import cKDTree

import guarded_carrier as guarded
from check_nine83_peaks import peak_masks
from evaluate_nine83 import DIRECTIONS, samples, snapshot
from evaluate_weave import projected_residual, read_json, write_json
from image_io_v3 import sha256_file
from project_comparison import ROOT, array_hash
from project_guarded import _restore, denoise
from project_weak import _subtract


TARGETS = ("axis_x_E@0.045", "axis_x_E@0.080", "axis_y_E@0.045")


def spectrum(image):
    return np.fft.fft2(image, norm="ortho") / np.sqrt(image.size)


def stages(observed, clean, diag, variant, amplitude):
    h, w = observed.shape
    p = diag["evidence"]["guarded_carrier"]
    params = np.r_[np.asarray(p["center_frequency"])*[h,w], np.asarray(p["hessian"])*[h*h,h*w,w*w]]
    y, x = np.indices((h, w), dtype=float)
    y -= (h-1)/2
    x -= (w-1)/2
    basis = np.stack((y/h, x/w, .5*(y/h)**2, (y/h)*(x/w), .5*(x/w)**2), axis=-1)
    full = np.real(p["amplitude"]*np.exp(1j*p["final_phase_radians"])*np.exp((2j*np.pi)*(basis@params)))
    peaks, _, powers = guarded._spectrum(observed)
    members, lattice = guarded._fit_lattice(peaks,powers,np.array((h,w)))
    nodes = np.vstack((peaks[members], -peaks[members], guarded._node_bins(lattice)))
    field = guarded._field(params,basis[...,:2])
    taper, protection = guarded._collision_taper(field,cKDTree(nodes),observed.shape)
    wave = diag["carrier_wave"]
    if not np.allclose(wave, full*taper, rtol=0, atol=2e-14):
        raise AssertionError("Reconstructed wave does not match guarded diagnostics")
    v = variant
    true_phase = (v["fy"]*y+v["fx"]*x+v["dy"]*y*y/(2*(h-1))+v["dx"]*x*x/(2*(w-1))
                  +v["cross"]*x*y/np.sqrt((h-1)*(w-1)))
    true_wave = amplitude*np.sin(2*np.pi*true_phase + np.random.default_rng(v["seed"]).uniform(-np.pi,np.pi))
    if not np.array_equal(np.clip(clean+true_wave,0,1),observed):
        raise AssertionError("Unclipped generative wave differs from the formal protocol")
    actual_added = observed-clean
    trained = np.array(p["training_parameters_bins"])
    _, cells, brightness, weights, _ = guarded._measurements(observed)
    train = np.indices((4,4)).sum(axis=0).ravel()%2 == 0
    # Frequency and phase are estimated without reference data. Replay the independent
    # training-only estimate before the bounded all-image calibration.
    _, coef = guarded._verify(trained,basis,cells,brightness,weights,train)
    pre_calibration = np.real(coef*np.exp((2j*np.pi)*(basis@trained)))
    final = _restore(observed,wave,1.)
    gain, _ = projected_residual(observed-final,actual_added)
    full_gain, _ = projected_residual(full,actual_added)
    # Only this offline comparison can match a *true* projection gain.
    # Restore endpoints at every trial as in the frozen production path.
    def gain_error(scale):
        restored = _restore(observed, scale*full, 1.)
        return projected_residual(observed-restored,actual_added)[0] - gain
    scale = brentq(gain_error,0.,1.,xtol=1e-14)
    outputs = dict(observed=observed, fit_raw=observed-full, fit_restore=_restore(observed,full,1.),
                   pre_calibration_raw=observed-pre_calibration, tapered_raw=observed-wave,
                   tapered_range_clip=np.clip(observed-wave,0,1), tapered_midpoint=_subtract(observed,wave,1.),
                   final=final, true_tapered_raw=observed-true_wave*taper,
                   true_tapered_restore=_restore(observed,true_wave*taper,1.),
                   true_full_restore=_restore(observed,true_wave,1.),
                   uniform_fit_matched_gain=observed-full*(gain/full_gain),
                   uniform_fit_restore_matched_gain=_restore(observed,scale*full,1.),
                   oracle_postclip_matched_gain=observed-gain*actual_added)
    interior = (observed>0) & (observed<1)
    if (abs(gain_error(scale)) > 1e-10
            or not np.array_equal(outputs["tapered_midpoint"][interior],final[interior])):
        raise AssertionError("Intervention altered the matched gain or the non-endpoint control")
    notes = dict(protection=protection, observed_endpoints=int(np.sum((observed==0)|(observed==1))),
                 estimated_amplitude=p["amplitude"], phase_refined=p["phase_refined"],
                 mean_taper=float(taper.mean()), final_gain=gain, uniform_calibrated_scale=scale,
                 unmasked_fit_raw_wave_rmse=float(np.mean((full-true_wave)**2)**.5),
                 unmasked_fit_raw_wave_cos=float(np.vdot(full,true_wave)/(np.linalg.norm(full)*np.linalg.norm(true_wave))))
    return outputs, notes, field


def diagnose():
    old_folder = ROOT/"results/nine83"
    frozen = read_json(old_folder/"freeze.json")
    manifest = read_json(old_folder/"run.json")
    if frozen != snapshot() or any(sha256_file(old_folder/n) != d for n,d in manifest["artifact_sha256"].items()):
        raise ValueError("Baseline code or artifact hash differs from the archived run")
    with (old_folder/"metrics.csv").open(encoding="utf8", newline="") as fh:
        old = {(r["cohort"], r["image_id"], r["kind"],r["variant"]):r for r in csv.DictReader(fh)
               if r["method"]=="current"}
    with (old_folder/"per_peak.csv").open(encoding="utf8", newline="") as fh:
        per_peak = {(r["variant"],int(r["peak"])):r for r in csv.DictReader(fh)}
    plan = frozen["protocol"]
    variants = {v["name"]:v for v in plan["new_variants"]}
    results = {}
    for case, amplitude in samples(plan,frozen["endpoint_controls"]):
        cohort, image_id, kind, variant, clean, observed, _ = case
        is_clean = (cohort,image_id,kind) == ("archived_photo",plan["target"],"clean")
        if not is_clean and (cohort!=DIRECTIONS or variant not in TARGETS):
            continue
        output, diag = denoise(observed,return_diagnostics=True)
        hashes = [array_hash(a) for a in (clean,observed,output)]
        record = old[cohort,image_id,kind,str(variant)]
        if hashes != [record[k] for k in ("reference_sha256","observed_sha256","output_sha256")]:
            raise AssertionError(f"The archived source, input or output differs from {variant}")
        if is_clean:
            results["clean"] = dict(changed_pixels=int(np.sum(output!=clean)),route=diag["route"],reason=diag["reason"],
                                    sha256=hashes)
            continue
        peaks, disks = peak_masks(clean,manifest["reference_texture_peaks_in_fft_bins"][hashes[0]])
        outputs, notes, field = stages(observed,clean,diag,variants[variant.split("@")[0]],amplitude)
        if not np.array_equal(outputs["final"],output):
            raise AssertionError("Intermediate replay did not reconstruct the official result")
        by_stage = {}
        noise = observed-clean
        for name, restored in outputs.items():
            gain, off = projected_residual(observed-restored, noise)
            energy = abs(spectrum(restored-clean))**2
            by_stage[name] = dict(mse=float(np.mean((restored-clean)**2)), gain=gain, off_noise_mse=off,
                                  peak_mse=[float(energy[d].sum()) for d in disks])
        for i in range(len(peaks)):
            for stage, key in (("observed","input_mse"), ("final","current_mse")):
                if not np.isclose(by_stage[stage]["peak_mse"][i], float(per_peak[variant,i+1][key]),rtol=1e-12,atol=1e-23):
                    raise AssertionError("Peak MSE does not match the prior per-peak audit")
        distances = [float(np.minimum(np.linalg.norm(field-peak,axis=-1),np.linalg.norm(field+peak,axis=-1)).min())
                     for peak in map(np.asarray, peaks)]
        # Exact identity B(M-Q)-B(M) = ||Q||² - 2 Re <M,Q> for the post-clipping
        # added signal M and the actual full reconstruction residual Q on each disk.
        ns, removed = spectrum(noise), spectrum(observed-output)
        decomposition = []
        for i,d in enumerate(disks):
            rr = float(np.sum(abs(removed[d])**2))
            term = float(-2*np.vdot(ns[d],removed[d]).real)
            delta = by_stage["final"]["peak_mse"][i]-by_stage["observed"]["peak_mse"][i]
            if not np.isclose(rr+term,delta,rtol=1e-11,atol=1e-21):
                raise AssertionError("Error decomposition failed")
            decomposition.append(dict(removal_energy=rr, negative_twice_noise_dot_removal=term, increase=delta))
        results[variant] = dict(sha256=hashes,peak_bins=peaks,closest_frequency_to_peak_bins=distances,
                                diagnostics=notes,stages=by_stage,final_peak_error_delta_terms=decomposition)
    if set(results) != {"clean",*TARGETS}:
        raise ValueError("Missing diagnosis case")
    return results


def report(results):
    lines = [
        "# woven_0083 三个织纹峰误差上升条件的原因",
        "",
        "在同一张 256×256 清晰织物及三个含噪观测上重放 `project_guarded`。"
        "清晰参考、观测和旧版输出均匹配原实验 SHA-256。以下推理只读取观测图，"
        "真值对照及频带评价在得到结果后单独计算。频带的每个 FFT 格点等于 1/256 周期/像素。",
        "",
        "六组峰的中心依次为 (9,−2)、(19,0)、(9,6)、(0,8)、(0,15)、(10,−13)，"
        "按 (纵向, 横向) 记录，包含共轭中心。每组统计两个半径为 2 格的圆盘，共 26 个频谱点。"
        "原始未加噪照片的改动像素数为 " + str(results["clean"]["changed_pixels"]) + "。",
        "",
        "## 误差在哪一步增加",
        "",
        "从每例旧版的同一次运行取出各阶段波形，逐步重放减除、空间加权与端点复原。"
        "表中 MSE 的数值均须乘以 10⁻⁹，"
        "单位为归一亮度平方。各项都按整图像素数计。未限幅减除的中间结果允许暂时超出 [0,1]。",
        "",
        "| 条件／峰组 | 输入 | 训练波形直接减 | 精化波形直接减 | 叠加空间包络 | 限幅至合法值 | 端点取中点 | 旧版最终端点估计 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def value(r, stage, peak):
        return r["stages"][stage]["peak_mse"][peak-1]*1e9
    target_peaks = ((TARGETS[0],5),(TARGETS[0],6),(TARGETS[1],5),(TARGETS[2],6))
    for name, peak in target_peaks:
        r = results[name]
        stages = ("observed","pre_calibration_raw","fit_raw","tapered_raw","tapered_range_clip","tapered_midpoint","final")
        lines.append(f"| {name}／{peak} | " + " | ".join(f"{value(r,s,peak):.3f}" for s in stages) + " |")
    lines += [
        "",
        "横向的两个幅值均在施加空间包络时越过输入误差。在幅值 0.045 时，第 5 和第 6 组同时受损，"
        "幅值 0.080 时，第 5 组受损。包络只在图像部分位置减去波形，产生的频谱调制分布到原干扰轨迹以外。"
        "普通亮度限幅在横向两例的这些频带没有可见影响，旧版端点估计挽回了部分误差。",
        "",
        "纵向幅值 0.045 的第 6 组对拟合和端点估计都敏感。训练阶段波形直接减除时该组已经高于输入，"
        "精化及空间包络进一步改变该组误差。固定同一次拟合和包络，端点中点法使误差低于输入，"
        "更换为旧版使用的相反相位经验估计后重新超过输入。两种端点方法在非端点像素逐值相同。",
        "",
        "## 同一波形、同一去条纹强度的反事实对照",
        "",
        "下表仍以 10⁻⁹ 为单位，所有列均使用同一套旧端点估计（最后一项理想对照除外）。"
        "真实波形按原生成参数重建，加到参考上再截幅须与正式观测逐像素一致。"
        "均匀等比例对照使用相同拟合波形，离线求一个空间常数，使最终输出的真实投影减除比例和旧版相同。"
        "它会触及旧版禁止修改的位置。三个条件的匹配增益依次为 0.45360、0.46567 和 0.76121。"
        "理想等比例对照直接减去该比例的实际新增量，使用了清晰参考。",
        "",
        "| 条件／峰组 | 冻结算法 | 拟合波形，无包络 | 真实波形，同包络 | 真实波形，无包络 | 拟合波形，均匀等比例 | 理想等比例 | 最近轨迹距该峰（格） |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, peak in target_peaks:
        r = results[name]
        stages = ("final","fit_restore","true_tapered_restore","true_full_restore",
                  "uniform_fit_restore_matched_gain","oracle_postclip_matched_gain")
        lines.append(f"| {name}／{peak} | " + " | ".join(f"{value(r,s,peak):.3f}" for s in stages)
                     + f" | {r['closest_frequency_to_peak_bins'][peak-1]:.2f} |")
    lines += ["",
              "横向两例在同一拟合与端点算法下，仅移去空间包络就消除了上述主峰的误差上升。"
              "固定包络而把拟合换成完美真实波形，两个幅值仍损伤相同主峰。"
              "统一衰减并匹配原去条纹比例后，横向目标峰误差仍低于输入。"
              "这些干预共同支持空间权重在当前分支、当前横向条件下足以造成观察到的频带损伤。"
              "直接移去包络还会提高去除强度和放开原禁止处理像素，不能作为安全的修复策略。"
              "现有探针没有单独改变 FFT 边界条件，无法把误差进一步分摊给图像边界或包络内部的过渡带。","",
              "纵向条件保持拟合和包络时，仅把端点经验估计换成中点即可消除第 6 峰相对输入的微小增量。"
              "在不改包络、端点的第二个探针中，换入真实波形同样消除了增量。"
              "因此两种独立的变动都能跨过输入误差，不能把全部增量唯一归给端点，也不能据此决定永久使用中点法。"
              "训练波形与精化波形的直接减除对比只隔离未加包络、未作端点重建时的拟合变化。","",
              "三个受损条件中，载波轨迹到受损峰中心至少相距 21 个整图 FFT 格。这只衡量轨迹与六个评价峰的距离。"
              "算法在轨迹附近还可能检出其他织纹及推算格点。以上真值替换和增益匹配均只用于验证，均未参与候选判断或推理。", "",
              "`diagnosis.json` 还保留六组完整频带、各阶段整图 MSE、实际减除比例、正交误差、"
              "包络覆盖像素数，以及逐峰的误差差值分解。令 M 为真实新增量、Q 为实际减除量，"
              "每个频带的误差增量等于 ‖Q‖² − 2Re〈M,Q〉。只要两项不抵消，"
              "波形与真实干扰在整图接近也可能扩大某一织纹频带的误差。", "",
              "复现命令为 `python diagnose_peak83.py --out results/peak83_reproduction`。[逐阶段数据](diagnosis.json)"]
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"results/peak83")
    args = parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    data = diagnose()
    write_json(args.out/"diagnosis.json",data)
    (args.out/"diagnosis.md").write_text(report(data),encoding="utf8")
