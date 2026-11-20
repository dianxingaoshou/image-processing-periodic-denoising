"""Replay the frozen weak-carrier search and attribute missed woven_0083 components."""
import argparse
import csv
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.ndimage import maximum_filter

from diagnose_woven_weak import old_cases
from evaluate_weak83 import extra_cases
from evaluate_weave import read_json, write_json
from image_io_v3 import sha256_file
from project_comparison import ROOT, array_hash
from project_weak import denoise
import weak_carrier as old
from weave_guard import MAX_ORDER, _fit_lattice, _spectrum


ARCHIVE = ROOT / "results/weak83"
BOUNDS = (.025, .45)


def observations():
    plan, axial = (read_json(ARCHIVE/name) for name in ("protocol.json", "single_axis_protocol.json"))
    specs = {s["name"]: s for s in plan["new_variants"] + axial["new_variants"]}
    ref = None
    for kind, variant, clean, obs, field in old_cases():
        if kind not in ("clean", "stationary_A", "chirp"):
            continue
        ref = clean
        truth = None
        if field is not None:
            fy, fx = field
            truth = dict(fy=float(np.mean(fy)), fx=float(np.mean(fx)),
                         dy=float(fy[-1,0]-fy[0,0]), dx=float(fx[0,-1]-fx[0,0]), cross=0.)
        yield kind, variant, clean, obs, truth
    for _, _, kind, variant, clean, obs, _ in extra_cases(ref, plan, axial):
        if kind != "clean":
            yield kind, variant, clean, obs, specs[variant.split("@")[0]]


def lattice_parts(image):
    shape = np.asarray(image.shape)
    peaks, _, power = _spectrum(image)
    members, basis = _fit_lattice(peaks, power, shape)
    orders = np.array(list(product(range(-MAX_ORDER,MAX_ORDER+1), repeat=2)))
    nodes = orders[np.any(orders != 0, axis=1)] @ basis
    return np.vstack((peaks[members], -peaks[members])), nodes


def frequency_field(params, positions):
    """Return gradient in native Fourier bins, in centered scaled image coordinates."""
    y, x = positions[..., 0], positions[..., 1]
    cross = params[4] if len(params) == 5 else 0.
    return np.stack((params[0]+params[2]*y+cross*x, params[1]+params[3]*x+cross*y), axis=-1)


def field_error(params, truth, corners):
    estimate, target = frequency_field(params,corners), frequency_field(truth,corners)
    return float(min(np.linalg.norm(estimate-sign*target,axis=1).max() for sign in (-1,1)))


def clearance(params, shape, peaks):
    # Exact continuous line segment distance reproduces the stationary/axial guard.
    ends = old._field(params,shape)[[0,-1]] * shape
    direction = ends[1] - ends[0]
    relative = peaks-ends[0]
    fraction = np.clip(relative@direction/max(direction@direction,1e-20),0,1)
    distances = np.linalg.norm(relative-fraction[:,None]*direction, axis=1)
    j = int(distances.argmin())
    return dict(distance_bins=float(distances[j]), closest_bin=peaks[j].tolist())


def target_clearance(params, basis, peaks):
    if params[4] == 0.:
        return clearance(params[:4],np.array(basis.shape[:2]),peaks)["distance_bins"]
    # The full two-dimensional target can have an interior collision. Sample all
    # pixel centers for this reference-only test of the out-of-model waveform.
    field = frequency_field(params,basis[:,:,:2])
    return float(min(np.min(np.hypot(field[:,:,0]-py,field[:,:,1]-px)) for py,px in peaks))


def phase_checks(phase, cells, signal, weights, training):
    phasor = np.exp(-2j*np.pi*phase)
    cell = cells.ravel()
    v = (2*weights*signal*phasor).ravel()
    coeff = (np.bincount(cell,weights=v.real)+1j*np.bincount(cell,weights=v.imag))
    coeff /= np.bincount(cell,weights=weights.ravel())
    fitted = coeff[training].mean()
    predicted = np.real(fitted*phasor.conj())
    gain = np.bincount(cell,weights=(weights*predicted*(2*signal-predicted)).ravel())
    ratio = coeff / fitted if abs(fitted) > 0 else np.zeros_like(coeff)
    held = np.where(training, np.nan+0j, ratio).reshape(4,4)
    row_col = [int((abs(np.nanmean(held,axis=a)-1)<=old.MAX_COEFFICIENT_ERROR).sum()) for a in (1,0)]
    passes = old._verify(np.ones(1),phase[:,:,None],cells,signal,weights,training) is not None
    return dict(passed=passes, amplitude=float(abs(fitted)), positive_check_cells=int((gain[~training]>0).sum()),
                training_inphase_cells=int((ratio[training].real>.25).sum()), row_column_hits=row_col)


def trace_stage(curved, basis, cells, sig, wt, discovery, measured, nodes, truth, corners):
    shape = np.array(discovery.shape)
    protected = np.vstack((measured,nodes))
    # Capture the production seeds immediately before the clearance filter and
    # 12-fit budget. The transform and its 16-peak budget remain identical.
    with patch.object(old,"MAX_FITS",10000), patch.object(old,"_clear",return_value=True):
        raw = old._seeds(discovery,basis,protected,BOUNDS,curved)
    eligible = [i for i,(_,p,_) in enumerate(raw) if curved or old._clear(p,shape,protected,BOUNDS)]
    selected = old._seeds(discovery,basis,protected,BOUNDS,curved)
    kept = eligible[:old.MAX_FITS]
    if len(selected) != len(kept) or any(not np.array_equal(raw[i][1],s[1]) for i,s in zip(kept,selected)):
        raise AssertionError("Instrumented candidate selection disagrees with frozen search")
    tr = np.indices((4,4)).sum(axis=0).ravel()%2==0
    train = tr[cells]
    coords, train_signal = basis[train], 2*(sig*wt)[train]/wt[train].sum()
    rows = []
    for i, (_, p, axis) in enumerate(raw):
        err = field_error(p,truth,corners) if truth is not None else None
        if i not in kept and (err is None or err>2.):
            continue
        fit = old._optimize(p,axis,coords,train_signal)
        # Testing phase here even on protected/discarded seeds is a diagnostic
        # counterfactual, never a change to the production result.
        checks = phase_checks(basis@fit,cells,sig,wt,tr)
        direct_check = old._verify(fit,basis,cells,sig,wt,tr) is not None
        if direct_check != checks["passed"]:
            raise AssertionError("Phase checks disagree with the frozen implementation")
        rows.append(dict(raw_rank=i+1, eligible_rank=eligible.index(i)+1 if i in eligible else None,
                         budget_selected=i in kept, seed=p.tolist(), seed_error=err,
                         coarse_clear=old._clear(p,shape,protected,BOUNDS),
                         refined=fit.tolist(), refined_error=field_error(fit,truth,corners) if truth is not None else None,
                         refined_clear=old._clear(fit,shape,protected,BOUNDS),
                         measured_clearance=clearance(fit,shape,measured), node_clearance=clearance(fit,shape,nodes),
                         curvature_pass=not curved or bool(abs(fit[2+axis])>=old.SWEEP_STEP),
                         phase=checks))
    return dict(model="axial" if curved else "fixed", pooled_raw=len(raw), selected=len(kept), traced=rows)


def known_sweep_rank(discovery, basis, truth):
    """Use only the known curvature, leaving central frequency and amplitude unknown."""
    angle = basis[:,:,2:]@truth[2:4] + truth[4]*basis[:,:,0]*basis[:,:,1]
    power = abs(np.fft.fft2(discovery*np.exp(-2j*np.pi*angle)))
    h,w = power.shape
    fy,fx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
    radial = np.hypot(fy,fx)
    mask = (power==maximum_filter(power,7,mode="wrap")) & (radial>=BOUNDS[0]) & (radial<=BOUNDS[1])
    loc = np.argwhere(mask)[np.argsort(-power[mask],kind="stable")]
    bins = np.column_stack((np.fft.fftfreq(h)[loc[:,0]]*h, np.fft.fftfreq(w)[loc[:,1]]*w))
    distance = np.linalg.norm(bins-truth[:2],axis=1)
    hits = np.flatnonzero(distance<=1.)
    return int(hits[0]+1) if len(hits) else None


def diagnose():
    with (ARCHIVE/"metrics.csv").open(encoding="utf8",newline="") as stream:
        compare = {(r["kind"],r["variant"]):r for r in csv.DictReader(stream)
                   if r["image_id"]=="dtd_woven_0083" and r["method"]=="current"}
    records = []
    for kind,var,ref,obs,spec in observations():
        output, diag = denoise(obs,return_diagnostics=True)
        a = compare[kind,var]
        if [a[k] for k in ("reference_sha256","observed_sha256","output_sha256","route","reason")] != [
                array_hash(ref),array_hash(obs),array_hash(output),diag["route"],diag["reason"]]:
            raise AssertionError(f"Unexpected historical replay for {kind} {var}")
        basis,cells,sig,wt,disc = old._measurements(obs)
        measured,nodes = lattice_parts(obs)
        clean_peaks,clean_nodes = lattice_parts(ref)
        shape = np.asarray(ref.shape)
        corners = np.array(list(product((-.5,.5),repeat=2)))*(shape-1)/shape
        truth = None
        row = dict(kind=kind,variant=var,route=diag["route"],reason=diag["reason"],
                   retry_executed="weak_carrier" in diag["evidence"], old_weak_decision=diag["evidence"].get("weak_carrier"),
                   observed_sha256=a["observed_sha256"], reference_sha256=a["reference_sha256"],output_sha256=a["output_sha256"])
        e = diag["evidence"]
        row["prior_route_evidence"] = dict(candidate_tiles=e.get("general_candidate_tiles"),
                corroborated_tiles=e.get("general_corroborated_tiles"), accepted_tiles=e.get("general_accepted_tiles"),
                lattice_matches=e.get("lattice_screen",{}).get("matched_regions"),
                alternate_carrier=e.get("alternative_carrier"))
        if spec is not None:
            truth = np.array([spec[k] for k in ("fy","fx","dy","dx","cross")])*np.r_[
                shape,shape**2/(shape-1),np.prod(shape)/np.sqrt(np.prod(shape-1))]
            row.update(target=truth.tolist(), exact_curvature_peak_rank=known_sweep_rank(disc,basis,truth),
                       observed_measured_distance=target_clearance(truth,basis,measured),
                       observed_node_distance=target_clearance(truth,basis,nodes),
                       clean_measured_distance=target_clearance(truth,basis,clean_peaks),
                       clean_node_distance=target_clearance(truth,basis,clean_nodes))
            tr = np.indices((4,4)).sum(axis=0).ravel()%2==0
            phase = basis@truth[:4]+truth[4]*basis[:,:,0]*basis[:,:,1]
            row["known_phase_check"] = phase_checks(phase,cells,sig,wt,tr)
        row["stages"] = [trace_stage(curved,basis,cells,sig,wt,disc,measured,nodes,truth,corners)
                         for curved in (False,True)]
        records.append(row)
    return dict(archive_sha256=sha256_file(ARCHIVE/"metrics.csv"), source_sha256=sha256_file(Path(__file__)),
                trace="Both stages replayed even if routing skipped them; all production outputs verified separately",
                bound="Corner-max frequency-field error with conjugate equivalence; seed <= 2 native bins, refined <= 1",
                records=records)


def report(data):
    lines = ["# woven_0083 的九个剩余条件为何没有处理", "",
             "重放上版 13 组输入，包括九个失败例和四个对照。参考图、输入图、输出图的 SHA-256、"
             "最终路径和原因逐项匹配历史记录。所有频率真值只进入离线评分。", "",
             "表中候选排名来自拟合用的八个交错块。定频谱和各扫频谱各取前 16 峰，合并同一模型阶段的"
             "峰后最多精拟合 12 项。将预测与真值在四角的二维频率相差均不超过两个整图 FFT 格点的种子"
             "计为接近，允许整体正负共轭。每格为 1/256 周期/像素。", "",
             "| 旧条件 | 候选在原始合并榜排名 | 进入精拟合预算 | 精化后最大频率误差（格） | 轨迹与观测实测织纹／推算格点的最近距离（格） | 精化相位通过 |",
             "|---|---:|---|---:|---|---|"]
    for r in data["records"]:
        if not r["variant"].startswith(("fixed_", "axis_", "sweep_")):
            continue
        relevant = r["stages"][int(r["kind"]!="fixed")]["traced"]
        matched = [t for t in relevant if t["seed_error"]<=2.]
        first = matched[0] if matched else None
        name = r["variant"].split("@")[0]
        amp = .075 if name.startswith("axis_") else (.05 if r["variant"].endswith("0.05") else .085)
        rank = str(first["raw_rank"]) if first else "无"
        selected = ("是" if first["budget_selected"] else "否") if first else "—"
        error = f"{first['refined_error']:.3f}" if first else "—"
        phase = ("是" if first["phase"]["passed"] else "否") if first else "—"
        distance = f"{r['observed_measured_distance']:.2f}／{r['observed_node_distance']:.2f}"
        lines.append(f"| {name} {amp:g} | {rank} | {selected} | {error} | {distance} | {phase} |")
    lines += ["", "距离列使用完整真实轨迹，定频和单轴用连续线段计算，含交叉项的双轴用所有像素中心。"
              "对于预算外、被保护或路由跳过的种子，精化与相位列是单独重放旧拟合器的对照，实际运行未放行。", "",
              "**定频。** C 的真实频率为 (0.063, −0.136)，D 为 (0.143, 0.083) 周期/像素。"
              "四个条件均在前五名中检出。投影到二维频谱后，新增频率被观测图的格点拟合本身归为"
              "实测织纹成员。定频粗筛就在 1.25 格的清距门槛处排除了这些峰。C 的 0.085 条件还提前"
              "触发原来的织纹保护，31 个已有空间支撑的图块全被归入织纹格点，原替代载波提出 3 项、通过 0 项，"
              "后续弱载波重试未运行。在诊断中越过清距后，四个候选沿用原相位核验均通过。"
              "同一张未加噪原图中，C、D 的真实频率分别距最近实测格点 18.73、17.84 格。D 距一个未实测的推算"
              "格点仅 1.23 格，单靠理想格点位置也会拒绝这个真实新增分量。", "",
              "**单轴渐变。** 横向扫频的近真种子排第 9，进入预算并获得准确相位，但精化轨迹穿过观测"
              "实测格点的保护区。纵向扫频的两个近真种子排第 53 和 56，合并排序只留 12 个。单独"
              "精化第 53 个种子后，格点与原相位核验均通过。横向扫频轨迹与最近实测格点的距离在同图"
              "加噪前为 15.54 格，加噪后为 0.01 格。两例的首要阻碍分别是格点保护与候选截断。", "",
              "**双轴渐变。** 三个未处理条件和一个既有小幅改善对照都含两个非零轴向曲率与交叉项。"
              "旧搜索只允许一条轴改变频率。在四角场误差的判据下均没有接近的模型，D 的真轨迹还经过"
              "推算格点附近。诊断中若仅提供真实曲率消除扫频，再从训练块搜索未知中心频率，三个失败例的目标峰"
              "依次排第 9、7、1。C 的 0.085 已由旧广域路径取得少量改善，单独列作对照。", "",
              "**相位核验对照。** 将生成器提供的真实曲率和中心频率作为单独的离线探针，仍从观测训练"
              "半区拟合复振幅。九个目标均通过原有的八块留出正收益、行列一致性检查。这个结果说明若能"
              "找到相应波形，既有相位条件可以接受它。实际算法还需要仅凭观测找到候选，并判断它与织纹的归属。", "",
              "在项目根目录运行 `python diagnose_woven_nine.py --out results/nine83_reproduction` 可重建。"
              "[逐候选、同图未加噪频率距离、相位证据和历史哈希](diagnosis.json)"]
    return "\n".join(lines)+"\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=ROOT/"results/nine83_reproduction")
    folder=parser.parse_args().out
    folder.mkdir(parents=True,exist_ok=True)
    data=diagnose()
    write_json(folder/"diagnosis.json",data)
    (folder/"diagnosis.md").write_text(report(data),encoding="utf8")
