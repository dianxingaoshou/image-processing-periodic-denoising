"""Locate the two vertical residuals in the frozen peak83 woven observations."""
import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

from check_nine83_peaks import peak_masks
from evaluate_nine83 import DIRECTIONS, samples
from evaluate_weave import read_json, write_json
from guarded_carrier import TAPER_BINS, _node_bins
from image_io_v3 import sha256_file
from peak_guard import _carrier_geometry
from project_comparison import ROOT, array_hash
from project_guarded import _restore
from project_peak import denoise
from weave_guard import LATTICE_TOLERANCE_BINS, MAX_ORDER, _fit_lattice, _spectrum
from weak_carrier import _measurements


CASES = ("axis_x_E@0.045", "axis_x_E@0.080")


def geometry(image):
    peaks, _, powers = _spectrum(image)
    members, lattice = _fit_lattice(peaks, powers, np.asarray(image.shape))
    measured = np.vstack((peaks[members], -peaks[members]))
    orders = np.array([ij for ij in product(range(-MAX_ORDER, MAX_ORDER+1), repeat=2) if any(ij)])
    return dict(peaks=peaks, measured=measured, lattice=lattice, orders=orders, predicted=_node_bins(lattice))


def weights(field, nodes):
    distance, nearest = cKDTree(nodes).query(field.reshape(-1, 2))
    ramp = np.clip((distance-LATTICE_TOLERANCE_BINS)/TAPER_BINS, 0, 1)
    return (ramp*ramp*(3-2*ramp)).reshape(field.shape[:2]), nearest.reshape(field.shape[:2])


def region_stats(mask, clean, obs, output):
    addition = (obs-clean)[mask]
    removed = (obs-output)[mask]
    errors = (output-clean)[mask]
    return dict(pixels=int(mask.sum()), mse=float(np.mean(errors**2)), input_mse=float(np.mean(addition**2)),
                removal_gain=float(np.dot(removed,addition)/np.dot(addition,addition)))


def intervals(included):
    changes = np.flatnonzero(np.diff(np.r_[False, included, False])).reshape(-1,2)
    return [[int(first),int(stop-1)] for first,stop in changes]


def at_node(point, g):
    # Both signs correspond to one true frequency for a real image.
    def distance(array):
        return float(np.linalg.norm(np.vstack((array,-array))-point,axis=1).min())
    return dict(nearest_detected_bin=distance(g["peaks"]), nearest_measured_member_bin=distance(g["measured"]),
                nearest_predicted_bin=distance(g["predicted"]))


def one_case(clean, obs, old, diag, ref_peaks):
    fit = diag["evidence"]["guarded_carrier"]
    wave, field = _carrier_geometry(clean.shape, fit)
    full = _restore(obs,wave,1.)
    lattices = {name:geometry(frame) for name,frame in (("observed",obs),("clean",clean),("estimated_clean",full))}
    g = lattices["observed"]
    all_nodes = np.vstack((g["measured"],g["predicted"]))
    taper, nearest = weights(field, all_nodes)
    if not np.allclose(taper*wave,diag["carrier_wave"],rtol=0,atol=2e-14):
        raise ValueError("Reconstructed envelope differs from the previous carrier")
    if not np.array_equal(diag["support_mask"], taper*wave != 0):
        raise ValueError("Reconstructed support differs from the previous path")
    labels, count = label(taper < 1.)
    if count != 2:
        raise ValueError("Historical target should have two separated collision regions")
    components = sorted((labels==i for i in range(1,count+1)), key=lambda m: np.where(m)[1].mean())
    _,cells,_,_,_ = _measurements(obs)
    held = ((cells//4 + cells%4) % 2) == 1
    signs = (obs==1.).astype(float) - (obs==0.)
    signed_phase = signs*wave/fit["amplitude"]
    clean_taper, _ = weights(field, np.vstack((lattices["clean"]["measured"],lattices["clean"]["predicted"])))
    measured_taper, _ = weights(field, g["measured"])
    previous_envelope = _restore(obs, wave*taper, 1.)
    probes = dict(input=obs, previous_peak83=old, guarded_envelope=previous_envelope,
                  clean_lattice_envelope=_restore(obs,wave*clean_taper,1.),
                  measured_nodes_only=_restore(obs,wave*measured_taper,1.), full_fit=full)
    regions = dict(all=np.ones_like(taper,bool),clear=taper==1)
    nodes = []
    info = []
    for idx, m in enumerate(components, start=1):
        region = f"band_{idx}"
        regions[region] = m
        regions[region+"_zero"] = m & (taper == 0)
        regions[region+"_transition"] = m & (taper > 0)
        ids, counts = np.unique(nearest[m], return_counts=True)
        for loc, pixels in zip(ids,counts):
            point = all_nodes[loc]
            j = loc-len(g["measured"])
            details = dict(band=idx, pixels=int(pixels), position_yx=point.tolist(),
                           source="predicted" if j>=0 else "measured",
                           integer_order=g["orders"][j].tolist() if j>=0 else None,
                           distance={name:at_node(point, lattice) for name,lattice in lattices.items()})
            nodes.append(details)
        y, x = np.where(m)
        info.append(dict(index=idx, rows_extent=[int(y.min()),int(y.max())], columns_extent=[int(x.min()),int(x.max())],
                         majority_columns=intervals(m.mean(0)>=.5),
                         majority_zero_columns=intervals(regions[region+"_zero"].mean(0)>=.5)))
        events = m & held & (signs!=0)
        info[-1]["phase_support"] = dict(held_endpoint_count=int(events.sum()),
            held_mean_signed_phase=float(signed_phase[events].mean()),
            held_endpoints_per_cell_row=np.bincount(cells[events]//4,minlength=4).tolist())
        # Remove only the nodes actually selected in this component, retaining
        # all other nodes and the fitted waveform and endpoint estimator.
        others = np.delete(all_nodes, ids, axis=0)
        alternate, _ = weights(field, others)
        probes["withhold_band_"+str(idx)+"_nodes"] = _restore(obs,wave*alternate,1.)
    _, disks = peak_masks(clean,ref_peaks)
    stages = {}
    for name, predicted in probes.items():
        freq_error = abs(np.fft.fft2(predicted-clean,norm="ortho"))**2/clean.size
        stages[name] = dict(regions={region:region_stats(m,clean,obs,predicted) for region,m in regions.items()},
                            weave_peak_mse=[float(freq_error[d].sum()) for d in disks])
    return dict(sha256=[array_hash(a) for a in (clean,obs,old)], band_geometry=info, nearest_nodes=nodes,
                components_definition="Four-connected components where fitted old taper is below 1",
                lattice_bases={k:g["lattice"].tolist() for k,g in lattices.items()},
                measured_member_pairs={k:len(g["measured"])//2 for k,g in lattices.items()},
                column_profile=dict(mean_old_taper=taper.mean(0).tolist(),
                                    input_rms=np.sqrt(np.mean((obs-clean)**2,axis=0)).tolist(),
                                    previous_rms=np.sqrt(np.mean((old-clean)**2,axis=0)).tolist()),
                stages=stages)


def diagnose():
    study = ROOT/"results/peak83"
    manifest = read_json(study/"run.json")
    if sha256_file(study/"metrics.csv") != manifest["artifact_sha256"]["metrics.csv"]:
        raise ValueError("Previous metrics changed")
    with (study/"metrics.csv").open(encoding="utf8",newline="") as stream:
        rows = {r["variant"]:r for r in csv.DictReader(stream) if r["cohort"]==DIRECTIONS and r["method"]=="current"}
    prior = ROOT/"results/nine83"
    freeze = read_json(prior/"freeze.json")
    ref_peaks = read_json(prior/"run.json")["reference_texture_peaks_in_fft_bins"]
    results = {}
    for case, _ in samples(freeze["protocol"],freeze["endpoint_controls"]):
        group, name, _, variant, clean, obs, _ = case
        if group != DIRECTIONS or name != "dtd_woven_0083" or variant not in CASES:
            continue
        old, diag = denoise(obs,return_diagnostics=True)
        result = one_case(clean,obs,old,diag,ref_peaks[array_hash(clean)])
        if result["sha256"] != [rows[variant][k] for k in ("reference_sha256","observed_sha256","output_sha256")]:
            raise ValueError(f"Historical replay failed for {variant}")
        results[variant] = result
    if set(results) != set(CASES):
        raise ValueError("Missing historical inputs")
    return results


def report(results):
    text = ["# axis_x_E 两处残留怎样形成", "",
            "同一张 256×256 织物的清晰参考、含噪观测、上版恢复图均匹配上一轮正式 SHA-256。"
            "以拟合条纹在每个像素的局部频率查询旧格点，离最近节点不足 1.25 个整图 FFT 格时权重为零，"
            "1.25 至 2.25 格之间平滑过渡。下表将旧权重小于 1 的像素划为两个四连通区，"
            "行列从零开始，区间包含两端。列范围按该列至少半数行符合条件记录。", "",
            "| 幅值 | 残留区 | 受抑制列 | 零权重列 | 零权重／过渡像素 | 原处理去除比例 | 输入 → 上版区域 MSE |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for name,r in results.items():
        metrics = r["stages"]["previous_peak83"]["regions"]
        for b in r["band_geometry"]:
            idx = b["index"]
            k = f"band_{idx}"
            lo, hi = b["majority_columns"][0]
            z0, z1 = b["majority_zero_columns"][0]
            v = metrics[k]
            text.append(f"| {name.split('@')[-1]} | {idx} | {lo}–{hi} | {z0}–{z1} | "
                        f"{metrics[k+'_zero']['pixels']}／{metrics[k+'_transition']['pixels']} | "
                        f"{v['removal_gain']:.3f} | {v['input_mse']:.6f} → {v['mse']:.6f} |")
    text += ["","两处均跨过全部 256 行。零权重区输出逐值保留观测，条纹也留在原位。上一版主峰修正只能在"
             "已有支撑内调整减除量。两处都由高阶整数推算的格点触发，最近的节点均未列入实测织纹成员。", "",
             "| 幅值 | 区 | 拟合节点（纵、横格） | 格点基向量的整数倍数 | 与最近检出峰距离（观测／原图） | 原图重拟最近推算点的偏移 |",
             "|---|---:|---|---|---:|---:|"]
    for name,r in results.items():
        for node in r["nearest_nodes"]:
            a,b = node["position_yx"]
            d = node["distance"]
            i,j = node["integer_order"]
            text.append(f"| {name.split('@')[-1]} | {node['band']} | ({a:.2f}, {b:.2f}) | ({i},{j}) | "
                        f"{d['observed']['nearest_detected_bin']:.2f}／{d['clean']['nearest_detected_bin']:.2f} | "
                        f"{d['clean']['nearest_predicted_bin']:.3f} |")
    text += ["", "将同一条频率轨迹分别与含噪图、未加噪同图、减掉完整候选波形后的图像各自拟合出的格点比较。"
             "三者都重建出几乎相同的两个推算节点。第一处在加噪后附近出现较强峰，但并未成为触发保护的实测成员。"
             "即使使用清晰参考的格点，旧规则也会留下相同两条带。旧峰检测有强度、峰距与数量限制，"
             "未列入实测成员不能证明该处完全没有真实织纹。", "",
             "在旧拟合器划分的核验图块上，按黑白截幅端点的实际符号核查最终波形。"
             "下表列出空间覆盖，留块坐标沿用旧划分，最终已精化相位使用过全图。", "",
             "| 幅值 | 区 | 有符号相位余弦均值 | 核验端点数 | 从上到下四排的端点数 |",
             "|---|---:|---:|---:|---|"]
    for name,r in results.items():
        for b in r["band_geometry"]:
            p = b["phase_support"]
            text.append(f"| {name.split('@')[-1]} | {b['index']} | {p['held_mean_signed_phase']:.3f} | "
                        f"{p['held_endpoint_count']} | "+"/".join(map(str,p["held_endpoints_per_cell_row"]))+" |")
    text += ["", "低幅右带范围内，33 个核验端点中 32 个集中在最上方一排。整条带的相位均值无法单独证明下方也有条纹。"
             "独立局部亮度核验有助于限制向缺少证据的清晰位置外推。", "", "## 固定同一拟合波形的节点消融", "",
             "下表复用同一频率、相位、幅值与端点复原。各格为整图真实减除投影比例／整图 MSE。"
             "改变包络的探针从端点复原直接评分，未叠加后续主峰修正。单点删除和仅保留实测节点可揭示误伤来源，"
             "不能直接推广为默认运行规则。干净参考与真实干扰仅进入诊断。", "",
             "| 幅值 | 上版输出 | 原包络 | 原图重拟格点包络 | 只去掉左节点 | 只去掉右节点 | 只保留实测节点 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name,r in results.items():
        fields = ("previous_peak83","guarded_envelope","clean_lattice_envelope","withhold_band_1_nodes",
                  "withhold_band_2_nodes","measured_nodes_only")
        values = [r["stages"][field]["regions"]["all"] for field in fields]
        text.append("| "+name.split("@")[-1]+" | "+" | ".join(f"{v['removal_gain']:.3f}／{v['mse']:.3g}" for v in values)+" |")
    text += ["", "[列误差曲线、标注坐标、分区能量与各消融结果](diagnosis.json)",
             "", "复现命令为 `python diagnose_collision83.py --out results/collision83_reproduction`。"]
    return "\n".join(text)+"\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"results/collision83")
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    data = diagnose()
    write_json(args.out/"diagnosis.json",data)
    (args.out/"diagnosis.md").write_text(report(data),encoding="utf8")
