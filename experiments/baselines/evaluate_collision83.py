"""Reproduce the prior study and score residual bands, real stripe removal and weave fidelity."""
import argparse
import ast
import csv
from datetime import datetime, timezone
from pathlib import Path
import platform

import numpy as np
import scipy

from check_nine83_peaks import peak_masks
from evaluate_narrow83 import check_output, write_csv
from evaluate_nine83 import DIRECTIONS, generated_carriers, old_focus, selection
from evaluate_peak83 import PHASES as PRIOR_PHASES, samples as previous_samples, snapshot as previous_snapshot, squared_spectrum
from evaluate_project_refinement import identity
from evaluate_weak83 import checks_69022, measure
from evaluate_weave import projected_residual, read_json, summarize, write_json
from image_io_v3 import sha256_file
from peak_guard import _carrier_geometry
from project_comparison import ROOT, array_hash
from project_peak import denoise as previous
from project_pipeline import load_config


ARCHIVE = ROOT / "results/peak83"
NINE = ROOT / "results/nine83"
PLAN = ROOT / "results/collision83/protocol.json"
PHASES = "collision83_phases"
ARTIFACTS = ("freeze.json", "metrics.csv", "regions.csv", "matched.csv", "per_peak.csv", "parity.csv", "RESULTS.md", "run.json")
EPS = 1e-18


def snapshot():
    old = read_json(ARCHIVE/"freeze.json")
    if old != previous_snapshot():
        raise ValueError("The previous experiment's dependencies changed")
    files = dict(old["source_sha256"])
    run = read_json(ARCHIVE/"run.json")
    for name, digest in run["artifact_sha256"].items():
        if sha256_file(ARCHIVE/name) != digest:
            raise ValueError(f"Historical artifact changed: {name}")
        files[(ARCHIVE/name).relative_to(ROOT).as_posix()] = digest
    queue = [Path(__file__).resolve(), ROOT/"project_collision.py"]
    while queue:
        path = queue.pop()
        name = path.relative_to(ROOT).as_posix()
        if name in files:
            continue
        files[name] = sha256_file(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            queue.extend(dep for m in names if (dep := ROOT/(m.replace(".", "/")+".py")).is_file())
    for path in (PLAN, ARCHIVE/"run.json"):
        files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return dict(protocol=read_json(PLAN), configuration=old["configuration"], source_sha256=dict(sorted(files.items())))


def freeze(folder):
    folder.mkdir(parents=True, exist_ok=True)
    if any((folder/n).exists() for n in ARTIFACTS):
        raise FileExistsError("Choose a fresh experiment directory")
    write_json(folder/"freeze.json", snapshot())


def samples(plan):
    old_plan = read_json(NINE/"protocol.json")
    original = None
    for case, amplitude in previous_samples(read_json(ARCHIVE/"protocol.json"), old_plan):
        if case[:3] == ("archived_photo", plan["target"], "clean"):
            original = case[4]
        yield case, amplitude
    if original is None:
        raise ValueError("Missing original photograph")
    axis_x = next(v for v in old_plan["new_variants"] if v["name"] == "axis_x_E")
    for seed in plan["new_phase_seeds"]:
        direction = axis_x | dict(name=f"axis_x_E_s{seed}", seed=seed)
        yield from generated_carriers(original, plan["target"], PHASES, [direction], plan["amplitudes"])


def group_for(ident, plan, nine):
    if ident[0] == DIRECTIONS and ident[3] in plan["target_variants"]:
        return "target"
    if ident in nine:
        return "nine"
    if ident[0] == PRIOR_PHASES:
        return "prior_phases"
    return "new_phases" if ident[0] == PHASES else "other"


def old_strips(diag):
    if "carrier_wave" not in diag:
        return []
    record = diag["evidence"]["guarded_carrier"]
    applied = diag["carrier_wave"]
    full, _ = _carrier_geometry(applied.shape, record)
    weight = np.divide(applied, full, out=np.full_like(full, np.nan),
                       where=abs(full) > 1e-8*record["amplitude"])
    cut = np.nanmedian(weight, axis=0) <= .5
    changes = np.flatnonzero(np.diff(np.r_[False, cut, False])).reshape(-1, 2)
    return [(int(a), int(b-1)) for a,b in changes if b-a >= 8]


def region_scores(key, clean, observed, old, new, fixed):
    adaptive = old_strips(old[1])
    rows = []
    for source, ranges in (("fixed",fixed), ("old_taper",adaptive)):
        for index,(left,right) in enumerate(ranges, start=1):
            window = np.s_[:, left:right+1]
            added = (observed-clean)[window]
            row = dict(**key, mask_source=source, strip=index, first_column=left, last_column=right,
                       pixels=added.size, old_strip_count=len(adaptive))
            for label, output in (("input",observed), ("previous",old[0]), ("current",new[0])):
                removed = (observed-output)[window]
                row[label+"_mse"] = float(np.mean((added-removed)**2))
                row[label+"_gain"] = projected_residual(removed, added)[0]
            rows.append(row)
    return rows


def spectral_scores(key, clean, observed, baseline, candidate, records, frequencies, disks):
    injection = observed-clean
    energy = {"input_mse": squared_spectrum(injection)}
    gains = [records[m]["noise_gain"] for m in ("previous", "current")]
    common = min(gains) if all(g != "" and g > 1e-8 for g in gains) else ""
    for label, output, gain in zip(("previous","current"),(baseline,candidate),gains):
        removed = observed-output
        energy[label+"_mse"] = squared_spectrum(output-clean)
        if gain != "":
            energy[label+"_excess"] = squared_spectrum(removed-gain*injection)
        if common != "":
            # Scaling down is a convex combination of the observed and restored images.
            scaled = (common/gain)*removed
            measured = projected_residual(scaled, injection)[0]
            if not np.isclose(measured, common, rtol=1e-12, atol=1e-12):
                raise ValueError(f"Failed matched-gain check: {key}/{label}")
            energy["matched_"+label+"_mse"] = squared_spectrum(injection-scaled)
            energy["matched_"+label+"_excess"] = squared_spectrum(scaled-common*injection)
    columns = ("input_mse","previous_mse","current_mse","previous_excess","current_excess",
               "matched_previous_mse","matched_current_mse","matched_previous_excess","matched_current_excess")
    rows = []
    for index, (frequency, disk) in enumerate(zip(frequencies, disks),start=1):
        row = dict(**key, peak=index, fy_bin=frequency[0], fx_bin=frequency[1], common_gain=common)
        row.update({k:float(energy[k][disk].sum()) if k in energy else "" for k in columns})
        rows.append(row)
    for m in ("input","previous","current"):
        if not np.isclose(sum(r[m+"_mse"] for r in rows), records[m]["texture_band_mse"], rtol=1e-12, atol=1e-22):
            raise ValueError(f"Peak sum mismatch for {key}/{m}")
    return rows


def evaluate(folder):
    from project_collision import denoise as current
    sealed = read_json(folder/"freeze.json")
    if sealed != snapshot() or any((folder/n).exists() for n in ARTIFACTS[1:]):
        raise ValueError("Choose a matching frozen directory with no completed outputs")
    plan = sealed["protocol"]
    nine = old_focus(read_json(NINE/"protocol.json"))
    known_peaks = read_json(NINE/"run.json")["reference_texture_peaks_in_fft_bins"]
    cfg = load_config()
    with (ARCHIVE/"metrics.csv").open(encoding="utf8", newline="") as stream:
        archive = selection(csv.DictReader(stream))
    pairs, masks_by_ref = {}, {}
    checked, input_hashes = set(), set()
    table, regions, matched, peaks, parity = [], [], [], [], []
    for case, amplitude in samples(plan):
        group, name, kind, variant, clean, observed, masks = case
        ident = (group, name, kind, str(variant))
        key = dict(group=group_for(ident, plan, nine), cohort=group, image_id=name, kind=kind, variant=str(variant))
        if ident in pairs:
            raise ValueError(f"Duplicate identity {ident}")
        ref, inp = array_hash(clean), array_hash(observed)
        if group == PHASES and inp in input_hashes:
            raise ValueError(f"New condition duplicates a measured observation at {ident}")
        input_hashes.add(inp)
        old = previous(observed, cfg, return_diagnostics=True)
        if array_hash(observed) != inp:
            raise ValueError("Previous method mutated input")
        if group != PHASES:
            replay = (ref, inp, array_hash(old[0]), old[1]["route"], old[1]["reason"])
            if replay != tuple(archive[ident][k] for k in ("reference_sha256","observed_sha256","output_sha256","route","reason")):
                raise ValueError(f"Old output diverged at {ident}")
            checked.add(ident)
        new = current(observed, cfg, return_diagnostics=True)
        if key["group"] == "target" and not np.array_equal(current(observed,cfg),new[0]):
            raise ValueError(f"Regular and diagnostic entry disagree for {ident}")
        if array_hash(observed) != inp:
            raise ValueError("Current method mutated input")
        parity.extend(checks_69022(ident,clean,observed,masks,cfg,previous,current,old,new))
        band = None
        if name == plan["target"]:
            if ref not in masks_by_ref:
                frequencies, disks = peak_masks(clean,known_peaks[ref])
                masks_by_ref[ref] = frequencies, disks, np.logical_or.reduce(disks)
            frequencies, disks, band = masks_by_ref[ref]
        pairs[ident] = records = {}
        for label,(out,diag) in (("input",(observed,{})),("previous",old),("current",new)):
            if diag:
                check_output(observed,out,diag)
            row = dict(**key, method=label, exact_amplitude=amplitude, route=diag.get("route",""),
                       reason=diag.get("reason",""), reference_sha256=ref, observed_sha256=inp, output_sha256=array_hash(out))
            row.update(measure(clean,observed,out,masks,band))
            table.append(row)
            records[label] = row
        if band is not None:
            case_peaks = spectral_scores(key,clean,observed,old[0],new[0],records,frequencies,disks)
            peaks.extend(case_peaks)
            summary = dict(**key, common_gain=case_peaks[0]["common_gain"],
                           previous_gain=records["previous"]["noise_gain"], current_gain=records["current"]["noise_gain"])
            for label in ("previous", "current"):
                for metric in ("mse", "excess"):
                    field = f"matched_{label}_{metric}"
                    summary[field] = sum(r[field] for r in case_peaks) if summary["common_gain"] != "" else ""
            matched.append(summary)
        if key["group"] in ("target","prior_phases","new_phases") and str(variant).startswith("axis_x_E"):
            regions.extend(region_scores(key,clean,observed,old,new,plan["fixed_column_ranges_inclusive"]))
        print(len(pairs), *ident, flush=True)
    sizes = {g:sum(p["current"]["group"] == g for p in pairs.values()) for g in ("target","nine","prior_phases","new_phases")}
    if (checked != archive.keys() or len(checked) != plan["historical_inputs"] or len(parity) != 10 or
            sizes != dict(target=2,nine=9,prior_phases=6,new_phases=6) or snapshot() != sealed):
        raise ValueError("Incomplete run or changed frozen sources")
    stats = {g:summarize([p for p in pairs.values() if p["current"]["group"] == g],1e-12) for g in sizes}
    stats["historical"] = summarize([pairs[k] for k in checked],1e-12)
    for name, data in (("metrics",table),("matched",matched),("per_peak",peaks),("regions",regions),("parity",parity)):
        write_csv(folder/(name+".csv"),data)
    write_report(folder, plan, pairs, regions, peaks, matched, parity, stats)
    write_json(folder/"run.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(), inputs=len(pairs),
               historical_baselines_checked=len(checked), rows=len(table), peak_rows=len(peaks), region_rows=len(regions),
               environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
               groups=stats, artifact_sha256={n:sha256_file(folder/n) for n in ARTIFACTS if n != "run.json"}))


def change_counts(rows, before, after):
    differences = [r[after]-r[before] for r in rows if r[before] != "" and r[after] != ""]
    return f"{sum(x < -EPS for x in differences)} 降、{sum(abs(x) <= EPS for x in differences)} 平、{sum(x > EPS for x in differences)} 升"


def fmt(x):
    return "—" if x == "" else f"{x:.5g}"


def write_report(folder, plan, pairs, regions, peaks, matched, parity, stats):
    lines = ["# woven_0083 两处条纹残留的配对实验","",
             "每个观测图交给两个完整入口。用裁剪后的含噪图减清晰原图得到真实新增量 n，"
             "用含噪图减恢复图得到实际减除量 r。投影 a = ⟨r,n⟩/⟨n,n⟩ 衡量沿条纹方向的去除比例。"
             "参考图、真实条纹或筛选用的残留区均不传给算法。",
             "","## 两个幅值与预登记的新相位","",
             "两档目标是设计条件。此前六例相位包含四个横向、两个纵向渐变。另为横向渐变预登记三个相位，每个相位分别用两个幅值，"
             "在冻结后生成六个新观测。相同相位两档共用同一底图和频率场。","",
             "| 分组／条件 | 去除比例 上版 → 本版 | 整图 MSE 输入／上版／本版 | 六峰合计 MSE 上版 → 本版 | 等比例六峰额外误差 上版 → 本版 |",
             "|---|---:|---:|---:|---:|"]
    focus = [p for p in pairs.values() if p["current"]["group"] in ("target","prior_phases","new_phases")]
    matched_by_id = {identity(row): row for row in matched}
    for p in focus:
        i,o,n = (p[k] for k in ("input","previous","current"))
        match = matched_by_id[identity(n)]
        lines.append(f"| {n['group']}／{n['variant']} | {o['noise_gain']:.5f} → {n['noise_gain']:.5f} | "
                     f"{i['mse']:.5g}／{o['mse']:.5g}／{n['mse']:.5g} | "
                     f"{o['texture_band_mse']:.5g} → {n['texture_band_mse']:.5g} | "
                     f"{fmt(match['matched_previous_excess'])} → {fmt(match['matched_current_excess'])} |")
    lines += ["","## 残留区与可见效果","",
              "按上版结果，两档原始输入的公共残留列为 38–111 和 199–255（从零计，含端点，计全部行）。"
              "以下汇报固定列的结果。另按每个输入独立重建上版空间权重，保留列中位数不超过 0.5、连续至少八列的区间；"
              "其边界与每种方法的区域 MSE 和去除比例见 [regions.csv](regions.csv)。","",
              "| 条件 | 旧权重区间数 | 左区去除比例 上版 → 本版 | 右区去除比例 上版 → 本版 | 左区 MSE 上版 → 本版 | 右区 MSE 上版 → 本版 | 达到预定效果量 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    qualified = {}
    for p in focus:
        n,o = p["current"],p["previous"]
        pair = [r for r in regions if r["mask_source"]=="fixed" and r["variant"]==n["variant"] and r["cohort"]==n["cohort"]]
        if not pair:
            continue
        left,right = pair
        yes = (n["noise_gain"]-o["noise_gain"] >= plan["minimum_visible_gain_change"] and
               all(r["current_mse"] <= (1-plan["minimum_visible_regional_mse_reduction"])*r["previous_mse"] for r in pair))
        qualified.setdefault(n["group"],[]).append(yes)
        cells = [f'{r["previous_"+k]:.5g} → {r["current_"+k]:.5g}' for k in ("gain","mse") for r in pair]
        lines.append(f'| {n["variant"]} | {left["old_strip_count"]} | '+" | ".join(cells)+" | "+("是" if yes else "否")+" |")
    lines += ["","预定效果量同时要求整图真实去除比例提高至少 0.10、两处固定残留区各自 MSE 降低至少 25%。"
              +"；".join(f"{name} 达标 {sum(flags)}/{len(flags)}" for name,flags in qualified.items())+"。"]
    lines += ["","## 第 5、6 组织纹峰","",
              "两组峰中心为 (0,15) 与 (10,−13) 整图 FFT 格。各组含共轭中心的两个半径 2 格圆盘，按全图像素数归一化。"
              "原始峰 MSE 与去除强度均保留原值。为单独比较误减，在两个正投影中取较小的一个作为共同强度，"
              "把两个版本的减除量分别缩小至该强度，统计与理想同等去除量的频带误差。"
              "若某一方没有正投影，等强度栏留空。","",
              "| 条件／峰号 | 峰 MSE 含噪／上版／本版 | 同去除比例额外误差 上版 → 本版 |",
              "|---|---:|---:|"]
    for r in peaks:
        if r["peak"] in plan["peak_groups"] and r["group"] in ("target","prior_phases","new_phases"):
            lines.append(f"| {r['variant']}／{r['peak']} | {fmt(r['input_mse'])}／{fmt(r['previous_mse'])}／{fmt(r['current_mse'])} | "
                         f"{fmt(r['matched_previous_excess'])} → {fmt(r['matched_current_excess'])} |")
    for group in ("target","prior_phases","new_phases","nine"):
        subset = [r for r in peaks if r["group"] == group and r["peak"] in plan["peak_groups"]]
        lines += ["",f"{group} 的第 5、6 组原始峰 MSE 相对上版 {change_counts(subset,'previous_mse','current_mse')}，"
                  f"相对含噪输入 {change_counts(subset,'input_mse','current_mse')}。"
                  f"共同去除比例下的额外误差相对上版 {change_counts(subset,'matched_previous_excess','matched_current_excess')}。"]
    lines += ["","## 历史回归与干净对照","",
              "| 集合 | 输入数 | 整图 MSE 优于／持平／劣于上版 | 输出逐值相同 | 保留上版已有净收益 |",
              "|---|---:|---:|---:|---:|"]
    for group in ("historical","target","nine","prior_phases","new_phases"):
        s = stats[group]
        lines.append(f"| {group} | {s['inputs']} | {s['better_than_previous']}／{s['tied_with_previous']}／{s['worse_than_previous']} | "
                     f"{s['exact_same_as_previous']} | {s['retained_net_improvements']}/{s['previous_net_improvements']} |")
    clean = [p for k,p in pairs.items() if k[2]=="clean"]
    nine_peaks = [r for r in peaks if r["group"]=="nine"]
    lines += ["",f"原先九例的 54 个原始峰误差相对上版 {change_counts(nine_peaks,'previous_mse','current_mse')}，"
              f"相对含噪输入 {change_counts(nine_peaks,'input_mse','current_mse')}。"
              f"{len(clean)} 个无新增条纹输入中，有像素变化的条件数由 "
              f"{sum(p['previous']['changed_pixels']>0 for p in clean)} 变成 {sum(p['current']['changed_pixels']>0 for p in clean)}。"
              "历史集合包含目标、原九例和此前相位。全部新相位均来自同一张照片。"]
    worse = [p for p in pairs.values() if p["current"]["mse"] > p["previous"]["mse"]+1e-12]
    for p in worse:
        o,n = p["previous"],p["current"]
        lines.append(f"- 整图误差上升：{n['image_id']}／{n['kind']}／{n['variant']}，{o['mse']:.10g} → {n['mse']:.10g}。")
    lines += ["","## 69022 边缘窄条","",
              "| 边缘／宽度 | 污染区 MSE 输入 → 上版 → 本版 | 相邻干净区 MSE 上版／本版 | 全部干净区 MSE 上版／本版 |",
              "|---|---:|---:|---:|"]
    for k,p in pairs.items():
        if k[1] == "69022" and k[2] in ("edge8","edge16"):
            i,o,n = (p[m] for m in ("input","previous","current"))
            side = {"0":"左","2":"上","3":"下"}[k[3]]
            lines.append(f"| {side}／{k[2][4:]} px | {fmt(i['contaminated_mse'])} → {fmt(o['contaminated_mse'])} → "
                         f"{fmt(n['contaminated_mse'])} | {fmt(o['adjacent_clean_mse'])}／{fmt(n['adjacent_clean_mse'])} | "
                         f"{fmt(o['clean_mse'])}／{fmt(n['clean_mse'])} |")
    equal = sum(all(r[k] for k in ("route_equal","regions_equal","support_equal","output_equal","residual_equal")) for r in parity)
    lines += ["",f"合计 {equal}/{len(parity)} 项恢复、残差、掩膜、区域与路径逐值一致。相邻干净区为距污染边界不超过 16 像素的真实干净像素。",
              "","## 复现","","```powershell","python evaluate_collision83.py freeze --out results/collision83_reproduction",
              "python evaluate_collision83.py run --out results/collision83_reproduction","```","",
              "[逐图指标](metrics.csv) · [逐峰原始与等比例指标](per_peak.csv) · [等比例六峰合计](matched.csv) · [残留区域](regions.csv) · "
              "[窄条逐值检查](parity.csv) · [运行指纹](run.json)"]
    (folder/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("freeze","run"))
    parser.add_argument("--out",type=Path,default=ROOT/"results/collision83_reproduction")
    args = parser.parse_args()
    (freeze if args.stage == "freeze" else evaluate)(args.out)
