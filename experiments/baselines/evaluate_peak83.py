"""Compare the three textile cases, including each reference peak at matched stripe removal."""
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
from evaluate_nine83 import (DIRECTIONS, generated_carriers, old_focus, samples as historical,
                             selection, snapshot as previous_snapshot)
from evaluate_project_refinement import identity
from evaluate_weak83 import checks_69022, measure
from evaluate_weave import read_json, summarize, write_json
from image_io_v3 import sha256_file
from project_comparison import ROOT, array_hash
from project_guarded import denoise as previous
from project_pipeline import load_config


ARCHIVE = ROOT/"results/nine83"
PLAN = ROOT/"results/peak83/protocol.json"
PHASES = "peak83_phases"
ARTIFACTS = ("freeze.json", "metrics.csv", "matched.csv", "per_peak.csv", "parity.csv", "RESULTS.md", "run.json")
PEAK_EPS = 1e-18


def snapshot():
    historical_freeze = read_json(ARCHIVE/"freeze.json")
    if historical_freeze != previous_snapshot():
        raise ValueError("Previous dependencies changed")
    files = dict(historical_freeze["source_sha256"])
    manifest = read_json(ARCHIVE/"run.json")
    for name, digest in manifest["artifact_sha256"].items():
        if sha256_file(ARCHIVE/name) != digest:
            raise ValueError(f"Previous artifact changed: {name}")
    dependencies = [Path(__file__).resolve(), ROOT/"project_peak.py"]
    while dependencies:
        path = dependencies.pop()
        name = path.relative_to(ROOT).as_posix()
        if name in files:
            continue
        files[name] = sha256_file(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            dependencies.extend(dep for m in modules if (dep := ROOT/(m.replace(".", "/")+".py")).is_file())
    for path in (PLAN, *(ARCHIVE/n for n in (*manifest["artifact_sha256"], "run.json", "per_peak.csv"))):
        files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    return dict(protocol=read_json(PLAN), configuration=historical_freeze["configuration"],
                source_sha256=dict(sorted(files.items())))


def freeze(folder):
    folder.mkdir(parents=True, exist_ok=True)
    if any((folder/n).exists() for n in ARTIFACTS):
        raise FileExistsError("Use a new experiment directory")
    write_json(folder/"freeze.json", snapshot())


def samples(plan, old_plan):
    original = None
    for case, amplitude in historical(old_plan, read_json(ARCHIVE/"endpoint_controls.json")):
        if case[:3] == ("archived_photo", plan["target"], "clean"):
            original = case[4]
        yield case, amplitude
    if original is None:
        raise ValueError("Missing woven_0083 original")
    directions = {v["name"]:v for v in old_plan["new_variants"]}
    for name, seeds in plan["preregistered_phase_seeds"].items():
        for seed in seeds:
            trial = directions[name] | dict(name=f"{name}_s{seed}", seed=seed)
            yield from generated_carriers(original, plan["target"], PHASES, [trial], plan["phase_cases"][name])


def squared_spectrum(array):
    return abs(np.fft.fft2(array, norm="ortho"))**2 / array.size


def compare_peaks(key, group, focus_peaks, clean, observed, old, new, input_rows, masks, frequencies, tolerance):
    """Compute raw and matched-removal comparisons with the actual post-clipping injection."""
    inp, old_row, new_row = (input_rows[m] for m in ("input","previous","current"))
    gain_old, gain_new = old_row["noise_gain"], new_row["noise_gain"]
    added = observed-clean
    removed_old, removed_new = observed-old, observed-new
    matched = None
    scale = ""
    gain_change = "" if gain_old == "" else gain_new-gain_old
    scores = dict(group=group, **key, delta_gain=gain_change,
                  comparable_gain="" if gain_change == "" else abs(gain_change) <= tolerance,
                  previous_gain=gain_old, current_gain=gain_new,
                  previous_mse=old_row["mse"], current_mse=new_row["mse"],
                  previous_off_noise_mse=old_row["off_noise_mse"], current_off_noise_mse=new_row["off_noise_mse"],
                  previous_texture_excess=old_row["texture_excess_mse"], current_texture_excess=new_row["texture_excess_mse"],
                  previous_texture_band_mse=old_row["texture_band_mse"], current_texture_band_mse=new_row["texture_band_mse"])
    total = {"input": squared_spectrum(added), "previous": squared_spectrum(old-clean),
             "current": squared_spectrum(new-clean)}
    excess = {}
    if gain_old != "":
        excess = {"previous": squared_spectrum(removed_old-gain_old*added),
                  "current": squared_spectrum(removed_new-gain_new*added)}
        if gain_old > 1e-8:
            scale = gain_new/gain_old
            # Analytical comparison on the old output's removal direction. Do not
            # clip again: that changes the gain and makes the comparison inexact.
            scaled = scale*removed_old
            gained = np.mean(scaled*added)/np.mean(added**2)
            if not np.isclose(gained, gain_new, atol=1e-12, rtol=1e-12):
                raise ValueError(f"Inexact matched gain for {key}")
            matched = scaled-gain_new*added
            total["matched_previous"] = squared_spectrum(added-scaled)
            excess["matched_previous"] = squared_spectrum(matched)
    rows = []
    for index, ((fy, fx), disk) in enumerate(zip(frequencies, masks), start=1):
        row = dict(group=group, **key, peak=index, fy_bin=fy, fx_bin=fx,
                   previously_harmed_peak=index in focus_peaks)
        for label in ("input", "previous", "current", "matched_previous"):
            row[label+"_mse"] = float(total[label][disk].sum()) if label in total else ""
        for label in ("previous", "current", "matched_previous"):
            row[label+"_excess_mse"] = float(excess[label][disk].sum()) if label in excess else ""
        rows.append(row)
    for label, record in (("input", inp),("previous", old_row),("current", new_row)):
        if not np.isclose(sum(r[label+"_mse"] for r in rows), record["texture_band_mse"], atol=1e-22, rtol=1e-12):
            raise ValueError(f"Peak sum mismatch for {key}")
    scores.update(matched_scale=scale, matched_previous_mse="" if matched is None else float(np.mean((added-scaled)**2)),
                  matched_previous_off_noise_mse="" if matched is None else float(np.mean(matched**2)),
                  matched_previous_texture_band_mse="" if matched is None else sum(r["matched_previous_mse"] for r in rows),
                  matched_previous_texture_excess="" if matched is None else sum(r["matched_previous_excess_mse"] for r in rows))
    return scores, rows


def group_for(ident, plan, nine):
    if ident[0] == DIRECTIONS and ident[3] in plan["known_damage"]:
        return "three"
    if ident in nine:
        return "nine"
    if ident[0] == PHASES:
        return "new_phases"
    return "other"


def evaluate(folder):
    from project_peak import denoise as current

    frozen = read_json(folder/"freeze.json")
    if frozen != snapshot() or any((folder/n).exists() for n in ARTIFACTS[1:]):
        raise ValueError("Use a matching frozen directory with no existing results")
    plan, old_plan = frozen["protocol"], read_json(ARCHIVE/"protocol.json")
    nine, cfg = old_focus(old_plan), load_config()
    old_run = read_json(ARCHIVE/"run.json")
    with (ARCHIVE/"metrics.csv").open(encoding="utf8", newline="") as stream:
        archive = selection(csv.DictReader(stream))
    table, spectra, matched, peaks, parity, pairs = [], {}, [], [], [], {}
    checked, hashes = set(), set()
    for case, amp in samples(plan, old_plan):
        cohort, name, kind, variant, clean, observed, masks = case
        key = dict(cohort=cohort, image_id=name, kind=kind, variant=str(variant))
        ident = identity(key)
        if ident in pairs:
            raise ValueError(f"Duplicate identity {ident}")
        ref, inp = array_hash(clean), array_hash(observed)
        if cohort == PHASES and inp in hashes:
            raise ValueError(f"New observation duplicated an existing input: {ident}")
        hashes.add(inp)
        old = previous(observed, cfg, return_diagnostics=True)
        if array_hash(observed) != inp:
            raise ValueError(f"Old method mutated input {ident}")
        if cohort != PHASES:
            existing = archive[ident]
            replay = ref, inp, array_hash(old[0]), old[1]["route"], old[1]["reason"]
            if replay != tuple(existing[k] for k in ("reference_sha256","observed_sha256","output_sha256","route","reason")):
                raise ValueError(f"Failed to reproduce previous measurements for {ident}")
            checked.add(ident)
        new = current(observed, cfg, return_diagnostics=True)
        if array_hash(observed) != inp:
            raise ValueError(f"Current method mutated input {ident}")
        group = group_for(ident, plan, nine)
        if group == "three" and not np.array_equal(current(observed, cfg), new[0]):
            raise ValueError(f"Normal and diagnostic entries differ for {ident}")
        parity.extend(checks_69022(ident, clean, observed, masks, cfg, previous, current, old, new))
        mask = None
        if name == plan["target"]:
            if ref not in spectra:
                freq, disks = peak_masks(clean, old_run["reference_texture_peaks_in_fft_bins"][ref])
                spectra[ref] = freq, disks, np.logical_or.reduce(disks)
            freq, disks, mask = spectra[ref]
        pairs[ident] = rows = {}
        for label, (out, diag) in (("input",(observed,{})),("previous",old),("current",new)):
            if diag:
                check_output(observed,out,diag)
            row = dict(group=group, **key, method=label, exact_amplitude=amp, route=diag.get("route",""),
                       reason=diag.get("reason",""), reference_sha256=ref, observed_sha256=inp, output_sha256=array_hash(out))
            row.update(measure(clean, observed, out, masks, mask))
            rows[label] = row
            table.append(row)
        if mask is not None:
            harmed = plan["known_damage"].get(str(variant), []) if group == "three" else []
            match, comparisons = compare_peaks(key, group, harmed, clean, observed, old[0], new[0], rows,
                                                disks, freq, plan["near_gain_absolute_tolerance"])
            matched.append(match)
            peaks.extend(comparisons)
        print(len(pairs), group, name, kind, variant, flush=True)
    sizes = {g:sum(rows["current"]["group"] == g for rows in pairs.values()) for g in ("three","nine","new_phases")}
    if (checked != archive.keys() or len(checked) != plan["historical_inputs"] or len(parity) != 10
            or sizes != dict(three=3, nine=9, new_phases=6) or snapshot() != frozen):
        raise ValueError("Incomplete or inconsistent frozen replay")
    groups = {g:summarize([rows for rows in pairs.values() if rows["current"]["group"] == g], 1e-12) for g in sizes}
    groups["historical"] = summarize([p for key,p in pairs.items() if key in checked], 1e-12)
    write_csv(folder/"metrics.csv", table)
    write_csv(folder/"matched.csv", matched)
    write_csv(folder/"per_peak.csv", peaks)
    write_csv(folder/"parity.csv", parity)
    write_report(folder, plan, groups, pairs, matched, peaks, parity)
    write_json(folder/"run.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(), inputs=len(pairs),
               historical_baselines_checked=len(checked), rows=len(table), peak_rows=len(peaks), groups=groups,
               environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
               artifact_sha256={n:sha256_file(folder/n) for n in ARTIFACTS if n != "run.json"}))


def directions(rows, baseline, current="current_mse"):
    available = [r for r in rows if r[baseline] != "" and r[current] != ""]
    difference = np.asarray([r[current]-r[baseline] for r in available])
    return f"{int((difference < -PEAK_EPS).sum())} 降、{int((abs(difference) <= PEAK_EPS).sum())} 平、{int((difference > PEAK_EPS).sum())} 升"


def fmt(value):
    return "—" if value == "" else f"{value:.5g}"


def write_report(folder, plan, stats, pairs, matched, peak_rows, parity):
    lines = ["# woven_0083 的织纹主峰与等去条纹比例验证", "",
             "在同一张 woven_0083 上比较原图和加条纹后的恢复图。3 个目标均参与方案设计，另固定两组相位，"
             "分别重建两档横向扫频和一档纵向扫频，共 6 例。用干净变体、原先 9 个已改善条件和 69022 边缘窄条做回归。"
             "全部 148 个历史输入都重新运行上版，核对参考、观测及恢复图指纹，以及路径、原因。", "",
             "## 如何比较相近的去条纹效果", "",
             "以裁剪后的实际新增量 n = 含噪输入 − 原图、实际减除量 r = 含噪输入 − 恢复图计算投影 a = ⟨r,n⟩/⟨n,n⟩。"
             "预先规定两版的 a 相差至多 0.02 才记作强度相近。额外误差是 r − a·n 的均方能量。"
             "每组主峰各取半径 2 个 FFT 格点的峰和共轭圆盘，表中单位均为按整幅像素数归一化的 [0,1] 亮度平方。"
             "每幅参考照片的六个不重叠频带沿用上轮定义。纹带总 MSE 含未去净条纹与减除误差。", "",
             "另将旧版实际减除量乘以 a新/a旧，使其准确达到新版的投影。"
             "这个对照仅在评分时使用参考，计算时保留线性缩放后的亮度值。旧版 a≤10⁻⁸ 时留空。", "",
             "## 三个目标及其原先受损的主峰", ""]
    focus = [r for r in peak_rows if r["group"] == "three" and r["previously_harmed_peak"]]
    lines += ["| 条件 | 主峰（纵、横格） | 含噪输入 | 上版 | 等比例上版 | 本版 | 等比例上版／本版额外误差 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for r in focus:
        lines.append(f"| {r['variant']} | {r['peak']}（{r['fy_bin']}, {r['fx_bin']}） | {fmt(r['input_mse'])} | "
                     f"{fmt(r['previous_mse'])} | {fmt(r['matched_previous_mse'])} | {fmt(r['current_mse'])} | "
                     f"{fmt(r['matched_previous_excess_mse'])}／{fmt(r['current_excess_mse'])} |")
    comparable = sum(r["comparable_gain"] for r in matched if r["group"] == "three")
    others = [r for r in peak_rows if r["group"] == "three" and not r["previously_harmed_peak"]]
    lines += ["",f"{comparable}/3 例符合强度相近的预定标准。四个受损主峰与上版比较为 {directions(focus,'previous_mse')}，"
              f"与等比例上版比较为 {directions(focus,'matched_previous_mse')}，与含噪输入比较为 {directions(focus,'input_mse')}。"
              f"其余 14 项与等比例上版比较为 {directions(others,'matched_previous_mse')}。"
              "主峰方向计数的绝对持平公差为 10⁻¹⁸。", ""]
    for group, title in (("three","三个设计条件"), ("nine","原先九个已改善条件"), ("new_phases","六个预登记的新相位条件")):
        entries = [r for r in matched if r["group"] == group]
        rows = [r for r in peak_rows if r["group"] == group]
        lines += [f"### {title}","",
                  "| 方向、幅值和相位标识 | a 上版 → 本版 | Δa | 整图 MSE 上版 → 本版 | 纹带总 MSE 上版／等比例上版／本版 | 等比例上版／本版纹带额外误差 |",
                  "|---|---:|---:|---:|---:|---:|"]
        for r in entries:
            lines.append(f"| {r['variant']} | {fmt(r['previous_gain'])} → {fmt(r['current_gain'])} | {fmt(r['delta_gain'])} | "
                         f"{fmt(r['previous_mse'])} → {fmt(r['current_mse'])} | {fmt(r['previous_texture_band_mse'])}／"
                         f"{fmt(r['matched_previous_texture_band_mse'])}／{fmt(r['current_texture_band_mse'])} | "
                         f"{fmt(r['matched_previous_texture_excess'])}／{fmt(r['current_texture_excess'])} |")
        lines += ["",f"{len(rows)} 项单峰比较中，本版相对上版为 {directions(rows,'previous_mse')}，"
                  f"相对等比例上版为 {directions(rows,'matched_previous_mse')}，"
                  f"相对含噪输入为 {directions(rows,'input_mse')}。采用等比例上版的项目数为 "
                  f"{sum(r['matched_previous_mse'] != '' for r in rows)}。",""]
    lines += ["## 同图无条纹及全集回归", "",
              "| 输入分组 | 输入数 | 整图 MSE 优于／持平／劣于上版 | 输出逐像素相同 | 保留上版净收益 | 优于含噪输入 |",
              "|---|---:|---:|---:|---:|---:|"]
    for g, label in (("three","三个目标"),("nine","原九例"),("historical","全部历史输入"),("new_phases","新增相位")):
        s = stats[g]
        lines.append(f"| {label} | {s['inputs']} | {s['better_than_previous']}／{s['tied_with_previous']}／"
                     f"{s['worse_than_previous']} | {s['exact_same_as_previous']} | {s['retained_net_improvements']}/"
                     f"{s['previous_net_improvements']} | {s['current_net_improvements']} |")
    clean = [p for key,p in pairs.items() if key[2] == "clean"]
    original = pairs["archived_photo",plan["target"],"clean",""]
    lines += ["",f"同一张 woven_0083 原图改动像素数为 {original['previous']['changed_pixels']} → {original['current']['changed_pixels']}。"
              f"全部 {len(clean)} 个无新增条纹对照中，被修改的输入数由 {sum(p['previous']['changed_pixels']>0 for p in clean)} "
              f"变为 {sum(p['current']['changed_pixels']>0 for p in clean)}。包含同图旋转、形变、亮度及带曝光截幅的对照。"
              "两个设计子集均计入历史输入总数。整图 MSE 持平公差为 10⁻¹²。", ""]
    worse = [p for p in pairs.values() if p["current"]["mse"] > p["previous"]["mse"] + 1e-12]
    if worse:
        lines += ["整图 MSE 高于上版的全部条件如下。", "",
                  "| 照片与条件 | 上版 | 本版 |", "|---|---:|---:|"]
        for p in worse:
            r = p["current"]
            lines.append(f"| {r['image_id']} / {r['cohort']} / {r['variant']} | {fmt(p['previous']['mse'])} | {fmt(r['mse'])} |")
        lines.append("")
    lines += ["## 69022 的窄条及相邻 16 像素干净区", "",
              "| 边缘／宽度 | 污染区 MSE 输入／上版／本版 | 邻区 MSE 上版／本版 | 全部干净区 MSE 上版／本版 |",
              "|---|---:|---:|---:|"]
    for key,p in pairs.items():
        if key[1] != "69022" or key[2] not in ("edge8","edge16"):
            continue
        inp,old,new = (p[m] for m in ("input","previous","current"))
        side = {"0":"左","2":"上","3":"下"}[key[3]]
        lines.append(f"| {side}／{key[2][4:]} | {fmt(inp['contaminated_mse'])}／{fmt(old['contaminated_mse'])}／"
                     f"{fmt(new['contaminated_mse'])} | {fmt(old['adjacent_clean_mse'])}／{fmt(new['adjacent_clean_mse'])} | "
                     f"{fmt(old['clean_mse'])}／{fmt(new['clean_mse'])} |")
    same = sum(all(r[k] for k in ("route_equal","regions_equal","support_equal","output_equal","residual_equal")) for r in parity)
    lines += ["",f"默认强度的六种位置与宽度组合，加上底边另两档强度检查，共 {same}/{len(parity)} 组路径、"
              "接受区间、掩膜、恢复图和实际减除量均与上版相同。邻区只计入实际干净像素。", "",
              "## 复现", "", "```powershell", "python evaluate_peak83.py freeze --out results/peak83_reproduction",
              "python evaluate_peak83.py run --out results/peak83_reproduction", "```","",
              "[协议和冻结源码指纹](freeze.json) · [逐图指标](metrics.csv) · [等去条纹比例](matched.csv) · "
              "[逐主峰指标](per_peak.csv) · [69022 核对](parity.csv) · [运行和哈希](run.json)"]
    (folder/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze","run"))
    parser.add_argument("--out", type=Path, default=ROOT/"results/peak83_reproduction")
    args = parser.parse_args()
    (freeze if args.stage == "freeze" else evaluate)(args.out)
