"""Freeze the candidate method and compare known and new carriers on woven_0083."""
import argparse
import ast
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.ndimage import map_coordinates

from evaluate_narrow83 import check_output, write_csv
from evaluate_project_refinement import identity
from evaluate_weak83 import (band_energy, checks_69022, measure, samples as historical_samples,
                             texture_band)
from evaluate_weave import new_cases, read_json, summarize, write_json
from image_io_v3 import sha256_file
from project_comparison import ROOT, array_hash
from project_pipeline import load_config
from project_weak import _subtract as midpoint_subtraction


ARCHIVE = ROOT/"results/weak83"
PLAN = ROOT/"results/nine83/protocol.json"
ENDPOINT_PLAN = PLAN.with_name("endpoint_controls.json")
OUTPUTS = ("freeze.json", "metrics.csv", "parity.csv", "RESULTS.md", "run.json")
DIRECTIONS = "nine83_directions"
TEXTURES = "nine83_clean_controls"
CLIPPED = "nine83_clipped_clean"
INTERIOR = "nine83_no_endpoints"


def snapshot():
    old = read_json(ARCHIVE/"freeze.json")
    run = read_json(ARCHIVE/"run.json")
    for name, digest in run["artifact_sha256"].items():
        if sha256_file(ARCHIVE/name) != digest:
            raise ValueError(f"Historical artifact changed: {name}")
    hashes = dict(old["source_sha256"])
    for name, digest in hashes.items():
        if sha256_file(ROOT/name) != digest:
            raise ValueError(f"Historical dependency changed: {name}")
    queue = [Path(__file__).resolve(), ROOT/"project_guarded.py"]
    while queue:
        path = queue.pop()
        name = path.relative_to(ROOT).as_posix()
        if name in hashes:
            continue
        hashes[name] = sha256_file(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            queue.extend(dep for m in modules if (dep := ROOT/(m.replace(".", "/")+".py")).is_file())
    for path in (PLAN, ENDPOINT_PLAN, *(ARCHIVE/n for n in (*run["artifact_sha256"], "run.json"))):
        hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    config = json.loads(json.dumps(asdict(load_config()), allow_nan=False))
    return dict(protocol=read_json(PLAN), endpoint_controls=read_json(ENDPOINT_PLAN),
                configuration=config, source_sha256=dict(sorted(hashes.items())))


def freeze(folder):
    folder.mkdir(parents=True, exist_ok=True)
    if any((folder/n).exists() for n in OUTPUTS):
        raise FileExistsError("Choose a fresh experiment directory")
    write_json(folder/"freeze.json", snapshot())


def old_samples():
    return historical_samples(read_json(ARCHIVE/"protocol.json"), read_json(ARCHIVE/"single_axis_protocol.json"))


def old_focus(plan):
    return {(group, plan["target"], kind, variant):amp for group, kind, variant, amp in plan["focus_nine"]}


def generated_carriers(clean, target, group, variants, amplitudes):
    # Use one direction and amplitude per call so the exact amplitude is
    # carried alongside the array, independent of the legacy display format.
    for v in variants:
        for amp in amplitudes:
            gen = new_cases({target:clean}, dict(target_ids=[target], new_variants=[v], amplitudes=[amp]))
            (_, name, kind, _, ref, observed, masks), = gen
            yield (group, name, kind, f'{v["name"]}@{amp:.3f}', ref, observed, masks), amp


def clean_case(group, target, variant, ref):
    mask = np.ones(ref.shape, bool)
    return (group, target, "clean", variant, ref, ref.copy(), dict(clean=mask, contaminated=~mask)), 0.


def new_samples(clean, plan, extra):
    target, variants = plan["target"], plan["new_variants"]
    yield from generated_carriers(clean, target, DIRECTIONS, variants, plan["amplitudes"])
    y, x = np.indices(clean.shape, dtype=float)
    shear = [y+1.6*np.sin(2*np.pi*(x+.23*y)/83), x+2.2*np.cos(2*np.pi*(y-.17*x)/127)]
    controls = dict(clockwise90=np.rot90(clean,-1).copy(), gamma_1p25=clean**1.25,
                    cross_shear=map_coordinates(clean,shear,order=1,mode="reflect",prefilter=False))
    for name in plan["clean_controls"]:
        yield clean_case(TEXTURES, target, name, controls[name])
    exposures = dict(high_contrast_clip=np.clip(.5+1.35*(clean-.5),0,1),
                     bright_exposure_clip=np.clip(1.20*clean+.045,0,1))
    for name in extra["clean_controls"]:
        ref = exposures[name]
        if not np.any((ref == 0.) | (ref == 1.)):
            raise ValueError(f"Exposure control has no clipped endpoint: {name}")
        yield clean_case(CLIPPED, target, name, ref)
    by_name = {v["name"]:v for v in variants}
    interior = .18+.64*clean
    for case, amp in generated_carriers(interior,target,INTERIOR,
                        [by_name[n] for n in extra["no_endpoint_variants"]],[extra["amplitude"]]):
        if not np.all((case[5] > 0) & (case[5] < 1)):
            raise ValueError("Interior-intensity observation reached a clipping endpoint")
        yield case, amp


def samples(plan, extra):
    amplitudes = old_focus(plan)
    group, kind, variant, amp = plan["previous_small_gain"]
    target = plan["target"]
    amplitudes[group,target,kind,variant] = amp
    reference = None
    for case in old_samples():
        cohort, name, kind, v, clean, _, _ = case
        if (cohort,name,kind) == ("archived_photo",target,"clean"):
            reference = clean
        yield case, amplitudes.get((cohort,name,kind,str(v)), 0. if kind == "clean" else "")
    if reference is None:
        raise ValueError("Missing original woven reference")
    yield from new_samples(reference, plan, extra)


def selection(rows):
    return {identity(row):row for row in rows if row["method"] == "current"}


def midpoint_comparison(clean, observed, output, diag, band):
    """Change only endpoint restoration, keeping the verified wave, support and gain."""
    stats = dict(midpoint_mse="", midpoint_texture_band_mse="", midpoint_endpoint_mse="",
                 current_endpoint_mse="", endpoint_ablation_pixels_changed="")
    if "carrier_wave" not in diag:
        return stats
    wave = np.asarray(diag["carrier_wave"])
    if wave.shape != observed.shape or not np.isfinite(wave).all():
        raise ValueError("Expected the verified finite wave on its reconstruction support")
    gain = diag["accepted_regions"][0]["gain"]
    midpoint = midpoint_subtraction(observed,wave,gain)
    endpoints = (observed == 0.) | (observed == 1.)
    if not np.array_equal(midpoint[~endpoints],output[~endpoints]):
        raise AssertionError("Midpoint intervention changed a non-endpoint pixel")
    delta = midpoint-clean
    stats.update(midpoint_mse=float(np.mean(delta**2)),
                 midpoint_texture_band_mse=band_energy(delta,band) if band is not None else "",
                 endpoint_ablation_pixels_changed=int(np.count_nonzero(midpoint != output)))
    if endpoints.any():
        stats.update(midpoint_endpoint_mse=float(np.mean(delta[endpoints]**2)),
                     current_endpoint_mse=float(np.mean((output[endpoints]-clean[endpoints])**2)))
    return stats


def evaluate(folder):
    from project_guarded import denoise as current
    from project_weak import denoise as previous

    frozen = read_json(folder/"freeze.json")
    if frozen != snapshot() or any((folder/n).exists() for n in OUTPUTS[1:]):
        raise ValueError("Choose the matching frozen directory with no completed results")
    cfg = load_config()
    plan, endpoints = frozen["protocol"], frozen["endpoint_controls"]
    with (ARCHIVE/"metrics.csv").open(newline="",encoding="utf8") as f:
        archive = selection(csv.DictReader(f))
    focus = old_focus(plan)
    pairs, bands, checked, parity, table = {}, {}, set(), [], []
    observed_hashes = set()
    observed_groups = {}
    for case, amp in samples(plan,endpoints):
        cohort, name, kind, variant, clean, observed, masks = case
        key = dict(cohort=cohort,image_id=name,kind=kind,variant=str(variant))
        ident = identity(key)
        if ident in pairs:
            raise ValueError(f"Duplicate input {ident}")
        ref_hash, in_hash = array_hash(clean), array_hash(observed)
        fresh = cohort in (DIRECTIONS,TEXTURES,CLIPPED,INTERIOR)
        if fresh and in_hash in observed_hashes:
            raise AssertionError(f"New case duplicates a previously measured observation: {ident}")
        observed_hashes.add(in_hash)
        old = previous(observed,cfg,return_diagnostics=True)
        if array_hash(observed) != in_hash:
            raise AssertionError(f"Baseline mutated input {ident}")
        if not fresh:
            record = archive[ident]
            actual = (ref_hash,in_hash,array_hash(old[0]),old[1]["route"],old[1]["reason"])
            if actual != tuple(record[n] for n in ("reference_sha256","observed_sha256","output_sha256","route","reason")):
                raise AssertionError(f"Historical result mismatch: {ident}")
            if ident in focus and not np.array_equal(old[0],observed):
                raise AssertionError(f"Expected an unprocessed target: {ident}")
            checked.add(ident)
        new = current(observed,cfg,return_diagnostics=True)
        if array_hash(observed) != in_hash:
            raise AssertionError(f"New method mutated input {ident}")
        parity.extend(checks_69022(ident,clean,observed,masks,cfg,previous,current,old,new))
        band = None
        if name == plan["target"]:
            if ref_hash not in bands:
                bands[ref_hash] = texture_band(clean)
            band = bands[ref_hash][0]
        if ident in focus and not np.array_equal(current(observed,cfg),new[0]):
            raise AssertionError(f"Regular and diagnostic entry disagree: {ident}")
        group = {}
        low, high = int((observed==0).sum()), int((observed==1).sum())
        for method,(output,diag) in (("input",(observed,{})), ("previous",old), ("current",new)):
            if diag:
                check_output(observed,output,diag)
            verification = diag.get("evidence",{}).get("guarded_carrier",{})
            row = key | dict(method=method,exact_amplitude=amp,low_endpoints=low,high_endpoints=high,
                        route=diag.get("route",""),reason=diag.get("reason",""),
                        guarded_reason=verification.get("reason",""),
                        reference_sha256=ref_hash,observed_sha256=in_hash,output_sha256=array_hash(output))
            row.update(measure(clean,observed,output,masks,band))
            row.update(midpoint_comparison(clean,observed,output,diag if method=="current" else {},band))
            group[method] = row
            table.append(row)
        pairs[ident] = group
        observed_groups[cohort] = observed_groups.get(cohort,0)+1
        print(len(pairs),name,kind,variant,old[1]["route"],new[1]["route"],flush=True)
    expected_new = {DIRECTIONS:len(plan["new_variants"])*len(plan["amplitudes"]),
                    TEXTURES:len(plan["clean_controls"]),CLIPPED:len(endpoints["clean_controls"]),
                    INTERIOR:len(endpoints["no_endpoint_variants"])}
    if (len(archive) != plan["historical_inputs"] or checked != archive.keys() or not focus.keys() <= checked
            or any(observed_groups.get(g,0) != n for g,n in expected_new.items())
            or len(parity) != 10 or frozen != snapshot()):
        raise AssertionError("Incomplete evaluation or changed frozen sources")
    groups = dict(historical=[p for k,p in pairs.items() if k in checked],
                  focus_nine=[pairs[k] for k in focus])
    groups.update({g:[p for k,p in pairs.items() if k[0] == g] for g in expected_new})
    stats = {name:summarize(cases,plan["mse_tie_tolerance"]) for name,cases in groups.items()}
    write_csv(folder/"metrics.csv",table)
    write_csv(folder/"parity.csv",parity)
    write_report(folder,plan,pairs,parity,stats)
    write_json(folder/"run.json",dict(completed_utc=datetime.now(timezone.utc).isoformat(),
               inputs=len(pairs),rows=len(table),historical_baselines_checked=len(checked),
               groups=stats,targeted_normal_entry_checks=len(focus),
               environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
               reference_texture_peaks_in_fft_bins={ref:info[1] for ref,info in bands.items()},
               artifact_sha256={n:sha256_file(folder/n) for n in OUTPUTS if n != "run.json"}))


def write_report(folder, plan, pairs, parity, stats):
    focus = old_focus(plan)
    selected = [(k,pairs[k]) for k in focus]
    held_out = [(k,p) for k,p in pairs.items() if k[0]==DIRECTIONS]
    cohort, kind, variant, _ = plan["previous_small_gain"]
    old_gain = (cohort,plan["target"],kind,variant)
    tol = plan["mse_tie_tolerance"]
    lines = ["# woven_0083 九个已知失败和新方向的同图验证","",
             f'九个既有未处理条件中有 {stats["focus_nine"]["better_than_previous"]} 个整图均方误差低于上版。'
             f'预登记的 16 个新方向与幅度组合中，有 {stats[DIRECTIONS]["better_than_previous"]} 个优于上版、'
             f'{stats[DIRECTIONS]["worse_than_previous"]} 个劣于上版。',"",
             "用同一张清晰图合成不同方向的条纹，并把每个观测图分别交给上版和本版。"
             "九个已知失败参与了本轮设计。另有八个新方向，各含两个幅值，方向与相位在运行前登记。"
             "检查织纹保留时，另用同图的旋转、灰度及形变，以及带亮暗裁剪端点的真实织物对照。"
             "各载波条件采用实际送入算法的数值幅度。","",
             "## 已知的九个未处理条件","",
             "像素范围为 [0,1]。实际注入量取裁剪后的输入减清晰参考。投影系数 a 表示实际减除量沿注入量方向的比例。"
             "纹带 MSE 为恢复图与清晰参考在六组织纹主峰及共轭邻域内的均方误差。"
             "纹带额外误差取实际减除量减 a 倍注入量后在相同频带的能量。原图、频带和真值仅用于评分。",""]

    def add_cases(items):
        lines.extend(["| 条件 | 幅值 | 整图 MSE 上版 → 本版 | a 上版 → 本版 | 纹带 MSE 上版 → 本版 | 本版正交误差 | 本版纹带额外误差 |",
                      "|---|---:|---:|---:|---:|---:|---:|"])
        for key,p in items:
            o,n = p["previous"],p["current"]
            def changed(field):
                return f'{o[field]:.4g} → {n[field]:.4g}'
            lines.append(f'| {key[3].split("@")[0]} | {n["exact_amplitude"]:g} | {changed("mse")} | '
                         f'{changed("noise_gain")} | {changed("texture_band_mse")} | {n["off_noise_mse"]:.4g} | '
                         f'{n["texture_excess_mse"]:.4g} |')
    add_cases(selected)
    band_better = sum(p["current"]["texture_band_mse"] < p["previous"]["texture_band_mse"] for _,p in selected)
    band_worse = sum(p["current"]["texture_band_mse"] > p["previous"]["texture_band_mse"] for _,p in selected)
    lines += ["",f'九例中 {stats["focus_nine"]["better_than_previous"]} 例整图 MSE 下降，'
              f'{band_better} 例织纹频带总误差下降、{band_worse} 例上升。'
              "已停止处理的旧版投影和额外误差均为零；本版有净收益时仍可能有波形偏差。","",
              "另列上一轮已取得小幅收益的条件。",""]
    add_cases([(old_gain,pairs[old_gain])])
    interventions = [(k,p["current"]) for k,p in pairs.items() if p["current"]["midpoint_mse"] != ""]
    lines += ["","## 同一条纹估计的端点复原对照","",
              "对本轮新通过的载波固定波形、支撑、减除强度，用上版中点法重新估计原观测中灰度恰为 0 或 1 的像素。"
              "每组均逐值检查非端点处两种复原相同。比较的是端点估计带来的变化，完整条件的两种端点误差见逐图表。"]
    if interventions:
        midpoint = np.array([r["midpoint_mse"] for _,r in interventions])
        paired = np.array([r["mse"] for _,r in interventions])
        textured = [r for _,r in interventions if r["midpoint_texture_band_mse"] != ""]
        lines += ["",f"共 {len(interventions)} 个条件可作此对照。更换端点估计后整图 MSE 下降 "
                  f"{int(np.sum(paired < midpoint-tol))} 例，上升 {int(np.sum(paired > midpoint+tol))} 例。"
                  f"整图 MSE 均值为 {midpoint.mean():.5g} → {paired.mean():.5g}。"]
        if textured:
            m = np.array([r["midpoint_texture_band_mse"] for r in textured])
            v = np.array([r["texture_band_mse"] for r in textured])
            lines += [f"其中 {len(textured)} 个织物条件的纹带 MSE 均值为 {m.mean():.5g} → {v.mean():.5g}，"
                      f"{int(np.sum(v<m))} 例下降，{int(np.sum(v>m))} 例上升。"]
    else:
        lines += ["","本轮无新成功路径可开展该对照。"]
    lines += ["","## 新方向和无端点对照","",
              "下表由预登记的八个新方向与两个幅值生成。位置变量采用像素坐标，测量保留同一方向两档强度的相位。",""]
    add_cases(held_out)
    wins = sum(p["current"]["texture_band_mse"] < p["previous"]["texture_band_mse"] for _,p in held_out)
    losses = sum(p["current"]["texture_band_mse"] > p["previous"]["texture_band_mse"] for _,p in held_out)
    lines += ["",f"16 个新组合中，织纹频带总误差较上版下降 {wins} 例，上升 {losses} 例，"
              f"其余 {len(held_out)-wins-losses} 例相同。"]
    lines += ["","另将同一张原图亮度压到 [0.18,0.82]，重加上表指定的两种 0.08 幅值载波。"
              "与原强度范围的同名案例使用相同方向、相位、幅值，亮度压缩同时改变了背景织纹对比度。",""]
    add_cases([(k,p) for k,p in pairs.items() if k[0]==INTERIOR])
    lines += ["","## 无新增条纹的织纹对照","",
              "| 对照 | 黑／白端点数 | 改动像素 上版 → 本版 | 整图 MAE 上版 → 本版 |",
              "|---|---:|---:|---:|"]
    for k,p in pairs.items():
        if k[1] == plan["target"] and k[2]=="clean":
            o,n = p["previous"],p["current"]
            lines.append(f'| {k[3] or "同一张原图"} | {n["low_endpoints"]}/{n["high_endpoints"]} | '
                         f'{o["changed_pixels"]} → {n["changed_pixels"]} | {o["mae"]:.4g} → {n["mae"]:.4g} |')
    clean = [p for k,p in pairs.items() if k[2] == "clean"]
    lines += ["",f'全部 {len(clean)} 个无新增条纹对照中，改动至少一个像素的输入数由 '
              f'{sum(p["previous"]["changed_pixels"]>0 for p in clean)} 变为 '
              f'{sum(p["current"]["changed_pixels"]>0 for p in clean)}。',"","## 69022 窄条与邻区","",
              "| 位置 | 宽度 | 污染区 MSE 输入／上版／本版 | 相邻干净区 MSE 上版／本版 | 全部干净区 MSE 上版／本版 |",
              "|---|---:|---:|---:|---:|"]
    for k,p in pairs.items():
        if k[1]!="69022" or k[2] not in ("edge8","edge16"):
            continue
        i,o,n = p["input"],p["previous"],p["current"]
        side = {"0":"左","2":"上","3":"下"}[k[3]]
        lines.append(f'| {side} | {k[2][4:]} px | {i["contaminated_mse"]:.5g}／{o["contaminated_mse"]:.5g}／'
                     f'{n["contaminated_mse"]:.5g} | {o["adjacent_clean_mse"]:.5g}／{n["adjacent_clean_mse"]:.5g} | '
                     f'{o["clean_mse"]:.5g}／{n["clean_mse"]:.5g} |')
    identical = sum(all(r[k] for k in ("output_equal","residual_equal","support_equal","route_equal","regions_equal"))
                    for r in parity)
    lines += ["",f'默认强度的六个输入及底边另外两档强度合计 {identical}/{len(parity)} 组输出、残差、掩膜、'
              '路径和接受区间逐值相同。相邻干净区取真实干净区域内距污染边缘 16 像素以内的位置。[逐项核对](parity.csv)',
              "","## 全集回归","",
              "| 集合 | 输入数 | 优于／持平／劣于上版 | 恢复图逐值相同 | 保留上版净收益 | 本版优于输入 |",
              "|---|---:|---:|---:|---:|---:|"]
    for g,s in stats.items():
        lines.append(f'| {g} | {s["inputs"]} | {s["better_than_previous"]}／{s["tied_with_previous"]}／'
                     f'{s["worse_than_previous"]} | {s["exact_same_as_previous"]} | {s["retained_net_improvements"]}/'
                     f'{s["previous_net_improvements"]} | {s["current_net_improvements"]} |')
    regressions = [k for k,p in pairs.items() if p["current"]["mse"] > p["previous"]["mse"]+tol]
    lines += ["","focus_nine 是 historical 的子集。历史 125 个输入逐一核对了参考、输入及上版输出的 SHA-256、"
              "路径和原因。MSE 持平公差为 1e−12。新增方向和图像变体均来自同一张已知照片。"]
    if regressions:
        lines += ["","上版整图 MSE 更低的案例为 "+"、".join("/".join(k) for k in regressions)+"。"]
    lines += ["","## 本地复现","","```powershell","python evaluate_nine83.py freeze --out results/nine83_reproduction",
              "python evaluate_nine83.py run --out results/nine83_reproduction","```","",
              "[逐图指标](metrics.csv) · [运行与校验值](run.json)"]
    (folder/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze","run"))
    parser.add_argument("--out",type=Path,default=ROOT/"results/nine83_reproduction")
    args = parser.parse_args()
    (freeze if args.stage=="freeze" else evaluate)(args.out)
