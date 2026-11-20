"""Freeze and measure blind phase interpolation on the same woven photograph."""
import argparse
import ast
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import evaluate_collision83 as legacy
from check_nine83_peaks import peak_masks
from evaluate_narrow83 import check_output, write_csv
from evaluate_nine83 import generated_carriers, old_focus, selection
from evaluate_weak83 import checks_69022, measure
from evaluate_weave import read_json, summarize, write_json
from image_io_v3 import sha256_file
from phase_refit_controls import controls
from project_collision import denoise as previous
from project_comparison import ROOT, array_hash
from project_pipeline import load_config

ARCHIVE = ROOT/"results/collision83"
PLAN = ROOT/"results/phase_refit/protocol.json"
FRESH = "phase_refit_phases"
OUTPUTS = ("freeze.json", "metrics.csv", "per_peak.csv", "regions.csv", "parity.csv",
           "texture_controls.csv", "alternative_references.csv", "RESULTS.md", "run.json")


def snapshot():
    past = read_json(ARCHIVE/"freeze.json")
    if legacy.snapshot() != past:
        raise ValueError("Previous inference or its scoring dependencies changed")
    paths = set(past["source_sha256"])
    for name, digest in read_json(ARCHIVE/"run.json")["artifact_sha256"].items():
        path = ARCHIVE/name
        if sha256_file(path) != digest:
            raise ValueError(f"Changed historical result at {path}")
        paths.add(path.relative_to(ROOT).as_posix())
    for path in (ARCHIVE/"run.json", ARCHIVE/"texture_controls.json", PLAN, Path(__file__).resolve()):
        paths.add(path.relative_to(ROOT).as_posix())
    pending = [ROOT/name for name in ("project_phase.py","phase_refit_controls.py")]
    while pending:
        path = pending.pop()
        relative = path.relative_to(ROOT).as_posix()
        if relative in paths:
            continue
        paths.add(relative)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            names = ([a.name for a in node.names] if isinstance(node,ast.Import) else
                     [node.module] if isinstance(node,ast.ImportFrom) and node.module else [])
            pending.extend(dep for name in names if (dep:=ROOT/(name.replace(".","/")+".py")).is_file())
    return dict(protocol=read_json(PLAN), configuration=past["configuration"],
                source_sha256={name:sha256_file(ROOT/name) for name in sorted(paths)})


def freeze(folder):
    folder.mkdir(parents=True, exist_ok=True)
    if any((folder/n).exists() for n in OUTPUTS):
        raise FileExistsError("Choose a fresh experiment directory")
    write_json(folder/"freeze.json", snapshot())


def samples(plan):
    clean = None
    for case, amplitude in legacy.samples(read_json(legacy.PLAN)):
        if case[:3] == ("archived_photo", plan["target"], "clean"):
            clean = case[4]
        yield case, amplitude
    if clean is None:
        raise ValueError("Missing clean woven reference")
    spec = next(v for v in read_json(legacy.NINE/"protocol.json")["new_variants"] if v["name"] == "axis_x_E")
    for seed in plan["new_phase_seeds"]:
        variant = spec | dict(name=f"axis_x_E_s{seed}", seed=seed)
        yield from generated_carriers(clean, plan["target"], FRESH, [variant], plan["amplitudes"])


def group_for(key, plan, nine):
    if key[0] == FRESH:
        return "new_phases"
    if key[0] == legacy.PHASES and any(key[3].startswith(f"axis_x_E_s{s}@") for s in plan["target_seeds"]):
        return "target"
    return "nine" if key in nine else "other"


def region_rows(fields, ref, obs, outputs, bands):
    for i, (left,right) in enumerate(bands, 1):
        section = np.s_[:,left:right+1]
        r = dict(**fields, strip=i, first_column=left, last_column=right)
        addition = (obs-ref)[section]
        for label, (out,_) in outputs.items():
            r[label+"_mse"] = float(np.mean((out[section]-ref[section])**2))
            r[label+"_gain"] = legacy.projected_residual((obs-out)[section], addition)[0]
        yield r


def report(folder, cases, stats, peaks, regions, parity, textures, alternatives, protocol):
    lines = ["# 四个相位残留案例与冻结后的新相位验证", "",
             "整图去除比例为实际减除量在真实新增条纹上的投影。固定左右带分别取第 38–111、199–255 列的全部行。"
             "取两个方法较小的正投影作为共同强度，分别缩放各自减除量，衡量相同去除比例下的织纹峰额外误差。",
             "", "| 条件 | 幅值 | 去除比例（上版 → 本版） | 左带 MSE（上版 → 本版） | 右带 MSE（上版 → 本版） | 达到效果量／两峰兼顾 |",
             "|---|---:|---:|---:|---:|---|"]
    effects = {}
    for key, records in cases.items():
        o, n = records["previous"], records["current"]
        group = n["group"]
        if group not in ("target","new_phases"):
            continue
        pair = [r for r in regions if r["group"]==group and r["variant"]==n["variant"]]
        focus = [r for r in peaks if r["group"]==group and r["variant"]==n["variant"] and r["peak"] in (5,6)]
        visible = (n["noise_gain"]-o["noise_gain"] >= protocol["visible_gain_increase"] and
                   all(r["current_mse"] <= (1-protocol["visible_regional_mse_decrease"])*r["previous_mse"] for r in pair))
        guarded = all(r[a]!="" and r[b]!="" and r[b] <= r[a]+legacy.EPS for r in focus for a,b in (
            ("previous_mse","current_mse"), ("matched_previous_excess","matched_current_excess")))
        effects[key[3]] = dict(group=group, visible=bool(visible), visible_and_peaks=bool(visible and guarded))
        cells = [f"{r['previous_mse']:.6g} → {r['current_mse']:.6g}" for r in pair]
        lines.append(f"| {group} / {key[3].split('@')[0]} | {n['exact_amplitude']:.3f} | "
                     f"{o['noise_gain']:.2%} → {n['noise_gain']:.2%} | "+" | ".join(cells)+
                     f" | {int(visible)}／{int(visible and guarded)} |")
    lines += ["","效果量要求整图投影提高至少 10 个百分点且两带 MSE 均下降至少 25%。两峰兼顾还要求第 5、6 峰各自原始误差和共同强度额外误差均未增加。"
              "输入和上版均作为逐峰比较对象。所有新相位都在同一块已知织物上生成。","",
              "| 分组 | 第 5、6 峰 MSE 对上版 | 对含噪输入 | 同比例额外误差对上版 |","|---|---|---|---|"]
    for name in ("target","new_phases","nine"):
        focus = [r for r in peaks if r["group"]==name and r["peak"] in (5,6)]
        lines.append("| "+name+" | "+" | ".join(legacy.change_counts(focus,a,b) for a,b in (
            ("previous_mse","current_mse"), ("input_mse","current_mse"), ("matched_previous_excess","matched_current_excess")))+" |")
    exceptions=[r for r in peaks if r["group"] in ("target","new_phases") and r["peak"] in (5,6)
                and (r["current_mse"]>r["previous_mse"]+legacy.EPS or r["current_mse"]>r["input_mse"]+legacy.EPS
                     or (r["matched_previous_excess"]!="" and r["matched_current_excess"]>r["matched_previous_excess"]+legacy.EPS))]
    if exceptions:
        lines += ["","以下条件在三种逐峰比较中至少有一项上升。",
                  "","| 条件／峰 | 输入、上版、本版峰 MSE | 上版、本版同强度额外误差 |","|---|---:|---:|"]
        for r in exceptions:
            lines.append(f"| {r['variant']}／{r['peak']} | {r['input_mse']:.6g}，{r['previous_mse']:.6g}，{r['current_mse']:.6g} | "
                         f"{legacy.fmt(r['matched_previous_excess'])}，{legacy.fmt(r['matched_current_excess'])} |")
    lines += ["","| 集合 | 样本数 | 整图 MSE 改善／持平／上升 | 新旧逐像素一致 |","|---|---:|---:|---:|"]
    for name, s in stats.items():
        lines.append(f"| {name} | {s['inputs']} | {s['better_than_previous']}／{s['tied_with_previous']}／{s['worse_than_previous']} | {s['exact_same_as_previous']} |")
    worse=[(k,r) for k,r in cases.items() if r["current"]["mse"]>r["previous"]["mse"]+1e-12]
    if worse:
        lines += ["","整图 MSE 高于上版的输入如下。",""]
        for k,r in worse:
            lines.append(f"- {k[1]}／{k[2]}／{k[3]}，{r['previous']['mse']:.6g} → {r['current']['mse']:.6g}。")
    prior=[p for k,p in cases.items() if k[0] in (legacy.PRIOR_PHASES,legacy.PHASES,legacy.DIRECTIONS)
           and k[3].startswith("axis_x_E") and p["current"]["group"]!="target"]
    lines += ["","其余既有横向相位与原相位如下。","",
              "| 既有条件 | 去除比例（上版 → 本版） | 整图 MSE（上版 → 本版） |","|---|---:|---:|"]
    for p in prior:
        o,n=p["previous"],p["current"]
        lines.append(f"| {n['variant']} | {o['noise_gain']:.2%} → {n['noise_gain']:.2%} | {o['mse']:.6g} → {n['mse']:.6g} |")
    clean = [v["current"] for k,v in cases.items() if k[2]=="clean"]
    exact = sum(all(r[k] for k in ("route_equal","regions_equal","support_equal","output_equal","residual_equal")) for r in parity)
    lines += ["",f"{len(clean)} 组无新增干扰对照中，{sum(r['changed_pixels']>0 for r in clean)} 组出现像素改变。69022 的 8／16 像素与强度对照共 {len(parity)} 组，"
              f"输出、残差、掩膜、路径和接受区间全同 {exact} 组。", "",
              "| 69022 边缘／宽度 | 污染区 MSE（上版 → 本版） | 邻近 16 像素干净区 MSE（上版 → 本版） |","|---|---:|---:|"]
    for k,p in cases.items():
        if k[1]=="69022" and k[2] in ("edge8","edge16"):
            o,n=p["previous"],p["current"]
            side = {"0":"左","2":"上","3":"下"}[k[3]]
            lines.append(f"| {side}／{k[2][4:]} | {o['contaminated_mse']:.6g} → {n['contaminated_mse']:.6g} | "
                         f"{o['adjacent_clean_mse']:.6g} → {n['adjacent_clean_mse']:.6g} |")
    lines += ["","## 附加的局部真实织纹对照","",
              "同图加入幅值 0.035 的局部定频织纹与相同渐变条纹，计算有织纹和无织纹两次输出的差相对于合法织纹新增量的投影和 MSE。"
              "投影越接近 1、MSE 越小，保真越好。干净带另查条带内错误改动。","",
              "| 相位／幅值 | 位置 | 真实织纹增量保留比例（上版 → 本版） | 增量 MSE（上版 → 本版） |","|---|---|---:|---:|"]
    for r in textures:
        if r["group"]=="target" and r["kind"]=="stationary_weave_plus_stripe":
            lines.append(f"| {r['variant']} | {r['side']} | {r['previous_texture_gain']:.3f} → {r['current_texture_gain']:.3f} | "
                         f"{r['previous_texture_mse']:.6g} → {r['current_texture_mse']:.6g} |")
    gaps = [r for r in textures if r['kind'] in ('clean_gap','stationary_weave_gap')]
    lines += ["",f"{len(gaps)} 项带内无新干扰的历史与目标附加对照中，旧版和本版分别改动了 "
              f"{sum(r['previous_inside_changes']>0 for r in gaps)} 和 {sum(r['current_inside_changes']>0 for r in gaps)} 项的干净带。"
              "历史的同一观测双重解释也保留在原始表中。","",
              "另作事后敏感性分析，复用四个目标的观测和处理结果，改把各带内完全同幅、同频、同相的波形归为织物的一部分。"
              f"此时带内输入本来正确，新版相对旧版的带内误差共有 "
              f"{sum(r['current_inside_mse'] > r['previous_inside_mse']+1e-12 for r in alternatives)}/{len(alternatives)} 项上升。"
              "这些重新解释并未产生新的输入图片，单列于[同观测不同真值](alternative_references.csv)。","",
              "[逐图指标](metrics.csv) · [逐峰误差](per_peak.csv) · [固定区域](regions.csv) · [69022](parity.csv) · "
              "[纹理控制](texture_controls.csv) · [预登记](protocol.json)"]
    (folder/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf8")
    return effects


def evaluate(folder):
    from project_phase import denoise as current
    seal = read_json(folder/"freeze.json")
    if seal != snapshot() or any((folder/n).exists() for n in OUTPUTS[1:]):
        raise ValueError("Start with a matching freeze in a fresh output directory")
    plan = seal["protocol"]
    nine = old_focus(read_json(legacy.NINE/"protocol.json"))
    known_peaks = read_json(legacy.NINE/"run.json")["reference_texture_peaks_in_fft_bins"]
    with (ARCHIVE/"metrics.csv").open(encoding="utf8", newline="") as f:
        archive = selection(csv.DictReader(f))
    cfg = load_config()
    pairs, geometry, target_outputs = {}, {}, {}
    table, regions, peaks, parity = [], [], [], []
    checked, seen = set(), set()
    example_dir = folder/"examples"
    example_dir.mkdir(exist_ok=True)
    for case, amp in samples(plan):
        cohort, name, kind, variant, clean, obs, masks = case
        ident = cohort,name,kind,str(variant)
        group = group_for(ident, plan, nine)
        key = dict(group=group, cohort=cohort, image_id=name, kind=kind, variant=str(variant))
        ref, inp = array_hash(clean), array_hash(obs)
        if ident in pairs or (cohort==FRESH and inp in seen):
            raise ValueError(f"Duplicate {ident}")
        seen.add(inp)
        old = previous(obs,cfg,return_diagnostics=True)
        if array_hash(obs)!=inp:
            raise ValueError(f"Previous entry changed input {ident}")
        if cohort != FRESH:
            row = archive[ident]
            if (ref,inp,array_hash(old[0]),old[1]["route"],old[1]["reason"]) != tuple(
                    row[k] for k in ("reference_sha256","observed_sha256","output_sha256","route","reason")):
                raise ValueError(f"Archive replay mismatch {ident}")
            checked.add(ident)
        new = current(obs,cfg,return_diagnostics=True)
        if array_hash(obs)!=inp:
            raise ValueError(f"Current entry changed input {ident}")
        parity.extend(checks_69022(ident,clean,obs,masks,cfg,previous,current,old,new))
        outputs = dict(input=(obs,{}), previous=old, current=new)
        pairs[ident] = rows = {}
        if name==plan["target"] and ref not in geometry:
            frequencies, disks = peak_masks(clean,known_peaks[ref])
            geometry[ref] = frequencies,disks,np.logical_or.reduce(disks)
        band = geometry[ref][2] if name==plan["target"] else None
        for label,(out,d) in outputs.items():
            if d:
                check_output(obs,out,d)
            evidence = d.get("evidence",{})
            row = dict(**key, method=label, exact_amplitude=amp, route=d.get("route",""), reason=d.get("reason",""),
                       collision_decision=evidence.get("collision_review",{}).get("reason",""),
                       refit_decision=evidence.get("phase_refit",{}).get("reason",""),
                       reference_sha256=ref, observed_sha256=inp, output_sha256=array_hash(out))
            row.update(measure(clean,obs,out,masks,band))
            table.append(row)
            rows[label]=row
        if band is not None:
            peaks.extend(legacy.spectral_scores(key,clean,obs,old[0],new[0],rows,*geometry[ref][:2]))
        if group in ("target","new_phases"):
            regions.extend(region_rows(key,clean,obs,outputs,plan["fixed_column_ranges_inclusive"]))
            np.savez_compressed(example_dir/(variant+".npz"), clean=clean, **{label:out for label,(out,_) in outputs.items()})
            if group=="target":
                target_outputs[variant] = clean,obs,old[0],new[0]
        if len(pairs)%10==0:
            print(f"{len(pairs)} inputs", flush=True)
    sizes = {g:sum(p["current"]["group"]==g for p in pairs.values()) for g in ("target","new_phases","nine")}
    if (checked != archive.keys() or len(checked)!=plan["historical_inputs"] or len(parity)!=10 or
        sizes!={"target":4, "new_phases":6, "nine":9} or sum(k[2]=="clean" for k in pairs)!=33):
        raise ValueError("Incomplete cohorts")
    texture_rows, alternatives = controls(plan, target_outputs, previous, current)
    if snapshot()!=seal:
        raise ValueError("Source changed during evaluation")
    stats = {"historical":summarize([p for k,p in pairs.items() if k in checked],1e-12)}
    stats.update({g:summarize([p for p in pairs.values() if p["current"]["group"]==g],1e-12) for g in sizes})
    for name, data in (("metrics",table),("per_peak",peaks),("regions",regions),("parity",parity),
                       ("texture_controls",texture_rows),("alternative_references",alternatives)):
        write_csv(folder/(name+".csv"),data)
    effects = report(folder,pairs,stats,peaks,regions,parity,texture_rows,alternatives,plan)
    write_json(folder/"run.json",dict(completed_utc=datetime.now(timezone.utc).isoformat(), inputs=len(pairs),
               historical_baselines_checked=len(checked), texture_challenges=len(texture_rows),
               alternative_reference_checks=len(alternatives), groups=stats,
               effects=effects, artifact_sha256={n:sha256_file(folder/n) for n in OUTPUTS if n!="run.json"}))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("freeze","run"))
    p.add_argument("--out",type=Path,default=ROOT/"results/phase_refit_reproduction")
    args=p.parse_args()
    (freeze if args.stage=="freeze" else evaluate)(args.out)
