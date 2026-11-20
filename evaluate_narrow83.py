"""Replay the local woven_0083 and 69022 comparison under a fixed protocol."""
import argparse
import ast
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import platform
from pathlib import Path

import numpy as np
import PIL
import scipy
import skimage

from evaluate_project_refinement import extended_cases, identity
from evaluate_weave import (new_cases, projected_residual, read_json, summarize,
                            write_json)
from image_io_v3 import preprocess_image, sha256_file
from project_comparison import AMP, ROOT, array_hash, general_sample, scores
from project_pipeline import load_config
from support_data import make_sample as narrow_sample


ARCHIVE = ROOT / "results/weave"
PLAN = ROOT / "results/narrow83/protocol.json"
OUTPUTS = ("freeze.json", "metrics.csv", "parity.csv", "run.json", "RESULTS.md")


def current_dependencies():
    """Include transitive local module imports, including imports under functions."""
    queue = [Path(__file__).resolve(), ROOT/"project_stripes.py"]
    seen = set()
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                candidate = ROOT / (name.replace(".", "/") + ".py")
                if candidate.is_file() and candidate not in seen:
                    queue.append(candidate)
    return {p.relative_to(ROOT).as_posix() for p in seen}


def snapshot():
    old = read_json(ARCHIVE/"freeze.json")
    for name, digest in old["source_sha256"].items():
        if sha256_file(ROOT/name) != digest:
            raise ValueError(f"Historical dependency changed: {name}")
    run, photos = read_json(ARCHIVE/"run.json"), read_json(ARCHIVE/"photos_run.json")
    for name, digest in (("metrics.csv", run["metrics_sha256"]), ("freeze.json", run["freeze_sha256"]),
                         ("photos_metrics.csv", photos["metrics_sha256"]),
                         ("photos_selection.json", photos["selection_sha256"])):
        if sha256_file(ARCHIVE/name) != digest:
            raise ValueError(f"Historical archive changed: {name}")
    names = set(old["source_sha256"]) | current_dependencies() | {
        PLAN.relative_to(ROOT).as_posix(), "original/periodic_notch_starter/demo.py",
        *["results/weave/" + name for name in ("freeze.json", "metrics.csv", "run.json",
           "photos_selection.json", "photos_metrics.csv", "photos_run.json")]}
    return dict(source_sha256={s: sha256_file(ROOT/s) for s in sorted(names)},
                configuration=json.loads(json.dumps(asdict(load_config()))), protocol=read_json(PLAN))


def freeze(folder):
    folder.mkdir(parents=True, exist_ok=True)
    if any((folder/name).exists() for name in OUTPUTS):
        raise FileExistsError("Freeze into a fresh directory")
    write_json(folder/"freeze.json", snapshot())


def archive_rows():
    expected = {identity(r): dict(reference_sha256=r["reference_sha256"], observed_sha256=r["observed_sha256"])
                for r in read_json(ARCHIVE/"run.json")["cases"]}
    with (ARCHIVE/"metrics.csv").open(encoding="utf8", newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] == "current":
                expected[identity(r)].update({n: r[n] for n in ("output_sha256", "route", "reason")})
    sources = {r["filename"]: r for r in read_json(ARCHIVE/"photos_run.json")["sources"]}
    with (ARCHIVE/"photos_metrics.csv").open(encoding="utf8", newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] == "current":
                ident = ("archived_photo", "dtd_"+Path(r["photo"]).stem, r["kind"], "")
                expected[ident] = dict(reference_sha256=sources[r["photo"]]["reference_sha256"],
                                       **{n: r[n] for n in ("observed_sha256", "output_sha256", "route", "reason")})
    return expected


def samples(plan):
    refs = {}
    weave_plan = read_json(ROOT/"weave_protocol.json")
    for case in extended_cases():
        if case[2] == "clean" and case[1] in [*weave_plan["target_ids"], "69022"]:
            refs[case[1]] = case[4]
        yield case
    yield from new_cases(refs, weave_plan)
    for source in read_json(ARCHIVE/"photos_run.json")["sources"]:
        path = ROOT/"data/textures_v3/source"/source["filename"]
        if sha256_file(path) != source["jpeg_sha256"]:
            raise ValueError(f"Unexpected DTD file: {path}")
        clean, _ = preprocess_image(path, size=256)
        image_id = "dtd_" + path.stem
        if image_id == plan["target"]:
            refs[image_id] = clean
        for kind in ("clean", "global", "chirp"):
            observed, _, masks = general_sample(clean, image_id, kind, 0 if kind == "clean" else AMP)
            yield "archived_photo", image_id, kind, "", clean, observed, masks
    clean = refs[plan["target"]]
    yy, xx = np.indices(clean.shape, dtype=float)
    yy -= (clean.shape[0]-1)/2
    xx -= (clean.shape[1]-1)/2
    full = np.ones(clean.shape, bool)
    for carrier in plan["new_woven"]["carriers"]:
        phase = np.random.default_rng(carrier["phase_seed"]).uniform(-np.pi, np.pi)
        wave = np.sin(2*np.pi*(carrier["fy"]*yy + carrier["fx"]*xx) + phase)
        for amp in plan["new_woven"]["amplitudes"]:
            observed = np.clip(clean+amp*wave, 0, 1)
            yield ("new_woven", plan["target"], carrier["name"], f"A{amp:.2f}", clean, observed,
                   dict(contaminated=full, clean=~full))
    spec = plan["new_69022"]
    clean = refs["69022"]
    for variant in spec["variants"]:
        for kind in spec["kinds"]:
            observed, _, masks = narrow_sample(clean, "69022", kind, spec["amplitude"], variant)
            yield "new_69022", "69022", kind, str(variant), clean, observed, masks


def check_output(observed, output, diag):
    mask = diag["support_mask"]
    if (mask.dtype != bool or mask.shape != observed.shape or
            not np.array_equal(output[~mask], observed[~mask]) or
            not np.array_equal(diag["residual"], observed-output)):
        raise AssertionError("Support or residual does not describe output")


def parity_row(ident, strength, observed, old, new, masks, clean):
    (a, da), (b, db) = old, new
    for output, diag in (old, new):
        check_output(observed, output, diag)
    mask = da["support_mask"]
    m = scores(clean, observed, a, masks)
    equal = dict(output_equal=np.array_equal(a, b),
                 support_equal=np.array_equal(mask, db["support_mask"]),
                 residual_equal=np.array_equal(da["residual"], db["residual"]),
                 route_equal=da["route"] == db["route"],
                 regions_equal=da["accepted_regions"] == db["accepted_regions"])
    if not all(equal.values()):
        raise AssertionError(f"69022 default/strength parity failed: {ident}, {strength}, {equal}")
    return dict(variant=ident[3], kind=ident[2], strength=strength, route=db["route"],
                **equal, changed_pixels=m["changed_pixels"], support_pixels=int(mask.sum()),
                contaminated_mse=m["contaminated_mse"], adjacent_clean_mse=m["adjacent_clean_mse"],
                clean_mse=m["clean_mse"], support_sha256=array_hash(mask), output_sha256=array_hash(b),
                residual_sha256=array_hash(db["residual"]))


def write_csv(path, rows):
    with path.open("x", encoding="utf8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def evaluate(folder):
    from project_weave import denoise as previous
    from project_stripes import denoise as current

    sealed = read_json(folder/"freeze.json")
    if sealed != snapshot() or any((folder/n).exists() for n in OUTPUTS[1:]):
        raise ValueError("Use a frozen empty output directory with matching sources")
    plan, cfg = sealed["protocol"], load_config()
    expected, seen = archive_rows(), set()
    table, pairs, parity = [], {}, []
    clean_residual = {}
    normal_entry_checks = 0
    for cohort, image_id, kind, variant, clean, observed, masks in samples(plan):
        key = dict(cohort=cohort, image_id=image_id, kind=kind, variant=str(variant))
        ident = identity(key)
        ref_hash, obs_hash = array_hash(clean), array_hash(observed)
        if ident in pairs:
            raise ValueError(f"Duplicate case {ident}")
        if cohort not in ("new_woven", "new_69022"):
            record = expected[ident]
            if (ref_hash, obs_hash) != (record["reference_sha256"], record["observed_sha256"]):
                raise AssertionError(f"Historical input changed: {ident}")
            seen.add(ident)
        pairs[ident] = triplet = {}
        old = previous(observed, cfg, return_diagnostics=True)
        new = current(observed, cfg, return_diagnostics=True)
        if image_id == plan["target"] or (image_id == "69022" and kind in plan["required_parity"]["kinds"]):
            if not np.array_equal(current(observed, cfg), new[0]):
                raise AssertionError(f"Normal inference and diagnostic output disagree: {ident}")
            normal_entry_checks += 1
        if array_hash(observed) != obs_hash:
            raise AssertionError(f"Inference modified its input: {ident}")
        for method, (output, diag) in (("input", (observed, {})), ("previous", old), ("current", new)):
            if diag:
                check_output(observed, output, diag)
            row = key | dict(method=method, route=diag.get("route", ""), reason=diag.get("reason", ""),
                             reference_sha256=ref_hash, observed_sha256=obs_hash, output_sha256=array_hash(output))
            if ident in seen and method == "previous":
                if any(row[f] != record[f] for f in ("route", "reason", "output_sha256")):
                    raise AssertionError(f"Historical baseline changed: {ident}")
            row.update(scores(clean, observed, output, masks))
            row.update(mse_reduction_vs_input=triplet.get("input", row)["mse"]-row["mse"],
                       noise_gain="", off_noise_mse="", paired_gain="", paired_off_noise_mse="")
            if np.any(observed != clean):
                residual, added = observed-output, observed-clean
                gain, off = projected_residual(residual, added)
                row.update(noise_gain=gain, off_noise_mse=off)
                if not np.isclose(row["mse"], (1-gain)**2*np.mean(added**2)+off, rtol=1e-12, atol=2e-15):
                    raise ValueError(f"MSE decomposition failed for {ident}, {method}")
                if image_id == plan["target"]:
                    dg, de = projected_residual(residual-clean_residual[method], added)
                    row.update(paired_gain=dg, paired_off_noise_mse=de)
            elif image_id == plan["target"]:
                clean_residual[method] = observed-output
            triplet[method] = row
            table.append(row)
        spec = plan["required_parity"]
        if (cohort == "regression" and image_id == spec["image_id"] and kind in spec["kinds"]
                and str(variant) == str(spec["variant"])):
            parity.append(parity_row(ident, cfg.narrow.strength, observed, old, new, masks, clean))
            for strength in (0., .5):
                c = replace(cfg, narrow=replace(cfg.narrow, strength=strength))
                parity.append(parity_row(ident, strength, observed, previous(observed, c, return_diagnostics=True),
                                         current(observed, c, return_diagnostics=True), masks, clean))
        print(len(pairs), image_id, kind, variant, old[1]["route"], new[1]["route"], flush=True)
    groups = {"weave_92": [p for k,p in pairs.items() if k in seen and k[0] != "archived_photo"],
              "photos_12": [p for k,p in pairs.items() if k[0] == "archived_photo"],
              **{group:[p for k,p in pairs.items() if k[0] == group] for group in ("new_woven", "new_69022")}}
    if (seen != expected.keys() or [len(v) for v in groups.values()] !=
            [plan["historical"]["weave_study"], plan["historical"]["additional_photos"],
             len(plan["new_woven"]["carriers"])*len(plan["new_woven"]["amplitudes"]),
             len(plan["new_69022"]["kinds"])*len(plan["new_69022"]["variants"])]
            or len(parity) != 3*len(plan["required_parity"]["kinds"]) or snapshot() != sealed):
        raise AssertionError("Incomplete comparison or changed sources")
    stats = {g: summarize(v, plan["mse_tie_tolerance"]) for g,v in groups.items()}
    write_csv(folder/"metrics.csv", table)
    write_csv(folder/"parity.csv", parity)
    write_report(folder, pairs, stats, plan)
    files = {n:sha256_file(folder/n) for n in ("freeze.json", "metrics.csv", "parity.csv", "RESULTS.md")}
    write_json(folder/"run.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(),
               inputs=len(pairs), historical_baselines_checked=len(seen), rows=len(table),
               targeted_normal_entry_checks=normal_entry_checks,
               groups=stats, environment=dict(python=platform.python_version(), platform=platform.platform(),
               numpy=np.__version__, scipy=scipy.__version__, scikit_image=skimage.__version__,
               pillow=PIL.__version__), artifact_sha256=files))


def write_report(folder, pairs, stats, plan):
    header = ["# woven_0083 同图与 69022 窄条验证", "",
              "使用相同的单张灰度输入和默认配置比较 project_weave 与 project_stripes。"
              "92 个既有输入和追加四张 DTD 照片的 12 个条件均按参考、观测和上版输出哈希逐一核对。"
              "另外两组各四个新退化条件事先固定，使用的两张照片均已参与本次问题定位。", "",
              "## woven_0083", "",
              "| 条件 | 上版→本轮路径 | 修改像素 上→新 | MSE 输入／上版／本轮 | 实际噪声减除比例 上→新 | 正交误差 上→新 |",
              "|---|---|---:|---:|---:|---:|"]
    rows = []
    for (_, image_id, kind, variant), t in pairs.items():
        if image_id != plan["target"]:
            continue
        i, p, c = (t[m] for m in ("input", "previous", "current"))
        proj = "—" if c["noise_gain"] == "" else f"{p['noise_gain']:.4f} → {c['noise_gain']:.4f}"
        off = "—" if c["off_noise_mse"] == "" else f"{p['off_noise_mse']:.3g} → {c['off_noise_mse']:.3g}"
        rows.append(f"| {kind} {variant} | {p['route']} → {c['route']} | {p['changed_pixels']} → {c['changed_pixels']} | "
                    f"{i['mse']:.6g}／{p['mse']:.6g}／{c['mse']:.6g} | {proj} | {off} |")
        if kind == "clean":
            clean_mae = f"{p['mae']:.6g} → {c['mae']:.6g}"
    rows += ["", f"干净原图 MAE 为 {clean_mae}。对含噪图，实际新增量 n = Y − X，减除量 r = Y − F(Y)。"
             "减除比例 a = 〈r,n〉/〈n,n〉，正交误差为 mean((r − a·n)²)。总 MSE = (1 − a)²·mean(n²) + 正交误差。"
             "比例较高仍需配合整图 MSE 判断净收益。metrics.csv 还给出减去干净图处理响应后的配对投影。", "",
             "## 69022 边缘窄条", "",
             "| 边界 | 宽度 | 路径 上→新 | 污染区 MSE 输入／上版／本轮 | 相邻干净区 MSE 上→新 | 全部干净区 MSE 上→新 |",
             "|---|---|---|---:|---:|---:|"]
    for (group, img, kind, variant), t in pairs.items():
        if img != "69022" or kind not in ("edge8", "edge16"):
            continue
        name = {"0":"左","2":"上","3":"下"}[variant]
        i, p, c = (t[m] for m in ("input", "previous", "current"))
        rows.append(f"| {name} | {kind} | {p['route']} → {c['route']} | {i['contaminated_mse']:.5g}／"
                    f"{p['contaminated_mse']:.5g}／{c['contaminated_mse']:.5g} | "
                    f"{p['adjacent_clean_mse']:.5g} → {c['adjacent_clean_mse']:.5g} | "
                    f"{p['clean_mse']:.5g} → {c['clean_mse']:.5g} |")
    rows += ["", "历史下边界 8、16 像素在窄条强度 0、0.5、1 下均分别核对了输出、残差、支持掩膜、"
             "接受区间与路径完全相同。相邻干净区取真实干净掩膜内距边界 16 像素以内的位置。"
             "污染区采用合成包络大于 0.01 的位置，干净区取包络等于零的位置。", "",
             "## 总体回归", "",
             "| 集合 | 个数 | 对上版更好／持平／更差 | 逐像素相同 | 保留上版净改善／上版净改善 | 本轮净改善 | 比输入更差 上→新 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for group, s in stats.items():
        rows.append(f"| {group} | {s['inputs']} | {s['better_than_previous']}／{s['tied_with_previous']}／"
                    f"{s['worse_than_previous']} | {s['exact_same_as_previous']} | {s['retained_net_improvements']}／"
                    f"{s['previous_net_improvements']} | {s['current_net_improvements']} | "
                    f"{s['previous_worse_than_input']} → {s['current_worse_than_input']} |")
    clean_controls = [p for k, p in pairs.items() if k[2] == "clean"]
    changed_counts = [sum(p[m]["changed_pixels"] > 0 for p in clean_controls)
                      for m in ("previous", "current")]
    rows += ["", f"{len(clean_controls)} 个未加噪对照中，至少修改一个像素的图像数为 "
             f"{changed_counts[0]} → {changed_counts[1]}。"]
    worse = [(k, p) for k, p in pairs.items()
             if p["current"]["mse"] > p["previous"]["mse"] + plan["mse_tie_tolerance"]]
    if worse:
        rows += ["", "### 本轮整图 MSE 超过上版的条件", "",
                 "| 集合 | 图像 | 条件 | MSE 上版／本轮 |", "|---|---|---|---:|"]
        for (group, img, kind, variant), p in worse:
            rows.append(f"| {group} | {img} | {kind} {variant} | {p['previous']['mse']:.6g}／{p['current']['mse']:.6g} |")
    rows += ["", "净改善按整图 MSE 相对输入下降超过 1e−12 计。零改动可以保护原图，也可能留下完整干扰。"
             "新退化仅检验同图上强度、载波或边界位置改变时的表现。", "",
             "## 复现", "", "使用本地数据和锁定依赖，从项目根目录运行下面两条命令。"
             "首次命令在新目录记录源码、配置、历史档案和实验协议的 SHA-256 指纹。", "",
             "```text", "python evaluate_narrow83.py freeze --out results/narrow83_reproduction",
             "python evaluate_narrow83.py run --out results/narrow83_reproduction", "```", "",
             "逐条件整图与区域质量见 metrics.csv，独立的强度对照见 parity.csv，运行环境及分组统计见 run.json。"]
    (folder/"RESULTS.md").write_text("\n".join(header+rows)+"\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("--out", type=Path, default=ROOT/"results/narrow83_reproduction")
    args = parser.parse_args()
    (freeze if args.stage == "freeze" else evaluate)(args.out)
