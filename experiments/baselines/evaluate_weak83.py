"""Freeze and replay the woven_0083 weak-carrier and narrow-edge paired study."""
import argparse
import ast
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.ndimage import map_coordinates, maximum_filter

from evaluate_narrow83 import check_output, samples as historical_samples, write_csv
from evaluate_project_refinement import identity
from evaluate_weave import new_cases, projected_residual, read_json, summarize, write_json
from image_io_v3 import sha256_file
from local_spectral import _detrend
from project_comparison import ROOT, array_hash, scores
from project_pipeline import load_config


ARCHIVE = ROOT/"results/narrow83"
PLAN = ROOT/"results/weak83/protocol.json"
SINGLE_AXIS_PLAN = PLAN.with_name("single_axis_protocol.json")
OUTPUTS = ("freeze.json", "metrics.csv", "parity.csv", "RESULTS.md", "run.json")
NEW_GROUPS = ("weak83_carriers", "weak83_clean_controls", "weak83_single_axis")


def snapshot():
    old = read_json(ARCHIVE/"freeze.json")
    previous_run = read_json(ARCHIVE/"run.json")
    for file, digest in previous_run["artifact_sha256"].items():
        if sha256_file(ARCHIVE/file) != digest:
            raise ValueError(f"Historical result changed: {file}")
    names = set(old["source_sha256"])
    for name in names:
        if sha256_file(ROOT/name) != old["source_sha256"][name]:
            raise ValueError(f"Historical source changed: {name}")
    # Discover the new local module graph while the preceding graph is also
    # checked against its own recorded source hashes.
    queue = [Path(__file__).resolve(), ROOT/"project_weak.py"]
    while queue:
        path = queue.pop()
        name = path.relative_to(ROOT).as_posix()
        if name in names:
            continue
        names.add(name)
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Import):
                modules = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            else:
                continue
            for module in modules:
                dep = ROOT/(module.replace(".", "/")+".py")
                if dep.is_file():
                    queue.append(dep)
    names |= {(ARCHIVE/n).relative_to(ROOT).as_posix() for n in (*previous_run["artifact_sha256"], "run.json")}
    names.add(PLAN.relative_to(ROOT).as_posix())
    names.add(SINGLE_AXIS_PLAN.relative_to(ROOT).as_posix())
    return dict(source_sha256={n:sha256_file(ROOT/n) for n in sorted(names)},
                configuration=json.loads(json.dumps(asdict(load_config()))), protocol=read_json(PLAN),
                single_axis_protocol=read_json(SINGLE_AXIS_PLAN))


def freeze(folder):
    folder.mkdir(parents=True, exist_ok=True)
    if any((folder/n).exists() for n in OUTPUTS):
        raise FileExistsError("Use a fresh output directory")
    write_json(folder/"freeze.json", snapshot())


def extra_cases(clean, plan, single_axis):
    target = plan["target"]
    for _, name, kind, variant, ref, image, masks in new_cases({target:clean}, dict(
            target_ids=[target], new_variants=plan["new_variants"], amplitudes=plan["amplitudes"])):
        yield NEW_GROUPS[0], name, kind, variant, ref, image, masks
    for _, name, kind, variant, ref, image, masks in new_cases({target:clean}, dict(
            target_ids=[target], new_variants=single_axis["new_variants"], amplitudes=single_axis["amplitudes"])):
        yield NEW_GROUPS[2], name, kind, variant, ref, image, masks
    for variant in plan["clean_controls"]:
        if variant == "transpose":
            ref = clean.T.copy()
        else:
            source = clean if variant == "elastic_warp" else np.rot90(clean)
            sign = 1 if variant == "elastic_warp" else -1
            y, x = np.indices(source.shape, dtype=float)
            moved = [y+sign*2.5*np.sin(2*np.pi*x/113), x+sign*1.75*np.sin(2*np.pi*y/97+.4)]
            ref = map_coordinates(source, moved, order=1, mode="reflect", prefilter=False)
        clean_mask = np.ones(ref.shape, bool)
        yield NEW_GROUPS[1], target, "clean", variant, ref, ref.copy(), dict(contaminated=~clean_mask, clean=clean_mask)


def samples(plan, single_axis):
    target = None
    for case in historical_samples(read_json(ARCHIVE/"protocol.json")):
        if case[1] == plan["target"] and case[2] == "clean":
            target = case[4]
        yield case
    if target is None:
        raise ValueError("Missing woven reference")
    yield from extra_cases(target, plan, single_axis)


def texture_band(clean):
    """Select reference-only peak disks on the two-sided native FFT grid."""
    h, w = clean.shape
    wy = np.rint(np.fft.fftfreq(h)*h).astype(int)
    wx = np.rint(np.fft.fftfreq(w)*w).astype(int)
    y, x = np.meshgrid(wy, wx, indexing="ij")
    amplitude = abs(np.fft.fft2(_detrend(clean)*np.outer(np.hanning(h), np.hanning(w))))
    radius = np.hypot(y/h, x/w)
    keep = (radius >= .025) & (radius <= .45) & ((y > 0) | ((y == 0) & (x > 0)))
    keep &= amplitude == maximum_filter(amplitude, size=3, mode="wrap")
    indices = np.flatnonzero(keep)
    ranked = indices[np.argsort(-amplitude.ravel()[indices], kind="stable")]
    selected = []
    mask = np.zeros(clean.shape, bool)
    for p in ranked:
        peak = np.array([y.flat[p], x.flat[p]])
        if any(min(np.linalg.norm(peak-old), np.linalg.norm(peak+old)) < 6 for old in selected):
            continue
        for sign in (-1, 1):
            mask |= (y-sign*peak[0])**2 + (x-sign*peak[1])**2 <= 4
        selected.append(peak)
        if len(selected) == 6:
            break
    if len(selected) != 6:
        raise ValueError("Insufficient resolved reference texture peaks")
    return mask, [p.tolist() for p in selected]


def band_energy(image, mask):
    # Unwindowed squared distance in this frequency subspace; Parseval
    # normalization is for a two-sided FFT and mean over *all* image pixels.
    spectrum = np.fft.fft2(image, norm="ortho")
    return float(np.sum(abs(spectrum[mask])**2)/image.size)


def measure(clean, observed, output, masks, band):
    row = scores(clean, observed, output, masks)
    row.update(noise_gain="", off_noise_mse="", texture_band_mse="", texture_excess_mse="")
    if band is not None:
        row["texture_band_mse"] = band_energy(output-clean, band)
    added = observed-clean
    if np.any(added):
        removal = observed-output
        gain, off = projected_residual(removal, added)
        row.update(noise_gain=gain, off_noise_mse=off)
        if not np.isclose(row["mse"], (1-gain)**2*np.mean(added*added)+off, rtol=1e-12, atol=1e-15):
            raise AssertionError("Noise projection and residual do not reconstruct the true MSE")
        if band is not None:
            row["texture_excess_mse"] = band_energy(removal-gain*added, band)
    return row


def checks_69022(ident, clean, observed, masks, cfg, previous, current, old, new):
    variants = {"0", "2", "3"}
    if ident[1] != "69022" or ident[2] not in ("edge8", "edge16") or ident[3] not in variants:
        return []
    def comparison(strength, first, second):
        a, da = first
        b, db = second
        row = dict(variant=ident[3], kind=ident[2], strength=strength, previous_route=da["route"],
                   current_route=db["route"], output_equal=np.array_equal(a,b),
                   residual_equal=np.array_equal(da["residual"],db["residual"]),
                   support_equal=np.array_equal(da["support_mask"],db["support_mask"]),
                   regions_equal=da["accepted_regions"] == db["accepted_regions"], route_equal=da["route"] == db["route"])
        for label, (output, diag) in (("previous",first), ("current",second)):
            check_output(observed, output, diag)
            quality = scores(clean, observed, output, masks)
            row.update({label+"_"+field:quality[field] for field in (
                "changed_pixels", "contaminated_mse", "adjacent_clean_mse", "clean_mse")})
            row.update({label+"_"+field+"_sha256":array_hash(array) for field, array in
                        (("output",output), ("support",diag["support_mask"]), ("residual",diag["residual"]))})
        return row

    rows = [comparison(cfg.narrow.strength, old, new)]
    if ident[3] == "3":
        for strength in (0., .5):
            c = replace(cfg, narrow=replace(cfg.narrow, strength=strength))
            rows.append(comparison(strength, previous(observed, c, return_diagnostics=True),
                                   current(observed, c, return_diagnostics=True)))
    return rows


def evaluate(folder):
    from project_stripes import denoise as previous
    from project_weak import denoise as current

    sealed = read_json(folder/"freeze.json")
    if sealed != snapshot() or any((folder/n).exists() for n in OUTPUTS[1:]):
        raise ValueError("Use the matching frozen output directory")
    plan, cfg = sealed["protocol"], load_config()
    with (ARCHIVE/"metrics.csv").open(newline="", encoding="utf8") as f:
        archive = {identity(row):row for row in csv.DictReader(f) if row["method"] == "current"}
    pairs, table, parity, spectra, seen = {}, [], [], {}, set()
    normal_checks = 0
    for cohort, name, kind, variant, clean, observed, masks in samples(plan, sealed["single_axis_protocol"]):
        key = dict(cohort=cohort, image_id=name, kind=kind, variant=str(variant))
        ident = identity(key)
        if ident in pairs:
            raise ValueError(f"Duplicate key {ident}")
        ref_hash, obs_hash = array_hash(clean), array_hash(observed)
        old = previous(observed, cfg, return_diagnostics=True)
        if array_hash(observed) != obs_hash:
            raise AssertionError(f"Baseline mutated {ident}")
        if cohort not in NEW_GROUPS:
            a = archive[ident]
            actual = (ref_hash, obs_hash, array_hash(old[0]), old[1]["route"], old[1]["reason"])
            if actual != tuple(a[n] for n in ("reference_sha256", "observed_sha256", "output_sha256", "route", "reason")):
                raise AssertionError(f"Archived result changed at {ident}")
            seen.add(ident)
        new = current(observed, cfg, return_diagnostics=True)
        if array_hash(observed) != obs_hash:
            raise AssertionError(f"Current method mutated {ident}")
        if name == plan["target"] and (kind in ("clean", "chirp") or variant == "A0.07") and cohort not in NEW_GROUPS:
            if not np.array_equal(current(observed, cfg), new[0]):
                raise AssertionError(f"Normal entry disagrees with diagnostics: {ident}")
            normal_checks += 1
        parity.extend(checks_69022(ident, clean, observed, masks, cfg, previous, current, old, new))
        band = None
        if name == plan["target"]:
            if ref_hash not in spectra:
                spectra[ref_hash] = texture_band(clean)
            band = spectra[ref_hash][0]
        pairs[ident] = group = {}
        for method, (out, diag) in (("input",(observed,{})), ("previous",old), ("current",new)):
            if diag:
                check_output(observed, out, diag)
            row = key | dict(method=method, route=diag.get("route",""), reason=diag.get("reason",""),
                    reference_sha256=ref_hash, observed_sha256=obs_hash, output_sha256=array_hash(out))
            row.update(measure(clean, observed, out, masks, band))
            group[method] = row
            table.append(row)
        print(len(pairs), name, kind, variant, old[1]["route"], new[1]["route"], flush=True)
    ncarriers = len(plan["new_variants"])*len(plan["amplitudes"])
    if (len(archive) != plan["historical_inputs"] or seen != archive.keys() or
            sum(k[0] == NEW_GROUPS[0] for k in pairs) != ncarriers or
            sum(k[0] == NEW_GROUPS[2] for k in pairs) != 2 or
            sum(k[0] == NEW_GROUPS[1] for k in pairs) != len(plan["clean_controls"]) or len(parity) != 10
            or normal_checks != 4 or sealed != snapshot()):
        raise ValueError("Incomplete evaluation or changed sources")
    groups = {"historical": [p for k,p in pairs.items() if k in seen],
              **{g:[p for k,p in pairs.items() if k[0] == g] for g in NEW_GROUPS}}
    stats = {g:summarize(rows, plan["mse_tie_tolerance"]) for g,rows in groups.items()}
    write_csv(folder/"metrics.csv", table)
    write_csv(folder/"parity.csv", parity)
    write_report(folder, pairs, parity, stats, plan)
    write_json(folder/"run.json", dict(completed_utc=datetime.now(timezone.utc).isoformat(),
               inputs=len(pairs), historical_baselines_checked=len(seen), rows=len(table),
               targeted_normal_entry_checks=normal_checks, groups=stats,
               reference_texture_peaks_in_fft_bins={k:v[1] for k,v in spectra.items()},
               environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
               artifact_sha256={n:sha256_file(folder/n) for n in OUTPUTS if n != "run.json"}))


def write_report(folder, pairs, parity, stats, plan):
    lines = ["# woven_0083 弱定频、渐变与纹理保护验证", "",
             "两个方法逐张接收相同的浮点观测图。全部 112 个历史输入核对了清晰参考、输入、上轮输出的 SHA-256 及其处理路径。"
             "另外 8 个加噪条件和 3 个无新增条纹对照按预先登记的方向、种子及变换生成。"
             "确认新分支只搜索单轴扫频后，在生成新增输入前补充登记了横向和纵向扫频各一例，单列为 weak83_single_axis。"
             "上述条件均复用同一张已知照片。", "",
             "## 同图配对", "",
             "像素范围为 [0,1]。设实际新增量 n 为输入减清晰图，实际减除量 r 为输入减输出。"
             "表中 a 为 r 在 n 上的投影系数，正交误差为 mean((r−a·n)²)。"
             "纹理频带由清晰照片选定，为六个最强且分离的参考峰及其共轭各取半径 2 格的圆盘。"
             "纹理额外误差为 r−a·n 在这些频带的均方能量，以相同去条纹比例 a 的理想减除 a·n 为参照。"
             "该量包含波形和幅度失配。纹理总误差为输出减原图在相同频带的均方能量，也含残余干扰。", "",
             "| 条件 | 路径 上版 → 本轮 | 整图 MSE 输入／上版／本轮 | a 上版 → 本轮 | 正交误差 上版 → 本轮 | 纹理额外误差 上版 → 本轮 | 纹理总误差 上版 → 本轮 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    def change(old, new, key):
        return "—" if new[key] == "" else f"{old[key]:.4g} → {new[key]:.4g}"
    for (cohort, name, kind, variant), p in pairs.items():
        if name != plan["target"]:
            continue
        i, o, n = p["input"], p["previous"], p["current"]
        lines.append(f"| {kind} {variant} | {o['route']} → {n['route']} | {i['mse']:.6g}／{o['mse']:.6g}／{n['mse']:.6g} | "
                f"{change(o,n,'noise_gain')} | {change(o,n,'off_noise_mse')} | {change(o,n,'texture_excess_mse')} | "
                f"{change(o,n,'texture_band_mse')} |")
    focus = [p for (cohort,name,kind,variant),p in pairs.items() if name == plan["target"]
             and ((cohort == "archived_photo" and kind == "chirp") or
                  (cohort == "new_woven" and variant == "A0.07"))]
    if len(focus) != 3:
        raise AssertionError("Three known target failures are required for the paired conclusion")
    original = pairs[("archived_photo", plan["target"], "clean", "")]
    improved = sum(p["current"]["mse"] < p["input"]["mse"]-plan["mse_tie_tolerance"] for p in focus)
    positive = sum(p["current"]["texture_excess_mse"] > p["previous"]["texture_excess_mse"] for p in focus)
    largest = max(p["current"]["texture_excess_mse"] for p in focus)
    lines += ["", f"woven_0083 原图的修改像素数为 {original['previous']['changed_pixels']} → "
              f"{original['current']['changed_pixels']}。旧版未去除的两个弱定频及一个渐变条件中，"
              f"{improved}/3 的整图误差低于含噪输入。"
              f"{positive}/3 的纹理频带额外误差高于上版，这三例本轮最大值为 {largest:.6g}。"
              "停止处理时额外误差与减除比例均为零。含噪图上的这项比较需要和实际减除比例一起阅读。"]
    lines += ["", "## 69022 的 8／16 像素窄条", "",
              "| 位置 | 宽度 | 污染区 MSE 输入／上版／本轮 | 相邻干净区 MSE 上版／本轮 | 全部干净区 MSE 上版／本轮 |",
              "|---|---|---:|---:|---:|"]
    for (_, name, kind, variant), p in pairs.items():
        if name != "69022" or kind not in ("edge8","edge16"):
            continue
        i, o, n = p["input"], p["previous"], p["current"]
        side = {"0":"左","2":"上","3":"下"}[variant]
        lines.append(f"| {side} | {kind[4:]} px | {i['contaminated_mse']:.6g}／{o['contaminated_mse']:.6g}／{n['contaminated_mse']:.6g} | "
                     f"{o['adjacent_clean_mse']:.4g}／{n['adjacent_clean_mse']:.4g} | {o['clean_mse']:.4g}／{n['clean_mse']:.4g} |")
    identical = sum(all(row[key] for key in ("output_equal","residual_equal","support_equal","route_equal","regions_equal"))
                    for row in parity)
    lines += ["", "三个边界的两种宽度在默认强度下逐值核对输出、实际减除量、支撑掩膜、路径及接受区间。"
              f"底边另核对 0 与 0.5 两档窄条强度，{identical}/{len(parity)} 组所有项目完全一致。"
              "相邻干净区取真实干净掩膜内距污染边界 16 像素以内的位置。[逐项核对](parity.csv)", "",
              "## 全集回归", "",
              "| 分组 | 输入 | 优于／持平／劣于上版 | 输出逐值相同 | 上版净收益保留 | 净优于输入 | 劣于输入 上版 → 本轮 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for group, s in stats.items():
        lines.append(f"| {group} | {s['inputs']} | {s['better_than_previous']}／{s['tied_with_previous']}／{s['worse_than_previous']} | "
                     f"{s['exact_same_as_previous']} | {s['retained_net_improvements']}/{s['previous_net_improvements']} | "
                     f"{s['current_net_improvements']} | {s['previous_worse_than_input']} → {s['current_worse_than_input']} |")
    controls = [p for k,p in pairs.items() if k[2] == "clean"]
    lines += ["", f"{len(controls)} 个无新增条纹对照中，至少一个像素被修改的个数为 "
              f"{sum(p['previous']['changed_pixels']>0 for p in controls)} → {sum(p['current']['changed_pixels']>0 for p in controls)}。"
              "比较整图 MSE 使用 1e−12 的持平公差。", ""]
    worse = [(" / ".join(k), p) for k,p in pairs.items()
             if p['current']['mse'] > p['previous']['mse']+plan['mse_tie_tolerance']]
    if worse:
        lines += ["本轮劣于上版的全部条件如下。", "", "| 输入 | 上版 MSE | 本轮 MSE |", "|---|---:|---:|"]
        lines += [f"| {k} | {p['previous']['mse']:.6g} | {p['current']['mse']:.6g} |" for k,p in worse]
    lines += ["", "## 复现", "", "使用同一数据和本地依赖，在新目录依次运行以下命令。", "", "```powershell",
              "python evaluate_weak83.py freeze --out results/weak83_reproduction",
              "python evaluate_weak83.py run --out results/weak83_reproduction", "```", "",
              "[逐图指标](metrics.csv) · [运行与哈希](run.json)"]
    (folder/"RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze","run"))
    parser.add_argument("--out", type=Path, default=ROOT/"results/weak83_reproduction")
    args = parser.parse_args()
    (freeze if args.stage == "freeze" else evaluate)(args.out)
