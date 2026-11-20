"""Freeze the local comparison and measure paired clean/noisy texture restoration."""
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from evaluate_project_refinement import extended_cases, identity
from image_io_v3 import sha256_file
from project_comparison import ROOT, array_hash, scores
from project_pipeline import load_config
from project_refined import denoise as previous


ARCHIVE = ROOT / "results/project_refinement"
PROTOCOL = ROOT / "weave_protocol.json"
METHODS = ("input", "previous", "current")
OUTPUTS = ("metrics.csv", "run.json", "RESULTS.md")


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf8")


def snapshot():
    from project_weave import routing_parameters

    archive = read_json(ARCHIVE / "run.json")
    for name, digest in archive["source_sha256"].items():
        if sha256_file(ROOT / name) != digest:
            raise ValueError(f"Historical baseline dependency changed: {name}")
    names = set(archive["source_sha256"]) | {"results/project_refinement/run.json",
            "results/project_refinement/metrics.csv", "project_weave.py", "weave_guard.py",
            "weave_carrier.py", "evaluate_weave.py", "weave_protocol.json"}
    config = json.loads(json.dumps(asdict(load_config()), allow_nan=False))
    return dict(source_sha256={name: sha256_file(ROOT/name) for name in sorted(names)},
                configuration=config, routing_parameters=routing_parameters(), protocol=read_json(PROTOCOL))


def freeze(destination):
    destination.mkdir(parents=True, exist_ok=True)
    if (destination/"freeze.json").exists() or any((destination/f).exists() for f in OUTPUTS):
        raise FileExistsError("Choose a new evaluation directory")
    write_json(destination/"freeze.json", snapshot())


def new_cases(references, protocol):
    """Check a fixed menu of additional excitations at both strengths and on both textiles."""
    for name in protocol["target_ids"]:
        clean = references[name]
        h, w = clean.shape
        y, x = np.indices((h, w), dtype=float)
        y -= (h-1)/2
        x -= (w-1)/2
        masks = dict(contaminated=np.ones(clean.shape, bool), clean=np.zeros(clean.shape, bool))
        for v in protocol["new_variants"]:
            phase = (v["fy"]*y + v["fx"]*x + v["dy"]*y*y/(2*(h-1)) + v["dx"]*x*x/(2*(w-1))
                     + v["cross"]*x*y/np.sqrt((h-1)*(w-1)))
            offset = np.random.default_rng(v["seed"]).uniform(-np.pi, np.pi)
            wave = np.sin(2*np.pi*phase + offset)
            for amp in protocol["amplitudes"]:
                observed = np.clip(clean + amp*wave, 0, 1)
                yield "new_realization", name, v["kind"], f'{v["name"]}@{amp:.2f}', clean, observed, masks


def projected_residual(residual, added):
    """Uncentered projection onto the actual post-clipping added signal."""
    energy = np.mean(added*added)
    if energy <= 0:
        raise ValueError("An added signal with positive energy is required")
    gain = float(np.mean(residual*added)/energy)
    off = residual - gain*added
    return gain, float(np.mean(off*off))


def summarize(pairs, tol):
    """Count gains against the input separately from improvements against the old method."""
    old = np.array([p["previous"]["mse"] for p in pairs])
    new = np.array([p["current"]["mse"] for p in pairs])
    inp = np.array([p["input"]["mse"] for p in pairs])
    old_ok, new_ok = old < inp-tol, new < inp-tol
    return dict(inputs=len(pairs), better_than_previous=int(sum(new<old-tol)),
                tied_with_previous=int(sum(np.abs(new-old)<=tol)), worse_than_previous=int(sum(new>old+tol)),
                exact_same_as_previous=sum(p["previous"]["output_sha256"] == p["current"]["output_sha256"]
                                           for p in pairs),
                previous_net_improvements=int(old_ok.sum()), retained_net_improvements=int((old_ok & new_ok).sum()),
                new_net_improvements=int((new_ok & ~old_ok).sum()), current_net_improvements=int(new_ok.sum()),
                previous_worse_than_input=int(sum(old>inp+tol)), current_worse_than_input=int(sum(new>inp+tol)),
                input_mean_mse=float(inp.mean()), previous_mean_mse=float(old.mean()), current_mean_mse=float(new.mean()))


def write_report(destination, protocol, pairs, stats):
    targets = set(protocol["target_ids"])
    old_key = lambda name: next(k for k in pairs if k[1] == name and k[2] == "clean")
    lines = ["# 针织与编织纹理的同图配对验证", "",
             "比较上次本地版与本轮流程。原76个输入逐一核对了上次结果的观测、参考和输出哈希、"
             "处理路径及区域质量指标。新增16个输入复用这两张照片，在算法冻结后使用事先登记的"
             "四种载波、相位种子与两档强度；这些是新退化条件，照片本身已用于诊断。", "",
             "## 干净原图", "", "| 纹理 | 上版MAE | 本轮MAE | 修改像素，上版→本轮 |",
             "|---|---:|---:|---:|"]
    for name in sorted(targets):
        p = pairs[old_key(name)]
        b, c = p["previous"], p["current"]
        lines.append(f"| {name} | {b['mae']:.6f} | {c['mae']:.6f} | {b['changed_pixels']}→{c['changed_pixels']} |")
    lines += ["", "## 同图加噪", "",
              "令 X 为原始照片，Y 为裁剪后的含噪输入，F 为待评方法。实际新增干扰 n=Y−X，"
              "实际减除量 r=Y−F(Y)。投影系数 a=〈r,n〉/〈n,n〉刻画沿实际新增干扰方向减去的比例，"
              "0 表示该方向没有减除，1 表示该方向足量减除。正交误差 e=mean((r−a·n)²) "
              "同时反映错误分量和波形不匹配。整图MSE恰为 (1−a)²·mean(n²)+e。"
              "因此仅有较大 a 仍可能损坏纹理。下表的均方误差使用 [0,1] 像素范围。", "",
              "另将 r 与对未加噪同图的减除量 X−F(X) 作差，得到新增干扰引起的响应变化。"
              "逐图记录中 `paired_gain` 与 `paired_off_noise_mse` 使用这个差值计算，"
              "应结合干净原图的MAE判断。", ""]
    for is_new, title in ((False, "已有0.12幅值回归"), (True, "新载波及两档幅值")):
        lines += [f"### {title}", "",
                  "| 纹理 | 条件 | 输入MSE | 上版MSE | 本轮MSE | a 上版→本轮 | 正交误差 上版→本轮 |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for (cohort, name, kind, variant), p in pairs.items():
            if (cohort == "new_realization") != is_new or name not in targets or kind == "clean":
                continue
            b, c = p["previous"], p["current"]
            lines.append(f"| {name.split('_')[1]} | {variant or kind} | {p['input']['mse']:.6f} | "
                         f"{b['mse']:.6f} | {c['mse']:.6f} | {b['noise_gain']:.3f}→{c['noise_gain']:.3f} | "
                         f"{b['off_noise_mse']:.6f}→{c['off_noise_mse']:.6f} |")
        lines.append("")
    lines += ["## 原有去条纹效果与损伤", "",
              "表中“净改善”指相对各自的含噪输入，整图MSE至少下降超过1e−12。"
              "“保留”统计上版已净改善的案例里本轮也有净改善的数量。", "",
              "| 组别 | 输入数 | 对上版更好／持平／更差 | 输出逐像素不变 | 保留／上版净改善 | 本轮净改善 | 比输入更差，上版→本轮 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for group, label in (("historical","完整历史回归"),("historical_targets","目标两图历史条件"),
                         ("historical_others","其余历史输入"),("new","新载波检查")):
        s = stats[group]
        lines.append(f"| {label} | {s['inputs']} | {s['better_than_previous']}／{s['tied_with_previous']}／"
                     f"{s['worse_than_previous']} | {s['exact_same_as_previous']} | "
                     f"{s['retained_net_improvements']}／{s['previous_net_improvements']} | {s['current_net_improvements']} | "
                     f"{s['previous_worse_than_input']}→{s['current_worse_than_input']} |")
    lines += ["", "69022 的8／16／32像素边缘条纹在本轮的含噪区MSE依次为 "
              + "、".join(f"{pairs[('regression','69022',f'edge{w}','3')]['current']['contaminated_mse']:.6f}"
                        for w in (8,16,32)) + "，相邻干净区MSE依次为 "
              + "、".join(f"{pairs[('regression','69022',f'edge{w}','3')]['current']['adjacent_clean_mse']:.4g}"
                        for w in (8,16,32)) + "。", "",
              "每类纹理在此处只有一张照片，相同照片的不同合成退化不能增加独立纹理样本数。"
              "所有恢复仅接收观测图与固定配置；已知频率、相位、参考图及掩膜仅用于构造输入和评价。", "",
              "详细记录见[逐图指标](metrics.csv)、[来源冻结](freeze.json)和[输入及核验](run.json)。", ""]
    (destination/"RESULTS.md").write_text("\n".join(lines), encoding="utf8")


def comparison(destination):
    sealed = read_json(destination/"freeze.json")
    if snapshot() != sealed:
        raise ValueError("The freeze does not match current sources or inputs")
    if any((destination/f).exists() for f in OUTPUTS):
        raise FileExistsError("Choose a new output directory and freeze it first")
    from project_weave import denoise as current

    protocol = sealed["protocol"]
    targets = set(protocol["target_ids"])
    cfg = load_config()
    archive_inputs = {identity(r): r for r in read_json(ARCHIVE/"run.json")["cases"]}
    with (ARCHIVE/"metrics.csv").open(encoding="utf8", newline="") as f:
        archive_rows = {identity(r): r for r in csv.DictReader(f) if r["method"] == "refined"}
    if len(archive_rows) != protocol["historical_cases"] or archive_inputs.keys() != archive_rows.keys():
        raise RuntimeError("Incomplete historical inventory")
    references, clean_residuals = {}, {}
    def inputs():
        for case in extended_cases():
            if case[1] in targets and case[2] == "clean":
                references[case[1]] = case[4]
            yield case
        yield from new_cases(references, protocol)

    rows, records, pairs, checked = [], [], {}, set()
    for cohort, name, kind, variant, clean, observed, masks in inputs():
        key = dict(cohort=cohort, image_id=name, kind=kind, variant=str(variant))
        ident = identity(key)
        if ident in pairs:
            raise RuntimeError(f"Duplicate input: {ident}")
        src = key | dict(reference_sha256=array_hash(clean), observed_sha256=array_hash(observed))
        if cohort != "new_realization":
            if any(archive_inputs[ident][f] != src[f] for f in ("reference_sha256","observed_sha256")):
                raise AssertionError(f"Historical inputs changed for {ident}")
            checked.add(ident)
        pairs[ident] = {}
        added = observed-clean
        for method, algorithm in zip(METHODS, (None, previous, current)):
            output, dia = (observed, {}) if algorithm is None else algorithm(observed, cfg, return_diagnostics=True)
            if algorithm is not None:
                support = dia["support_mask"]
                if (support.shape != observed.shape or support.dtype != np.bool_
                        or not np.array_equal(output[~support], observed[~support])
                        or not np.array_equal(dia["residual"], observed-output)):
                    raise AssertionError(f"Invalid support or residual: {ident}, {method}")
            if method == "current" and name in targets:
                evidence = dia.get("evidence", {})
                src["lattice_screen"] = evidence.get("lattice_screen")
                src["alternative_carrier"] = evidence.get("alternative_carrier")
            row = key | dict(method=method, route=dia.get("route",""), reason=dia.get("reason",""),
                             output_sha256=array_hash(output)) | scores(clean, observed, output, masks)
            row.update(noise_gain="", off_noise_mse="", paired_gain="", paired_off_noise_mse="")
            if name in targets:
                correction = observed-output
                if kind == "clean":
                    clean_residuals[name,method] = correction
                else:
                    gain, off_mse = projected_residual(correction, added)
                    row["noise_gain"], row["off_noise_mse"] = gain, off_mse
                    row["paired_gain"], row["paired_off_noise_mse"] = projected_residual(
                        correction-clean_residuals[name,method], added)
                    if not np.isclose(row["mse"], (1-gain)**2 * np.mean(added*added) + off_mse,
                                      atol=2e-15, rtol=1e-12):
                        raise AssertionError("Post-clipping projection does not reconstruct the observed MSE")
            if ident in checked and method == "previous":
                prior = archive_rows[ident]
                if any(row[f] != prior[f] for f in ("output_sha256","route","reason")):
                    raise AssertionError(f"Historical output changed for {ident}")
                for field in ("mse","mae","psnr_db","ssim","clean_mse","adjacent_clean_mse",
                              "contaminated_mse","broad_only_mse","narrow_only_mse","changed_pixels"):
                    value, old = row[field], prior[field]
                    if value == "" or old == "":
                        equal = value == old
                    else:
                        equal = np.isclose(float(value), float(old), atol=1e-14, rtol=1e-12)
                    if not equal:
                        raise AssertionError(f"{ident}: changed historical metric {field}")
            pairs[ident][method] = row
            rows.append(row)
        records.append(src)
        print(len(records), name, kind, variant, pairs[ident]["current"]["route"], flush=True)
    n_new = len(targets)*len(protocol["new_variants"])*len(protocol["amplitudes"])
    if checked != archive_inputs.keys() or len(records) != len(checked)+n_new:
        raise AssertionError("Incomplete paired evaluation")
    if snapshot() != sealed:
        raise RuntimeError("A frozen comparison dependency changed during evaluation")
    groups = dict(historical=[p for k,p in pairs.items() if k in checked],
                  historical_others=[p for k,p in pairs.items() if k in checked and k[1] not in targets],
                  historical_targets=[p for k,p in pairs.items() if k in checked and k[1] in targets],
                  new=[p for k,p in pairs.items() if k not in checked])
    for target in sorted(targets):
        groups["new_"+target] = [p for k,p in pairs.items() if k not in checked and k[1] == target]
    stats = {name: summarize(p, protocol["mse_tie_tolerance"]) for name,p in groups.items()}
    with (destination/"metrics.csv").open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    report = dict(inputs=len(records), rows=len(rows), archived_outputs_verified=len(checked),
                  freeze_sha256=sha256_file(destination/"freeze.json"),
                  metrics_sha256=sha256_file(destination/"metrics.csv"), groups=stats, cases=records)
    write_json(destination/"run.json", report)
    write_report(destination, protocol, pairs, stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "run"))
    parser.add_argument("--out", type=Path, default=ROOT/"results/weave_reproduction")
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.out)
    else:
        comparison(args.out)
