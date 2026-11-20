"""Paired timings of the two regular inference entries, excluding image generation and scoring."""
import argparse
import csv
import os
from pathlib import Path
import platform
from statistics import median
from time import perf_counter_ns

import numpy as np
import scipy

import collision_review
import peak_guard
from evaluate_nine83 import DIRECTIONS, old_focus, samples
from evaluate_weave import read_json, write_json
from image_io_v3 import sha256_file
from project_collision import denoise as current
from project_comparison import ROOT, array_hash
from project_guarded import denoise as guarded
from project_peak import denoise as previous


def inputs():
    plan = read_json(ROOT / "results/nine83/protocol.json")
    original_nine = old_focus(plan)
    extra = read_json(ROOT / "results/nine83/endpoint_controls.json")
    for case, _ in samples(plan, extra):
        cohort, name, kind, variant, _, observed, _ = case
        key = cohort, name, kind, str(variant)
        if cohort == DIRECTIONS and variant in ("axis_x_E@0.045", "axis_x_E@0.080"):
            group = "two_amplitudes"
        elif key in original_nine:
            group = "original_nine"
        elif name == "69022" and kind in ("edge8", "edge16") and str(variant) == "3":
            group = "narrow_edges"
        elif key == ("archived_photo", plan["target"], "clean", ""):
            group = "clean"
        elif cohort == DIRECTIONS and variant == "coupled_E@0.080":
            group = "prior_abstention"
        else:
            continue
        yield group, key, observed


def time_final_stage(cases, reference):
    """Time only the final reconstruction on the identical, already verified carrier."""
    stats = {}
    for group, key, observed in cases:
        if group != "two_amplitudes":
            continue
        base, diag = guarded(observed, return_diagnostics=True)
        def old_stage():
            return peak_guard.refine(observed, base, diag)[0]
        def new_stage():
            output, _, _ = collision_review.review(observed, diag)
            return old_stage() if output is None else output
        durations = {"previous": [], "current": []}
        methods = (("previous", old_stage), ("current", new_stage))
        for rep in range(13):
            for name, method in (methods if rep%2 else methods[::-1]):
                start = perf_counter_ns()
                output = method()
                elapsed = (perf_counter_ns()-start)/1e6
                if array_hash(output) != reference[*key,name]["output_sha256"]:
                    raise ValueError("Final stage differs from the regular entry")
                if rep:
                    durations[name].append(elapsed)
        stats[key[-1]] = {name+"_median_ms": median(values) for name, values in durations.items()}
    return stats


def benchmark(folder, rounds):
    if rounds < 2 or rounds % 2:
        raise ValueError("Use an even number of paired rounds")
    run = read_json(folder / "run.json")
    if sha256_file(folder / "metrics.csv") != run["artifact_sha256"]["metrics.csv"]:
        raise ValueError("Metrics differ from the completed study")
    with (folder / "metrics.csv").open(encoding="utf8", newline="") as f:
        reference = {(r["cohort"], r["image_id"], r["kind"], r["variant"], r["method"]): r
                     for r in csv.DictReader(f)}
    cases = list(inputs())
    assert len(cases) == 15
    methods = (("previous", previous), ("current", current))
    groups = sorted({group for group, _, _ in cases})
    totals = {g: np.zeros((rounds, 2)) for g in [*groups, "all_15"]}
    records = []
    # Warm up each method on each image. No I/O, hashing or scoring in the timed interval.
    for _, key, observed in cases:
        for name, method in methods:
            if array_hash(method(observed)) != reference[*key, name]["output_sha256"]:
                raise ValueError(f"Warm-up did not reproduce {key}, {name}")
    rng = np.random.default_rng(214077)
    for rep in range(rounds):
        for i in rng.permutation(len(cases)):
            group, key, observed = cases[i]
            order = methods if (rep+i) % 2 == 0 else methods[::-1]
            for place, (name, method) in enumerate(order):
                start = perf_counter_ns()
                restored = method(observed)
                elapsed = (perf_counter_ns()-start)/1e6
                if array_hash(restored) != reference[*key, name]["output_sha256"]:
                    raise ValueError(f"Timed output changed for {key}, {name}")
                records.append([rep+1, group, *key, place+1, name, elapsed])
                col = int(name == "current")
                totals[group][rep, col] += elapsed
                totals["all_15"][rep, col] += elapsed
    with (folder/"timing.csv").open("w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "group", "cohort", "image_id", "kind", "variant", "order", "method", "ms"])
        writer.writerows(records)
    summary = {}
    for group, t in totals.items():
        ratios = t[:, 1]/t[:, 0]
        summary[group] = dict(previous_median_ms=median(t[:, 0]), current_median_ms=median(t[:, 1]),
                              paired_median_ratio=median(ratios), paired_range=[min(ratios), max(ratios)])
    write_json(folder/"timing.json", dict(rounds=rounds, distinct_inputs=len(cases),
               environment=dict(cpu=os.environ.get("PROCESSOR_IDENTIFIER"), python=platform.python_version(),
                                numpy=np.__version__, scipy=scipy.__version__),
               measurements="Uncached normal API, one warm-up per input and method, balanced serial AB/BA, "
                            "per-round summed group times; ratio current/previous. Image loading and scoring excluded.",
               final_stage_12_pairs_ms=time_final_stage(cases, reference),
               source_sha256=sha256_file(Path(__file__).resolve()), metrics_sha256=sha256_file(folder/"metrics.csv"),
               timing_sha256=sha256_file(folder/"timing.csv"), groups=summary))
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT/"results/collision83")
    parser.add_argument("--rounds", type=int, default=6)
    args = parser.parse_args()
    benchmark(args.results, args.rounds)
