"""Replay the old woven/69022 routes and isolate the effect of narrow-first routing."""
import argparse
import csv
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

import project_refined
import project_weave
from evaluate_weave import projected_residual, write_json
from image_io_v3 import preprocess_image, sha256_file
from narrow_profiles import _make_profile
from narrow_support import build_support
from project_comparison import AMP, ROOT, array_hash, cases, general_sample
from project_directional import _frequencies
from project_pipeline import load_config


def inputs():
    name = "woven_0083"
    ref, _ = preprocess_image(ROOT / f"data/textures_v3/source/woven/{name}.jpg", size=256)
    for kind in ("clean", "global", "chirp"):
        observed, _, masks = general_sample(ref, "dtd_" + name, kind, 0 if kind == "clean" else AMP)
        yield name, kind, ref, observed, masks
    for cohort, name, kind, _, ref, observed, masks in cases():
        if cohort == "regression" and name == "69022" and kind in ("edge8", "edge16"):
            yield name, kind, ref, observed, masks


def historical_hashes():
    archived = {}
    for filename in ("photos_metrics.csv", "metrics.csv"):
        with (ROOT / "results/weave" / filename).open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                name = Path(row["photo"]).stem if "photo" in row else row["image_id"]
                if name in ("woven_0083", "69022") and row["method"] in ("input", "current"):
                    archived[name, row["kind"], row["method"]] = row["output_sha256"]
    return archived


def proposals(image, cfg):
    """Recompute split scores on each proposal, including rejected proposals normally pruned early."""
    candidates = []
    for axis, rows in enumerate((image, image.T)):
        for rank, frequency in enumerate(_frequencies(rows, cfg.profiles), start=1):
            profile = _make_profile(rows, axis, frequency)
            support = build_support(profile, cfg.support, use_gate=cfg.use_gate, compact_support=cfg.compact_support)
            intervals = support["intervals"]
            accepted = []
            for interval in intervals:
                if not interval["accepted"]:
                    continue
                idx = np.arange(len(rows))
                far = ((idx < interval["context_start"] - cfg.support.flank_width)
                       | (idx >= interval["context_stop"] + cfg.support.flank_width))
                amplitudes = profile["amplitude"][far]
                half = interval["median_amplitude"] / 2
                accepted.append(dict(interval, far_lines=int(far.sum()),
                                     far_lines_at_least_half_amplitude=int((amplitudes >= half).sum())))
            candidates.append(dict(normal_axis=axis, rank=rank, tangent_frequency=frequency,
                                   qualified_lines=support["qualified_rows"],
                                   interval_reasons=dict(Counter(r["reason"] for r in intervals)),
                                   accepted=accepted))
    return candidates


def outcome(clean, observed, result, decision, masks):
    residual = observed - result
    changed = residual != 0
    if np.any(changed & ~decision["support_mask"]):
        raise AssertionError("Old router changed a pixel outside reported support")
    added = observed - clean
    gain, off = projected_residual(residual, added) if added.any() else (None, None)
    metrics = dict(route=decision["route"], reason=decision["reason"], sha256=array_hash(result),
                   mse=float(np.mean((result-clean)**2)), changed_pixels=int(changed.sum()),
                   projection=gain, orthogonal_mse=off,
                   narrow_plane_rmse=decision["evidence"].get("narrow_plane_rmse"),
                   global_abstain=bool(decision["global_assessment"]["abstain"]))
    for region in ("contaminated", "clean", "adjacent_clean"):
        mask = masks.get(region)
        if mask is not None and mask.any():
            metrics[region + "_mse"] = float(np.mean((result[mask] - clean[mask])**2))
    regions = decision["accepted_regions"]
    if decision["route"] == "narrow":
        metrics["selected_regions"] = regions
    elif decision["route"] == "general":
        metrics["tiles"] = len(regions)
    return metrics


def run(destination):
    cfg = load_config()
    archived = historical_hashes()
    records, woven_residuals = [], {}
    for name, kind, clean, observed, masks in inputs():
        old, decision = project_weave.denoise(observed, cfg, return_diagnostics=True)
        if (array_hash(observed) != archived[name, kind, "input"]
                or array_hash(old) != archived[name, kind, "current"]):
            raise ValueError(f"Historical output/input hash disagrees on {name} {kind}")
        # Offline intervention at the previous router's branch point, scoped to this call only.
        with patch.object(project_refined, "narrow_residual", return_value=(None, None, [])):
            broad, broad_decision = project_weave.denoise(observed, cfg, return_diagnostics=True)
        candidates = proposals(observed, cfg.narrow)
        traced = sorted((r["normal_axis"], i["start"], i["stop"], r["tangent_frequency"])
                        for r in candidates for i in r["accepted"])
        selected = sorted((r["normal_axis"], r["start"], r["stop"], r["tangent_frequency"])
                          for r in decision["accepted_regions"] if "normal_axis" in r)
        if traced != selected:
            raise AssertionError(f"Detailed narrow trace disagrees with baseline on {name} {kind}")
        truth = {key: [float(masks[key].min()), float(masks[key].max())]
                 for key in ("true_fy", "true_fx") if key in masks}
        records.append(dict(image=name, kind=kind, clean_sha256=array_hash(clean), observed_sha256=array_hash(observed),
                            frequency_ranges_cycles_per_pixel=truth, input_mse=float(np.mean((observed-clean)**2)),
                            old=outcome(clean, observed, old, decision, masks),
                            skip_narrow=outcome(clean, observed, broad, broad_decision, masks),
                            candidates=candidates))
        if name == "woven_0083":
            woven_residuals[kind] = observed - old
    raw, noisy = (woven_residuals[k] for k in ("clean", "global"))
    cosine = float(np.sum(raw*noisy)/np.sqrt(np.sum(raw**2)*np.sum(noisy**2)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json(destination, dict(baseline="project_weave.denoise", script_sha256=sha256_file(Path(__file__)),
               same_photo_clean_global_residual_cosine=cosine, hash_checks="5 historical input and output pairs",
               controls="skip_narrow disables the previous directional proposal for one diagnostic call only",
               axes="0: index rows and fit along x; 1: index columns and fit along y. Intervals are zero-based, half-open.",
               far_lines="outside amplitude context plus flank_width on either side; check same-frequency amplitude >= half the supported median",
               records=records))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/narrow83/diagnosis.json")
    run(parser.parse_args().out)
