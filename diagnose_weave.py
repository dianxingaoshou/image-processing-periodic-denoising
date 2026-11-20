"""Explain the previous local pipeline on six fixed paired DTD inputs (evaluation only)."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

import denoiser_v3 as general
from image_io_v3 import sha256_file
from project_comparison import AMP, ROOT, array_hash, general_sample
from project_directional import narrow_residual
from project_pipeline import load_config
from project_refined import _restore_general
from project_texture import AMBIGUOUS_FRACTION_LIMIT, HARMONIC_SCORE_LIMIT, STRUCTURE_LOSS_LIMIT, screen_tiles


def frequency(candidate):
    return np.array([candidate["fy"], candidate["fx"]])


def distance(a, b):
    return min(np.linalg.norm(a-b), np.linalg.norm(a+b))


def candidate_coverage(tiles, indices, truth, clean_choices):
    """Count candidate availability separately from the final frequency choice."""
    proposed = chosen = prior = 0
    for i in indices:
        t = tiles[i]
        y, x = t["y"] + t["h"]//2, t["x"] + t["w"]//2
        target = np.array([truth["true_fy"][y, x], truth["true_fx"][y, x]])
        tolerance = 1 / min(t["h"], t["w"])
        proposed += any(distance(frequency(c), target) <= tolerance for c in t["candidates"])
        if t["selected"] is not None:
            f = frequency(t["selected"])
            chosen += int(distance(f, target) <= tolerance)
            prior += int(clean_choices[i] is not None and distance(f, clean_choices[i]) <= tolerance)
    return dict(tiles=len(indices), pool_contains_added=proposed, selected_added=chosen,
                selected_matches_same_tile_clean=prior)


def center_example(tiles, shape, truth):
    """Use the image-center tile in every case. Merge near-identical proposals across scales."""
    h, w = shape
    t = min(tiles, key=lambda t: (t["y"] + t["h"]/2-h/2)**2 + (t["x"] + t["w"]/2-w/2)**2)
    modes = []
    for c in sorted(t["candidates"], key=lambda c: -c["selection_score"]):
        f = frequency(c)
        if all(distance(f, np.asarray(m["fy_fx"])) > 1/min(t["h"], t["w"]) for m in modes):
            modes.append(dict(fy_fx=f.tolist(), local_amplitude=c["local_amplitude"]))
    y, x = t["y"] + t["h"]//2, t["x"] + t["w"]//2
    return dict(center_yx=[y, x], selected_fy_fx=frequency(t["selected"]).tolist(),
                distinct_modes=modes, true_fy_fx=[truth["true_fy"][y, x], truth["true_fx"][y, x]]
                if "true_fx" in truth else None)


def distortion(clean, observed, restored):
    added, removed = observed-clean, observed-restored
    noise_mse = np.mean(added**2)
    measured = dict(input_mse=float(noise_mse), restored_mse=float(np.mean((restored-clean)**2)),
                    restored_mae=float(np.mean(np.abs(restored-clean))), removed_rms=float(np.sqrt(np.mean(removed**2))))
    if noise_mse > 0:
        projection = float(np.mean(removed*added) / noise_mse)
        off_power = float(np.mean((removed-projection*added)**2))
        if not np.isclose(measured["restored_mse"], (1-projection)**2*noise_mse+off_power, atol=1e-14, rtol=1e-12):
            raise ValueError("Paired residual decomposition mismatch")
        measured.update(removed_projection_on_added=projection, nonaligned_removed_rms=float(np.sqrt(off_power)))
    return measured


def compare_case(clean, observed, truth, cfg, clean_choices):
    _, support, _ = narrow_residual(observed, cfg.narrow)
    prepared = general.prepare(observed, cfg.general)
    if support is not None or prepared.global_assessment["abstain"]:
        raise ValueError("Expected the six baseline cases to reach the general route")
    gains = general.effective_gains(prepared.fit_weights, prepared.safety_weights, cfg.general.strength, cfg.general.combination)
    allowed, screen = screen_tiles(observed, prepared, gains)
    tiles = prepared.estimation.tiles
    result, _, _ = _restore_general(observed, tiles, gains, allowed, False)
    record = dict(observed_sha256=array_hash(observed), output_sha256=array_hash(result),
                  narrow_supported_pixels=0, global_ambiguity_risk=prepared.global_assessment["ambiguity_risk"],
                  positive_gain_tiles=int(np.count_nonzero(gains)), total_tiles=len(tiles),
                  spatially_eligible_tiles=screen["spatially_eligible_tiles"], texture_vetoed_tiles=screen["texture_vetoed_tiles"],
                  accepted_tiles=int(allowed.sum()), components=[], example=center_example(tiles, observed.shape, truth),
                  paired_errors=distortion(clean, observed, result))
    for c in screen["components"]:
        indices = c["tile_indices"]
        decs = [prepared.decisions[i] for i in indices]
        record["components"].append(dict(tiles=len(indices), decision=c["reason"],
            median_harmonic_score=c["median_harmonic_score"], near_pure_loss_fraction=c["near_pure_loss_fraction"],
            median_local_periodic_fraction=float(np.median([d["local_periodic_fraction"] for d in decs])),
            median_structure_loss=float(np.median([d["structure_loss"] for d in decs]))))
    if "true_fy" in truth:
        record["all_tile_coverage"] = candidate_coverage(tiles, range(len(tiles)), truth, clean_choices)
        record["accepted_tile_coverage"] = candidate_coverage(tiles, np.flatnonzero(allowed), truth, clean_choices)
    return record, [frequency(t["selected"]) if t["selected"] is not None else None for t in tiles]


def diagnose(destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Choose a fresh output file: {destination}")
    manifest = json.loads((ROOT / "data/v3_manifest.json").read_text(encoding="utf8"))
    archive = ROOT / "results/project_refinement/metrics.csv"
    with archive.open(encoding="utf8", newline="") as f:
        expected = {(r["image_id"], r["kind"]): r["output_sha256"] for r in csv.DictReader(f) if r["method"] == "refined"}
    cfg = load_config()
    records = []
    for r in manifest["textures"]:
        if r["category"] not in ("knitted", "woven"):
            continue
        path = ROOT/"data"/r["npy_path"]
        if sha256_file(path) != r["clean_sha256"]:
            raise ValueError(f"Unexpected texture file: {path}")
        clean = np.load(path, allow_pickle=False)
        clean_choices = None
        for kind in ("clean", "global", "chirp"):
            observed, meta, truth = general_sample(clean, r["id"], kind, 0 if kind == "clean" else AMP)
            record, selected = compare_case(clean, observed, truth, cfg, clean_choices)
            if record["output_sha256"] != expected[r["id"], kind]:
                raise ValueError("Reconstructed baseline differs from the last local run")
            if kind == "clean":
                clean_choices = selected
            record.update(image_id=r["id"], kind=kind, seed=meta["seed"], reference_sha256=array_hash(clean))
            records.append(record)
    if len(records) != 6:
        raise ValueError("Expected two texture photographs, each with three conditions")
    audit = dict(cases=records, verified_against="results/project_refinement/metrics.csv; exact hashes for six outputs",
                 prior_metrics_sha256=sha256_file(archive), source_sha256=sha256_file(Path(__file__)),
                 frequency="cycles/pixel; local field sampled at (origin + size//2); distance modulo conjugate sign; "
                           "within 1/min(tile height,width), 1/64 in this audit",
                 projection="removed vs actual (post-clipping) addition. Orthogonal energy also includes waveform estimation error.",
                 old_thresholds=dict(harmonic_score=HARMONIC_SCORE_LIMIT, periodic_fraction=cfg.general.guard.local_ambiguity_high,
                                     structure_loss=STRUCTURE_LOSS_LIMIT, minimum_ambiguous_fraction=AMBIGUOUS_FRACTION_LIMIT))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, ensure_ascii=False, allow_nan=False, indent=2)+"\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"results/weave_reproduction/diagnosis.json")
    diagnose(parser.parse_args().out)
