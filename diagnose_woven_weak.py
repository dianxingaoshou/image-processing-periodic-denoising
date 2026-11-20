"""Trace the frozen narrow/2-D router on five previous same-photo observations."""
import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

import denoiser_v3 as general
import project_stripes
from diagnose_weave import distance, frequency
from evaluate_narrow83 import PLAN, check_output, samples
from evaluate_weave import projected_residual, read_json, write_json
from image_io_v3 import sha256_file
from project_comparison import ROOT, array_hash
from project_pipeline import _agree_on_overlap, load_config
from project_texture import screen_tiles
from weave_guard import _frequency_distance, analyze_lattice


ARCHIVE = ROOT / "results/narrow83"


def old_cases():
    """Use exactly the previous study's images and generation, including clipping and seeds."""
    plan = read_json(PLAN)
    fixed = {c["name"]: c for c in plan["new_woven"]["carriers"]}
    for _, name, kind, variant, clean, obs, masks in samples(plan):
        if name != plan["target"] or variant not in ("", "A0.07"):
            continue
        if kind in fixed:
            dy, dx = fixed[kind]["fy"], fixed[kind]["fx"]
            field = [np.full(clean.shape, dy), np.full(clean.shape, dx)]
        elif kind == "clean":
            field = None
        else:
            field = [masks["true_fy"], masks["true_fx"]]
        yield kind, variant, clean, obs, field


def tile_records(prepared, gains, allowed, field, clean_fits, added):
    records = []
    for i, (tile, decision) in enumerate(zip(prepared.estimation.tiles, prepared.decisions)):
        y, x, h, w = (tile[k] for k in ("y", "x", "h", "w"))
        fit = None if tile["selected"] is None else frequency(tile["selected"])
        gate = {k: decision.get(k) for k in ("candidate_gate", "spectral_gate", "support_reason", "peak_score",
                "reliability_weight", "amplitude_gate", "extrapolation_gate", "safety_weight")}
        row = dict(y=y, x=x, selected_fy_fx=None if fit is None else fit.tolist(), gate=gate,
                   neighbor=tile["neighbor"], cross_window=tile["cross_window"],
                   gain=float(gains[i]), after_spatial_and_texture=bool(allowed[i]))
        if field is not None:
            target = np.array([f[y+h//2, x+w//2] for f in field])
            tolerance = 1 / min(w, h)
            matched = [c for c in tile["candidates"] if distance(frequency(c), target) <= tolerance]
            match = bool(fit is not None and distance(fit, target) <= tolerance)
            row.update(true_fy_fx=target.tolist(), matching_pool_scales=sorted({c["scale"] for c in matched}),
                       selected_in_tolerance=match,
                       selected_matches_clean=bool(fit is not None and clean_fits[i] is not None
                                                   and distance(fit, clean_fits[i]) <= tolerance))
            if match:
                patch = added[y:y+h, x:x+w]
                residual = tile["residual"]
                norm = np.linalg.norm(patch)*np.linalg.norm(residual)
                row["residual_added_cosine"] = float(np.sum(residual*patch)/norm) if norm > 0 else None
        records.append(row)
    return records


def retry_checks(obs, tiles, allowed, components, reviews):
    """Audit each existing broad-retry condition, including simultaneous failure reasons."""
    indices = np.flatnonzero(allowed)
    if not indices.size or not reviews:
        return []
    freqs = np.array([frequency(tiles[i]["selected"]) for i in indices])
    sizes = np.array([[tiles[i]["h"], tiles[i]["w"]] for i in indices])
    regs = [dict(height=h, width=w, frequency_y=fy, frequency_x=fx)
            for (fy, fx), (h, w) in zip(freqs, sizes)]
    peaks, _, _ = analyze_lattice(obs, regs)
    limit = project_stripes.SEPARATION_BINS
    audit = []
    for c in components:
        if c["reason"] != "accepted":
            continue
        members = c["tile_indices"]
        pos = np.searchsorted(indices, members)
        fs, sz = freqs[pos], sizes[pos]
        stationary = [tiles[i]["selected"]["model"] == "sinusoid" for i in members]
        locked = _frequency_distance(fs, fs[:1], sz[:, None])[:, 0] <= limit
        hits = _frequency_distance(fs, peaks, sz[:, None]).min(axis=1) <= limit if len(peaks) else np.zeros(len(fs), bool)
        conflicts = []
        for r in reviews:
            axis = 1-r["normal_axis"]
            for protected in [r["tangent_frequency"], *r["confirmed_companions"]]:
                separation = abs(abs(fs[:, axis])-protected) * sz[:, axis]
                conflicts.append(dict(protected_frequency=protected, tangent_axis="y" if axis==0 else "x",
                                      within_one_bin=int(np.sum(separation <= limit)),
                                      min_bins=float(separation.min()), max_bins=float(separation.max())))
        audit.append(dict(tiles=len(members), nonstationary=len(members)-sum(stationary), lattice_hits=int(hits.sum()),
                          unlocked=int(np.sum(~locked)), tangent_checks=conflicts))
    return audit


def component_extents(tiles, gains):
    """Record sizes and pixel spans even for components discarded by the old spatial gate."""
    grid = {(t["gy"], t["gx"]): i for i,t in enumerate(tiles) if gains[i] > 0}
    seen, records = set(), []
    for first in grid.values():
        if first in seen:
            continue
        members, queue = [], [first]
        seen.add(first)
        while queue:
            i = queue.pop()
            t = tiles[i]
            y, x = t["gy"], t["gx"]
            members.append(i)
            for point in ((y-1,x), (y+1,x), (y,x-1), (y,x+1)):
                j = grid.get(point)
                if j is not None and j not in seen and _agree_on_overlap(t, tiles[j]):
                    queue.append(j)
                    seen.add(j)
        dy = int(np.ptp([tiles[i]["y"] for i in members]))
        dx = int(np.ptp([tiles[i]["x"] for i in members]))
        records.append(dict(tiles=len(members), span_y=dy, span_x=dx,
                            sufficient=dy>=tiles[first]["h"] and dx>=tiles[first]["w"]))
    return sorted(records, key=lambda c: -c["tiles"])


def trace(kind, variant, clean, obs, field, clean_fits, cfg, expected):
    out, diag = project_stripes.denoise(obs, cfg, return_diagnostics=True)
    check_output(obs, out, diag)
    hashes = dict(reference_sha256=array_hash(clean), observed_sha256=array_hash(obs), output_sha256=array_hash(out))
    if any(expected[k] != v for k, v in hashes.items()) or expected["route"] != diag["route"] or expected["reason"] != diag["reason"]:
        raise ValueError(f"Historical inputs, output or decision differ on {kind}/{variant}")
    prepared = general.prepare(obs, cfg.general)
    gains = general.effective_gains(prepared.fit_weights, prepared.safety_weights, cfg.general.strength, cfg.general.combination)
    allowed, screen = screen_tiles(obs, prepared, gains)
    tiles = prepared.estimation.tiles
    extents = component_extents(tiles, gains)
    records = tile_records(prepared, gains, allowed, field, clean_fits, obs-clean)
    evidence = diag["evidence"]
    reviews = evidence.get("narrow_periodicity", [])
    if (int(allowed.sum()) != screen["spatially_eligible_tiles"]-screen["texture_vetoed_tiles"]
            or sum(c["tiles"] for c in extents if c["sufficient"]) != screen["spatially_eligible_tiles"]):
        raise ValueError("Inconsistent broad screening trace")
    if evidence["general_candidate_tiles"] != np.count_nonzero(gains) or evidence["general_corroborated_tiles"] != screen["spatially_eligible_tiles"]:
        raise ValueError("Independent stage reconstruction differs from actual routing")
    matches = [r for r in records if r.get("selected_in_tolerance")]
    counts = dict(tiles=len(tiles), positive_gain_all=int(np.count_nonzero(gains)),
                  spatial_all=screen["spatially_eligible_tiles"], texture_veto_all=screen["texture_vetoed_tiles"],
                  allowed_all=int(allowed.sum()), final_accepted=len(diag["accepted_regions"]))
    if field is not None:
        counts.update(proposed=sum(bool(r["matching_pool_scales"]) for r in records),
                      matched_per_scale={s:sum(s in r["matching_pool_scales"] for r in records) for s in cfg.general.estimation.scales},
                      selected=len(matches), selected_matches_clean=sum(r["selected_matches_clean"] for r in matches),
                      positive_gain_near_target=sum(r["gain"] > 0 for r in matches),
                      allowed_near_target=sum(r["after_spatial_and_texture"] for r in matches),
                      matching_gate_reasons=dict(Counter(r["gate"]["support_reason"] for r in matches)),
                      zero_gate_counts={g:sum(r["gate"][g] == 0 for r in matches) for g in
                      ("candidate_gate", "reliability_weight", "extrapolation_gate", "amplitude_gate", "safety_weight")},
                      matched_residual_cosines=[r["residual_added_cosine"] for r in matches])
    proj, off = projected_residual(obs-out, obs-clean) if field is not None else (None, None)
    rec = dict(kind=kind, variant=variant, **hashes, route=diag["route"], reason=diag["reason"],
               field_ranges=None if field is None else [[float(f.min()), float(f.max())] for f in field],
               global_abstain=prepared.global_assessment["abstain"], narrow_reviews=reviews,
               counts=counts, component_extents=extents, components=screen["components"],
               retry_checks=retry_checks(obs, tiles, allowed, screen["components"], reviews),
               tile_rows=records, input_mse=float(np.mean((obs-clean)**2)), output_mse=float(np.mean((out-clean)**2)),
               projection=proj, orthogonal_mse=off)
    fits = [None if t["selected"] is None else frequency(t["selected"]) for t in tiles]
    return rec, fits


def run(folder):
    dest = folder/"diagnosis.json"
    if dest.exists():
        raise FileExistsError(f"Choose a new diagnostic destination: {dest}")
    archive = ARCHIVE/"metrics.csv"
    if sha256_file(archive) != read_json(ARCHIVE/"run.json")["artifact_sha256"]["metrics.csv"]:
        raise ValueError("Historical study table changed")
    with archive.open(encoding="utf8", newline="") as f:
        expected = {(r["kind"], r["variant"]): r for r in csv.DictReader(f)
                    if r["image_id"] == "dtd_woven_0083" and r["method"] == "current"}
    rows, clean_fits = [], None
    for kind, variant, clean, obs, field in old_cases():
        row, fits = trace(kind, variant, clean, obs, field, clean_fits, load_config(), expected[kind,variant])
        if kind == "clean":
            clean_fits = fits
        rows.append(row)
    if len(rows) != 5:
        raise ValueError("Expected five historical cases")
    folder.mkdir(exist_ok=True, parents=True)
    write_json(dest, dict(comparator="project_stripes.denoise", source_sha256=sha256_file(Path(__file__)),
               archive_metrics_sha256=sha256_file(archive),
               units="(fy,fx), cycles/pixel. Match modulo conjugate sign within one synthesis bin of the true field at the tile center.",
               scope="Truth is used only after each model is fit, for scoring. A frequency match does not imply phase agreement.",
               records=rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT/"results/weak83_reproduction")
    run(p.parse_args().out)
