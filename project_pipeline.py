"""Single-image restoration using compact directional support or a corroborated 2-D fit."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import denoiser_v3 as general
import narrow_denoiser as narrow
from local_estimation_v3 import restore as restore_tiles
from texture_guard_v3 import assess_global


_CONFIGS = Path(__file__).resolve().parent / "configs"
DEFAULT_GENERAL_CONFIG = _CONFIGS / "v3_p0_coupled.json"
DEFAULT_NARROW_CONFIG = _CONFIGS / "context_context_full.json"
MIN_OVERLAP_AGREEMENT = 0.8


@dataclass(frozen=True)
class ProjectConfig:
    general: general.Config
    narrow: narrow.Config


def load_config() -> ProjectConfig:
    """Load the same versioned parameters used by the two component baselines."""
    def read(path):
        return json.loads(path.read_text(encoding="utf-8-sig"))["configuration"]

    return ProjectConfig(general.config_from_dict(read(DEFAULT_GENERAL_CONFIG)),
                         narrow.config_from_dict(read(DEFAULT_NARROW_CONFIG)))


def _agree_on_overlap(a, b):
    y0, x0 = max(a["y"], b["y"]), max(a["x"], b["x"])
    y1 = min(a["y"] + a["h"], b["y"] + b["h"])
    x1 = min(a["x"] + a["w"], b["x"] + b["w"])
    if y1 <= y0 or x1 <= x0:
        return False

    def overlap(tile):
        return tile["residual"][y0-tile["y"]:y1-tile["y"], x0-tile["x"]:x1-tile["x"]]

    first, second = overlap(a), overlap(b)
    energy = np.sum(first * first) + np.sum(second * second)
    # Agreement is sensitive to waveform sign, phase and amplitude.
    return energy > 0 and 2 * np.sum(first * second) >= MIN_OVERLAP_AGREEMENT * energy


def corroborated_tiles(tiles, gains):
    """Keep connected fits spanning a block displacement along both image axes."""
    by_grid = {(t["gy"], t["gx"]): i for i, t in enumerate(tiles) if gains[i] > 0}
    edges = {i: [] for i in by_grid.values()}
    for (y, x), i in by_grid.items():
        for neighbor in ((y + 1, x), (y, x + 1)):
            j = by_grid.get(neighbor)
            if j is not None and _agree_on_overlap(tiles[i], tiles[j]):
                edges[i].append(j)
                edges[j].append(i)

    keep = np.zeros(len(tiles), dtype=bool)
    seen = set()
    for i in edges:
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for j in edges[current]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        # The estimator uses a common synthesis shape, including edge tiles.
        # Actual pixel origins matter when the last stride is shortened.
        span_y = np.ptp([tiles[j]["y"] for j in component])
        span_x = np.ptp([tiles[j]["x"] for j in component])
        if span_y >= tiles[i]["h"] and span_x >= tiles[i]["w"]:
            keep[component] = True
    return keep


def denoise(image, config=None, *, return_diagnostics=False):
    """Restore one observed [0,1] gray image. Every route uses the original pixels."""
    cfg = load_config() if config is None else config
    directional = narrow.prepare(image, cfg.narrow)
    observed = directional.image
    narrow_result, profile = narrow.restore(directional, return_diagnostics=True)
    narrow_support = profile["support_mask"]
    has_narrow = bool(narrow_support.any())
    regions = []
    if has_narrow and return_diagnostics:
        for p in profile["profiles"]:
            for interval in p["intervals"]:
                if interval["accepted"]:
                    regions.append(dict(normal_axis=p["axis"], start=interval["start"], stop=interval["stop"],
                                        context_start=interval["context_start"], context_stop=interval["context_stop"],
                                        tangent_frequency=p["frequency_cycles_per_pixel"]))
    # Keep only the narrow proposal and its mask while fitting larger 2-D windows.
    del directional, profile
    broad = None
    if has_narrow:
        assessment = assess_global(observed, cfg.general.guard)
    else:
        del narrow_result
        broad = general.prepare(observed, cfg.general)
        assessment = broad.global_assessment

    result = observed
    support = np.zeros(observed.shape, dtype=bool)
    route = "unchanged"
    evidence = dict(narrow_supported_pixels=int(narrow_support.sum()),
                    minimum_overlap_agreement=MIN_OVERLAP_AGREEMENT,
                    general_candidate_tiles=None, general_corroborated_tiles=None)
    if assessment["abstain"]:
        reason = "global_periodic_ambiguity"
        regions.clear()
    elif has_narrow:
        route, reason = "narrow", "compact_directional_support"
        result, support = narrow_result, narrow_support
    else:
        tiles = broad.estimation.tiles
        gains = general.effective_gains(broad.fit_weights, broad.safety_weights,
                                        cfg.general.strength, cfg.general.combination)
        keep = corroborated_tiles(tiles, gains)
        evidence.update(general_candidate_tiles=int(np.count_nonzero(gains)),
                        general_corroborated_tiles=int(keep.sum()))
        if keep.any():
            route, reason = "general", "spatially_corroborated_tiles"
            # Preserve the original overlap denominator, including rejected blocks.
            result = restore_tiles(broad.estimation, weights=np.where(keep, gains, 0.0))
            if return_diagnostics:
                for i in np.flatnonzero(keep):
                    t = tiles[i]
                    support[t["y"]:t["y"]+t["h"], t["x"]:t["x"]+t["w"]] = True
                    c = t["selected"]
                    regions.append(dict(y=t["y"], x=t["x"], height=t["h"], width=t["w"],
                                        scale=c["scale"], frequency_y=c["fy"], frequency_x=c["fx"],
                                        gain=float(gains[i])))
        else:
            reason = "insufficient_spatial_support"
    if not return_diagnostics:
        return result
    return result, dict(route=route, reason=reason, global_assessment=assessment, evidence=evidence,
                        accepted_regions=regions, residual=observed-result, support_mask=support)
