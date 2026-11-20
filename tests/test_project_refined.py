"""Contracts specific to the refined local entry point."""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import filter_project
import project_refined as refined
from context_controls import make_controls
from local_estimation_v3 import restore as restore_tiles


def test_band_and_surroundings_both_have_to_match_one_plane():
    y, x = np.indices((128, 160))
    mask = y < 4
    field = .3 + .0004 * x + .0007 * y
    assert max(refined._plane_errors(field, mask).values()) < 1e-12
    field[mask] += .012 * np.cos(2*np.pi*x[mask]/8)
    errors = refined._plane_errors(field, mask)
    assert errors["outside"] < refined.PLANE_RMS_LIMIT < errors["inside"]
    assert refined._plane_errors(field.T, mask.T) == pytest.approx(errors)


@pytest.mark.parametrize("strength", [1., .5, .001, 0.])
def test_ambiguity_decision_does_not_depend_on_requested_strength(strength):
    _, _, observed, _, _ = next(case for case in make_controls(128) if case[0] == "compact_legitimate_stripe_clean")
    cfg = refined.load_config()
    cfg = replace(cfg, narrow=replace(cfg.narrow, strength=strength))
    output, diag = refined.denoise(observed, cfg, return_diagnostics=True)
    assert diag["reason"] == "narrow_affine_background_ambiguity"
    np.testing.assert_array_equal(output, observed)


def test_fractional_narrow_gain_clips_after_scaling_the_raw_wave(monkeypatch):
    obs = np.full((70, 80), .11)
    obs[:, 40:] = .75
    support = np.zeros(obs.shape, dtype=bool)
    support[:6, :] = True
    noise = .4 * support
    monkeypatch.setattr(refined, "narrow_residual", lambda *a, **k: (noise, support, []))
    cfg = refined.load_config()
    cfg = replace(cfg, narrow=replace(cfg.narrow, strength=.5))
    output, diag = refined.denoise(obs, cfg, return_diagnostics=True)
    assert diag["route"] == "narrow"
    np.testing.assert_array_equal(output, np.clip(obs - .5*noise, 0, 1))
    support[:] = True
    _, diag = refined.denoise(obs, cfg, return_diagnostics=True)
    assert diag["reason"] == "no_narrow_background"


def test_small_image_and_constant_field_skip_broad_fitting(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("This geometry or input cannot produce an eligible broad component")
    monkeypatch.setattr(refined.general, "prepare", unexpected)
    x = np.linspace(.3, .6, 110)
    for image, reason in ((np.broadcast_to(x, (80, 110)), "insufficient_image_extent"),
                          (np.full((150, 200), .3), "constant_image")):
        result, diag = refined.denoise(image, return_diagnostics=True)
        assert diag["reason"] == reason
        np.testing.assert_array_equal(result, image)
        assert not np.shares_memory(result, image)


def test_light_overlap_reconstruction_matches_original_at_edges_and_clipping():
    rng = np.random.default_rng(12)
    image = rng.uniform(size=(50, 76))
    tiles = []
    for y in (0, 16, 18):
        for x in (0, 16, 32, 44):
            selected = dict(scale=64, model="stationary", fy=.09, fx=.12)
            tiles.append(dict(y=y, x=x, h=32, w=32, selected=selected,
                              residual=rng.uniform(-1, 1, (32, 32))))
    gains = rng.uniform(.1, 1, len(tiles))
    allowed = np.arange(len(tiles)) % 3 == 0
    expected = restore_tiles(SimpleNamespace(image=image, tiles=tiles), weights=np.where(allowed, gains, 0.))
    result, mask, regions = refined._restore_general(image, tiles, gains, allowed, True)
    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(result[~mask], image[~mask])
    assert len(regions) == allowed.sum() and ((result == 0) | (result == 1)).any()


def test_cli_records_narrow_abstention_and_reproduces_previous_cli(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, _, observed, _, _ = next(case for case in make_controls(128) if case[0] == "compact_legitimate_stripe_clean")
    source = tmp_path / "stripes.png"
    Image.fromarray(np.rint(observed * 255).astype(np.uint8)).save(source)
    filter_project.main([str(source)])
    filter_project.main([str(source), "--previous"])
    filter_project.main([str(source), "--weave"])
    filter_project.main([str(source), "--stripes"])
    filter_project.main([str(source), "--weak"])
    filter_project.main([str(source), "--guarded"])
    filter_project.main([str(source), "--peak"])
    filter_project.main([str(source), "--collision"])
    filter_project.main([str(source), "--baseline"])
    root = tmp_path / "results"
    new = json.loads((root/"own_image_project"/"decisions.json").read_text(encoding="utf8"))
    old = json.loads((root/"own_image_project_baseline"/"decisions.json").read_text(encoding="utf8"))
    previous = json.loads((root/"own_image_project_previous"/"decisions.json").read_text(encoding="utf8"))
    weave = json.loads((root/"own_image_project_weave"/"decisions.json").read_text(encoding="utf8"))
    stripes = json.loads((root/"own_image_project_stripes"/"decisions.json").read_text(encoding="utf8"))
    weak = json.loads((root/"own_image_project_weak"/"decisions.json").read_text(encoding="utf8"))
    guarded = json.loads((root/"own_image_project_guarded"/"decisions.json").read_text(encoding="utf8"))
    peak = json.loads((root/"own_image_project_peak"/"decisions.json").read_text(encoding="utf8"))
    collision = json.loads((root/"own_image_project_collision"/"decisions.json").read_text(encoding="utf8"))
    assert new["algorithm"] == "project_phase" and old["algorithm"] == "project_pipeline"
    assert collision["algorithm"] == "project_collision" and collision["reason"] == new["reason"]
    assert peak["algorithm"] == "project_peak" and peak["reason"] == new["reason"]
    assert guarded["algorithm"] == "project_guarded" and guarded["reason"] == new["reason"]
    assert weak["algorithm"] == "project_weak" and weak["reason"] == new["reason"]
    assert stripes["algorithm"] == "project_stripes" and stripes["reason"] == new["reason"]
    assert weave["algorithm"] == "project_weave" and weave["reason"] == new["reason"]
    assert previous["algorithm"] == "project_refined" and previous["reason"] == new["reason"]
    assert new["reason"] == "narrow_affine_background_ambiguity"
    assert new["evidence"]["narrow_supported_pixels"] > 0 and not new["accepted_regions"]
    assert old["route"] == "narrow" and old["accepted_regions"]
    with np.load(root/"own_image_project"/"diagnostics.npz") as maps:
        assert not maps["support_mask"].any() and not maps["residual"].any()
