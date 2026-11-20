"""Local texture evidence can only veto an already corroborated broad component."""
from types import SimpleNamespace

import numpy as np
import pytest

import denoiser_v3
from context_controls import make_controls
from project_pipeline import corroborated_tiles, load_config
from project_texture import screen_tiles


def block_fixture(rows=3, columns=3):
    # Fields consumed by the graph and the existing texture guard.
    image = np.zeros(((rows+1)*8, (columns+1)*8))
    tiles, decisions = [], []
    for gy in range(rows):
        for gx in range(columns):
            tiles.append(dict(gy=gy, gx=gx, y=gy*8, x=gx*8, h=16, w=16,
                              residual=np.full((16, 16), 1.0)))
            decisions.append(dict(harmonic_support=0., local_periodic_fraction=0., structure_loss=0.))
    prepared = SimpleNamespace(estimation=SimpleNamespace(tiles=tiles, image=image), decisions=decisions,
                               config=load_config().general)
    return image, prepared, np.ones(len(tiles))


def test_neighboring_opposite_phase_components_keep_independent_texture_decisions():
    image, prepared, gains = block_fixture(columns=6)
    for tile, d in zip(prepared.estimation.tiles, prepared.decisions):
        if tile["gx"] < 3:
            tile["residual"] *= -1
            d["harmonic_support"] = .25
    keep, diagnostics = screen_tiles(image, prepared, gains)
    np.testing.assert_array_equal(keep, [t["gx"] >= 3 for t in prepared.estimation.tiles])
    assert diagnostics["spatially_eligible_tiles"] == 18
    assert diagnostics["texture_vetoed_tiles"] == 9
    assert {c["reason"] for c in diagnostics["components"]} == {"periodic_harmonics", "accepted"}


def test_an_isolated_harmonic_tile_does_not_veto_a_broad_component():
    image, prepared, gains = block_fixture()
    prepared.decisions[3]["harmonic_support"] = 1.
    keep, diagnostics = screen_tiles(image, prepared, gains)
    assert keep.all()
    assert diagnostics["components"][0]["median_harmonic_score"] == 0.


@pytest.mark.parametrize("joint, veto", [(False, False), (True, True)])
def test_near_pure_concentration_and_structure_loss_must_coincide(joint, veto):
    image, prepared, gains = block_fixture()
    for i, decision in enumerate(prepared.decisions):
        decision["local_periodic_fraction"] = .98 if i < 5 else .2
        decision["structure_loss"] = .8 if (i < 5 if joint else i >= 4) else 0.
    keep, diagnostics = screen_tiles(image, prepared, gains)
    assert np.all(keep == (not veto))
    assert diagnostics["components"][0]["reason"] == ("near_pure_structure_loss" if veto else "accepted")


def test_geometry_before_veto_matches_the_frozen_graph_with_irregular_edge_steps():
    image, prepared, gains = block_fixture(rows=7, columns=7)
    rng = np.random.default_rng(601)
    tiles = prepared.estimation.tiles
    for tile in tiles:
        if tile["gy"] == 6:
            tile["y"] -= 5
        if tile["gx"] == 6:
            tile["x"] -= 5
    for _ in range(10):
        for tile in tiles:
            tile["residual"].fill(rng.choice([-1., 1., 1.]))
        gains[:] = rng.choice([0., .5, 1.], size=len(gains))
        keep, diagnostics = screen_tiles(image, prepared, gains)
        expected = corroborated_tiles(tiles, gains)
        np.testing.assert_array_equal(keep, expected)
        assert diagnostics["spatially_eligible_tiles"] == expected.sum()


def test_existing_features_detect_the_broad_periodic_control():
    _, _, observed, _, _ = next(c for c in make_controls() if c[0] == "broad_phase_banded_clean")
    cfg = load_config().general
    prepared = denoiser_v3.prepare(observed, cfg)
    gains = denoiser_v3.effective_gains(prepared.fit_weights, prepared.safety_weights,
                                       cfg.strength, cfg.combination)
    keep, diagnostics = screen_tiles(observed, prepared, gains)
    assert diagnostics["spatially_eligible_tiles"] > 0
    assert not keep.any()
    assert {c["reason"] for c in diagnostics["components"]} == {"near_pure_structure_loss"}
