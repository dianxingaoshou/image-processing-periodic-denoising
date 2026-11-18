"""Regression tests for routing on the observed image and overlap corroboration."""
import numpy as np
import pytest

import narrow_denoiser
from context_controls import make_controls
from project_pipeline import corroborated_tiles, denoise, load_config


def tile_grid(ys, xs, size=16):
    """Crop each hypothetical candidate from the same spatially varying signal."""
    tiles = []
    for gy, y in enumerate(ys):
        for gx, x in enumerate(xs):
            yy, xx = np.mgrid[y:y + size, x:x + size]
            wave = np.sin(2 * np.pi * (.07 * yy + .13 * xx) + .3)
            tiles.append(dict(y=y, x=x, h=size, w=size, gy=gy, gx=gx,
                              residual=wave, selected=dict(fy=.07, fx=.13)))
    return tiles


def test_agreement_uses_same_image_coordinates_and_returns_one_flag_per_tile():
    tiles = tile_grid([0, 8, 16], [0, 8, 16])
    # Deliberately reorder rows so inference must respect tile coordinates.
    tiles = [tiles[i] for i in [4, 1, 8, 2, 5, 0, 3, 7, 6]]
    flags = corroborated_tiles(tiles, np.ones(len(tiles)))
    assert flags.dtype == np.bool_
    np.testing.assert_array_equal(flags, np.ones(9, dtype=bool))


@pytest.mark.parametrize(('ys', 'xs'), [
    ([0], [0, 8, 16, 24]),
    ([0, 8, 16, 24], [0]),
    ([0, 8, 9], [0, 8, 16]),
    ([0, 8, 16], [0, 8, 9]),
    ([0, 16, 32], [0, 16, 32]),
])
def test_one_dimensional_or_insufficient_physical_span_cannot_open_general_route(ys, xs):
    tiles = tile_grid(ys, xs)
    assert not corroborated_tiles(tiles, np.ones(len(tiles))).any()


def test_disconnected_patches_do_not_pool_evidence_across_zero_gain_tiles():
    tiles = tile_grid([0, 8, 16, 24, 32], [0, 8, 16])
    gains = np.array([float(t['gy'] in (0, 1, 3, 4)) for t in tiles])
    assert not corroborated_tiles(tiles, gains).any()


def test_only_members_of_the_supported_component_can_change_pixels():
    tiles = tile_grid([0, 8, 16], [0, 8, 16, 24, 32])
    gains = np.array([float(t['gx'] != 3) for t in tiles])
    flags = corroborated_tiles(tiles, gains)
    np.testing.assert_array_equal(flags, [t['gx'] <= 2 for t in tiles])


def test_diagonal_contacts_alone_cannot_authorize_a_broad_region():
    tiles = tile_grid([0, 8, 16], [0, 8, 16])
    gains = np.array([float(t['gy'] == t['gx']) for t in tiles])
    assert not corroborated_tiles(tiles, gains).any()


@pytest.mark.parametrize('disagreement', ['opposite_phase', 'amplitude_mismatch', 'zero_residual'])
def test_incompatible_waveforms_do_not_confirm_each_other(disagreement):
    tiles = tile_grid([0, 8, 16], [0, 8, 16])
    for t in tiles:
        if disagreement == 'zero_residual':
            t['residual'].fill(0)
        elif (t['gy'] + t['gx']) % 2:
            t['residual'] *= -1 if disagreement == 'opposite_phase' else 4
    assert not corroborated_tiles(tiles, np.ones(len(tiles))).any()


@pytest.mark.parametrize(('ratio', 'accepted'), [(2., True), (2.001, False)])
def test_overlap_agreement_at_the_documented_cutoff(ratio, accepted):
    tiles = tile_grid([0, 8, 16], [0, 8, 16])
    for t in tiles:
        t['residual'].fill(ratio if (t['gy'] + t['gx']) % 2 else 1.)
    # For proportional signals agreement is 2r/(1+r²), exactly 0.8 at r=2.
    flags = corroborated_tiles(tiles, np.ones(len(tiles)))
    np.testing.assert_array_equal(flags, np.full(len(tiles), accepted))


def assert_output(observed, output, diag):
    assert output.shape == observed.shape
    assert np.isfinite(output).all() and 0 <= output.min() <= output.max() <= 1
    assert diag['support_mask'].shape == observed.shape
    assert diag['support_mask'].dtype == np.bool_
    np.testing.assert_array_equal(diag['residual'], observed - output)
    outside = ~diag['support_mask']
    np.testing.assert_array_equal(output[outside], observed[outside])


def test_global_periodic_ambiguity_and_flat_input_remain_unchanged():
    y, x = np.indices((64, 96))
    for image in (np.full(x.shape, .46), .5 + .15 * np.sin(2 * np.pi * (x / 8 + y / 16))):
        output, diag = denoise(image, return_diagnostics=True)
        assert diag['route'] == 'unchanged'
        assert not diag['support_mask'].any()
        np.testing.assert_array_equal(output, image)
        assert_output(image, output, diag)


def test_global_ambiguity_veto_can_override_positive_narrow_evidence():
    _, _, observed, _, _ = next(row for row in make_controls(128)
                                if row[0] == 'broad_texture_edge8')
    output, diag = denoise(observed, return_diagnostics=True)
    assert diag['evidence']['narrow_supported_pixels'] > 0
    assert diag['reason'] == 'global_periodic_ambiguity'
    assert diag['route'] == 'unchanged'
    assert not diag['support_mask'].any()
    np.testing.assert_array_equal(output, observed)


def test_narrow_route_matches_standalone_result_and_preserves_rejected_rows():
    y, x = np.indices((96, 160))
    base = .45 + .07 * (x > 70)
    band = (y >= 20) & (y < 32)
    # These rows have a tangent phase discontinuity and fail split validation.
    holes = np.isin(y, (25, 26))
    angle = 2 * np.pi * (x / 8 + .075 * y) + .3
    image = base + band * .12 * np.sin(angle + np.where(holes & (x >= 80), np.pi/2, 0))
    original = image.copy()
    cfg = load_config()
    reference, reference_diag = narrow_denoiser.denoise(image, cfg.narrow, return_diagnostics=True)
    output, diag = denoise(image, cfg, return_diagnostics=True)
    assert diag['route'] == 'narrow'
    np.testing.assert_array_equal(output, reference)
    np.testing.assert_array_equal(diag['support_mask'], reference_diag['support_mask'])
    assert diag['support_mask'].any() and not diag['support_mask'][~band | holes].any()
    assert_output(image, output, diag)
    np.testing.assert_array_equal(denoise(image, cfg), output)
    np.testing.assert_array_equal(image, original)


def test_extended_interference_with_broadband_background_can_use_general_route():
    y, x = np.indices((144, 160))
    background = .5 + .045 * np.random.default_rng(503).uniform(-1, 1, x.shape)
    image = background + .12 * np.sin(2 * np.pi * (.094 * y + .125 * x) + .4)
    output, diag = denoise(image, return_diagnostics=True)
    assert diag['route'] == 'general'
    assert diag['support_mask'].any()
    assert_output(image, output, diag)
    assert np.mean((output - background)**2) < np.mean((image - background)**2)


@pytest.mark.parametrize('image', [
    np.full((31, 64), .4),
    np.zeros((64, 64, 3)),
    np.full((32, 32), np.nan),
    np.full((32, 32), 1.01),
])
def test_invalid_input_rejected_before_returning_a_route(image):
    with pytest.raises(ValueError):
        denoise(image)
