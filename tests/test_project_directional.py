"""Compare the integrated directional shortcuts with the original complete fit."""
from dataclasses import replace

import numpy as np
import pytest

import narrow_denoiser
from narrow_support import SupportConfig, build_support
from project_directional import _possible_rows, narrow_residual


def configuration(**changes):
    cfg = narrow_denoiser.Config(support=SupportConfig(contrast_context="amplitude"))
    return replace(cfg, **changes)


def crossing_bands():
    y, x = np.indices((131, 163))
    row_wave = .12*np.sin(2*np.pi*(.125*x+.06*y)+.2)
    col_wave = .08*np.sin(2*np.pi*(.136*y-.074*x)+.7)
    return (.49+.009*np.cos(.0165*y-.8)
            + row_wave*((y >= 11) & (y < 23))
            + col_wave*((x >= 74) & (x < 86)))


@pytest.mark.parametrize("changes", [
    {}, {"strength": .75}, {"strength": 0.}, {"use_gate": False}, {"compact_support": False},
    {"support": SupportConfig(contrast_context="qualified")},
    {"support": SupportConfig(contrast_context="amplitude", limit_context_width=False)},
])
def test_crossing_profiles_match_even_in_the_overlap(changes):
    image = crossing_bands()
    cfg = configuration(**changes)
    expected, diag = narrow_denoiser.denoise(image, cfg, return_diagnostics=True)
    raw, mask, regions = narrow_residual(image, cfg, return_regions=True)
    output = np.clip(image-cfg.strength*raw, 0, 1)
    assert {r["normal_axis"] for r in regions} == {0, 1}
    assert np.count_nonzero(raw[~mask]) == 0
    np.testing.assert_array_equal(raw*cfg.strength, diag["raw_residual"])
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(mask, diag["support_mask"])
    assert regions == [
        dict(normal_axis=p["axis"], start=r["start"], stop=r["stop"],
             context_start=r["context_start"], context_stop=r["context_stop"],
             tangent_frequency=p["frequency_cycles_per_pixel"])
        for p in diag["profiles"] for r in p["intervals"] if r["accepted"]
    ]


def test_noncontiguous_float32_input_uses_original_precision():
    image = crossing_bands().astype("float32")[::-1, ::-1]
    cfg = configuration()
    expected, diag = narrow_denoiser.denoise(image, cfg, return_diagnostics=True)
    raw, mask, regions = narrow_residual(image, cfg)
    output = np.clip(image.astype(float)-cfg.strength*raw, 0, 1)
    assert output.dtype == np.float64
    assert mask.any() and regions == []
    np.testing.assert_array_equal(output, expected)
    np.testing.assert_array_equal(mask, diag["support_mask"])


@pytest.mark.parametrize("shape", [(32, 43), (64, 130)])
def test_no_support_avoids_allocating_an_output(shape):
    image = np.full(shape, .5)
    assert narrow_residual(image, configuration(), return_regions=True) == (None, None, [])


def test_raw_fit_does_not_clip_before_applying_requested_strength():
    y, x = np.indices((96, 160))
    band = (y >= 30) & (y < 42)
    image = np.clip(.08+.18*band*np.sin(2*np.pi*(.125*x+.06*y)), 0, 1)
    # A single dark outlier within an otherwise bright crest makes full subtraction clip.
    image[35, 17] = 0.
    cfg = configuration(strength=.5)
    raw, mask, _ = narrow_residual(image, cfg)
    assert mask[35, 17] and np.any(image-raw < 0)
    full_raw, _, _ = narrow_residual(image, replace(cfg, strength=1.))
    np.testing.assert_array_equal(raw, full_raw)
    expected, diag = narrow_denoiser.denoise(image, cfg, return_diagnostics=True)
    np.testing.assert_array_equal(raw*.5, diag["raw_residual"])
    np.testing.assert_array_equal(np.clip(image-.5*raw, 0, 1), expected)


@pytest.mark.parametrize("context,limit", [("qualified", True), ("amplitude", False), ("amplitude", True)])
def test_screening_never_discards_a_qualified_run(context, limit):
    # Isolated rows, context-width boundaries, single-flank bands, and
    # deliberately fragmented stability masks exercise the necessary bound.
    rng = np.random.default_rng(32421)
    cfg = configuration(support=SupportConfig(contrast_context=context, limit_context_width=limit))
    for start, width in ((0, 1), (0, 3), (0, 16), (17, 31), (17, 32), (17, 33), (56, 8)):
        for outside in (0, .019, .0199):
            amplitude = np.full(64, outside, dtype=float)
            amplitude[start:start+width] = rng.choice([.03, .05, .08], width)
            stable = rng.choice([.7, .8, .9], 64)
            profile = dict(amplitude=amplitude, stability=stable, validation_gain=np.ones(64))
            old = build_support(profile, cfg.support)["mask"]
            assert not np.any(old & ~_possible_rows(amplitude, cfg))
