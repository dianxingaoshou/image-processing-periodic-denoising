"""Separate widespread harmonic texture from local carriers using only observed line profiles."""
import json

import numpy as np
import pytest

from narrow_periodicity import screen_narrow


FREQUENCY = 19 / 192
REGION = dict(normal_axis=0, start=58, stop=63, context_start=56, context_stop=65,
              tangent_frequency=FREQUENCY)


def texture(carriers, companions, ratio=.5):
    y, x = np.indices((len(carriers), 192))
    phase = 2*np.pi*FREQUENCY*x + .12*y
    return .48 + carriers[:, None]*np.cos(phase) + companions[:, None]*np.cos(ratio*phase+.3)


def check(image, *regions):
    decisions = screen_narrow(image, dict(accepted_regions=list(regions)))
    json.dumps(decisions, allow_nan=False)
    return decisions


@pytest.mark.parametrize("ratio", [.5, 2.])
@pytest.mark.parametrize("transpose", [False, True])
def test_same_resolved_harmonics_inside_and_widely_outside_support(ratio, transpose):
    carrier = np.full(128, .06)
    carrier[58:63] = .08
    image = texture(carrier, np.full(128, .07), ratio)
    result, = check(image.T if transpose else image, dict(REGION, normal_axis=int(transpose)))
    assert result["veto"] and result["companion_frequency"] == ratio*FREQUENCY
    assert result["remote_paired_lines"] == result["remote_lines"]
    assert result["inside_paired_lines"] == 5


def test_single_carrier_has_no_harmonic_evidence_even_if_it_is_widespread():
    result, = check(texture(np.full(128, .08), np.zeros(128)), REGION)
    assert not result["veto"] and result["remote_carrier_lines"] == result["remote_lines"]


def test_all_recurrent_companions_are_available_to_the_broad_retry():
    arr = np.full(128, .08)
    image = texture(arr, arr) + texture(np.zeros(128), arr, ratio=2.) - .48
    result, = check(image, REGION)
    assert result["confirmed_companions"] == [FREQUENCY/2, 2*FREQUENCY]
    assert result["companion_frequency"] == FREQUENCY/2


def test_harmonics_on_different_distant_lines_do_not_count_as_a_pair():
    carrier = .08*(np.arange(128) % 2)
    companion = .08-carrier
    carrier[58:63] = companion[58:63] = .08
    result, = check(texture(carrier, companion), REGION)
    assert not result["veto"] and result["remote_carrier_lines"] > 32
    assert result["tested_companions"][0]["remote_lines"] == 0


def test_local_band_and_one_additional_compact_copy_do_not_count_as_broad_texture():
    # All 32 remote matches are concentrated in one allowed strip.
    arr = np.zeros(128)
    arr[58:63] = arr[:32] = .08
    result, = check(texture(arr, arr), REGION)
    assert result["remote_carrier_lines"] == 32 and not result["veto"]


def test_decisions_preserve_region_order_and_frequency_boundaries():
    image = texture(np.full(128, .08), np.full(128, .07))
    regions = [dict(REGION, tangent_frequency=.499), REGION, dict(REGION, tangent_frequency=.001)]
    assert [r["veto"] for r in check(image, *regions)] == [False, True, False]
    assert check(image) == []


def test_no_remote_profiles_leaves_original_decision_in_place():
    image = texture(np.full(72, .08), np.full(72, .07))
    region = dict(REGION, context_start=0, context_stop=72)
    result, = check(image, region)
    assert result["remote_lines"] == 0 and not result["veto"]
