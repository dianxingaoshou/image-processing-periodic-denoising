"""Small-window adequacy without references, masks or parity aliasing."""
import json

import numpy as np
import pytest

from small_window import _balanced_mask, small_window_evidence


def _image(fy=.07, fx=.11, phase=.4, shape=(32, 32)):
    yy, xx = np.indices(shape)
    carrier = np.sin(2*np.pi*(fy*yy+fx*xx)+phase)
    background = .45+.015*yy/shape[0]-.02*xx/shape[1]
    return background+.1*carrier


@pytest.mark.parametrize('fy,fx', [(0., .25), (.25, 0.), (0., .2495), (.11, .07)])
@pytest.mark.parametrize('phase', [0., .5, np.pi/2])
def test_valid_small_window_keeps_all_quarter_frequency_parities(fy, fx, phase):
    decision = small_window_evidence(_image(fy, fx, phase), dict(fy=fy, fx=fx))
    assert decision['eligible']
    assert decision['fit_rank_min'] == 5
    assert decision['stability'] == pytest.approx(1., abs=1e-12)
    assert decision['amplitude'] == pytest.approx(.1, abs=1e-12)
    json.dumps(decision, allow_nan=False)


def test_cycles_are_physical_rotation_invariant_and_reject_short_evidence():
    axis = small_window_evidence(_image(0., .08), dict(fy=0., fx=.08))
    diagonal = small_window_evidence(_image(.08/np.sqrt(2), .08/np.sqrt(2)),
                                     dict(fy=.08/np.sqrt(2), fx=.08/np.sqrt(2)))
    assert axis['cycles'] == pytest.approx(diagonal['cycles'])
    assert not axis['eligible'] and not diagonal['eligible']
    assert axis['reason'] == 'insufficient_cycles'
    assert diagonal['phase_span'] > axis['phase_span']
    # The same physical frequency has enough periods in a larger window.
    large = small_window_evidence(_image(0., .08, shape=(64, 64)), dict(fy=0., fx=.08))
    assert large['eligible']


def test_localized_envelope_is_not_rejected_for_being_absent_on_one_side():
    yy, xx = np.indices((32, 32))
    envelope = np.where(xx < 8, .12, 0.)
    image = .5+envelope*np.sin(2*np.pi*(.13*yy+.08*xx)+.2)
    decision = small_window_evidence(image, dict(fy=.13, fx=.08))
    assert decision['eligible']
    assert decision['stability'] > .9


def test_disagreeing_sample_groups_fail_without_rng_or_truth():
    yy, xx = np.indices((32, 32))
    first = _balanced_mask((32, 32))
    carrier = .1*np.sin(2*np.pi*(.11*yy+.13*xx)+.4)
    image = .5+np.where(first, carrier, -carrier)
    np.random.seed(771); state = np.random.get_state()
    result = small_window_evidence(image, dict(fy=.11, fx=.13))
    after = np.random.get_state()
    np.testing.assert_array_equal(state[1], after[1])
    assert state[2:] == after[2:]
    assert not result['eligible'] and result['stability'] < .01
    assert result['reason'] == 'unstable_coefficients'


def test_stratified_sampling_is_balanced_in_every_cell_and_parity():
    mask = _balanced_mask((32, 32))
    assert not mask.flags.writeable
    for y in range(0, 32, 4):
        for x in range(0, 32, 4):
            cell = mask[y:y+4, x:x+4]
            assert cell.sum() == 8
            assert all(cell[dy::2, dx::2].sum() == 2 for dy in (0, 1) for dx in (0, 1))


def test_constant_patch_does_not_gain_stability_from_zero_coefficients():
    result = small_window_evidence(np.full((32, 32), .5), dict(fy=.11, fx=.13))
    assert not result['eligible']
    assert result['stability'] == 0
    assert result['reason'] == 'no_identifiable_component'


@pytest.mark.parametrize('candidate', [dict(fy=0., fx=0.), dict(fy=np.nan, fx=.1), dict(fx=.1)])
def test_invalid_frequency_is_rejected(candidate):
    with pytest.raises(ValueError):
        small_window_evidence(np.full((32, 32), .5), candidate)


@pytest.mark.parametrize('options', [dict(minimum_cycles=0), dict(minimum_cycles=np.inf),
                                    dict(minimum_stability=-.1), dict(minimum_stability=np.nan)])
def test_invalid_thresholds_are_rejected(options):
    with pytest.raises(ValueError):
        small_window_evidence(_image(), dict(fy=.07, fx=.11), **options)
