"""Analytic reference gains, local loss deltas and evaluation-only boundaries."""
import ast
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from denoiser_v3 import Config, Prepared
from gain_diagnostics import candidate_gain_rows, summarize_gain_rows
from local_estimation_v3 import _geometry


def fixture(candidate_scale=1., *, true_scale=1., q=1., s=1., mode='coupled', selected=True):
    y, x = np.indices((32, 40))
    wave = .06*np.sin(2*np.pi*(.125*y+.15*x)+.3)
    clean = np.full(wave.shape, .5)
    observed = clean+true_scale*wave
    tile = dict(y=0, x=0, h=32, w=40, selected={} if selected else None,
                residual=candidate_scale*wave)
    config = replace(Config(), strength=1., combination=mode)
    estimation = SimpleNamespace(image=observed, tiles=[tile])
    prepared = Prepared(config, estimation, np.array([q]), np.array([s]), {},
                        [dict(candidate_gate=True, texture_risk=1-s)])
    return observed, clean, prepared


@pytest.mark.parametrize('candidate_scale,oracle', [(1., 1.), (2., .5), (.5, 1.), (-1., 0.)])
def test_analytic_gain_and_weighted_delta(candidate_scale, oracle):
    observed, clean, prepared = fixture(candidate_scale)
    row = candidate_gain_rows(observed, clean, prepared)[0]
    assert row['oracle_gain'] == pytest.approx(oracle)
    weight = _geometry(observed.shape)[4]
    truth = observed-clean
    estimate = prepared.estimation.tiles[0]['residual']
    actual_delta = np.sum(weight*((truth-estimate)**2-truth**2))
    oracle_delta = np.sum(weight*((truth-oracle*estimate)**2-truth**2))
    assert row['actual_weighted_sse_delta'] == pytest.approx(actual_delta)
    assert row['oracle_weighted_sse_delta'] == pytest.approx(oracle_delta)
    assert row['avoidable_harm'] == (candidate_scale < 0)
    json.dumps(row, allow_nan=False)


def test_mixed_candidate_projects_onto_true_added_noise():
    observed, clean, prepared = fixture()
    y, x = np.indices(observed.shape)
    unrelated = .04*np.cos(2*np.pi*(.19*y-.07*x)+.7)
    prepared.estimation.tiles[0]['residual'] += unrelated
    residual = prepared.estimation.tiles[0]['residual']
    weight = _geometry(observed.shape)[4]
    expected = np.clip(np.sum(weight*residual*(observed-clean))/np.sum(weight*residual**2), 0, 1)
    assert candidate_gain_rows(observed, clean, prepared)[0]['oracle_gain'] == pytest.approx(expected)


def test_clean_reference_assigns_zero_gain_and_detects_texture_damage():
    observed, clean, prepared = fixture()
    # The exact same observed stripes now belong to the reference: they are
    # legitimate texture rather than added interference.  Candidate is unchanged.
    row = candidate_gain_rows(observed, observed.copy(), prepared)[0]
    assert row['oracle_gain'] == 0
    assert row['avoidable_harm']
    assert row['actual_local_mse_delta'] > 0
    assert row['over_subtraction'] == 1


def test_zero_candidate_is_unidentifiable_and_not_counted_as_oracle_zero():
    observed, clean, prepared = fixture(0.)
    row = candidate_gain_rows(observed, clean, prepared)[0]
    assert row['oracle_gain'] is None
    assert row['oracle_weighted_sse_delta'] is None
    assert not row['nonzero_candidate'] and not row['avoidable_harm']
    summary = summarize_gain_rows([row])
    assert summary['excluded_tiles'] == 1 and summary['eligible_nonzero_tiles'] == 0
    assert summary['overall']['mean_oracle_gain'] is None
    assert summary['safety_oracle_gain_correlation'] is None
    json.dumps(summary, allow_nan=False)


def test_actual_gain_respects_combination_and_absent_candidate():
    observed, clean, prepared = fixture(q=.8, s=.7)
    coupled = candidate_gain_rows(observed, clean, prepared, strength=2., combination='coupled')[0]
    decoupled = candidate_gain_rows(observed, clean, prepared, strength=2., combination='decoupled')[0]
    assert coupled['actual_gain'] == 1.
    assert decoupled['actual_gain'] == .7
    observed, clean, prepared = fixture(selected=False)
    row = candidate_gain_rows(observed, clean, prepared)[0]
    assert row['actual_gain'] == 0 and not row['eligible']


def test_bins_include_one_once_and_null_constant_correlation():
    rows = []
    for safety in (0., .2, .4, .6, .8, 1.):
        observed, clean, prepared = fixture(q=.7, s=safety)
        rows.extend(candidate_gain_rows(observed, clean, prepared))
    summary = summarize_gain_rows(rows)
    assert [row['count'] for row in summary['safety_bins']] == [1, 1, 1, 1, 2]
    assert summary['safety_oracle_gain_correlation'] is None
    assert summary['overall']['count'] == 6
    assert summary['overall']['harmful_fraction'] == 0.
    json.dumps(summary, allow_nan=False)
    with pytest.raises(ValueError): summarize_gain_rows(rows, bins=(0., .7, .6, 1.))


def test_diagnostic_does_not_modify_prepared_or_allow_wrong_observation():
    observed, clean, prepared = fixture()
    before = prepared.estimation.tiles[0]['residual'].copy()
    before_q = prepared.fit_weights.copy()
    candidate_gain_rows(observed, clean, prepared)
    np.testing.assert_array_equal(prepared.estimation.tiles[0]['residual'], before)
    np.testing.assert_array_equal(prepared.fit_weights, before_q)
    with pytest.raises(ValueError, match='belong'):
        candidate_gain_rows(observed+.001, clean, prepared)
    with pytest.raises(ValueError, match='shapes'):
        candidate_gain_rows(observed, clean[:-1], prepared)


def test_production_algorithm_does_not_import_reference_diagnostics():
    for path in ('denoiser_v3.py', 'local_estimation_v3.py', 'texture_guard_v3.py', 'filter_image_v3.py'):
        tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom): modules.append(node.module)
            elif isinstance(node, ast.Import): modules.extend(alias.name for alias in node.names)
        assert 'gain_diagnostics' not in modules
