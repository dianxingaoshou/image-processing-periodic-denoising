"""Small analytic checks of paired ablations and quality-matched reporting."""
from dataclasses import replace
import json

import numpy as np
import pytest

from denoiser_v3 import Config, config_dict
import refinement_experiments as study


def quality_rows(method, points):
    return [dict(image_id='analytic', kind='local', method=method,
                 strength=float(index+1), contaminated_mse=dirty, clean_mse=clean)
            for index, (dirty, clean) in enumerate(points)]


def test_matching_uses_same_image_common_quality_interval_only():
    rows = quality_rows('coupled_baseline', [(.08, .001), (.04, .004)])
    rows += quality_rows('structure_risk', [(.10, .0005), (.05, .003)])
    result = study.matched_quality(rows)
    assert len(result) == 1 and result[0]['matched']
    row = result[0]
    assert row['target_dirty_mse'] == pytest.approx(.065)
    assert row['baseline_clean_mse'] == pytest.approx(.002125)
    assert row['structure_clean_mse'] == pytest.approx(.00225)
    assert row['clean_mse_difference'] == pytest.approx(.000125)


def test_matching_keeps_unmatchable_cases_and_does_not_extrapolate():
    rows = quality_rows('coupled_baseline', [(.01, .002), (.02, .001)])
    rows += quality_rows('structure_risk', [(.04, .004), (.05, .003)])
    result = study.matched_quality(rows)
    assert len(result) == 1 and not result[0]['matched']
    assert 'common' in result[0]['reason']
    rows = quality_rows('coupled_baseline', [(.01, ''), (.02, '')])
    rows += quality_rows('structure_risk', [(.01, ''), (.02, '')])
    assert not study.matched_quality(rows)[0]['matched']


@pytest.mark.parametrize('dirty,reason', [([.08, .04, .06], 'nonmonotonic'),
                                        ([.08, .04, .04], 'duplicate')])
def test_matching_does_not_hide_ambiguous_strength_branches(dirty, reason):
    rows = quality_rows('coupled_baseline', list(zip(dirty, [.001, .003, .005])))
    rows += quality_rows('structure_risk', [(.09, .001), (.06, .002), (.03, .003)])
    result = study.matched_quality(rows)
    assert len(result) == 1 and not result[0]['matched']
    assert reason in result[0]['reason']


def test_risk_only_configuration_cannot_change_estimator_or_strength(monkeypatch):
    baseline = replace(Config(), combination='coupled')
    structure = replace(baseline, guard=replace(baseline.guard, risk_mode='structure'))
    payloads = {'v3_p0_coupled.json': dict(configuration=config_dict(baseline)),
                'refined_risk.json': dict(configuration=config_dict(structure))}
    monkeypatch.setattr(study, 'read', lambda path: payloads[path.name])
    configs = study.configs()
    assert configs['structure_risk'].estimation == configs['coupled_baseline'].estimation
    assert configs['decoupled_control'].guard == baseline.guard
    payloads['refined_risk.json'] = dict(configuration=config_dict(replace(structure, strength=.5)))
    with pytest.raises(ValueError, match='risk-only'):
        study.configs()


def test_aggregation_uses_one_fixed_strength_and_keeps_per_image_failure_fraction():
    def row(image_id, method, strength, mse, mae, degraded):
        return dict(image_id=image_id, kind='local', method=method, strength=strength,
                    mse=mse, input_mse=.1, mae=mae, degraded=degraded)
    rows = [row('a', 'coupled_baseline', 2., .05, .1, 0),
            row('b', 'coupled_baseline', 2., .2, .3, 1),
            row('a', 'coupled_baseline', .5, .001, .001, 0),
            row('a', 'input', 0., .1, .2, 0)]
    summary = {(r['kind'], r['method']): r for r in study.aggregate(rows)}
    actual = summary['local', 'coupled_baseline']
    assert actual['n'] == 2
    assert actual['mse'] == pytest.approx(.125)
    assert actual['mean_mse_change'] == pytest.approx(.025)
    assert actual['degraded_fraction'] == .5
    assert actual['tail20_mae'] == .3


@pytest.mark.parametrize('tamper', ['development', 'configuration'])
def test_frozen_study_checks_development_and_sealed_configuration(tmp_path, monkeypatch, tamper):
    # The runner obtains configurations from JSON, which represents scale
    # sequences as lists rather than the dataclass's tuple defaults.
    config = study.config_from_dict(json.loads(json.dumps(config_dict(replace(Config(), combination='coupled')))))
    configuration = {'coupled_baseline': config_dict(config)}
    plan = tmp_path/'protocol.json'
    freeze = tmp_path/'freeze.json'
    development = tmp_path/'development/metrics.csv'
    development.parent.mkdir()
    development.write_text('image_id,mse\na,.02\n', encoding='utf-8')
    study.dump(plan, dict(independent_check=[dict(id='unused')]))
    study.dump(freeze, dict(configurations=configuration,
        development_metrics_sha256=study.sha256_file(development),
        snapshot=dict(metadata=dict(configurations=configuration))))
    monkeypatch.setattr(study, 'OUT', tmp_path)
    monkeypatch.setattr(study, 'PLAN', plan)
    monkeypatch.setattr(study, 'FREEZE', freeze)
    monkeypatch.setattr(study, 'configs', lambda: {'coupled_baseline': config})
    # Source/runtime snapshot verification is separately tested exhaustively;
    # this test isolates the focused study's extra artifact/metadata checks.
    monkeypatch.setattr(study, 'verify_snapshot', lambda *args, **kwargs: None)
    assert study.verify_frozen()['configurations'] == configuration
    if tamper == 'development':
        development.write_text('image_id,mse\na,.01\n', encoding='utf-8')
        reason = 'Development metrics changed'
    else:
        payload = study.read(freeze)
        payload['configurations']['coupled_baseline']['strength'] = .5
        study.dump(freeze, payload)
        reason = 'sealed snapshot metadata'
    with pytest.raises(ValueError, match=reason):
        study.verify_frozen()


def test_reference_array_hash_is_checked_before_loading_samples(tmp_path, monkeypatch):
    reference = tmp_path/'clean.npy'
    np.save(reference, np.full((32, 32), .5))
    record = dict(id='development', npy_path='clean.npy', clean_sha256=study.sha256_file(reference))
    manifest = tmp_path/'v3_manifest.json'
    study.dump(manifest, dict(images=[record], textures=[]))
    plan = tmp_path/'protocol.json'
    study.dump(plan, dict(development_ids=['development'],
                        source_manifest_sha256=study.sha256_file(manifest)))
    monkeypatch.setattr(study, 'DATA', tmp_path)
    monkeypatch.setattr(study, 'PLAN', plan)
    # The manifest still has its frozen bytes, while the underlying array has
    # changed. Checking only the manifest would silently allow another input.
    np.save(reference, np.full((32, 32), .4))
    with pytest.raises(ValueError, match='Reference array changed'):
        next(study.samples('development'))
