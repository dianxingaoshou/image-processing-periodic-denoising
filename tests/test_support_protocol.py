"""Paired support-ablation checks, independent of image-data downloads."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

import support_experiments as study


def test_factorial_configs_only_change_the_three_declared_factors():
    configurations = study.factorial_configs()
    baseline = configurations['baseline']
    assert len(configurations) == len(study.FACTORS) == 8
    assert set(study.FACTORS.values()) == {
        (a, s, g) for a in (False, True) for s in (False, True) for g in (False, True)}
    for name, config in configurations.items():
        a32, s32, grid = study.FACTORS[name]
        expected = asdict(baseline.estimation)
        expected.update(scales=[32, 64, 128, 256] if a32 else [64, 128, 256],
                        synthesis_size=32 if s32 else 64, stride=16 if s32 else 32,
                        amplitude_model='grid' if grid else 'affine')
        assert asdict(config.estimation) == expected
        assert config.guard == baseline.guard
        assert config.guard.risk_mode == 'legacy'
        assert config.strength == baseline.strength == 2.
        assert config.combination == baseline.combination == 'coupled'


def primary_row(method, *, dirty=.04, clean=.001, adjacent=.003, image_id='analytic'):
    return dict(dataset='BSDS', image_id=image_id, kind='edge16', method=method,
                strength=2., mse=.02, contaminated_mse=dirty, clean_mse=clean,
                adjacent_clean_mse=adjacent, mae=.1)


def test_joint_success_requires_dirty_gain_without_either_kind_of_clean_damage():
    baseline = primary_row('baseline')
    rows = [baseline,
            primary_row('a32', dirty=.03),
            primary_row('s32', dirty=.03, clean=.0011),
            primary_row('grid', dirty=.03, adjacent=.004),
            primary_row('a32_s32', dirty=.04, clean=.0005)]
    result = {r['method']: r for r in study.paired_deltas(rows)}
    assert result['a32']['joint_success']
    assert result['a32']['contaminated_mse_delta'] == pytest.approx(-.01)
    assert result['s32']['clean_increased'] and not result['s32']['joint_success']
    assert result['grid']['adjacent_increased'] and not result['grid']['joint_success']
    assert not result['a32_s32']['dirty_improved']
    assert not result['a32_s32']['joint_success']


def test_missing_adjacent_evaluation_cannot_be_called_joint_success():
    rows = [primary_row('baseline', adjacent=''), primary_row('a32', dirty=.03, adjacent='')]
    row, = study.paired_deltas(rows)
    assert row['adjacent_clean_mse_delta'] == ''
    assert row['dirty_improved']
    assert not row['joint_success']


def test_pairing_does_not_mix_datasets_and_uses_only_primary_strength():
    a = primary_row('baseline', dirty=.04)
    b = primary_row('baseline', dirty=.10)
    b['dataset'] = 'BSDS_regression'
    treatment = primary_row('a32', dirty=.09)
    treatment['dataset'] = 'BSDS_regression'
    ignored = primary_row('a32', dirty=0.)
    ignored['strength'] = .5
    row, = study.paired_deltas([a, b, treatment, ignored])
    assert row['dataset'] == 'BSDS_regression'
    assert row['contaminated_mse_delta'] == pytest.approx(-.01)


def test_adjacent_metric_is_not_diluted_by_far_away_clean_pixels():
    clean = np.zeros((20, 20))
    adjacent = np.zeros(clean.shape, bool)
    adjacent[:, 3] = True
    dirty = np.zeros(clean.shape, bool)
    dirty[:, :3] = True
    clean_mask = ~dirty
    output = np.zeros_like(clean)
    output[adjacent] = .1
    metrics = study.spatial_metrics(clean, output,
        dict(clean=clean_mask, adjacent_clean=adjacent))
    assert metrics['adjacent_clean_mse'] == pytest.approx(.01)
    assert metrics['far_clean_mse'] == 0.
    assert np.mean((output[clean_mask]-clean[clean_mask])**2) < .001
    empty = study.spatial_metrics(clean, clean,
        dict(clean=clean_mask, adjacent_clean=np.zeros_like(adjacent)))
    assert empty['adjacent_clean_mse'] == ''


def quality_rows(method, points):
    return [dict(dataset='BSDS', image_id='analytic', kind='edge16', method=method,
                 strength=float(i+1), contaminated_mse=dirty,
                 adjacent_clean_mse=adjacent) for i, (dirty, adjacent) in enumerate(points)]


def test_quality_matching_uses_common_interval_and_preserves_unmatched_methods():
    rows = quality_rows('baseline', [(.08, .001), (.04, .004)])
    rows += quality_rows('a32', [(.10, .0005), (.05, .003)])
    result = {r['method']: r for r in study.matched_quality(rows)}
    assert len(result) == 7
    assert result['a32']['matched']
    assert result['a32']['target_dirty_mse'] == pytest.approx(.065)
    assert result['a32']['adjacent_mse_delta'] == pytest.approx(.000125)
    assert not result['grid']['matched']
    assert 'insufficient' in result['grid']['reason']


@pytest.mark.parametrize('points,reason', [
    ([(.10, .001), (.05, .002), (.07, .003)], 'nonmonotonic'),
    ([(.10, .001), (.05, .002), (.05, .003)], 'duplicate'),
    ([(.15, .001), (.12, .002)], 'common')])
def test_quality_matching_rejects_ambiguous_or_nonoverlapping_curves(points, reason):
    rows = quality_rows('baseline', [(.08, .001), (.04, .004)])
    rows += quality_rows('a32', points)
    actual = next(r for r in study.matched_quality(rows) if r['method'] == 'a32')
    assert not actual['matched']
    assert reason in actual['reason']


def test_summary_keeps_per_image_failure_and_tail_instead_of_strength_pseudoreplication():
    first = primary_row('a32', dirty=.01, clean=.001)
    first.update(degraded=0, psnr_db=20.)
    second = primary_row('a32', dirty=.20, clean=.004, image_id='second')
    second.update(degraded=1, psnr_db=10., mae=.3)
    ignored = dict(first, strength=.5, mae=0.)
    row, = study.summary([first, second, ignored])
    assert row['n'] == 2
    assert row['degraded_fraction'] == .5
    assert row['tail20_mae'] == .3
    assert row['clean_mse'] == pytest.approx(.0025)
    assert row['tail20_clean_mse'] == .004
    zero_control = dict(first, kind='clean', psnr_db='inf', mse=0., mae=0.)
    assert study.summary([zero_control])[0]['psnr_db'] == ''


def test_samples_checks_reference_array_hash_before_loading(tmp_path, monkeypatch):
    reference = tmp_path/'clean.npy'
    np.save(reference, np.full((32, 32), .5))
    record = dict(id='dev', npy_path='clean.npy', clean_sha256=study.sha256_file(reference))
    manifest = tmp_path/'v3_manifest.json'
    study.dump(manifest, dict(images=[record], textures=[]))
    plan = tmp_path/'protocol.json'
    study.dump(plan, dict(development_ids=['dev'], source_manifest_sha256=study.sha256_file(manifest)))
    monkeypatch.setattr(study, 'DATA', tmp_path)
    monkeypatch.setattr(study, 'PLAN', plan)
    np.save(reference, np.full((32, 32), .4))
    with pytest.raises(ValueError, match='Reference array changed'):
        next(study.samples('development'))


def test_check_source_hash_failure_happens_before_pixels_are_preprocessed(tmp_path, monkeypatch):
    source = tmp_path/'source.jpg'
    source.write_bytes(b'changed source')
    manifest = tmp_path/'v3_manifest.json'
    study.dump(manifest, dict(images=[], textures=[]))
    plan = tmp_path/'protocol.json'
    study.dump(plan, dict(independent_check=[dict(id='check', source_jpg='source.jpg', source_sha256='wrong')],
                         source_manifest_sha256=study.sha256_file(manifest)))
    monkeypatch.setattr(study, 'DATA', tmp_path)
    monkeypatch.setattr(study, 'PLAN', plan)
    def never_load(*args, **kwargs):
        pytest.fail('Changed source must be rejected before pixel access')
    monkeypatch.setattr(study, 'preprocess_image', never_load)
    with pytest.raises(ValueError, match='Source JPEG changed'):
        next(study.samples('check'))


def test_fixed_support_check_ids_exclude_all_previous_bsds_usage():
    manifest = study.read(study.DATA/'v3_manifest.json')
    previous = study.read(study.ROOT/'refinement_protocol.json')
    protocol = study.read(study.PLAN)
    used = {r['id'] for r in manifest['images']} | {r['id'] for r in previous['independent_check']}
    check = [r['id'] for r in protocol['independent_check']]
    assert len(check) == len(set(check)) == 4
    assert not set(check)&used
