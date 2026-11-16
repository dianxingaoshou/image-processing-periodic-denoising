"""Attribution must distinguish net restoration from failure mitigation."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import narrow_denoiser as narrow
import narrow_experiments as study


def row(method,*,dirty=.003,clean=0.,adjacent=0.,dataset='BSDS',image_id='analytic',strength=None):
    if strength is None:strength=0. if method=='input' else 2. if method.startswith('legacy') else 1.
    return dict(dataset=dataset,image_id=image_id,kind='edge16',method=method,strength=strength,
        mse=.001,contaminated_mse=dirty,clean_mse=clean,adjacent_clean_mse=adjacent)


def test_primary_rows_respect_old_and_new_strength_families():
    rows=[row('input'),row('legacy64'),row('legacy32_grid'),row('old_peaks_new_support'),row('full'),
          row('legacy32_grid',strength=1.),row('full',strength=.75),row('profile_gate',strength=.5)]
    actual=study.primary_rows(rows)
    assert len(actual)==5
    assert {r['method'] for r in actual}=={'input','legacy64','legacy32_grid','old_peaks_new_support','full'}


def test_ablation_configs_change_only_the_two_declared_flags():
    configs=study.load_configs();full=configs['full']
    assert full.use_gate is True and full.compact_support is True
    expected=dict(proposal_only=(False,False),profile_gate=(True,False),compact_only=(False,True),full=(True,True))
    assert set(configs)==set(expected)
    for name,cfg in configs.items():
        assert (cfg.use_gate,cfg.compact_support)==expected[name]
        assert cfg.profiles==full.profiles and cfg.support==full.support and cfg.strength==full.strength==1.


def test_full_config_cannot_silently_disable_both_ablation_factors(monkeypatch):
    collapsed=replace(narrow.Config(),use_gate=False,compact_support=False)
    monkeypatch.setattr(study.base,'read',lambda path:dict(configuration=narrow.config_dict(collapsed)))
    with pytest.raises(ValueError,match='full|gate|support|ablations'):
        study.load_configs()


def test_less_damage_than_baseline_is_not_input_net_restoration():
    rows=[row('input',dirty=.002819),row('legacy32_grid',dirty=.013712,clean=.02,adjacent=.01),
          row('full',dirty=.008989)]
    actual,=study.paired_rows(rows)
    assert actual['relative_joint']
    assert not actual['input_net_joint']
    assert actual['contaminated_mse_delta']==pytest.approx(.008989-.013712)


def test_input_net_restoration_requires_both_dirty_improvements_and_zero_clean_damage():
    base=[row('input'),row('legacy32_grid')]
    rows=base+[row('full',dirty=.002),row('profile_gate',dirty=.002,clean=.0001),
               row('compact_only',dirty=.002,adjacent=.0001),row('proposal_only',dirty=.003)]
    actual={r['method']:r for r in study.paired_rows(rows)}
    assert actual['full']['relative_joint'] and actual['full']['input_net_joint']
    assert not actual['profile_gate']['input_net_joint'] and actual['profile_gate']['new_clean_damage']
    assert not actual['compact_only']['input_net_joint'] and actual['compact_only']['new_adjacent_damage']
    assert not actual['proposal_only']['input_net_joint']


def test_lower_but_still_positive_clean_damage_is_not_zero_damage_to_input():
    actual,=study.paired_rows([row('input'),row('legacy32_grid',clean=.005,adjacent=.002),
        row('full',dirty=.002,clean=.001,adjacent=.0001)])
    assert actual['relative_joint']
    assert not actual['input_net_joint']


@pytest.mark.parametrize('field',['clean_mse','adjacent_clean_mse'])
def test_missing_clean_evaluation_never_counts_as_joint_success(field):
    baseline=row('legacy32_grid');candidate=row('full',dirty=.002)
    baseline[field]='';candidate[field]=''
    actual,=study.paired_rows([row('input'),baseline,candidate])
    assert not actual['relative_joint'] and not actual['input_net_joint']


def test_pairing_preserves_dataset_identity_and_ignores_nonprimary_strength():
    rows=[row('input'),row('legacy32_grid'),row('input',dataset='BSDS_regression',dirty=.01),
        row('legacy32_grid',dataset='BSDS_regression',dirty=.012),
        row('full',dataset='BSDS_regression',dirty=.009),
        row('full',dirty=0.,strength=.5)]
    actual,=study.paired_rows(rows)
    assert actual['dataset']=='BSDS_regression'
    assert actual['contaminated_mse_delta']==pytest.approx(-.003)
    assert actual['input_net_joint']


def test_reference_hash_failure_precedes_sample_loading(tmp_path,monkeypatch):
    path=tmp_path/'clean.npy';np.save(path,np.full((32,32),.5))
    record=dict(id='dev',npy_path=path.name,clean_sha256=study.sha256_file(path))
    manifest=tmp_path/'v3_manifest.json';study.base.dump(manifest,dict(images=[record],textures=[]))
    plan=tmp_path/'protocol.json';study.base.dump(plan,dict(development_ids=['dev'],source_manifest_sha256=study.sha256_file(manifest)))
    monkeypatch.setattr(study,'DATA',tmp_path);monkeypatch.setattr(study,'PLAN',plan)
    np.save(path,np.full((32,32),.4))
    with pytest.raises(ValueError,match='Reference NPY changed'):next(study.samples('development'))


def test_source_hash_failure_precedes_image_preprocessing(tmp_path,monkeypatch):
    source=tmp_path/'test.jpg';source.write_bytes(b'changed source')
    manifest=tmp_path/'v3_manifest.json';study.base.dump(manifest,dict(images=[],textures=[]))
    plan=tmp_path/'protocol.json';study.base.dump(plan,dict(independent_check=[dict(id='check',source_jpg=source.name,source_sha256='invalid')],
        source_manifest_sha256=study.sha256_file(manifest)))
    monkeypatch.setattr(study,'DATA',tmp_path);monkeypatch.setattr(study,'PLAN',plan)
    monkeypatch.setattr(study,'preprocess_image',lambda *args,**kwargs:pytest.fail('Changed source cannot be read'))
    with pytest.raises(ValueError,match='Source JPEG changed'):next(study.samples('check'))


def test_new_check_ids_exclude_all_previous_bsds_usage():
    manifest=study.base.read(study.DATA/'v3_manifest.json')
    used={r['id'] for r in manifest['images']}
    for name in ('refinement_protocol.json','support_protocol.json'):
        used|={r['id'] for r in study.base.read(study.ROOT/name)['independent_check']}
    ids=[r['id'] for r in study.base.read(study.PLAN)['independent_check']]
    assert len(ids)==len(set(ids))==4
    assert not set(ids)&used


def test_old_peak_control_uses_observed_frequencies_and_same_new_fitter(monkeypatch):
    image=np.full((96,128),.5);cfg=narrow.Config()
    candidates=[dict(fy=.08,fx=.12,selection_score=3.,amplitude=.1,evidence=1.),
                dict(fy=-.08,fx=-.12,selection_score=2.,amplitude=.1,evidence=1.)]
    called=[]
    def fit(a,axis,frequencies,config,*,refine=False):
        np.testing.assert_array_equal(a,image)
        assert config==cfg.profiles
        assert refine is True
        called.append((axis,frequencies))
        return [dict(axis=axis,frequency=f) for f in frequencies]
    monkeypatch.setattr(study,'profiles_from_frequencies',fit)
    prepared=study.old_peak_profiles(image,SimpleNamespace(tiles=[dict(candidates=candidates)]),cfg)
    assert called==[(0,[.12]),(1,[.08])]
    assert prepared.config==cfg and len(prepared.profiles)==2
