"""Context correction must recover rejected cases without hiding regressions."""
from dataclasses import asdict,replace
from types import SimpleNamespace

import numpy as np
import pytest

import narrow_denoiser as narrow
import rejection_experiments as study


def row(method,*,dirty=.003,clean=0.,adjacent=0.,strength=None,dataset='BSDS',image_id='analytic'):
    if strength is None:strength=0. if method=='input' else 1.
    return dict(dataset=dataset,image_id=image_id,kind='edge16',method=method,strength=strength,
        mse=.001,contaminated_mse=dirty,clean_mse=clean,adjacent_clean_mse=adjacent)


def test_context_factorial_changes_only_the_declared_evidence_context():
    configurations=study.load_configs();baseline=configurations['baseline']
    assert set(configurations)=={'baseline','ratio_relaxed','context_no_width','context_full'}
    assert baseline.support.contrast_context=='qualified' and baseline.support.max_flank_ratio==.35
    for name,config in configurations.items():
        expected=asdict(baseline.support)
        if name=='ratio_relaxed':expected['max_flank_ratio']=.9
        elif name=='context_no_width':expected.update(contrast_context='amplitude',limit_context_width=False)
        elif name=='context_full':expected.update(contrast_context='amplitude',limit_context_width=True)
        assert asdict(config.support)==expected
        assert config.profiles==baseline.profiles
        assert config.use_gate is True and config.compact_support is True
        assert config.strength==baseline.strength==1.
        assert config.support.min_amplitude==.02
        assert config.support.min_stability==.8
        assert config.support.min_validation_gain==.08
        assert config.support.min_width==3 and config.support.max_width==32
    # The proposed full method retains the original contrast threshold.
    assert configurations['context_full'].support.max_flank_ratio==.35


def test_recovering_an_old_rejection_is_separate_from_improving_old_success():
    rows=[row('input'),row('baseline'),row('context_full',dirty=.001)]
    actual,=study.paired_rows(rows)
    assert actual['input_net_joint'] and actual['new_recovery']
    assert not actual['old_success']
    rows=[row('input'),row('baseline',dirty=.001),row('context_full',dirty=.0008)]
    actual,=study.paired_rows(rows)
    assert actual['input_net_joint'] and actual['old_success']
    assert not actual['new_recovery']
    assert not actual['old_success_regressed'] and not actual['old_success_lost']


@pytest.mark.parametrize('new_dirty,lost',[(.002,False),(.003,True),(.004,True)])
def test_old_success_regression_is_counted_even_when_still_better_than_input(new_dirty,lost):
    actual,=study.paired_rows([row('input'),row('baseline',dirty=.001),row('context_full',dirty=new_dirty)])
    assert actual['old_success'] and actual['old_success_regressed']
    assert actual['old_success_lost'] is lost
    assert not actual['input_net_joint'] and not actual['new_recovery']


def test_less_bad_than_a_failed_baseline_is_not_new_recovery():
    actual,=study.paired_rows([row('input',dirty=.003),row('baseline',dirty=.01,clean=.005,adjacent=.002),
        row('context_full',dirty=.004)])
    assert actual['contaminated_mse_delta']<0
    assert not actual['input_net_joint'] and not actual['new_recovery']
    assert not actual['old_success']


@pytest.mark.parametrize('field',['clean_mse','adjacent_clean_mse'])
def test_any_new_clean_damage_blocks_strict_recovery_and_can_lose_old_success(field):
    candidate=row('context_full',dirty=.0005);candidate[field]=1e-8
    actual,=study.paired_rows([row('input'),row('baseline',dirty=.001),candidate])
    assert actual['old_success_lost'] and actual['old_success_regressed']
    assert not actual['input_net_joint'] and not actual['new_recovery']
    assert actual['clean_increased'] if field=='clean_mse' else actual['adjacent_increased']


@pytest.mark.parametrize('field',['clean_mse','adjacent_clean_mse'])
def test_missing_clean_evaluation_never_counts_as_a_recovery(field):
    baseline=row('baseline');candidate=row('context_full',dirty=.001)
    baseline[field]='';candidate[field]=''
    actual,=study.paired_rows([row('input'),baseline,candidate])
    assert not actual['local_evaluable']
    assert not actual['input_net_joint'] and not actual['new_recovery']


def test_primary_strength_and_dataset_pairing_are_preserved():
    rows=[row('input'),row('baseline'),row('context_full',dirty=0.,strength=.5),
        row('input',dataset='BSDS_regression',dirty=.01),row('baseline',dataset='BSDS_regression',dirty=.011),
        row('context_full',dataset='BSDS_regression',dirty=.009)]
    actual,=study.paired_rows(rows)
    assert actual['dataset']=='BSDS_regression' and actual['new_recovery']
    assert actual['contaminated_mse_delta']==pytest.approx(-.002)


def test_reference_array_hash_is_checked_before_loading(tmp_path,monkeypatch):
    reference=tmp_path/'clean.npy';np.save(reference,np.full((32,32),.5))
    record=dict(id='development',npy_path=reference.name,clean_sha256=study.sha256_file(reference))
    manifest=tmp_path/'v3_manifest.json';study.base.dump(manifest,dict(images=[record],textures=[]))
    protocol=tmp_path/'protocol.json';study.base.dump(protocol,dict(development_ids=['development'],
        source_manifest_sha256=study.sha256_file(manifest)))
    monkeypatch.setattr(study,'DATA',tmp_path);monkeypatch.setattr(study,'PLAN',protocol)
    np.save(reference,np.full((32,32),.4))
    with pytest.raises(ValueError,match='Reference NPY changed'):next(study.samples('development'))


def test_source_hash_is_checked_before_preprocessing(tmp_path,monkeypatch):
    jpeg=tmp_path/'input.jpg';jpeg.write_bytes(b'changed')
    manifest=tmp_path/'v3_manifest.json';study.base.dump(manifest,dict(images=[],textures=[]))
    protocol=tmp_path/'protocol.json';study.base.dump(protocol,dict(
        source_manifest_sha256=study.sha256_file(manifest),
        independent_check=[dict(id='check',source_jpg=jpeg.name,source_sha256='wrong')]))
    monkeypatch.setattr(study,'DATA',tmp_path);monkeypatch.setattr(study,'PLAN',protocol)
    monkeypatch.setattr(study,'preprocess_image',lambda *args,**kwargs:pytest.fail('Changed JPEG must not be read'))
    with pytest.raises(ValueError,match='Source JPEG changed'):next(study.samples('check'))


def test_new_check_is_disjoint_from_all_prior_bsds_checks():
    manifest=study.base.read(study.DATA/'v3_manifest.json')
    used={r['id'] for r in manifest['images']}
    for name in ('refinement_protocol.json','support_protocol.json','narrow_protocol.json'):
        used|={r['id'] for r in study.base.read(study.ROOT/name)['independent_check']}
    ids=[r['id'] for r in study.base.read(study.PLAN)['independent_check']]
    assert len(ids)==len(set(ids))==4
    assert not set(ids)&used


def test_all_ablations_share_profiles_and_added_pixel_harm_is_not_diluted(tmp_path,monkeypatch):
    root=tmp_path/'source';root.mkdir();(root/'one.py').write_text('fixed\n',encoding='utf-8')
    output=tmp_path/'result'
    y,x=np.indices((32,32));clean=np.full((32,32),.5)
    dirty=y<8;observed=clean+.1*dirty
    masks=dict(contaminated=dirty,clean=~dirty,adjacent_clean=(~dirty)&(y<24))
    monkeypatch.setattr(study,'ROOT',root);monkeypatch.setattr(study,'OUT',output);monkeypatch.setattr(study,'SOURCES',('one.py',))
    monkeypatch.setattr(study,'samples',lambda stage:iter([('analytic',clean,observed,masks,dict(kind='edge16'),{},'analytic',0)]))
    baseline=narrow.Config()
    configurations=dict(baseline=baseline,
        ratio_relaxed=replace(baseline,support=replace(baseline.support,max_flank_ratio=.9)),
        context_no_width=replace(baseline,support=replace(baseline.support,contrast_context='amplitude',limit_context_width=False)),
        context_full=replace(baseline,support=replace(baseline.support,contrast_context='amplitude',limit_context_width=True)))
    monkeypatch.setattr(study,'load_configs',lambda:configurations)
    profile_pool=[object()];prepare_calls=[];restore_calls=[]
    def prepare(image,config):
        prepare_calls.append(True)
        return SimpleNamespace(image=image.copy(),config=config,profiles=profile_pool)
    def restore(prepared,strength=None,return_diagnostics=False):
        assert prepared.profiles is profile_pool
        restore_calls.append(True)
        result=prepared.image.copy();support=np.zeros(result.shape,bool)
        cfg=prepared.config.support
        if cfg.contrast_context=='amplitude' and cfg.limit_context_width:
            support[10,10]=True;result[10,10]-=.1
        return result,dict(support_mask=support,profiles=[])
    monkeypatch.setattr(study.narrow,'prepare',prepare);monkeypatch.setattr(study.narrow,'restore',restore)
    study.run('development')
    assert len(prepare_calls)==1
    assert len(restore_calls)==1+len(study.METHODS)*len(study.STRENGTHS)
    rows=study.base.read_csv(output/'development/metrics.csv')
    full=next(r for r in rows if r['method']=='context_full' and float(r['strength'])==1.)
    assert int(full['added_clean_pixels'])==int(full['added_adjacent_pixels'])==1
    assert float(full['added_clean_mse'])==pytest.approx(.01)
    assert float(full['added_adjacent_mse'])==pytest.approx(.01)
    assert float(full['clean_mse'])<float(full['added_clean_mse'])/100
