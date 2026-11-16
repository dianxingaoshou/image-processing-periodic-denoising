"""Boundary checks for preprocessing, explicit configuration and freeze scope."""
import copy
import json
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from image_io_v3 import to_float_gray, load_gray, preprocess_image, structure_seed
from filter_image_v3 import resolve_config
from reproducibility_v3 import capture_snapshot, verify_snapshot


def test_rgb_cli_and_dataset_use_identical_float_luminance(tmp_path):
    from data_pipeline import _preprocess
    rng = np.random.default_rng(8)
    rgb = rng.integers(0, 256, size=(79, 113, 3), dtype=np.uint8)
    path = tmp_path / 'rgb.png'; Image.fromarray(rgb).save(path)
    cli = load_gray(path)
    np.testing.assert_array_equal(cli, to_float_gray(rgb))
    prepared, metadata = preprocess_image(path, 64)
    np.testing.assert_array_equal(prepared, _preprocess(path, 64))
    assert metadata['output_shape'] == [64, 64]
    assert metadata['contrast_normalization'] is False
    # Saturated red exposes the old convert('L')/floating-luminance mismatch.
    assert to_float_gray(np.array([[[255, 0, 0]]], np.uint8))[0, 0] == .2125


def test_rgba_alpha_policy_and_grayscale_precision(tmp_path):
    rgba = np.array([[[255, 0, 0, 0], [255, 0, 0, 255]]], np.uint8)
    np.testing.assert_allclose(to_float_gray(rgba), [[1., .2125]], atol=1e-16)
    np.testing.assert_allclose(to_float_gray(rgba, rgba_background=0.), [[0., .2125]], atol=1e-16)
    gray = np.array([[.123456789012345, .9]], dtype=np.float64)
    np.testing.assert_array_equal(to_float_gray(gray), gray)
    path = tmp_path / 'alpha.png'; Image.fromarray(rgba).save(path)
    loaded, metadata = load_gray(path, return_metadata=True)
    np.testing.assert_array_equal(loaded, to_float_gray(rgba))
    assert metadata['alpha_policy'] == 'composite before luminance'


@pytest.mark.parametrize('image', [np.array([[-.1]]), np.array([[np.nan]]),
                                  np.zeros((2, 2, 2)), np.ones((2, 2), dtype=np.int16)])
def test_invalid_or_ambiguous_intensities_are_rejected(image):
    with pytest.raises(ValueError):
        to_float_gray(image)


def test_missing_config_cannot_silently_change_operating_point(tmp_path):
    path = tmp_path / 'missing.json'
    with pytest.raises(FileNotFoundError, match='Configuration file not found'):
        resolve_config(path)
    config, provenance = resolve_config(path, use_default_config=True)
    assert config is None and provenance['source'] == 'explicit --use-default-config'
    cli = subprocess.run([sys.executable, 'filter_image_v3.py', str(tmp_path/'unused.png'), '--config', str(path)],
                         capture_output=True, text=True)
    assert cli.returncode != 0 and 'Configuration file not found' in cli.stderr
    assert 'ModuleNotFoundError' not in cli.stderr


def test_configuration_provenance_hash(tmp_path):
    import hashlib
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'configuration': {'strength': 2.}}), encoding='utf-8')
    config, provenance = resolve_config(path)
    assert config == {'strength': 2.}
    assert provenance['path'] == str(path.resolve())
    assert provenance['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_cli_output_records_shared_preprocessing_and_config_origin(tmp_path, monkeypatch):
    import types
    from filter_image_v3 import main
    class FakeConfig:
        def __init__(self, strength=1.):self.strength=strength
    fake=types.SimpleNamespace(Config=FakeConfig,
        config_from_dict=lambda d:FakeConfig(**d),config_dict=lambda c:{'strength':c.strength},
        denoise=lambda a,c,return_diagnostics:(a.copy(),{'global_assessment':{},'tiles':[]}))
    monkeypatch.setitem(sys.modules,'denoiser_v3',fake)
    image=tmp_path/'rgb.png'
    rgb=np.zeros((32,32,3),dtype=np.uint8);rgb[...,0]=255
    Image.fromarray(rgb).save(image)
    config=tmp_path/'frozen.json';config.write_text(json.dumps({'configuration':{'strength':.75}}),encoding='utf-8')
    out=tmp_path/'output'
    main([str(image),'--out',str(out),'--config',str(config)])
    np.testing.assert_array_equal(np.load(out/'restored.npy'),to_float_gray(rgb))
    result=json.loads((out/'decisions.json').read_text(encoding='utf-8'))
    assert result['configuration_provenance']['path']==str(config.resolve())
    assert result['configuration_provenance']['sha256']
    assert result['preprocessing']['grayscale_weights']==[.2125,.7154,.0721]
    assert 'no clean reference' in result['reference_metrics']
    assert 'psnr' not in result and 'ssim' not in result


def test_cli_without_config_uses_coupled_profile(tmp_path,monkeypatch):
    import types
    from filter_image_v3 import main
    # Different strengths expose accidental fallback to the experimental file.
    configs=tmp_path/'configs';configs.mkdir()
    (configs/'v3_p0_coupled.json').write_text(json.dumps({'configuration':{'strength':.75,'combination':'coupled'}}),encoding='utf-8')
    (configs/'v3_frozen.json').write_text(json.dumps({'configuration':{'strength':2.,'combination':'decoupled'}}),encoding='utf-8')
    fake=types.SimpleNamespace(Config=lambda:None,config_from_dict=lambda d:d,config_dict=lambda c:c,
        denoise=lambda a,c,return_diagnostics:(a.copy(),{'global_assessment':{},'tiles':[]}))
    monkeypatch.setitem(sys.modules,'denoiser_v3',fake)
    image=tmp_path/'input.png';Image.fromarray(np.full((32,32),128,dtype=np.uint8)).save(image)
    monkeypatch.chdir(tmp_path)
    main([str(image),'--out','output'])
    result=json.loads((tmp_path/'output/decisions.json').read_text(encoding='utf-8'))
    assert result['configuration']=={'strength':.75,'combination':'coupled'}
    assert result['configuration_provenance']['path']==str((configs/'v3_p0_coupled.json').resolve())


def test_structure_seed_and_executed_v2_amplitude_pairing():
    from run_experiments import sample_seed, make_sample
    clean = np.full((64, 64), .5)
    for kind in ('global', 'local', 'chirp', 'orientation'):
        samples = [make_sample(clean, 'pairing_check', kind, amplitude) for amplitude in (.06, .12, .18)]
        expected = structure_seed('pairing_check', kind)
        assert all(sample_seed('pairing_check', kind, amplitude) == expected for amplitude in (.06, .12, .18))
        assert all(meta['seed'] == expected for _, meta, _ in samples)
        assert all(meta['parameters'] == samples[0][1]['parameters'] for _, meta, _ in samples)
        for _, _, masks in samples[1:]:
            for key in ('envelope', 'true_fy', 'true_fx', 'contaminated', 'clean'):
                np.testing.assert_array_equal(masks[key], samples[0][2][key])
        np.testing.assert_allclose(samples[-1][0] - clean, 3 * (samples[0][0] - clean), atol=3e-16)


def test_freeze_checks_exact_audit_order_and_source_inventory(tmp_path):
    source = tmp_path / 'algorithm.py'; source.write_text('value = 1\n', encoding='utf-8')
    metrics = tmp_path / 'metrics.py'; metrics.write_text('data_range = 1\n', encoding='utf-8')
    paths = ['algorithm.py', 'metrics.py']
    frozen = capture_snapshot(paths, ['b', 'a'], dependency_names=(), base_dir=tmp_path)
    assert verify_snapshot(frozen, audit_ids=['b', 'a'], base_dir=tmp_path)['all_passed']
    result = verify_snapshot(frozen, audit_ids=['a', 'b'], base_dir=tmp_path, raise_on_error=False)
    assert not result['all_passed'] and not result['checks']['audit_ids_exact_order']
    result = verify_snapshot(frozen, paths=['algorithm.py'], audit_ids=['b', 'a'], base_dir=tmp_path, raise_on_error=False)
    assert not result['checks']['source_manifest_protocol_inventory']
    metrics.write_text('data_range = 255\n', encoding='utf-8')
    with pytest.raises(RuntimeError, match='source_manifest_protocol_inventory'):
        verify_snapshot(frozen, audit_ids=['b', 'a'], base_dir=tmp_path)


def test_freeze_rejects_corrupted_snapshot_and_missing_actual_ids(tmp_path):
    source = tmp_path/'source.py'; source.write_text('pass\n', encoding='utf-8')
    frozen = capture_snapshot([source], ['1', '2'], dependency_names=(), base_dir=tmp_path)
    changed = copy.deepcopy(frozen); changed['audit_ids'].reverse()
    result = verify_snapshot(changed, audit_ids=['2', '1'], base_dir=tmp_path, raise_on_error=False)
    assert not result['checks']['snapshot_integrity']
    result = verify_snapshot(frozen, base_dir=tmp_path, raise_on_error=False)
    assert not result['checks']['audit_ids_exact_order']


def test_dependency_freeze_can_include_installed_binary_artifacts(tmp_path):
    source=tmp_path/'source.py';source.write_text('pass\n',encoding='utf-8')
    frozen=capture_snapshot([source],['only'],dependency_names=('Pillow',),base_dir=tmp_path)
    assert frozen['dependencies']['Pillow']['artifact_count']>0
    assert len(frozen['dependencies']['Pillow']['artifact_digest'])==64
    assert verify_snapshot(frozen,audit_ids=['only'],base_dir=tmp_path)['all_passed']
    result=verify_snapshot(frozen,audit_ids=['only'],base_dir=tmp_path,hash_dependencies=False,raise_on_error=False)
    assert not result['checks']['dependency_hash_policy']
