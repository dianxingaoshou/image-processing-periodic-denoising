"""Exercise the one-image entry point and its saved decision record."""
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

import filter_project
from image_io_v3 import load_gray, sha256_file


def test_default_cli_from_other_directory_roundtrips_an_unchanged_image(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = tmp_path / 'colour.png'
    pixels = np.empty((80, 96, 4), dtype=np.uint8)
    pixels[:] = [21, 98, 174, 179]
    Image.fromarray(pixels).save(source)

    filter_project.main([str(source)])

    out = tmp_path / 'results' / 'own_image_project'
    restored = np.load(out / 'restored.npy')
    observed = load_gray(source)
    np.testing.assert_array_equal(restored, observed)
    with Image.open(out / 'restored.png') as preview:
        np.testing.assert_array_equal(np.asarray(preview), np.rint(restored * 255).astype(np.uint8))
    with np.load(out / 'diagnostics.npz', allow_pickle=False) as maps:
        np.testing.assert_array_equal(maps['residual'], observed - restored)
        assert maps['support_mask'].shape == observed.shape
        assert not maps['support_mask'].any()
    record = json.loads((out / 'decisions.json').read_text(encoding='utf-8'))
    assert record['route'] == 'unchanged'
    assert record['reason'] and isinstance(record['evidence'], dict)
    assert record['preprocessing']['input_sha256'] == sha256_file(source)
    assert record['preprocessing']['input_mode'] == 'RGBA'
    provenance = record['configuration_provenance']
    for branch, path in [('general', filter_project.DEFAULT_GENERAL_CONFIG),
                         ('narrow', filter_project.DEFAULT_NARROW_CONFIG)]:
        assert provenance[branch] == {'file': path.name, 'sha256': sha256_file(path)}
    encoded = json.dumps(record['configuration'], sort_keys=True, separators=(',', ':')).encode()
    assert provenance['effective_configuration_sha256'] == hashlib.sha256(encoded).hexdigest()


def test_cli_refuses_to_overwrite_the_input_image(tmp_path, capsys):
    source = tmp_path / 'restored.png'
    Image.new('L', (64, 64), color=125).save(source)
    before = source.read_bytes()
    with pytest.raises(SystemExit) as error:
        filter_project.main([str(source), '--out', str(tmp_path)])
    assert error.value.code == 2
    assert 'overwrite the input' in capsys.readouterr().err
    assert source.read_bytes() == before
    assert not (tmp_path / 'decisions.json').exists()


def test_invalid_input_exits_without_publishing_outputs(tmp_path):
    source = tmp_path / 'bad.png'
    source.write_bytes(b'not an image')
    out = tmp_path / 'result'
    with pytest.raises(SystemExit) as error:
        filter_project.main([str(source), '--out', str(out)])
    assert error.value.code == 2
    assert not out.exists()
