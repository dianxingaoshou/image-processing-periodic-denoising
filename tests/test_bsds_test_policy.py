"""Data availability policy checks use temporary manifests, never downloads."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from conftest import _missing_bsds_inputs


@pytest.mark.parametrize('dataset,filename,key',(('legacy','manifest.json','clean_npy'),('v3','v3_manifest.json','npy_path')))
def test_present_manifest_does_not_imply_present_reference_arrays(tmp_path,dataset,filename,key):
    data=tmp_path/'data';data.mkdir()
    path=data/filename
    path.write_text(json.dumps({'images':[{'split':'audit_v3',key:'clean/one.npy'}]}),encoding='utf-8')
    assert _missing_bsds_inputs(tmp_path,dataset)==[data/'clean/one.npy']
    (data/'clean').mkdir();(data/'clean/one.npy').write_bytes(b'present')
    assert _missing_bsds_inputs(tmp_path,dataset)==[]


def test_missing_manifest_is_reported_and_corruption_is_not_hidden(tmp_path):
    data=tmp_path/'data';data.mkdir()
    assert _missing_bsds_inputs(tmp_path,'legacy')==[data/'manifest.json']
    (data/'manifest.json').write_text('invalid json',encoding='utf-8')
    with pytest.raises(json.JSONDecodeError):_missing_bsds_inputs(tmp_path,'legacy')


def test_missing_data_is_explicitly_skipped_or_fails_full_validation(tmp_path):
    root=Path(__file__).resolve().parents[1]
    for name in ('conftest.py','pytest.ini'):
        shutil.copyfile(root/name,tmp_path/name)
    (tmp_path/'test_missing.py').write_text(
        "import pytest\n@pytest.mark.requires_bsds(dataset='legacy')\n"
        "def test_missing():\n    raise AssertionError('must not run without references')\n",encoding='utf-8')
    command=[sys.executable,'-m','pytest','-q','-rs','test_missing.py']
    offline=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert offline.returncode==0 and '1 skipped' in offline.stdout
    assert 'BSDS integration not verified' in offline.stdout
    strict=subprocess.run(command+['--require-bsds'],cwd=tmp_path,capture_output=True,text=True)
    assert strict.returncode!=0
    assert 'BSDS integration not verified' in strict.stdout
    assert '1 skipped' not in strict.stdout
