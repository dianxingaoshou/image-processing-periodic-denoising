"""End-to-end inference checks for compact coherent profile subtraction."""
from dataclasses import replace
import json
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from narrow_denoiser import Config,prepare,restore,denoise


@pytest.mark.parametrize('axis',(0,1))
def test_coherent_edge_band_restores_without_touching_outside_support(axis):
    y,x=np.indices((128,128));normal=y if axis==0 else x
    tangent=x if axis==0 else y
    noise=.12*(normal<8)*np.sin(2*np.pi*(.13*normal+.1*tangent)+.4)
    observed=.5+noise;output,diag=denoise(observed,return_diagnostics=True)
    assert np.mean((output-.5)**2)<1e-12
    np.testing.assert_array_equal(output[normal>=8],observed[normal>=8])
    assert not diag['support_mask'][normal>=8].any()
    json.dumps(diag['profiles'],allow_nan=False)


def test_broad_ambiguous_periodic_image_abstains_with_compact_support():
    y,x=np.indices((128,128));observed=.5+.12*np.sin(2*np.pi*(.13*y+.1*x))
    p=prepare(observed)
    np.testing.assert_array_equal(restore(p),observed)
    # This ablation is expected to remove genuine periodic content as well.
    unguarded=restore(p,use_gate=False,compact_support=False)
    assert np.mean(np.abs(unguarded-observed))>.05


def test_zero_strength_and_short_tangent_abstain():
    y,x=np.indices((32,32));observed=.5+.1*(x<8)*np.sin(2*np.pi*.12*y)
    np.testing.assert_array_equal(denoise(observed),observed)
    p=prepare(np.tile(observed,(4,4)))
    np.testing.assert_array_equal(restore(p,strength=0.),p.image)
    with pytest.raises(ValueError):restore(p,strength=1.1)


def test_same_profiles_are_reused_without_mutating_ablation_inputs():
    y,x=np.indices((96,96));a=.5+.1*(x<8)*np.cos(2*np.pi*.15*y)
    p=prepare(a);before=[r['coefficients'].copy() for r in p.profiles]
    restore(p,use_gate=False,compact_support=False)
    restore(p,use_gate=True,compact_support=True)
    for item,expected in zip(p.profiles,before):np.testing.assert_array_equal(item['coefficients'],expected)
    np.testing.assert_array_equal(p.image,a)


def test_cli_records_configuration_and_input_without_reference_metrics(tmp_path):
    y,x=np.indices((96,96));a=.5+.12*(x<8)*np.sin(2*np.pi*.13*y)
    path=tmp_path/'input.png';Image.fromarray(np.rint(a*255).astype('uint8')).save(path)
    output=tmp_path/'output'
    result=subprocess.run([sys.executable,'filter_project.py',str(path),'--out',str(output)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    record=json.loads((output/'decisions.json').read_text(encoding='utf-8'))
    assert record['algorithm']=='project_phase'
    assert record['configuration']['narrow']['compact_support']
    assert record['reference_metrics'].startswith('not computed')
    assert len(record['configuration_provenance']['effective_configuration_sha256'])==64
    restored=np.load(output/'restored.npy')
    assert restored.shape==a.shape and np.isfinite(restored).all()


@pytest.mark.parametrize('kwargs',(dict(strength=np.nan),dict(use_gate='yes'),dict(compact_support=1)))
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):Config(**kwargs)
