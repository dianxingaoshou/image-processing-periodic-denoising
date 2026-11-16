"""Scale ablation and conjugate-pair matching properties."""
import numpy as np
import pytest
from local_spectral import LocalConfig,prepare,_frequency_distance


def test_conjugate_frequency_matching_across_canonical_boundary():
    assert _frequency_distance(.0001,.1,.0001,-.1) == pytest.approx(.0002)
    assert _frequency_distance(.08,.12,-.08,-.12) == pytest.approx(0.)
    assert _frequency_distance(.08,.12,.08,.12) == pytest.approx(0.)


def test_selection_ablation_keeps_all_analysis_scales():
    y,x=np.indices((256,256))
    image=.5+.1*np.sin(2*np.pi*(.081*y+.143*x)+.7)+.02*np.random.default_rng(12).normal(size=y.shape)
    p=prepare(image,LocalConfig(selection_scale=128))
    selected=[t for t in p.tiles if t['selected'] is not None]
    assert selected
    assert all(t['selected']['requested_scale']==128 for t in selected)
    assert any({c['requested_scale'] for c in t['candidates']}=={64,128,256} for t in selected)
    with pytest.raises(ValueError):
        LocalConfig(fixed_scale=128,selection_scale=128)
