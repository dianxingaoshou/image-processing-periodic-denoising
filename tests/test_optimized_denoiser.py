import numpy as np
from optimized_denoiser import V2Config,config_dict,config_from_dict,prepare,restore


def test_config_roundtrip_and_no_reference_api():
    config=V2Config()
    assert config_dict(config_from_dict(config_dict(config)))==config_dict(config)
    image=np.full((65,83),.5)
    p=prepare(image,config)
    np.testing.assert_array_equal(restore(p),image)
    np.testing.assert_array_equal(restore(p,strength=0),image)


def test_ablation_preserves_raw_candidates_before_ambiguity_guard():
    y,x=np.indices((96,96))
    observed=.5+.12*np.sin(2*np.pi*(.12*y+.16*x)+.3)
    p=prepare(observed)
    protected=restore(p)
    unprotected=restore(p,conservative=False)
    assert p.global_assessment['ambiguity_risk']>.99
    assert np.mean((protected-observed)**2)<1e-12
    assert np.mean((unprotected-observed)**2)>.001
