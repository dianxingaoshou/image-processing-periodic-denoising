"""Protect paired experiments, intensity projection and independent failures."""
import numpy as np

from run_experiments import make_sample, metric_row, evaluate_methods


def test_amplitude_sweep_is_paired_and_reference_free():
    clean=np.full((64,64),.5)
    low,meta_low,_=make_sample(clean,'held_out_id','chirp',.06)
    high,meta_high,_=make_sample(clean,'held_out_id','chirp',.18)
    assert meta_low['seed']==meta_high['seed']
    assert meta_low['parameters']==meta_high['parameters']
    np.testing.assert_allclose(high-clean,3*(low-clean),atol=2e-16)


def test_absent_clean_region_is_not_zero_error():
    clean=np.full((32,32),.5)
    row=metric_row(clean,clean+.01,clean+.005,
                   dict(contaminated=np.ones_like(clean,bool),clean=np.zeros_like(clean,bool)))
    assert row['clean_mse']==''
    assert row['clean_modification_mae']==''
    assert row['contaminated_mse']>0


def test_published_exception_does_not_drop_later_methods(monkeypatch):
    import published_baseline
    def fail(*args,**kwargs):
        raise RuntimeError('deliberate independent-method failure')
    monkeypatch.setattr(published_baseline,'denoise',fail)
    results=list(evaluate_methods(np.full((64,64),.5),
                 dict(fixed_sigma=2.,local_strength=1.,fixed_local_strength=1.,local_config={})))
    by_name={r[0]:r for r in results}
    assert len(by_name)==8
    assert by_name['published_arpenos'][1] is None
    assert 'deliberate' in by_name['published_arpenos'][3]['error']
    assert by_name['local_adaptive'][1] is not None
    assert by_name['without_scale_selection'][1] is not None
