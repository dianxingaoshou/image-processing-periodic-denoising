"""Numerical properties that matter for reconstruction and experimental validity."""
import numpy as np
import pytest

from local_spectral import LocalConfig, prepare, restore, denoise


def test_odd_rectangle_constant_and_boundaries():
    image = np.full((79,113),0.42)
    result, diag = denoise(image,return_diagnostics=True)
    np.testing.assert_allclose(result,image,atol=1e-12)
    assert np.isfinite(result).all()
    assert diag['suppression'].shape == image.shape


def test_no_reference_interface_and_zero_strength_identity():
    y,x = np.indices((129,151))
    observed = .4 + .1*np.sin(2*np.pi*(.11*y+.17*x))
    prepared = prepare(observed)
    result = restore(prepared,strength=0)
    np.testing.assert_allclose(result,observed,atol=1e-12)


def test_reconstruction_removes_detectable_stripes_without_tile_seams():
    y,x = np.indices((160,192))
    clean = .35+.08*x/192+.08*y/160
    observed = clean + .12*np.sin(2*np.pi*(.13*y+.17*x)+.7)
    result = denoise(observed,LocalConfig(conservative=False))
    input_mse = np.mean((observed-clean)**2)
    assert np.mean((result-clean)**2) < .5*input_mse
    residual = result-clean
    # A gross overlap-normalization error would create a jump at stride boundaries.
    boundary = np.mean(np.abs(np.diff(residual,axis=1)[:,31::32]))
    average = np.mean(np.abs(np.diff(residual,axis=1)))
    assert boundary < 3*average+1e-6


def test_invalid_inputs():
    for a in (np.full((64,64),np.nan),np.ones((64,64))*2,np.zeros((32,32,3))):
        with pytest.raises(ValueError):
            denoise(a)


def test_published_constant_and_finite_odd_size():
    from published_baseline import denoise as published
    a = np.full((97,131),.5)
    b = published(a)
    assert b.shape == a.shape
    assert np.isfinite(b).all()
    np.testing.assert_allclose(b,a,atol=1e-8)
