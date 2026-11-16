"""Analytic fixed-phase checks; no clean information enters envelope fitting."""
import json

import numpy as np
import pytest

from local_estimation_v3 import _geometry
from spatial_envelope import bilinear_basis, fit_spatial_envelope


def geometry(shape=(64, 64)):
    y, x, u, v, weight, plane, _ = _geometry(shape)
    carrier = np.sin(2*np.pi*(.14*x+.06*y)+.4)
    return carrier, weight, plane, u, v


@pytest.mark.parametrize('shape', [(32, 32), (64, 64), (48, 80)])
def test_bilinear_envelope_recovers_known_lowdimensional_amplitude(shape):
    carrier, weight, plane, _, _ = geometry(shape)
    basis = bilinear_basis(shape)
    controls = np.array([[.025, .04, .065], [.08, .12, .09], [.035, .06, .045]])
    actual = basis@controls.ravel()
    image = actual*carrier+plane@np.array([.51, .035, -.024])
    fitted = fit_spatial_envelope(image, carrier, weight, plane, .08, smoothness=0., ridge=0.)
    np.testing.assert_allclose(fitted['field'], actual, atol=2e-14)
    np.testing.assert_allclose(fitted['background_coefficients'], [.51, .035, -.024], atol=2e-14)
    assert fitted['sse'] < 1e-23
    assert np.all(basis >= 0)
    np.testing.assert_allclose(basis.sum(axis=-1), 1., atol=1e-15)
    assert not basis.flags.writeable


def test_representable_edge_support_improves_dirty_fit_and_clean_leakage_vs_affine():
    carrier, weight, plane, u, v = geometry()
    _, x = np.indices(carrier.shape)
    actual = .12*np.maximum(1-2*x/(carrier.shape[1]-1), 0.)
    image = actual*carrier+plane@np.array([.5, .03, -.02])
    # Fixed phase/anchor isolates model capability, not candidate detection.
    fitted = fit_spatial_envelope(image, carrier, weight, plane, .12)
    affine = np.concatenate([carrier[..., None]*np.stack([np.ones_like(u), u, v], axis=-1), plane], axis=-1)
    root = np.sqrt(weight.ravel())
    coef = np.linalg.lstsq(affine.reshape(-1, 6)*root[:, None], image.ravel()*root, rcond=None)[0]
    affine_field = np.clip(coef[0]+coef[1]*u+coef[2]*v, 0., 2.5*.12)
    dirty, clean = actual > .001, actual == 0
    grid_error = (fitted['residual']-actual*carrier)**2
    affine_error = ((affine_field-actual)*carrier)**2
    assert grid_error[dirty].mean() < .1*affine_error[dirty].mean()
    assert grid_error[clean].mean() < .1*affine_error[clean].mean()
    assert np.all(fitted['field'] >= 0)
    assert np.all(fitted['field'] <= fitted['amplitude_upper']+1e-14)


def test_weights_remove_held_out_pixels_and_scalar_weight_scale_does_not_retune_regularization():
    carrier, weight, plane, _, _ = geometry((32, 48))
    controls = np.array([[.02, .07, .03], [.01, .09, .02], [.02, .05, .01]])
    image = carrier*(bilinear_basis(carrier.shape)@controls.ravel())+plane@np.array([.4, .02, -.03])
    mask = np.indices(image.shape).sum(axis=0) % 3 != 0
    first = fit_spatial_envelope(image, carrier, weight*mask, plane, .08)
    corrupted = image.copy(); corrupted[~mask] += .7
    second = fit_spatial_envelope(corrupted, carrier, 13*weight*mask, plane, .08)
    np.testing.assert_allclose(second['field'], first['field'], atol=3e-14)
    np.testing.assert_allclose(second['prediction'], first['prediction'], atol=3e-14)
    assert second['sse'] == pytest.approx(13*first['sse'], rel=1e-10)
    assert second['objective'] == pytest.approx(first['objective'], rel=1e-10)


def test_cap_prevents_free_amplitude_growth_and_reports_penalized_objective():
    carrier, weight, plane, _, _ = geometry((32, 32))
    image = .5+.2*carrier
    fitted = fit_spatial_envelope(image, carrier, weight, plane, .02)
    assert fitted['amplitude_upper'] == .05
    assert max(np.ravel(fitted['coefficients'])) <= .05+1e-14
    assert fitted['field'].max() <= .05+1e-14
    actual_sse = float(np.sum(weight*(image-fitted['prediction'])**2))
    assert fitted['sse'] == pytest.approx(actual_sse)
    assert fitted['objective'] == pytest.approx(fitted['data_mse']+fitted['smoothness_penalty']+fitted['amplitude_penalty'])
    json.dumps({k: v for k, v in fitted.items() if not isinstance(v, np.ndarray)}, allow_nan=False)


def test_zero_amplitude_returns_consistent_background_only_prediction():
    carrier, weight, plane, _, _ = geometry((32, 32))
    image = plane@np.array([.42, .04, -.03])
    fitted = fit_spatial_envelope(image, carrier, weight, plane, 0.)
    np.testing.assert_array_equal(fitted['residual'], np.zeros_like(image))
    np.testing.assert_allclose(fitted['prediction'], image, atol=1e-15)
    assert fitted['amplitude_upper'] == fitted['amplitude_penalty'] == fitted['smoothness_penalty'] == 0


@pytest.mark.parametrize('kwargs', [dict(amplitude=-.1), dict(smoothness=-1), dict(ridge=np.nan),
                                   dict(grid_size=1), dict(grid_size=6), dict(amplitude_cap=0)])
def test_invalid_regularization_and_geometry_are_rejected(kwargs):
    carrier, weight, plane, _, _ = geometry((32, 32))
    arguments = dict(amplitude=.1) | kwargs
    with pytest.raises(ValueError):
        fit_spatial_envelope(.5+.1*carrier, carrier, weight, plane, **arguments)


def test_invalid_weights_shapes_and_unconverged_solutions_are_not_used(monkeypatch):
    import spatial_envelope as module
    from types import SimpleNamespace
    carrier, weight, plane, _, _ = geometry((32, 32))
    image = .5+.1*carrier
    with pytest.raises(ValueError):
        fit_spatial_envelope(image, carrier, np.zeros_like(weight), plane, .1)
    with pytest.raises(ValueError):
        fit_spatial_envelope(image, carrier[:, :-1], weight, plane, .1)
    with pytest.raises(ValueError):
        fit_spatial_envelope(image, carrier, weight, np.zeros_like(plane), .1)
    monkeypatch.setattr(module, 'lsq_linear', lambda *args, **kwargs: SimpleNamespace(success=False, message='forced failure'))
    with pytest.raises(RuntimeError, match='did not converge'):
        fit_spatial_envelope(image, carrier, weight, plane, .1)
