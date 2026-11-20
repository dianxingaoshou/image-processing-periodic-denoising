"""Checks for full-field validation, lattice protection and axial chirp units."""
import json

import numpy as np
import pytest

import weak_carrier as weak


def pattern(shape=(192,225)):
    y,x = np.indices(shape)
    first,second = .04*y+.02*x, .013*y-.061*x
    texture = np.full(shape,.48)
    for a,b,amplitude in [(1,0,.05),(0,1,.05),(1,1,.057),(2,1,.045),(1,-1,.04),
                           (0,2,.035),(1,2,.04),(2,0,.04)]:
        texture += amplitude*np.cos(2*np.pi*(a*first+b*second)+a)
    return texture


def added(shape, sweep):
    y,x = np.indices(shape)
    h,w = shape
    cy,cx = y-(h-1)/2,x-(w-1)/2
    return .075*np.sin(2*np.pi*(.137*cy-.091*cx+.5*sweep*(cy/h)**2)+.84)


@pytest.mark.parametrize("sweep", [0,-12])
@pytest.mark.parametrize("transpose", [False, True])
def test_blind_carrier_on_odd_rectangle_checks_all_cells_and_matches_physical_units(sweep,transpose):
    texture = pattern()
    wave = added(texture.shape,sweep)
    image = texture+wave
    result,d = weak.find_carrier(image.T if transpose else image,(.025,.45))
    assert result is not None
    json.dumps(d,allow_nan=False)
    if transpose:
        result = result.T
    assert np.mean((wave-result)**2)<.01*np.mean(wave**2)
    freq = np.array(d["center_frequency"])
    hessian = np.array(d["hessian"])[::2]
    if transpose:
        freq,hessian = freq[::-1],hessian[::-1]
    sign = np.sign(freq[0])
    np.testing.assert_allclose(sign*freq,[.137,-.091],atol=.001)
    np.testing.assert_allclose(sign*hessian,[sweep/texture.shape[0]**2,0],atol=1e-5)
    if not sweep:
        assert d["hessian"]==[0,0,0]
    assert d["positive_gain_checking_cells"]==8
    assert d["amplitude"]<=d["training_amplitude"]
    assert d["final_frequency_change_bins"]<=.5
    if d["phase_refined"]:
        assert d["final_check"]["positive_gain_checking_cells"]==8


@pytest.mark.parametrize("portion", ["clean","left_half","bottom_quarter"])
def test_a_partial_stripe_does_not_authorize_subtraction_on_the_other_side(portion):
    texture = pattern()
    h,w = texture.shape
    wave = added(texture.shape,0.)
    mask = np.zeros(texture.shape,bool)
    if portion=="left_half":
        mask[:,:w//2]=True
    if portion=="bottom_quarter":
        mask[h//2:,:w//2]=True
    result,d = weak.find_carrier(texture+mask*wave,(.025,.45))
    assert result is None and d["lattice_peaks"]>=6


def test_quadratic_search_is_optional_and_large_images_skip_spectra(monkeypatch):
    texture = pattern()
    result,d = weak.find_carrier(texture+added(texture.shape,-12),(.025,.45),enable_chirp=False)
    assert result is None and [stage["model"] for stage in d["stages"]]==["stationary"]
    monkeypatch.setattr(weak,"_spectrum",lambda a: pytest.fail("outside the bounded search"))
    for shape in [(128,513),(127,200)]:
        assert weak.find_carrier(np.zeros(shape),(.025,.45))[1]["reason"]=="outside_full_field_search_geometry"


def test_entire_swept_segment_including_crossings_and_frequency_minimum_is_checked():
    shape=np.array([192,256])
    # Axis-x sweep from negative to positive x. A protected peak lies
    # strictly between samples on its path even though both endpoints clear it.
    params=np.array([24.,0.,0.,20.])
    assert not weak._clear(params,shape,np.array([[24.,.33]]),(.025,.45))
    assert weak._clear(params,shape,np.array([[48.,.33]]),(.025,.45))
    assert not weak._clear(params,shape,np.array([[48.,.33]]),(.13,.45))


def test_curvature_refinement_respects_the_declared_sweep_limit():
    h,w=128,132
    y,x=np.indices((h,w))
    cy,cx=(y-(h-1)/2)/h,(x-(w-1)/2)/w
    coords=np.stack((cy,cx,.5*cy**2,.5*cx**2),axis=-1).reshape(-1,4)
    signal=np.cos(2*np.pi*(coords@np.array([16.,18.,weak.MAX_SWEEP+1,0.])))
    fit=weak._optimize(np.array([16.,18.,float(weak.MAX_SWEEP),0.]),0,coords,2*signal/signal.size)
    assert fit[2]==pytest.approx(weak.MAX_SWEEP,abs=1e-6)
