import numpy as np
import pytest

import phase_refit as phase


def mixture():
    y,x = np.indices((95,87),dtype=float)
    carrier = np.exp(2j*np.pi*(.117*y+.13*x-.035*x*x/(2*87)))
    stationary = np.exp(2j*np.pi*(.144*y+.12*x))
    amp = .055
    combined = amp*carrier+.041j*stationary
    events = np.where(combined.real>.04,1,np.where(combined.real<-.04,-1,0))
    held = (events!=0)&((x.astype(int)//9+y.astype(int)//11)%2==1)
    return carrier,stationary,amp,combined,events,held


@pytest.mark.parametrize("axis",[0,1])
def test_stationary_texture_is_fitted_as_nuisance_and_retained(axis):
    chirp,tone,amp,combined,events,held = mixture()
    if axis==0:
        chirp,tone,combined,events,held = (a.T for a in (chirp,tone,combined,events,held))
    signal = combined.real
    prediction, record = phase._conditional_band(signal,chirp,tone,events,held,axis,amp)
    assert prediction.shape == signal.shape
    for half in record["halves"]:
        np.testing.assert_allclose(half["carrier_coefficient"],[1.,0.],atol=1e-10)
    assert record["held_signed_agreement"]>.8
    np.testing.assert_allclose(signal-amp*chirp.real,(.041j*tone).real,atol=1e-15)


def test_stationary_tone_alone_contains_no_resolvable_chirp():
    chirp,tone,amp,_,events,held=mixture()
    assert phase._conditional_band(tone.real,chirp,tone,events,held,1,amp) is None
    assert phase._conditional_band(tone.real,tone,tone,events,held,1,amp) is None


def test_held_sign_reversal_rejects_a_brightness_only_fit():
    chirp,tone,amp,combined,events,held=mixture()
    assert phase._conditional_band(combined.real,chirp,tone,-events,held,1,amp) is None


@pytest.mark.parametrize("axis",[0,1])
def test_exterior_fit_is_independent_of_protected_pixels(axis):
    shape=np.array([180,219])
    coords=[(np.arange(n)-(n-1)/2)/n for n in shape]
    freq=np.array([20.,24.])
    total=freq[0]*coords[0][:,None]+freq[1]*coords[1]
    total+=.5*12*(coords[0][:,None]**2 if axis==0 else coords[1]**2)
    observed=.47+.08*np.cos(2*np.pi*total+.3)
    weights=np.ones(shape[axis])
    weights[20:80]=0.
    changed=observed.copy()
    if axis==0:
        changed[20:80,:]=.2
    else:
        changed[:,20:80]=.2
    first,model1=phase._external_wave(observed,axis,coords,weights,freq,12.)
    second,model2=phase._external_wave(changed,axis,coords,weights,freq,12.)
    np.testing.assert_array_equal(first,second)
    assert model1==model2
    assert abs(model1["amplitude"]-.08)<.002


def test_no_refit_on_early_texture_or_endpoint_vetoes():
    image=np.full((160,160),.5)
    assert phase.refine(image,{},{"reason":"collision_with_measured_peak"})[0] is None
    assert phase.review_abstention(image,{"reason":"clean"})[0] is None
    diag=dict(reason="recurring_narrow_texture",evidence=dict(guarded_carrier=dict(reason="no_independent_verified_carrier")))
    assert phase.review_abstention(image,diag)[2]["reason"]=="insufficient_clip_events"
