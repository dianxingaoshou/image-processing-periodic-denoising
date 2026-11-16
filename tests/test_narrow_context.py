"""Context rows may affect contrast evidence without becoming output support."""
import numpy as np
import pytest

from narrow_context import amplitude_context,context_flanks


def test_unreliable_high_amplitude_endpoint_is_not_a_background_flank():
    # Only rows 248..251 hypothetically pass coherence. The entire 244..255
    # periodic amplitude run should define contrast context, not the output.
    amplitude=np.full(256,.003)
    amplitude[244:]=[.025,.04,.07,.09,.11,.12,.1,.08,.06,.05,.04,.03]
    seed=(248,252)
    context=amplitude_context(amplitude,*seed,.02)
    assert context==(244,256)
    assert context_flanks(amplitude,*context,16)==(.003,None)
    raw_left,raw_right=context_flanks(amplitude,*seed,16)
    assert raw_right>.02
    # The only outputs are two bounds and two medians. Original support and
    # amplitudes are unchanged; the caller must keep its qualified mask.
    assert seed==(248,252)
    assert isinstance(context,tuple) and all(isinstance(value,int) for value in context)


def test_amplitude_gap_is_not_bridged():
    amplitude=np.full(64,.003)
    amplitude[10:18]=.1;amplitude[19:25]=.08
    assert amplitude_context(amplitude,12,16,.02)==(10,18)
    assert amplitude_context(amplitude,20,23,.02)==(19,25)
    with pytest.raises(ValueError,match='seed interval'):
        amplitude_context(amplitude,15,22,.02)


def test_broad_context_remains_visible_for_callers_width_guard():
    amplitude=np.full(96,.07)
    context=amplitude_context(amplitude,40,45,.02)
    assert context==(0,96) and context[1]-context[0]>32
    assert context_flanks(amplitude,*context,16)==(None,None)


@pytest.mark.parametrize('start,stop,expected',[(0,8,(None,.003)),(56,64,(.003,None)),(24,32,(.003,.003))])
def test_edge_or_interior_observed_flanks(start,stop,expected):
    amplitude=np.full(64,.003);amplitude[start:stop]=.1
    assert context_flanks(amplitude,start,stop,16)==expected


def test_threshold_equality_and_readonly_array_are_preserved():
    amplitude=np.array([.003,.02,.02,.1,.02,.003]);amplitude.setflags(write=False)
    before=amplitude.copy()
    assert amplitude_context(amplitude,3,4,.02)==(1,5)
    np.testing.assert_array_equal(amplitude,before)
    assert not amplitude.flags.writeable


@pytest.mark.parametrize('amplitude,start,stop',[
    ([],0,1),([.1,np.nan],0,1),([.1,np.inf],0,1),([-.1,.1],1,2),
    ([[.1,.1]],0,1),([.1,.1],-1,1),([.1,.1],1,1),([.1,.1],0,3),
    ([.1,.1],True,2),([.1,.1],0,1.5)])
def test_invalid_context_arrays_and_bounds_fail(amplitude,start,stop):
    with pytest.raises(ValueError):amplitude_context(amplitude,start,stop,.02)
    with pytest.raises(ValueError):context_flanks(amplitude,start,stop,16)


@pytest.mark.parametrize('threshold',[0.,-.01,np.nan,np.inf,[.02]])
def test_invalid_amplitude_floor_fails(threshold):
    with pytest.raises(ValueError):amplitude_context([.1,.1],0,1,threshold)


@pytest.mark.parametrize('width',[0,-1,1.5,True])
def test_invalid_flank_width_fails(width):
    with pytest.raises(ValueError):context_flanks([.1,.1],0,1,width)
