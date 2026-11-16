"""Fixed analytic controls for support-context changes (evaluation only).

These controls are deliberately small and predetermined. They are not a
training set or a model-selection benchmark. Legitimate compact periodic
texture is included explicitly because identical pixels cannot prove noise
ownership. No production algorithm imports this module.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt,gaussian_filter


SEED=1729


def _mask_record(reference,observed,envelope=None,*,fy=None,fx=None):
    shape=reference.shape
    if envelope is None:
        envelope=np.zeros(shape,dtype=float)
    contaminated=envelope>.01
    clean=envelope==0.
    adjacent=clean&(distance_transform_edt(clean)<=16) if contaminated.any() else np.zeros(shape,bool)
    masks=dict(contaminated=contaminated,clean=clean,transition=~(contaminated|clean),
               adjacent_clean=adjacent,envelope=envelope.copy(),true_noise=observed-reference)
    if fy is not None and fx is not None:
        masks.update(true_fy=np.full(shape,float(fy)),true_fx=np.full(shape,float(fx)))
    return masks


def _edge_envelope(yy,width):
    result=np.where(yy<width,.5*(1+np.cos(np.pi*np.minimum(yy,width)/width)),0.)
    result[result<1e-6]=0.
    return result


def make_controls(size=256):
    """Return ``(id, clean, observed, masks, metadata)`` for ten fixed cases.

    ``clean`` means the reference interpretation, not a claim that actual
    photographs are physically noise-free. In particular, compact legitimate
    stripes intentionally receive an unchanged reference. Synthetic noise
    masks and carrier truth remain at this evaluation boundary only.

    ``broad_phase_banded_clean`` has a connected 80-row amplitude band with
    only its central 20 rows coherent across tangent halves. The other rows
    switch by pi/2 at the tangent midpoint. Thus qualified-row context can
    appear narrow while the observed amplitude context remains broad.
    """
    if isinstance(size,(bool,np.bool_)) or not isinstance(size,(int,np.integer)) or size<128:
        raise ValueError('Control dimensions must be integer >=128')
    size=int(size);yy,xx=np.indices((size,size),dtype=float)
    cases=[]

    def add(identifier,reference,observed=None,envelope=None,*,fy=None,fx=None,**metadata):
        reference=np.asarray(reference,dtype=np.float64).copy()
        observed=reference.copy() if observed is None else np.asarray(observed,dtype=np.float64).copy()
        if not (np.isfinite(reference).all() and np.isfinite(observed).all() and
                reference.min()>=0 and reference.max()<=1 and observed.min()>=0 and observed.max()<=1):
            raise RuntimeError('Fixed analytic intensities unexpectedly left [0,1]')
        record=dict(source='fixed analytic evaluation controls',seed=SEED,shape=[size,size],
                    reference_interpretation='legitimate content is retained in clean reference',
                    parameters_fixed_before_evaluation=True,**metadata)
        cases.append((identifier,reference,observed,_mask_record(reference,observed,envelope,fy=fy,fx=fx),record))

    band_start=(size-80)//2;band_stop=band_start+80
    core_start=(size-20)//2;core_stop=core_start+20
    broad=(yy>=band_start)&(yy<band_stop)
    coherent=(yy>=core_start)&(yy<core_stop)
    phase_jump=np.where(broad&~coherent&(xx>=size//2),np.pi/2,0.)
    phased=.5+.1*broad*np.sin(2*np.pi*(.03125*yy+.125*xx)+.3+phase_jump)
    add('broad_phase_banded_clean',phased,semantic_label='legitimate broad periodic content',
        condition='80-row amplitude context /20 coherent rows',normal_axis=0,
        band_start=band_start,band_stop=band_stop,coherent_start=core_start,coherent_stop=core_stop,
        tangent_frequency=.125,normal_frequency=.03125,outer_half_phase_jump=float(np.pi/2),
        amplitude=.1,ambiguity_control=False)

    global_texture=.5+.1*np.sin(2*np.pi*(.11*yy+.14*xx)+.4)
    add('global_periodic_clean',global_texture,semantic_label='legitimate global periodic texture',
        condition='globally coherent periodic control',normal_frequency=.11,tangent_frequency=.14,amplitude=.1)

    broad_slope=.5+(.045+.08*yy/(size-1))*np.sin(2*np.pi*(.03125*yy+.125*xx)+.2)
    add('broad_amplitude_slope_clean',broad_slope,semantic_label='legitimate broad periodic texture',
        condition='smooth broad normal amplitude slope',normal_frequency=.03125,tangent_frequency=.125,
        amplitude_min=.045,amplitude_max=.125)

    rng=np.random.default_rng(SEED)
    clutter=gaussian_filter(rng.normal(size=(size,size)),3.,mode='reflect')
    clutter/=max(float(np.max(np.abs(clutter))),1e-15)
    sharp=.32+.19*(xx>size*.53)+.08*(yy>size*.32)+.05*clutter
    add('sharp_edge_clutter_clean',sharp,semantic_label='legitimate nonperiodic scene structure',
        condition='sharp steps and seeded smooth clutter',clutter_sigma=3.)

    legitimate=.5+.12*(yy<8)*np.sin(2*np.pi*(.08*yy+.125*xx)+.4)
    add('compact_legitimate_stripe_clean',legitimate,semantic_label='legitimate compact periodic stripe',
        condition='single-image ambiguity counterexample',normal_frequency=.08,tangent_frequency=.125,
        support_width=8,amplitude=.12,ambiguity_control=True,
        caveat='Pixels also admit the interpretation of added noise on a flat field; protection is not identifiable from input alone.')

    for width in (8,16):
        envelope=_edge_envelope(yy,width)
        added=.12*envelope*np.sin(2*np.pi*(.09375*yy+.1875*xx)+.7)
        add(f'broad_texture_edge{width}',broad_slope,broad_slope+added,envelope,
            fy=.09375,fx=.1875,semantic_label='added narrow interference on legitimate periodic background',
            condition='paired broad background plus added edge band',reference_control='broad_amplitude_slope_clean',
            width=width,amplitude=.12,carrier_phase=.7,normal_frequency=.09375,tangent_frequency=.1875,
            background_tangent_frequency=.125,pairing_id='broad_texture_added_edge')

    envelope=_edge_envelope(yy,16)
    weak_noise=.012*envelope*np.sin(2*np.pi*(.09375*yy+.1875*xx)+.7)
    flat=np.full((size,size),.5)
    add('weak_edge16',flat,flat+weak_noise,envelope,fy=.09375,fx=.1875,
        semantic_label='weak added periodic interference',condition='amplitude below fixed .02 evidence floor',
        width=16,amplitude=.012,carrier_phase=.7)

    # The added band is intentionally not globally tangent-stationary on four
    # normal rows. It checks whether context changes silently erase the gate.
    changed_rows=np.isin(yy,[1,2,6,9])
    phase_mismatch=np.where(changed_rows&(xx>=size//2),np.pi/2,0.)
    fragmented_noise=.12*envelope*np.sin(2*np.pi*(.09375*yy+.1875*xx)+.7+phase_mismatch)
    add('fragmented_coherence_edge16',flat,flat+fragmented_noise,envelope,
        semantic_label='added interference with specified tangent phase mismatch',
        condition='fragmented cross-half coherence',width=16,amplitude=.12,
        normal_frequency=.09375,tangent_frequency=.1875,changed_normal_rows=[1,2,6,9],
        phase_jump=float(np.pi/2),frequency_truth='base carrier only; phase discontinuity is model mismatch')

    background=.45+.06*np.exp(-((xx-.6*size)**2+(yy-.4*size)**2)/(2*(.22*size)**2))+.03*(xx>.7*size)
    coherent_noise=.12*envelope*np.sin(2*np.pi*(.09375*yy+.1875*xx)+.7)
    add('smooth_background_edge16',background,background+coherent_noise,envelope,fy=.09375,fx=.1875,
        semantic_label='added coherent interference on smooth scene content',
        condition='positive narrow-band control',width=16,amplitude=.12,carrier_phase=.7)
    return cases


__all__=['make_controls']
