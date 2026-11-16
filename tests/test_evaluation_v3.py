import numpy as np
import pytest
from evaluation_v3 import tail_mean,estimation_metrics


def test_top20_tail_not_interpolated_quantile():
    assert tail_mean([0,0,0,0,0,0,0,0,2,4])==3


def test_missing_candidates_count_in_denominator_and_conjugate_equivalence():
    masks=dict(contaminated=np.ones((64,64),bool),clean=np.zeros((64,64),bool),
               true_fy=np.full((64,64),.1),true_fx=np.full((64,64),.15))
    base=dict(y=0,x=0,h=64,w=64,frequency_y=-.1,frequency_x=-.15,effective_gain=.5,amplitude=.1)
    d=estimation_metrics(dict(tiles=[dict(base,selected={},candidates=[dict(fy=-.1,fx=-.15)]),dict(base,selected=None,candidates=[])]),masks)
    assert d['candidate_block_coverage']==.5
    assert d['correct_candidate_coverage']==.5
    assert d['processed_block_coverage']==.5
    assert d['frequency_error_mean']==pytest.approx(0)
    assert d['clean_block_processing_rate']==''


def test_clean_control_still_has_block_and_safety_diagnostics():
    t=dict(y=0,x=0,h=64,w=64,selected=None,candidates=[],amplitude=0.,effective_gain=0.)
    masks=dict(clean=np.ones((64,64),bool),contaminated=np.zeros((64,64),bool))
    d=estimation_metrics(dict(tiles=[t],risky_full_subtraction_blocks=0,eligible_blocks=0),masks)
    assert d['total_blocks']==d['clean_block_centres']==1
    assert d['clean_block_processing_rate']==0
    assert d['risky_full_blocks']==0
    assert d['frequency_error_mean']==''
