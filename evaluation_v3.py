"""Reference metrics plus block-center estimation coverage (evaluation only)."""
import numpy as np
from scipy.ndimage import map_coordinates

from run_experiments import metric_row


def tail_mean(values,fraction=.2):
    values=np.asarray(values,float)
    if not len(values):raise ValueError('A tail statistic needs observations')
    return float(np.mean(np.sort(values)[-max(1,int(np.ceil(fraction*len(values)))):]))


def estimation_metrics(diag,masks,tolerance=.01,gain_threshold=.05):
    """Use actual tile-center frequencies; never compare the blended map to truth.

    Coverage denominators include every dirty/clean block centre, even when a
    candidate was absent or rejected. Frequency error alone is conditional on
    a candidate, so its sample count is returned alongside unconditional recall.
    """
    if not diag:return {}
    counts=dict(dirty=0,clean=0,detected_dirty=0,matched_dirty=0,selected_dirty=0,processed_dirty=0,processed_clean=0,detected_clean=0)
    errors=[];accepted_errors=[]
    for t in diag.get('tiles',[]):
        y=t['y']+(t['h']-1)/2;x=t['x']+(t['w']-1)/2
        centre=np.array([[y],[x]])
        dirty=bool(map_coordinates(np.asarray(masks['contaminated'],float),centre,order=0,mode='nearest')[0])
        clean=bool(map_coordinates(np.asarray(masks['clean'],float),centre,order=0,mode='nearest')[0])
        selected=t.get('selected') is not None
        candidates=t.get('candidates',[])
        valid_selected=selected and not str(t.get('model','')).startswith('rejected') and float(t.get('amplitude',0.))>1e-8
        gain=float(t.get('effective_gain',t.get('suppression',0.)))
        processed=valid_selected and gain>=gain_threshold
        if clean:
            counts['clean']+=1;counts['processed_clean']+=int(processed);counts['detected_clean']+=int(bool(candidates))
        if not dirty:continue
        counts['dirty']+=1;counts['processed_dirty']+=int(processed)
        counts['detected_dirty']+=int(bool(candidates))
        if 'true_fy' not in masks:continue
        fy=float(map_coordinates(np.asarray(masks['true_fy']),centre,order=1,mode='nearest')[0])
        fx=float(map_coordinates(np.asarray(masks['true_fx']),centre,order=1,mode='nearest')[0])
        matched=any(min(np.hypot(c['fy']-fy,c['fx']-fx),np.hypot(c['fy']+fy,c['fx']+fx))<=tolerance for c in candidates)
        counts['matched_dirty']+=int(matched)
        if not valid_selected:continue
        counts['selected_dirty']+=1
        py=float(t['frequency_y']);px=float(t['frequency_x'])
        error=min(float(np.hypot(py-fy,px-fx)),float(np.hypot(py+fy,px+fx)))
        errors.append(error)
        if processed:accepted_errors.append(error)
    def rate(n,d):return counts[n]/counts[d] if counts[d] else ''
    return dict(total_blocks=len(diag.get('tiles',[])),dirty_block_centres=counts['dirty'],clean_block_centres=counts['clean'],
        candidate_block_coverage=rate('detected_dirty','dirty'),correct_candidate_coverage=rate('matched_dirty','dirty'),
        selected_block_coverage=rate('selected_dirty','dirty'),
        processed_block_coverage=rate('processed_dirty','dirty'),clean_block_processing_rate=rate('processed_clean','clean'),
        clean_candidate_rate=rate('detected_clean','clean'),
        frequency_error_count=len(errors),frequency_error_mean=float(np.mean(errors)) if errors else '',
        accepted_frequency_error_mean=float(np.mean(accepted_errors)) if accepted_errors else '',
        risky_full_blocks=diag.get('risky_full_subtraction_blocks',''),
        eligible_blocks=diag.get('eligible_blocks',''))
