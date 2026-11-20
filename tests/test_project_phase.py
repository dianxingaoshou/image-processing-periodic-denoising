"""The phase retry shares the guarded fit and cannot bypass other texture vetoes."""
import numpy as np
import pytest

import project_phase as pipeline


@pytest.mark.parametrize("reason,accepted", [
    ("inconclusive_exterior_phase_fit", True),
    ("exterior_model_failed_local_confirmation", False),
    ("collision_with_measured_peak", False),
    ("persistent_local_periodic_texture", False),
])
def test_retry_and_fallback(monkeypatch, reason, accepted):
    observed = np.full((32,48), .6)
    prior_wave = np.zeros_like(observed)
    prior_wave[:,:10] = .01
    model = dict(center_frequency=[.11,.07], hessian=[0,0,-.0001], amplitude=.04, phase_radians=1.1)
    calls = []
    def guarded(image, config, *, return_diagnostics):
        calls.append("guarded")
        return image-prior_wave, dict(reason="clip_supported_independent_carrier", evidence={"guarded_carrier":
            dict(model="quadratic", protection={"tapered_pixels":5,"protected_pixels":3})},
            support_mask=prior_wave!=0, carrier_wave=prior_wave, accepted_regions=[{}])
    def old_review(image, diag, *, bounds):
        calls.append("old_review")
        return None, None, dict(reason=reason)
    def retry(image, diag, prior, *, bounds):
        calls.append("retry")
        assert prior is diag["evidence"]["collision_review"] and 0 < bounds[0] < bounds[1]
        if accepted:
            return image-.04, np.full_like(image,.04), dict(applied_model=model)
        return None, None, dict(reason="rejected")
    def previous_peaks(image, base, diag):
        calls.append("peaks")
        return image-prior_wave*.9, prior_wave!=0, {}
    monkeypatch.setattr(pipeline,"guarded",guarded)
    monkeypatch.setattr(pipeline.collision_review,"review",old_review)
    monkeypatch.setattr(pipeline.phase_refit,"refine",retry)
    monkeypatch.setattr(pipeline.peak_guard,"refine",previous_peaks)
    retried = reason in ("inconclusive_exterior_phase_fit","exterior_model_failed_local_confirmation")
    expected_calls = ["guarded","old_review"] + (["retry"] if retried else []) + ([] if accepted else ["peaks"])
    restored, diag = pipeline.denoise(observed,return_diagnostics=True)
    assert calls == expected_calls
    assert diag["reason"] == ("phase_refit_verified_chirp" if accepted else "texture_reviewed_chirp")
    assert ("phase_refit" in diag["evidence"]) == retried
    np.testing.assert_array_equal(diag["residual"], observed-restored)
    np.testing.assert_array_equal(restored[~diag["support_mask"]],observed[~diag["support_mask"]])
    assert diag["accepted_regions"][0]["support_pixels"] == diag["support_mask"].sum()
    if accepted:
        np.testing.assert_array_equal(diag["prior_carrier_wave"],prior_wave)
        assert diag["accepted_regions"][0]["phase_radians"] == model["phase_radians"]
    calls.clear()
    np.testing.assert_array_equal(pipeline.denoise(observed),restored)
    assert calls == expected_calls


@pytest.mark.parametrize("prior,accepted", [
    ("insufficient_clipping_evidence", False),
    ("no_independent_verified_carrier", False),
    ("no_independent_verified_carrier", True),
])
def test_recurring_narrow_veto_preserves_history_and_support(monkeypatch, prior, accepted):
    observed = np.full((64,72), .4)
    wave = np.zeros_like(observed)
    wave[:,5:] = .03
    model = dict(center_frequency=[.1,.15], hessian=[0,0,.001], amplitude=.03, phase_radians=.2)
    def guarded(*args, **kwargs):
        return observed, dict(reason="recurring_narrow_texture", route="unchanged", accepted_regions=[],
                              evidence={"guarded_carrier":{"reason":prior}}, support_mask=wave<0, residual=wave*0)
    calls = []
    def review(image, diag, *, bounds, gain):
        calls.append(gain)
        return (image-wave,wave,dict(applied_model=model)) if accepted else (None,None,dict(reason="rejected"))
    monkeypatch.setattr(pipeline,"guarded",guarded)
    monkeypatch.setattr(pipeline.phase_refit,"review_abstention",review)
    output,diag = pipeline.denoise(observed,return_diagnostics=True)
    assert bool(calls) == (prior=="no_independent_verified_carrier")
    assert diag["route"] == ("general" if accepted else "unchanged")
    np.testing.assert_array_equal(diag["residual"],observed-output)
    np.testing.assert_array_equal(output[~diag["support_mask"]],observed[~diag["support_mask"]])
    if accepted:
        assert diag["evidence"]["prior_reason"] == "recurring_narrow_texture"
        assert diag["accepted_regions"][0]["support_pixels"] == np.count_nonzero(wave)
    np.testing.assert_array_equal(pipeline.denoise(observed),output)
