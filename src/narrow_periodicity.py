"""Screen accepted narrow regions for a harmonic pair shared with distant profiles."""

import numpy as np

from narrow_profiles import _fit
from narrow_support import SupportConfig


COMPANION_RATIOS = (.5, 2.)
SIDEBAND_BINS = 2.
MIN_RELATIVE_AMPLITUDE = .5
MIN_INSIDE_FRACTION = .5
MIN_REMOTE_FRACTION = .25


def routing_parameters():
    return dict(companion_ratios=list(COMPANION_RATIOS), sideband_offset_bins=SIDEBAND_BINS,
                minimum_relative_amplitude=MIN_RELATIVE_AMPLITUDE, minimum_inside_fraction=MIN_INSIDE_FRACTION,
                minimum_remote_fraction=MIN_REMOTE_FRACTION, remote_count_must_exceed="support.max_width",
                remote_exclusion="amplitude context plus support.flank_width on each side")


def screen_narrow(observed, diagnostics, *, flank_width=SupportConfig.flank_width,
                  max_band_width=SupportConfig.max_width):
    """Return one input-only ambiguity decision per accepted region, in original order.

    Both frequencies must form resolved peaks on the *same* profiles, inside
    the proposal and well outside its measured flanks. A single other compact
    strip cannot supply more supporting lines than the maximum strip width.
    Insufficient remote evidence leaves the existing narrow decision in place.
    """
    axes = (observed, observed.T)
    amplitudes = {}

    def amplitude(axis, frequency):
        key = (axis, frequency)
        if key not in amplitudes:
            rows = axes[axis]
            length = rows.shape[1]
            coefficients, _ = _fit(rows, frequency, np.arange(length), length)
            amplitudes[key] = np.linalg.norm(coefficients[:, :2], axis=1)
        return amplitudes[key]

    def resolved_peak(axis, frequency, spacing, floor):
        center = amplitude(axis, frequency)
        return ((center >= floor) & (center > amplitude(axis, frequency-spacing))
                & (center > amplitude(axis, frequency+spacing)))

    decisions = []
    for region in diagnostics["accepted_regions"]:
        axis = int(region["normal_axis"])
        start, stop = int(region["start"]), int(region["stop"])
        frequency = float(region["tangent_frequency"])
        count, length = axes[axis].shape
        remote = np.ones(count, dtype=bool)
        left = max(0, int(region["context_start"])-flank_width)
        right = min(count, int(region["context_stop"])+flank_width)
        remote[left:right] = False
        remote_count = int(remote.sum())
        decision = dict(normal_axis=axis, start=start, stop=stop, tangent_frequency=frequency,
                        remote_lines=remote_count, remote_carrier_lines=None, companion_frequency=None,
                        inside_paired_lines=None, remote_paired_lines=None, tested_companions=[],
                        confirmed_companions=[], veto=False)
        decisions.append(decision)
        spacing = SIDEBAND_BINS/length
        if remote_count <= max_band_width:
            decision["reason"] = "insufficient_remote_extent"
            continue
        if not spacing < frequency < .5-spacing:
            decision["reason"] = "unresolvable_carrier"
            continue

        floor = MIN_RELATIVE_AMPLITUDE * float(np.median(amplitude(axis, frequency)[start:stop]))
        carrier = resolved_peak(axis, frequency, spacing, floor)
        carrier_remote = int(np.count_nonzero(carrier & remote))
        decision["remote_carrier_lines"] = carrier_remote
        if carrier_remote <= max_band_width or carrier_remote < MIN_REMOTE_FRACTION*remote_count:
            decision["reason"] = "carrier_not_recurrent"
            continue
        decision["reason"] = "no_recurrent_harmonic_pair"
        for ratio in COMPANION_RATIOS:
            companion = frequency * ratio
            # Keep both pairs of sideband probes away from DC/Nyquist and each
            # other's main lobe (the Hann window has first null at two bins).
            if not spacing < companion < .5-spacing or abs(companion-frequency) <= 2*spacing:
                continue
            paired = carrier & resolved_peak(axis, companion, spacing, floor)
            inside_hits = int(paired[start:stop].sum())
            remote_hits = int(np.count_nonzero(paired & remote))
            decision["tested_companions"].append(dict(frequency=companion, inside_lines=inside_hits,
                                                       remote_lines=remote_hits))
            if (inside_hits >= MIN_INSIDE_FRACTION*(stop-start) and remote_hits > max_band_width
                    and remote_hits >= MIN_REMOTE_FRACTION*remote_count):
                decision["confirmed_companions"].append(companion)
                if not decision["veto"]:
                    decision.update(companion_frequency=companion, inside_paired_lines=inside_hits,
                                    remote_paired_lines=remote_hits, reason="recurrent_harmonic_pair", veto=True)
    return decisions
