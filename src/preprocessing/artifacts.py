import mne
import numpy as np

def mark_bad_channels(raw: mne.io.Raw, bad_channels: list) -> mne.io.Raw:
    """Flag known-bad channels so later steps (filtering, ICA) exclude them.

    Marking (rather than dropping) preserves the channel in raw.ch_names
    so montage/topography plots stay consistent, while info['bads']
    signals to MNE functions like ICA's picks="eeg" to skip it.

    Args:
        raw: Raw signal to mark. Mutated and returned (not copied).
        bad_channels: Channel names to mark bad, e.g. from visual
            inspection or a known faulty electrode for this recording.

    Returns:
        The same Raw object, with info['bads'] set.

    Raises:
        ValueError: if any name in bad_channels isn't in raw.ch_names --
            fails loudly rather than silently marking nothing, since a
            typo here would otherwise leave a bad channel untouched.
    """
    invalid = [ch for ch in bad_channels if ch not in raw.ch_names]
    if invalid:
        raise ValueError(f"Not valid channel names in raw: {invalid}")
    raw.info['bads'] = bad_channels
    print(f"Marked {len(bad_channels)} channel(s) as bad: {bad_channels}")
    return raw

def reject_by_amplitude(epochs: mne.Epochs, peak_to_peak_thresh: dict = None) -> mne.Epochs:
    """Drop whole trials whose peak-to-peak amplitude is physically implausible.

    Runs before ICA (not after): amplitude thresholding removes the
    worst whole-trial contamination first, so ICA's decomposition isn't
    spent fitting components to trials that were always getting thrown
    out anyway.

    Args:
        epochs: Epochs to screen.
        peak_to_peak_thresh: Per-channel-type rejection thresholds in
            volts, e.g. {"eeg": 150e-6}. Defaults to a 150uV EEG
            threshold -- comfortably above real cortical signal
            amplitude, so it catches gross artifacts (movement, loose
            electrodes) without discarding genuine brain activity.

    Returns:
        A copy of epochs with over-threshold trials dropped; the input
        is left untouched so a before/after comparison stays possible.
    """
    if peak_to_peak_thresh is None:
        peak_to_peak_thresh = {"eeg": 150e-6}
    epochs_clean = epochs.copy()
    epochs_clean.drop_bad(reject=peak_to_peak_thresh)
    return epochs_clean

def log_rejection(epochs_before: mne.Epochs, epochs_after: mne.Epochs) -> None:
    """Print how many trials reject_by_amplitude() dropped, and what fraction.

    Args:
        epochs_before: Epochs prior to rejection.
        epochs_after: Epochs after rejection (reject_by_amplitude()'s output).
    """
    n_before = len(epochs_before)
    n_after = len(epochs_after)
    n_dropped = n_before - n_after
    pct_kept = 100 * n_after / max(n_before, 1)
    pct_dropped = 100 - pct_kept
    print(f"Epochs before rejection : {n_before}")
    print(f"Epochs after rejection  : {n_after}")
    print(f"Dropped                 : {n_dropped} ({pct_dropped:.1f}%)")
    print(f"Kept                    : {n_after} ({pct_kept:.1f}%)")