import mne 
import numpy as np

def mark_bad_channels(raw: mne.io.Raw, bad_channels: list) -> mne.io.Raw:
    raw.info['bads'] = bad_channels
    print(f"Marked {len(bad_channels)} channel(s) as bad: {bad_channels}")
    return raw

def reject_by_amplitude(epochs: mne.Epochs, peak_to_peak_thresh: dict = None) -> mne.Epochs:
    if peak_to_peak_thresh is None:
        peak_to_peak_thresh = {"eeg": 150e-6}
    epochs_clean = epochs.copy()
    epochs_clean.drop_bad(reject=peak_to_peak_thresh)
    return epochs_clean\

def log_rejection(epochs_before: mne.Epochs, epochs_after: mne.Epochs) -> None:
    n_before = len(epochs_before)
    n_after = len(epochs_after)
    n_dropped = n_before - n_after
    pct_kept = 100 * n_after / max(n_before, 1)
    pct_dropped = 100 - pct_kept
    print(f"Epochs before rejection : {n_before}")
    print(f"Epochs after rejection  : {n_after}")
    print(f"Dropped                 : {n_dropped} ({pct_dropped:.1f}%)")
    print(f"Kept                    : {n_after} ({pct_kept:.1f}%)")