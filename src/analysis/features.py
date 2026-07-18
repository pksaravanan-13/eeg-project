import numpy as np
import mne
from mne.time_frequency import tfr_morlet


def band_power(epochs: mne.Epochs, bands: dict) -> dict:
    psds, freqs = epochs.compute_psd(method="welch").get_data(return_freqs=True)
    result = {}
    for band, (fmin, fmax) in bands.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        result[band] = psds[:, :, idx].mean(axis=-1)
    return result


def compute_erp(epochs: mne.Epochs) -> mne.Evoked:
    # Looks trivial but isn't: the true evoked response is time-locked to
    # the event on every trial, so it survives this average intact, while
    # background noise is uncorrelated trial-to-trial and shrinks toward
    # zero (~1/sqrt(N)) — the "√N argument," M6_concepts.md section 2.
    # Averaging can't undo contamination that survived M2-M5, though —
    # garbage trials just get averaged in alongside good ones.
    return epochs.average()


def compare_conditions(epochs: mne.Epochs, conditions: list) -> dict:
    # Averaging each condition separately (instead of one grand average)
    # preserves real differences between them — a grand average would blend
    # conditions together and could cancel out effects that point opposite ways.
    result = {}
    for condition in conditions:
        result[condition] = epochs[condition].average()  # epochs[tag] selects matching trials
    return result


def compute_tfr(epochs: mne.Epochs, freqs: np.ndarray, n_cycles: int = 7) -> mne.time_frequency.AverageTFR:
    return tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False)
