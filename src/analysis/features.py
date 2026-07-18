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
    return epochs.average()


def compute_tfr(epochs: mne.Epochs, freqs: np.ndarray, n_cycles: int = 7) -> mne.time_frequency.AverageTFR:
    # Morlet wavelets (not a single FFT) because they use a frequency-
    # dependent window — short at high frequencies, longer at low ones —
    # so power can be resolved in time as well as frequency instead of
    # collapsing the whole epoch into one spectrum (M7_concepts.md §3).
    # This catches effects like alpha suppression that are time-locked but
    # not phase-locked, which ERP averaging (M6) would cancel out as noise.
    #
    # n_cycles is the time/frequency tradeoff knob: more cycles sharpen
    # frequency resolution but blur timing; 7 is a common middle ground
    # for this frequency range (M7_concepts.md §4).
    return tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False)
    # return_itc=False: ITC (phase consistency across trials) is a separate
    # question from power, computed via its own return_itc=True call in the notebook.
