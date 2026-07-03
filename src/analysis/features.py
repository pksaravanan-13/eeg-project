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
    return tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False)
