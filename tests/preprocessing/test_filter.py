import numpy as np
import mne

from src.preprocessing.filter import apply_filters, save_processed


def _make_tone_raw(freqs_amps, sfreq=250.0, duration=8.0):
    n_samples = int(sfreq * duration)
    t = np.arange(n_samples) / sfreq
    data = np.zeros((1, n_samples))
    for freq, amp in freqs_amps:
        data[0] += amp * np.sin(2 * np.pi * freq * t)
    info = mne.create_info(["EEG 001"], sfreq=sfreq, ch_types=["eeg"])
    return mne.io.RawArray(data, info, verbose=False)


def test_apply_filters_preserves_channels(synthetic_raw):
    raw = synthetic_raw.copy()
    filtered = apply_filters(raw, l_freq=1.0, h_freq=40.0, notch_freq=60.0)
    assert filtered.ch_names == synthetic_raw.ch_names


def test_apply_filters_attenuates_out_of_band_power():
    raw = _make_tone_raw([(2.0, 1.0), (55.0, 1.0)])
    filtered = apply_filters(raw, l_freq=1.0, h_freq=40.0, notch_freq=60.0)

    psd = filtered.compute_psd(fmin=0.5, fmax=100, verbose=False)
    freqs = psd.freqs
    power = psd.get_data()[0]

    idx_2hz = np.argmin(np.abs(freqs - 2.0))
    idx_55hz = np.argmin(np.abs(freqs - 55.0))

    assert power[idx_55hz] < power[idx_2hz] * 0.01


def test_save_processed_creates_nested_dirs_and_roundtrips(tmp_path, synthetic_epochs):
    out_path = tmp_path / "nested" / "dir" / "sub-test-epo.fif"
    save_processed(synthetic_epochs, str(out_path))

    assert out_path.exists()
    reloaded = mne.read_epochs(str(out_path), verbose=False)
    assert len(reloaded) == len(synthetic_epochs)
