import numpy as np
import mne
import pytest

from src.preprocessing.epoching import make_epochs
from src.preprocessing.ica import fit_ica, auto_detect_eog, apply_ica


def _add_correlated_eog(raw, seed=1, scale=3.0):
    # Correlate the synthetic EOG channel with the first EEG channel (scaled +
    # noise) so find_bads_eog has something plausible to score against, rather
    # than pure independent noise.
    rng = np.random.RandomState(seed)
    eeg_data = raw.get_data(picks=[0])
    eog_data = eeg_data * scale + rng.normal(0, 5e-6, size=eeg_data.shape)
    eog_info = mne.create_info(["EOG 001"], sfreq=raw.info["sfreq"], ch_types=["eog"])
    eog_raw = mne.io.RawArray(eog_data, eog_info, verbose=False)
    return raw.copy().add_channels([eog_raw], force_update_info=True)


def test_fit_ica_returns_fitted_ica(synthetic_epochs):
    ica = fit_ica(synthetic_epochs, n_components=4, random_state=42)

    assert isinstance(ica, mne.preprocessing.ICA)
    assert ica.n_components_ == 4
    assert ica.current_fit != "unfitted"


def test_fit_ica_n_components_larger_than_channels_raises(synthetic_epochs):
    # synthetic_epochs (conftest) has 8 EEG channels; fit_ica's own default of
    # 20 exceeds that, so requesting more components than available channels
    # must fail loudly rather than silently truncate.
    with pytest.raises(ValueError):
        fit_ica(synthetic_epochs, n_components=20)


def test_auto_detect_eog_returns_list_when_eog_present(synthetic_raw):
    raw_with_eog = _add_correlated_eog(synthetic_raw)
    epochs = make_epochs(raw_with_eog, tmin=-0.2, tmax=0.4)

    ica = fit_ica(epochs, n_components=4, random_state=42)
    eog_indices = auto_detect_eog(ica, epochs)

    assert isinstance(eog_indices, list)


def test_auto_detect_eog_no_eog_channel_raises(synthetic_epochs):
    # synthetic_epochs (conftest) has no EOG channel at all -- find_bads_eog
    # has nothing to correlate against and must fail rather than return [].
    ica = fit_ica(synthetic_epochs, n_components=4, random_state=42)

    with pytest.raises(RuntimeError):
        auto_detect_eog(ica, synthetic_epochs)


def test_apply_ica_excludes_components_without_mutating_input(synthetic_epochs):
    ica = fit_ica(synthetic_epochs, n_components=4, random_state=42)
    original_data = synthetic_epochs.get_data(copy=True)

    cleaned = apply_ica(ica, synthetic_epochs, exclude=[0])

    assert ica.exclude == [0]
    assert cleaned is not synthetic_epochs
    # apply_ica must work on a copy -- the caller's epochs object is untouched.
    np.testing.assert_array_equal(synthetic_epochs.get_data(copy=True), original_data)
    assert cleaned.get_data(copy=True).shape == original_data.shape
