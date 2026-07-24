import numpy as np
import mne
import pytest

from src.analysis.classifier import extract_band_power_features, decode_with_lda


def _n_data_channels(epochs):
    return len(mne.pick_types(epochs.info, eeg=True))


def _make_two_class_epochs(sfreq: float = 250.0, n_epochs_per_class: int = 20, seed: int = 0) -> tuple:
    # Class A oscillates strongly at 10 Hz (mu band) on channel 0, class B is
    # flat noise there -- a synthetic stand-in for "channel 0 shows real ERD
    # between conditions" without needing real motor-imagery data. Duration
    # (2s) and sfreq (250 Hz) just need to give compute_psd(method="welch")
    # enough samples for a stable 8-30 Hz estimate; not tied to EEGBCI's real
    # 160 Hz.
    duration = 2.0
    n_samples = int(sfreq * duration)
    times = np.arange(n_samples) / sfreq
    rng = np.random.RandomState(seed)
    n_total = n_epochs_per_class * 2

    data = rng.normal(0, 1e-6, size=(n_total, 2, n_samples))
    for i in range(n_epochs_per_class):
        data[i, 0, :] += 5e-6 * np.sin(2 * np.pi * 10.0 * times + rng.uniform(0, 2 * np.pi))

    info = mne.create_info(["EEG1", "EEG2"], sfreq=sfreq, ch_types="eeg")
    epochs = mne.EpochsArray(data, info, tmin=0.0, verbose=False)

    y = np.array([1] * n_epochs_per_class + [2] * n_epochs_per_class)
    return epochs, y


def test_extract_band_power_features_returns_one_row_per_epoch_one_column_per_channel(synthetic_epochs):
    X = extract_band_power_features(synthetic_epochs, fmin=8.0, fmax=13.0)

    assert isinstance(X, np.ndarray)
    assert X.shape == (len(synthetic_epochs), _n_data_channels(synthetic_epochs))


def test_extract_band_power_features_fmax_above_nyquist_raises(synthetic_epochs):
    # Different failure mode than band_power() in features.py: that function
    # computes the full PSD first and filters columns afterward (so an
    # out-of-range band just yields NaN), but extract_band_power_features()
    # passes fmin/fmax straight into compute_psd(), which validates fmax
    # against the data's own Nyquist limit and raises immediately -- a
    # louder, earlier failure for the same kind of mistake.
    with pytest.raises(ValueError):
        extract_band_power_features(synthetic_epochs, fmin=500.0, fmax=600.0)


def test_decode_with_lda_returns_one_score_per_fold():
    epochs, y = _make_two_class_epochs()
    X = extract_band_power_features(epochs, fmin=8.0, fmax=13.0)

    scores = decode_with_lda(X, y, n_splits=5)

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (5,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_decode_with_lda_separable_classes_clear_chance():
    # Wiring-bug check per the task spec's own suggestion: feed a class
    # difference concentrated on one channel that's large relative to the
    # noise floor, and confirm the pipeline actually detects it rather than
    # silently returning ~chance due to a scaling/CV bug.
    epochs, y = _make_two_class_epochs(n_epochs_per_class=25)
    X = extract_band_power_features(epochs, fmin=8.0, fmax=13.0)

    scores = decode_with_lda(X, y, n_splits=5)

    assert scores.mean() > 0.8


def test_decode_with_lda_is_reproducible_across_calls():
    # random_state=42 is fixed inside decode_with_lda specifically so the
    # reported accuracy doesn't drift between identical runs -- verify that
    # guarantee actually holds rather than assuming the hardcoded seed works.
    epochs, y = _make_two_class_epochs()
    X = extract_band_power_features(epochs, fmin=8.0, fmax=13.0)

    scores_1 = decode_with_lda(X, y, n_splits=5)
    scores_2 = decode_with_lda(X, y, n_splits=5)

    np.testing.assert_array_equal(scores_1, scores_2)
