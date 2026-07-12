import mne
import pytest

from src.preprocessing.artifacts import mark_bad_channels, reject_by_amplitude, log_rejection


def _spiked_epochs(epochs, spike_epoch_idx=0, spike_uv=500e-6):
    data = epochs.get_data(copy=True)
    data[spike_epoch_idx, 0, data.shape[2] // 2] = spike_uv
    return mne.EpochsArray(
        data, epochs.info, events=epochs.events, tmin=epochs.tmin,
        event_id=epochs.event_id, verbose=False,
    )


def test_mark_bad_channels_valid(synthetic_raw):
    ch = synthetic_raw.ch_names[0]
    raw = mark_bad_channels(synthetic_raw.copy(), [ch])
    assert raw.info["bads"] == [ch]


def test_mark_bad_channels_invalid_name_raises(synthetic_raw):
    with pytest.raises(ValueError):
        mark_bad_channels(synthetic_raw.copy(), ["NOT_A_CHANNEL"])


def test_reject_by_amplitude_drops_spiked_epoch(synthetic_epochs):
    spiked = _spiked_epochs(synthetic_epochs, spike_epoch_idx=0)

    clean = reject_by_amplitude(spiked, {"eeg": 150e-6})

    assert len(clean) == len(spiked) - 1
    dropped_event_sample = spiked.events[0, 0]
    remaining_samples = clean.events[:, 0]
    assert dropped_event_sample not in remaining_samples


def test_log_rejection_reports_correct_counts(synthetic_epochs, capsys):
    spiked = _spiked_epochs(synthetic_epochs, spike_epoch_idx=0)
    clean = reject_by_amplitude(spiked, {"eeg": 150e-6})

    log_rejection(spiked, clean)
    captured = capsys.readouterr()

    n_before, n_after = len(spiked), len(clean)
    n_dropped = n_before - n_after
    assert f"Epochs before rejection : {n_before}" in captured.out
    assert f"Epochs after rejection  : {n_after}" in captured.out
    assert f"Dropped                 : {n_dropped}" in captured.out
