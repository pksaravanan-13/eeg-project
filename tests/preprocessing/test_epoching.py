import numpy as np
import mne

from src.preprocessing.epoching import make_epochs, summarize_epochs, DEFAULT_EVENT_ID


def _raw_with_stim(ch_name="STI 014", code=1, sfreq=250.0, duration=8.0, n_events=6):
    rng = np.random.RandomState(0)
    n_samples = int(sfreq * duration)
    info = mne.create_info(["EEG 001", "EEG 002", ch_name], sfreq=sfreq, ch_types=["eeg", "eeg", "stim"])
    eeg_data = rng.normal(0, 20e-6, size=(2, n_samples))
    stim_data = np.zeros((1, n_samples))
    event_samples = np.linspace(int(0.5 * sfreq), n_samples - int(0.5 * sfreq), n_events).astype(int)
    stim_data[0, event_samples] = code
    data = np.vstack([eeg_data, stim_data])
    return mne.io.RawArray(data, info, verbose=False), event_samples


def test_make_epochs_default_event_id_and_stim_channel(synthetic_raw):
    # synthetic_raw (conftest) cycles through every DEFAULT_EVENT_ID trigger code on
    # channel 'STI 014', so calling make_epochs with no event_id/stim_channel override
    # must still find every default event type — the backward-compat check for the
    # event_id/stim_channel parameterization fix.
    epochs = make_epochs(synthetic_raw, tmin=-0.2, tmax=0.4)
    assert set(epochs.event_id.keys()) == set(DEFAULT_EVENT_ID.keys())
    assert len(epochs) > 0


def test_make_epochs_custom_event_id_and_stim_channel():
    raw, event_samples = _raw_with_stim(ch_name="MyTrigger", code=7)
    epochs = make_epochs(
        raw, tmin=-0.2, tmax=0.4,
        event_id={"custom": 7},
        stim_channel="MyTrigger",
    )
    assert len(epochs) == len(event_samples)
    assert list(epochs.event_id) == ["custom"]


def test_summarize_epochs_drop_count_matches(synthetic_epochs, capsys):
    summarize_epochs(synthetic_epochs)
    captured = capsys.readouterr()
    expected_dropped = len(synthetic_epochs.drop_log) - len(synthetic_epochs)
    assert f"Epochs dropped : {expected_dropped}" in captured.out
