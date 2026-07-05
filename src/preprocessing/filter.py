import mne
from pathlib import Path
from src.preprocessing.loader import load_raw

def apply_filters(raw: mne.io.Raw, l_freq: float, h_freq: float, notch_freq: float) -> mne.io.Raw:
    raw.notch_filter(notch_freq)
    raw.filter(l_freq, h_freq)
    return raw

def make_epochs(raw: mne.io.Raw, tmin: float, tmax: float) -> mne.Epochs:
    events, event_id = mne.events_from_annotations(raw)
    return mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, preload=True)

def save_processed(epochs: mne.Epochs, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    epochs.save(out_path, overwrite=True)
