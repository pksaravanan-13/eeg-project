import mne
from pathlib import Path
from src.preprocessing.loader import load_raw

def apply_filters(raw: mne.io.Raw, l_freq: float, h_freq: float, notch_freq: float) -> mne.io.Raw:
    raw.notch_filter(notch_freq)
    raw.filter(l_freq, h_freq)
    return raw

def make_epochs(raw: mne.io.Raw, tmin: float, tmax: float) -> mne.Epochs:
    events = mne.find_events(raw, stim_channel='STI 014')
    event_id = {
        'auditory/left': 1,
        'auditory/right': 2,
        'visual/left': 3,
        'visual/right': 4,
        'smiley': 5,
        'buttonpress': 32,
    }
    return mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, preload=True)

def save_processed(epochs: mne.Epochs, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    epochs.save(out_path, overwrite=True)
