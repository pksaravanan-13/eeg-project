import mne
#mne for all EEG processing objects and methods

from pathlib import Path
#used in save_processed to safely create output directories

from src.preprocessing.loader import load_raw
#load_raw lives in loader.py and is exported here to be used in the main preprocessing script

from src.preprocessing.epoching import make_epochs

def apply_filters(raw: mne.io.Raw, l_freq: float, h_freq: float, notch_freq: float) -> mne.io.Raw:
    #Notch first: line noise is a narrow, high-amplitude signal that should be removed so that the bandpass filter does not have to deal with it. The bandpass filter is then applied to remove low and high frequency noise.
    #This order of filtering is important as in cases where the h_freq is set to a value greater than 60 Hz, the bandpass filter will not remove the line noise at 60 Hz, which can then be amplified by the bandpass filter and cause artifacts in the data.
    #So to prevent more noise in the data, the notch filter is applied first
    raw.notch_filter(notch_freq)
    raw.filter(l_freq, h_freq)
    return raw

def save_processed(epochs: mne.Epochs, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    epochs.save(out_path, overwrite=True)
