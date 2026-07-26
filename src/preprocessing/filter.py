import mne
#mne for all EEG processing objects and methods

from pathlib import Path
#used in save_processed to safely create output directories

from src.preprocessing.loader import load_raw
#load_raw lives in loader.py and is exported here to be used in the main preprocessing script

from src.preprocessing.epoching import make_epochs

def apply_filters(raw: mne.io.Raw, l_freq: float, h_freq: float, notch_freq: float) -> mne.io.Raw:
    """Remove power-line hum and out-of-band drift/noise from a Raw signal.

    Notch-filters first, then bandpass-filters. Order matters: line noise
    is a narrow, high-amplitude signal, and if h_freq is set above the
    line frequency (e.g. 60 Hz), the bandpass filter alone won't remove
    it -- worse, its own passband can pass that noise through mostly
    intact, leaving it to contaminate everything downstream. Notching it
    out first means the bandpass filter only ever has to deal with
    already-clean broadband signal.

    Args:
        raw: Raw signal to filter. Mutated and returned (not copied).
        l_freq: Bandpass low cutoff in Hz -- removes slow drift below this.
        h_freq: Bandpass high cutoff in Hz -- removes noise above this.
        notch_freq: Power-line frequency to notch out (60 Hz in the US,
            50 Hz in Europe).

    Returns:
        The same Raw object, filtered in place.
    """
    raw.notch_filter(notch_freq)
    raw.filter(l_freq, h_freq)
    return raw

def save_processed(epochs: mne.Epochs, out_path: str) -> None:
    """Save processed epochs to disk, creating parent directories as needed.

    Args:
        epochs: Cleaned epochs to persist (typically post-ICA).
        out_path: Destination .fif path. Overwrites any existing file at
            this path -- callers that want provenance tracking of what
            produced a given file should also write a sidecar record
            (see pipeline.py's provenance functions).
    """
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    epochs.save(out_path, overwrite=True)
