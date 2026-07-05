import mne
from pathlib import Path

def load_raw(file_path: str, preload: bool = True) -> mne.io.Raw:
    path = Path(file_path)
    if path.suffix == ".fif":
        raw = mne.io.read_raw_fif(file_path, preload=preload)
    elif path.suffix == ".edf":
        raw = mne.io.read_raw_edf(file_path, preload=preload)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return raw

def inspect_raw(raw: mne.io.Raw) -> None:
    print(f"Channels  : {len(raw.ch_names)}")
    print(f"Sampling Rate: {raw.info['sfreq']} Hz")
    print(f"Duration  : {raw.times[-1]:.1f} seconds")
    print(f"Channel types: {set(raw.get_channel_types())}")