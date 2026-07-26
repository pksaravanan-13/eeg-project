import mne
from pathlib import Path

def load_raw(file_path: str, preload: bool = True) -> mne.io.Raw:
    """Load an EEG recording from disk into an MNE Raw object.

    Supports .fif (MNE native) and .edf (clinical/PhysioNet standard)
    formats. This is the single entry point for the entire pipeline --
    every downstream module (filtering onward) starts from the Raw
    object this returns.

    Args:
        file_path: Path to the EEG file. Extension determines the reader.
        preload: If True, loads the full signal into RAM immediately.
            Defaults to True because this pipeline always filters
            immediately after loading, so deferring the load buys
            nothing and would just produce a confusing error at the
            filtering step instead of here.

    Returns:
        The loaded Raw object, with metadata (channel types, sampling
        rate, events) already populated from the file header.

    Raises:
        FileNotFoundError: if file_path does not exist.
        ValueError: if file_path's extension is neither .fif nor .edf.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw EEG file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".fif":
        raw = mne.io.read_raw_fif(file_path, preload=preload)
    elif suffix == ".edf":
        raw = mne.io.read_raw_edf(file_path, preload=preload)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return raw

def inspect_raw(raw: mne.io.Raw) -> None:
    """Print a quick summary of a loaded Raw object's basic properties.

    A cheap sanity check to run right after load_raw() -- confirms the
    channel count, sampling rate, duration, and channel types look like
    what's expected for this dataset before spending time on filtering
    or epoching a file that loaded wrong (wrong montage, truncated
    recording, unexpected channel types).

    Args:
        raw: A loaded Raw object, as returned by load_raw().
    """
    print(f"Channels  : {len(raw.ch_names)}")
    print(f"Sampling Rate: {raw.info['sfreq']} Hz")
    print(f"Duration  : {raw.times[-1]:.1f} seconds")
    print(f"Channel types: {set(raw.get_channel_types())}")