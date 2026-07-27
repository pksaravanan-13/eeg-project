import mne

DEFAULT_EVENT_ID = {
    'auditory/left': 1,
    'auditory/right': 2,
    'visual/left': 3,
    'visual/right': 4,
    'smiley': 5,
    'buttonpress': 32,
}

def make_epochs(
    raw: mne.io.Raw,
    tmin: float,
    tmax: float,
    baseline: tuple = (None, 0),
    event_id: dict = None,
    stim_channel: str = 'STI 014',
) -> mne.Epochs:
    """Cut a continuous, filtered Raw signal into event-locked trial windows.

    Finds events on the stimulus channel and slices out a fixed [tmin, tmax]
    window around each one. Must run after apply_filters() -- epoching first
    would let filter edge-artifacts and drift contaminate trial boundaries
    that can no longer be distinguished from real signal.

    Args:
        raw: Filtered Raw signal to epoch.
        tmin: Window start in seconds relative to each event (negative for
            a pre-event baseline period).
        tmax: Window end in seconds relative to each event.
        baseline: Baseline-correction interval, subtracted per-channel from
            each epoch. Defaults to (None, 0) -- from the start of tmin
            through the event onset -- the standard pre-stimulus baseline,
            so post-stimulus amplitude reflects the true evoked response
            rather than a channel's arbitrary pre-existing offset.
        event_id: Maps condition labels to trigger codes. Defaults to
            DEFAULT_EVENT_ID (this module's sample-dataset auditory/visual
            codes) -- pass a dataset-specific mapping for any other recording.
        stim_channel: Name of the channel carrying trigger codes.

    Returns:
        Preloaded Epochs, one per detected event whose window fits inside
        the recording.
    """
    if event_id is None:
        event_id = DEFAULT_EVENT_ID
    events = mne.find_events(raw, stim_channel=stim_channel)

    return mne.Epochs(
        raw,
        events,
        event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
    )

def summarize_epochs(epochs: mne.Epochs) -> None:
    """Print how many events were found, kept, and dropped, with sample reasons.

    A quick sanity check after epoching -- a very low keep rate usually means
    the event_id mapping doesn't match the recording's actual trigger codes,
    not that the data is unusually noisy.

    Args:
        epochs: Epochs to summarize (typically straight from make_epochs(),
            before any amplitude rejection).
    """
    n_events = len(epochs.drop_log)
    n_kept = len(epochs)
    n_dropped = n_events - n_kept
    print(f"Events found : {n_events}")
    print(f"Epochs kept  : {n_kept}")
    print(f"Epochs dropped : {n_dropped}")
    if n_dropped > 0:
        print("Sample drop reasons:", [r for r in epochs.drop_log if r][:3])

