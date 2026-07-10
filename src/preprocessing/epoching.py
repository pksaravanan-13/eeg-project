import mne

def make_epochs(raw: mne.io.Raw, tmin: float, tmax: float, baseline: tuple = (None, 0),) -> mne.Epochs:
    events = mne.find_events(raw, stim_channel='STI 014')
    event_id = {
        'auditory/left': 1,
        'auditory/right': 2,
        'visual/left': 3,
        'visual/right': 4,
        'smiley': 5,
        'buttonpress': 32,
    }

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
    n_events = len(epochs.drop_log)
    n_kept = len(epochs)
    n_dropped = n_events - n_kept
    print(f"Events found : {n_events}")
    print(f"Epochs kept  : {n_kept}")
    print(f"Epochs dropped : {n_dropped}")
    if n_dropped > 0:
        print("Sample drop reasons:", [r for r in epochs.drop_log if r][:3])

