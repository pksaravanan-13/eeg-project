import mne

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

    return mne.Epochs(
        raw,
        events,
        event_id,
        tmin=tmin,
        tmax=tmax,
        preload=True,
    )