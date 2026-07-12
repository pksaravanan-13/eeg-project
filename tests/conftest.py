import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import mne
import pytest

from src.preprocessing.epoching import make_epochs, DEFAULT_EVENT_ID

SFREQ = 250.0
DURATION_SEC = 10.0
# Real standard_1020 channel names (not generic 'EEG 00N') so set_montage can supply
# digitization points — plot_topomap raises RuntimeError without them.
STANDARD_CH_NAMES = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]
N_EEG = len(STANDARD_CH_NAMES)
# mne.Epochs raises if any event_id key has zero matches, so cycle through every
# default trigger code (not just one) to keep synthetic data usable with make_epochs' defaults.
CODES = list(DEFAULT_EVENT_ID.values())
N_CYCLES = 2
N_EVENTS = len(CODES) * N_CYCLES


def make_synthetic_raw(seed: int = 0) -> mne.io.RawArray:
    rng = np.random.RandomState(seed)
    n_samples = int(SFREQ * DURATION_SEC)

    ch_names = STANDARD_CH_NAMES + ["STI 014"]
    ch_types = ["eeg"] * N_EEG + ["stim"]
    info = mne.create_info(ch_names, sfreq=SFREQ, ch_types=ch_types)

    # Small noise amplitude (5uV std) so peak-to-peak stays well under the 150uV
    # rejection threshold by chance — deliberate spikes injected by tests should be
    # the only thing that crosses it.
    eeg_data = rng.normal(0, 5e-6, size=(N_EEG, n_samples))
    stim_data = np.zeros((1, n_samples))
    event_samples = np.linspace(int(0.5 * SFREQ), n_samples - int(0.5 * SFREQ), N_EVENTS).astype(int)
    stim_data[0, event_samples] = CODES * N_CYCLES

    data = np.vstack([eeg_data, stim_data])
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage("standard_1020", on_missing="ignore")
    return raw


@pytest.fixture
def synthetic_raw():
    return make_synthetic_raw()


@pytest.fixture
def synthetic_epochs(synthetic_raw):
    return make_epochs(synthetic_raw, tmin=-0.2, tmax=0.4)
