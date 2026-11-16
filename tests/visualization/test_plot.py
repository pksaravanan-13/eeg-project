import matplotlib

matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt
import mne
import numpy as np
import pytest

import src.visualization.plot as plot_mod
from src.visualization.plot import plot_itc, plot_tfr

CH_NAMES = ["Fp1", "Fp2", "F3", "F4", "C3"]
FREQS = np.array([6.0, 10.0, 14.0, 20.0])
TIMES = np.linspace(-0.2, 0.5, 36)


def _tfr_array(seed: int, low: float, high: float) -> mne.time_frequency.AverageTFRArray:
    """Real MNE AverageTFRArray: 5 EEG channels of synthetic values in [low, high)."""
    rng = np.random.default_rng(seed)
    info = mne.create_info(CH_NAMES, sfreq=100.0, ch_types="eeg")
    data = rng.uniform(low, high, size=(len(CH_NAMES), FREQS.size, TIMES.size))
    return mne.time_frequency.AverageTFRArray(info=info, data=data, times=TIMES, freqs=FREQS, nave=10)


@pytest.fixture
def figure_counter(monkeypatch):
    """Counts every matplotlib Figure constructed, and open pyplot figures when _save_or_show runs."""
    plt.close("all")
    counts = {"created": 0, "open_at_save": []}

    original_init = matplotlib.figure.Figure.__init__

    def counting_init(self, *args, **kwargs):
        counts["created"] += 1
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "__init__", counting_init)

    original_save = plot_mod._save_or_show

    def spying_save(fig, out_path):
        counts["open_at_save"].append(len(plt.get_fignums()))
        original_save(fig, out_path)

    monkeypatch.setattr(plot_mod, "_save_or_show", spying_save)
    yield counts
    plt.close("all")


def _assert_single_figure(counts, out_path):
    assert counts["created"] == 1, f"{counts['created']} figures created, expected 1"
    assert counts["open_at_save"] == [1], f"open figures at save: {counts['open_at_save']}, expected [1]"
    assert out_path.exists() and out_path.stat().st_size > 0
    assert plt.get_fignums() == [], "figures left open after the call"


def test_plot_tfr_creates_exactly_one_figure_for_multichannel_power(tmp_path, figure_counter):
    power = _tfr_array(seed=0, low=0.5, high=2.0)  # positive power, logratio-safe
    out_path = tmp_path / "tfr.png"

    plot_tfr(power, baseline=(None, 0), out_path=str(out_path))

    _assert_single_figure(figure_counter, out_path)


def test_plot_itc_creates_exactly_one_figure_for_multichannel_itc(tmp_path, figure_counter):
    itc = _tfr_array(seed=1, low=0.0, high=1.0)  # ITC lies in [0, 1]
    out_path = tmp_path / "itc.png"

    plot_itc(itc, out_path=str(out_path))

    _assert_single_figure(figure_counter, out_path)
