import mne
import matplotlib.pyplot as plt
from pathlib import Path


def plot_raw(raw: mne.io.Raw, out_path: str = None) -> None:
    fig = raw.plot(show=False)
    _save_or_show(fig, out_path)


def plot_erp(evoked: mne.Evoked, out_path: str = None) -> None:
    fig = evoked.plot(show=False)
    _save_or_show(fig, out_path)


def plot_topomap(evoked: mne.Evoked, times: list, out_path: str = None) -> None:
    fig = evoked.plot_topomap(times=times, show=False)
    _save_or_show(fig, out_path)


def plot_psd(epochs: mne.Epochs, out_path: str = None) -> None:
    fig = epochs.compute_psd().plot(show=False)
    _save_or_show(fig, out_path)


def plot_tfr(power: mne.time_frequency.AverageTFR, baseline: tuple, mode: str = "logratio", out_path: str = None) -> None:
    # mode="logratio": log10(power / baseline_power) per time/frequency
    # point -- 0 = no change, negative = below baseline, positive = above.
    # Chosen over plain percent-change because the log keeps increases and
    # decreases symmetric in magnitude, so the color scale stays readable.
    fig = power.plot(baseline=baseline, mode=mode, show=False)
    if isinstance(fig, list):
        # Some MNE versions return a list of figures (one per pick) instead
        # of a single figure -- normalize so _save_or_show gets one either way.
        fig = fig[0]

    _save_or_show(fig, out_path)


def _save_or_show(fig, out_path: str) -> None:
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
