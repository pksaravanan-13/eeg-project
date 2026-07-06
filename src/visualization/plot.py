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


def _save_or_show(fig, out_path: str) -> None:
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
