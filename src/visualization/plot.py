import mne
import matplotlib.pyplot as plt
from pathlib import Path


def plot_raw(raw: mne.io.Raw, out_path: str = None) -> None:
    """Plot a scrollable view of the raw continuous signal.

    Args:
        raw: Raw signal to plot.
        out_path: If given, saves the figure here instead of displaying it
            interactively (see _save_or_show()).
    """
    fig = raw.plot(show=False)
    _save_or_show(fig, out_path)


def plot_erp(evoked: mne.Evoked, out_path: str = None) -> None:
    """Plot the averaged event-related potential waveform, one line per channel.

    Args:
        evoked: Evoked object, e.g. from compute_erp().
        out_path: If given, saves the figure here instead of displaying it.
    """
    fig = evoked.plot(show=False)
    _save_or_show(fig, out_path)


def plot_topomap(evoked: mne.Evoked, times: list, out_path: str = None) -> None:
    """Plot scalp topography snapshots of an evoked response at given times.

    Args:
        evoked: Evoked object to visualize.
        times: Timepoints in seconds to snapshot (e.g. [0.1, 0.2, 0.3]).
        out_path: If given, saves the figure here instead of displaying it.
    """
    fig = evoked.plot_topomap(times=times, show=False)
    _save_or_show(fig, out_path)


def plot_psd(epochs: mne.Epochs, out_path: str = None) -> None:
    """Plot power spectral density, one line per channel per channel-type panel.

    Args:
        epochs: Epochs to compute and plot PSD for.
        out_path: If given, saves the figure here instead of displaying it.
    """
    fig = epochs.compute_psd().plot(show=False)
    _save_or_show(fig, out_path)


def plot_tfr(power: mne.time_frequency.AverageTFR, baseline: tuple, mode: str = "logratio", out_path: str = None) -> None:
    """Plot time-frequency power relative to a pre-stimulus baseline.

    Args:
        power: TFR power, e.g. from compute_tfr().
        baseline: Baseline interval to normalize against, e.g. (None, 0)
            for pre-stimulus.
        mode: Baseline normalization mode. Defaults to "logratio" --
            log10(power / baseline_power) per time/frequency point, so 0 =
            no change, negative = below baseline, positive = above. Chosen
            over plain percent-change because the log keeps increases and
            decreases symmetric in magnitude, so the color scale stays
            readable.
        out_path: If given, saves the figure here instead of displaying it.
    """
    figs = power.plot(baseline=baseline, mode=mode, show=False)
    if isinstance(figs, list):
        # Some MNE versions return a list of figures (one per pick) instead
        # of a single figure -- normalize so _save_or_show gets one either way.
        # Close the rest explicitly: leaving dozens of per-channel Tk figures
        # open (one per EEG channel) exhausts the backend's pixmap resources
        # and crashes the process on a long-enough channel list.
        fig = figs[0]
        for leftover in figs[1:]:
            plt.close(leftover)
    else:
        fig = figs

    _save_or_show(fig, out_path)


def plot_itc(itc: mne.time_frequency.AverageTFR, fmin: float = 4.0, fmax: float = 40.0, out_path: str = None) -> None:
    """Plot inter-trial coherence across time and frequency.

    Args:
        itc: ITC values, e.g. from compute_itc().
        fmin: Lower frequency bound to display in Hz.
        fmax: Upper frequency bound to display in Hz. Defaults (4.0, 40.0)
            are a fallback for standalone/notebook use -- pipeline.py passes
            these explicitly from config.yaml's analysis.tfr_freq_min/max so
            the displayed range always matches what was actually computed.
        out_path: If given, saves the figure here instead of displaying it.
    """
    fig = itc.plot(fmin=fmin, fmax=fmax, combine="mean", show=False)[0]
    # combine="mean" collapses the picked channels into one figure, so
    # plot() returns a one-element list -- [0] unwraps it to the single
    # fig _save_or_show() expects.
    _save_or_show(fig, out_path)


def _save_or_show(fig, out_path: str) -> None:
    """Save a figure to disk if out_path is given, otherwise display it interactively.

    Args:
        fig: Matplotlib figure to save or show.
        out_path: Destination path, or None to display interactively instead.
    """
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
