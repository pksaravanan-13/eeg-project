import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.visualization.plot import plot_tfr


class _FakeMultiFigTFR:
    """Stand-in for AverageTFR.plot() returning one figure per channel --
    the MNE behavior that caused plot_tfr() to leak dozens of unclosed Tk
    figures and crash on a real multi-channel run."""

    def __init__(self, n_figs: int):
        self.n_figs = n_figs

    def plot(self, baseline, mode, show):
        return [plt.figure() for _ in range(self.n_figs)]


def test_plot_tfr_closes_every_leftover_figure_when_power_plot_returns_a_list(tmp_path):
    power = _FakeMultiFigTFR(n_figs=5)
    out_path = tmp_path / "tfr.png"

    plot_tfr(power, baseline=(None, 0), out_path=str(out_path))

    assert out_path.exists()
    assert plt.get_fignums() == []  # every figure, including the saved one, is closed
