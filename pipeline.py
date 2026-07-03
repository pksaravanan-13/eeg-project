import yaml
import numpy as np
from pathlib import Path

from src.preprocessing.filter import load_raw, apply_filters, make_epochs, save_processed
from src.analysis.features import band_power, compute_erp, compute_tfr
from src.visualization.plot import plot_erp, plot_topomap, plot_psd


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(subject: str, raw_file: str, cfg: dict) -> None:
    pp = cfg["preprocessing"]
    paths = cfg["paths"]

    # --- Preprocessing ---
    raw = load_raw(raw_file)
    raw = apply_filters(raw, pp["bandpass"]["l_freq"], pp["bandpass"]["h_freq"], pp["notch_freq"])
    epochs = make_epochs(raw, pp["epoch_tmin"], pp["epoch_tmax"])

    processed_path = f"{paths['processed_data']}/{subject}-epo.fif"
    save_processed(epochs, processed_path)

    # --- Analysis ---
    bands = {k: tuple(v) for k, v in cfg["analysis"]["frequency_bands"].items()}
    powers = band_power(epochs, bands)
    erp = compute_erp(epochs)
    tfr = compute_tfr(epochs, freqs=np.arange(4, 40, 1))

    # --- Visualization ---
    fig_dir = paths["figures"]
    plot_erp(erp, out_path=f"{fig_dir}/{subject}_erp.png")
    plot_topomap(erp, times=[0.1, 0.2, 0.3], out_path=f"{fig_dir}/{subject}_topo.png")
    plot_psd(epochs, out_path=f"{fig_dir}/{subject}_psd.png")

    print(f"[{subject}] done — figures saved to {fig_dir}/")
    return powers


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="Subject ID, e.g. sub-01")
    parser.add_argument("--file", required=True, help="Path to raw EEG file")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(args.subject, args.file, cfg)
