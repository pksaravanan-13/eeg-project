import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml
import numpy as np
import mne

from src.preprocessing.filter import load_raw, apply_filters, make_epochs, save_processed
from src.preprocessing.artifacts import mark_bad_channels, reject_by_amplitude, log_rejection
from src.analysis.features import band_power, compute_erp, compute_tfr
from src.visualization.plot import plot_erp, plot_topomap, plot_psd

logger = logging.getLogger(__name__)

REJECT_THRESH = {"eeg": 150e-6}


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _hash_file(path: str, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _provenance_path(processed_path: Path) -> Path:
    return processed_path.with_suffix(processed_path.suffix + ".json")


def _write_provenance(processed_path: Path, params: dict, input_hash: str, raw_file: str) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_file": str(raw_file),
        "input_sha256": input_hash,
        "preprocessing": params,
    }
    prov_path = _provenance_path(processed_path)
    prov_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prov_path, "w") as f:
        json.dump(record, f, indent=2)


def _provenance_matches(processed_path: Path, params: dict, input_hash: str) -> bool:
    prov_path = _provenance_path(processed_path)
    if not prov_path.exists():
        return False
    try:
        with open(prov_path) as f:
            record = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    return record.get("input_sha256") == input_hash and record.get("preprocessing") == params


def _figure_is_stale(fig_path: Path, upstream_path: Path) -> bool:
    if not fig_path.exists():
        return True
    return fig_path.stat().st_mtime < upstream_path.stat().st_mtime


def run(subject: str, raw_file: str, cfg: dict, bad_channels: list = None, force: bool = False) -> dict:
    bad_channels = bad_channels or []
    pp = cfg["preprocessing"]
    paths = cfg["paths"]

    # --- Preprocessing (resumable) ---
    processed_path = Path(paths["processed_data"]) / f"{subject}-epo.fif"
    current_params = {
        "bandpass": pp["bandpass"],
        "notch_freq": pp["notch_freq"],
        "epoch_tmin": pp["epoch_tmin"],
        "epoch_tmax": pp["epoch_tmax"],
        "bad_channels": sorted(bad_channels),
        "reject_thresh": REJECT_THRESH,
    }
    input_hash = _hash_file(raw_file)

    if not force and processed_path.exists() and _provenance_matches(processed_path, current_params, input_hash):
        logger.info(f"[{subject}] preprocessing up to date — loading cached epochs from {processed_path}")
        epochs_clean = mne.read_epochs(processed_path)
    else:
        logger.info(f"[{subject}] running preprocessing")
        raw = load_raw(raw_file)
        raw = apply_filters(raw, pp["bandpass"]["l_freq"], pp["bandpass"]["h_freq"], pp["notch_freq"])
        if bad_channels:
            raw = mark_bad_channels(raw, bad_channels)
        epochs = make_epochs(raw, pp["epoch_tmin"], pp["epoch_tmax"])
        epochs_clean = reject_by_amplitude(epochs, REJECT_THRESH)
        log_rejection(epochs, epochs_clean)

        save_processed(epochs_clean, str(processed_path))
        _write_provenance(processed_path, current_params, input_hash, raw_file)
        logger.info(f"[{subject}] preprocessing complete -> {processed_path}")

    # --- Analysis ---
    bands = {k: tuple(v) for k, v in cfg["analysis"]["frequency_bands"].items()}
    powers = band_power(epochs_clean, bands)
    erp = compute_erp(epochs_clean)
    # n_cycles scaled with frequency (not a fixed 7) — a fixed n_cycles=7 needs a 1.75s
    # wavelet window at 4 Hz, longer than this pipeline's ~1s epochs, and crashes compute_tfr.
    tfr_freqs = np.arange(4, 40, 1)
    tfr = compute_tfr(epochs_clean, freqs=tfr_freqs, n_cycles=tfr_freqs / 2.0)

    # --- Visualization (resumable) ---
    fig_dir = Path(paths["figures"])
    fig_specs = {
        "erp": (fig_dir / f"{subject}_erp.png", lambda p: plot_erp(erp, out_path=str(p))),
        "topo": (fig_dir / f"{subject}_topo.png", lambda p: plot_topomap(erp, times=[0.1, 0.2, 0.3], out_path=str(p))),
        "psd": (fig_dir / f"{subject}_psd.png", lambda p: plot_psd(epochs_clean, out_path=str(p))),
    }
    for name, (fig_path, plot_fn) in fig_specs.items():
        if force or _figure_is_stale(fig_path, processed_path):
            plot_fn(fig_path)
            logger.info(f"[{subject}] wrote {name} figure -> {fig_path}")
        else:
            logger.info(f"[{subject}] skipping {name} figure — up to date")

    logger.info(f"[{subject}] done — figures saved to {fig_dir}/")
    return powers


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", help="Subject ID, e.g. sub-01. Omit together with --file to batch-run config.yaml's subjects list.")
    parser.add_argument("--file", help="Path to raw EEG file")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--bad-channels", default="", help="Comma-separated bad channel names, e.g. 'EEG 053'")
    parser.add_argument("--force", action="store_true", help="Recompute even if up-to-date output already exists")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bad_channels = [c.strip() for c in args.bad_channels.split(",") if c.strip()]

    if args.subject and args.file:
        run(args.subject, args.file, cfg, bad_channels=bad_channels, force=args.force)
    else:
        raw_data_dir = cfg["paths"]["raw_data"]
        for subject in cfg.get("subjects", []):
            raw_file = f"{raw_data_dir}/{subject}_raw.fif"
            try:
                run(subject, raw_file, cfg, bad_channels=bad_channels, force=args.force)
            except Exception:
                logger.exception(f"[{subject}] failed — continuing with next subject")
