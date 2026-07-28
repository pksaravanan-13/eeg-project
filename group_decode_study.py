import logging
from pathlib import Path

import mne
import numpy as np
from scipy import stats

from src.preprocessing.loader import load_raw
from src.preprocessing.filter import apply_filters
from src.preprocessing.ica import fit_ica, apply_ica
from src.analysis.classifier import extract_band_power_features, decode_with_lda
from pipeline import load_config

logger = logging.getLogger(__name__)

RUNS = [4, 8, 12]  # imagined left/right fist movement runs, same as M9
EOG_PROXY_CHANNEL = "Fpz"  # EEGBCI ships no true EOG channel (all 64 channels are type "eeg");
# Fpz is the frontal-pole channel closest to eye-blink signal -- same class of substitution
# the README already documents for the missing ECG reference on EEG-only recordings.


def load_subject_epochs(subject: int, cfg: dict) -> mne.Epochs:
    """Load, filter, and epoch one EEGBCI subject's left/right motor-imagery runs."""
    pp = cfg["preprocessing"]
    edf_paths = mne.datasets.eegbci.load_data(subject, RUNS, update_path=False)
    raws = [load_raw(str(p)) for p in edf_paths]
    raw = mne.concatenate_raws(raws)
    raw.rename_channels(lambda ch: ch.strip("."))
    montage = mne.channels.make_standard_montage("standard_1005")
    raw.set_montage(montage, match_case=False)

    raw = apply_filters(raw, pp["bandpass"]["l_freq"], pp["bandpass"]["h_freq"], pp["notch_freq"])

    events, event_id = mne.events_from_annotations(raw)
    target_event_id = {label: event_id[label] for label in ("T1", "T2")}
    return mne.Epochs(
        raw, events, event_id=target_event_id, tmin=-1.0, tmax=4.0, baseline=None, preload=True
    )


def decode_accuracy(epochs: mne.Epochs, cfg: dict) -> float:
    """Mean cross-validated LDA accuracy on mu/beta band-power features."""
    clf_cfg = cfg["classifier"]
    X = extract_band_power_features(epochs, fmin=clf_cfg["feature_fmin"], fmax=clf_cfg["feature_fmax"])
    y = epochs.events[:, -1]
    scores = decode_with_lda(X, y, n_splits=clf_cfg["cv_folds"])
    return scores.mean()


def run_subject(subject: int, cfg: dict) -> tuple:
    """Return (no_cleaning_accuracy, ica_cleaned_accuracy) for one subject.

    Two arms on the same epochs: as-is (no cleaning at all -- the baseline
    M9's own notebook already uses), versus ICA-cleaned. Answers whether
    ICA-based cleaning helps decode accuracy, not just whether the pipeline
    runs.

    Deliberately skips config.yaml's reject_threshold_uv (150uV) amplitude
    rejection stage for this dataset: confirmed empirically that it drops
    100% of trials here, because it was tuned against the sample dataset's
    ~1s auditory epochs, and peak-to-peak amplitude over EEGBCI's much
    longer 5s imagery window blows past the same fixed threshold on every
    trial. A real, discovered limitation of a fixed-threshold criterion
    across differing epoch durations, not a bug to silently work around.
    """
    pp = cfg["preprocessing"]
    epochs = load_subject_epochs(subject, cfg)

    acc_no_cleaning = decode_accuracy(epochs, cfg)

    ica = fit_ica(epochs, n_components=pp["ica"]["n_components"], random_state=pp["ica"]["random_state"])
    eog_indices, _ = ica.find_bads_eog(epochs, ch_name=EOG_PROXY_CHANNEL)
    epochs_cleaned = apply_ica(ica, epochs, exclude=eog_indices)

    acc_ica_cleaned = decode_accuracy(epochs_cleaned, cfg)
    return acc_no_cleaning, acc_ica_cleaned


def paired_test(acc_no_cleaning: np.ndarray, acc_ica_cleaned: np.ndarray) -> dict:
    """Wilcoxon signed-rank test: does ICA cleaning change decode accuracy?"""
    stat, p_value = stats.wilcoxon(acc_ica_cleaned, acc_no_cleaning)
    return {
        "n_subjects": len(acc_no_cleaning),
        "mean_no_cleaning": float(np.mean(acc_no_cleaning)),
        "mean_ica_cleaned": float(np.mean(acc_ica_cleaned)),
        "wilcoxon_stat": float(stat),
        "p_value": float(p_value),
    }


def _format_report(result: dict) -> str:
    verdict = "improves" if result["mean_ica_cleaned"] > result["mean_no_cleaning"] else "does not improve"
    significance = (
        "This is below the conventional 0.05 threshold."
        if result["p_value"] < 0.05
        else "This does not clear the conventional 0.05 threshold -- read as a null result, "
        "not a negative one, given the small subject count."
    )
    return f"""# Group Decode Study: Does ICA-Based Artifact Removal Improve Decode Accuracy?

**Hypothesis:** ICA-based artifact-component removal (EOG-proxy correlation, since PhysioNet
EEGBCI ships no true EOG channel) improves mu/beta band-power decode accuracy relative to no
cleaning at all, across subjects.

**Note on scope:** this study omits config.yaml's amplitude-rejection stage (150uV
peak-to-peak) for this dataset specifically -- confirmed empirically that threshold drops
100% of trials here, since it was tuned against the sample dataset's ~1s auditory epochs, and
peak-to-peak amplitude over EEGBCI's much longer 5s imagery window exceeds it on every trial.
A discovered limitation of a fixed-threshold criterion across differing epoch durations, not
a bug worked around silently.

**Subjects:** {result['n_subjects']} (PhysioNet EEGBCI, IDs: {result['subjects']})

**Mean cross-validated accuracy:**
- No cleaning (baseline): {result['mean_no_cleaning']:.3f}
- ICA-cleaned: {result['mean_ica_cleaned']:.3f}

**Paired Wilcoxon signed-rank test:** statistic={result['wilcoxon_stat']:.3f}, p={result['p_value']:.4f}

**Interpretation:** Across this sample, ICA-based cleaning {verdict} mean decode accuracy
(p={result['p_value']:.4f}). {significance}
"""


def main(subjects: list = None, cfg_path: str = "config.yaml") -> dict:
    subjects = subjects or list(range(1, 16))  # 15 subjects
    cfg = load_config(cfg_path)

    acc_no_cleaning, acc_ica_cleaned, ran = [], [], []
    for subject in subjects:
        try:
            no_clean, ica_clean = run_subject(subject, cfg)
        except Exception:
            logger.exception(f"subject {subject} failed -- continuing with next subject")
            continue
        acc_no_cleaning.append(no_clean)
        acc_ica_cleaned.append(ica_clean)
        ran.append(subject)
        logger.info(f"subject {subject}: no_cleaning={no_clean:.3f}  ica_cleaned={ica_clean:.3f}")

    result = paired_test(np.array(acc_no_cleaning), np.array(acc_ica_cleaned))
    result["subjects"] = ran

    report_path = Path(cfg["paths"]["reports"]) / "group_decode_study.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_format_report(result))
    logger.info(f"wrote report -> {report_path}")

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
