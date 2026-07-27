import numpy as np
import mne

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score


def extract_band_power_features(epochs: mne.Epochs, fmin: float = 8.0, fmax: float = 30.0) -> np.ndarray:
    """Extract per-channel mu/beta band power as classifier input features.

    Args:
        epochs: Cleaned epochs to extract features from.
        fmin: Band lower bound in Hz. Defaults to 8.0 (mu band start).
        fmax: Band upper bound in Hz. Defaults to 30.0 (beta band end) --
            together, fmin/fmax span mu (8-13 Hz) + beta (13-30 Hz), the
            range imagined-movement event-related desynchronization (ERD)
            shows up across.

    Returns:
        An (n_epochs, n_channels) array -- one power value per channel, not
        a whole-scalp average, since collapsing channels would erase the
        left-vs-right lateralized asymmetry the classifier depends on.
    """
    psds = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax).get_data()
    return psds.mean(axis=-1)


def decode_with_lda(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """Cross-validate an LDA decoder on band-power features.

    Args:
        X: (n_epochs, n_channels) feature array, e.g. from
            extract_band_power_features().
        y: Per-trial class labels (event codes).
        n_splits: Stratified K-fold count. Defaults to 5 -- stratified
            because with only a few dozen trials, a random split can easily
            hand one fold a lopsided class mix by chance.

    Returns:
        Per-fold cross-validated accuracy scores. Uses LDA specifically
        (over a more flexible classifier) because with ~64 features and
        only a few dozen trials, anything more flexible has enough freedom
        to fit noise instead of the true mu/beta suppression pattern.
        Scaling happens inside the cross-validation pipeline (not on X
        upfront) so the scaler refits on each training fold only --
        scaling the full dataset first would leak held-out information
        into training. random_state=42 keeps the fold assignment (and
        reported accuracy) reproducible run to run.
    """
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv)
    return scores
