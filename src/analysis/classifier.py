import numpy as np
import mne

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score


def extract_band_power_features(epochs: mne.Epochs, fmin: float = 8.0, fmax: float = 30.0) -> np.ndarray:
    # fmin/fmax span mu (8-13 Hz) + beta (13-30 Hz): imagined-movement ERD
    # shows up across both. Keeping one value per channel (not a whole-scalp
    # average) preserves the left-vs-right lateralized asymmetry the
    # classifier depends on.
    psds = epochs.compute_psd(method="welch", fmin=fmin, fmax=fmax).get_data()
    return psds.mean(axis=-1)


def decode_with_lda(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> np.ndarray:
    # LDA over a more flexible classifier: with ~64 features and only a few
    # dozen trials, anything more flexible has enough freedom to fit noise
    # instead of the true mu/beta suppression pattern.
    clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())
    # Pipeline (not scaling X upfront) so the scaler refits on the training
    # fold only each time -- scaling the full dataset first would leak
    # held-out information into training.

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # Stratified because with this few trials, a random split can easily
    # hand one fold a lopsided left/right mix by chance. random_state=42
    # keeps the folds (and reported accuracy) reproducible.

    scores = cross_val_score(clf, X, y, cv=cv)
    return scores
