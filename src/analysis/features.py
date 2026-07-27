import numpy as np
import mne
from mne.time_frequency import tfr_morlet


def band_power(epochs: mne.Epochs, bands: dict) -> dict:
    """Compute average power in named frequency bands, per trial and channel.

    Args:
        epochs: Epochs to analyze.
        bands: Maps a band name (e.g. "alpha") to an (fmin, fmax) tuple in Hz.

    Returns:
        Dict mapping each band name to an (n_epochs, n_channels) array of
        mean PSD power within that band's frequency range. A band whose
        range matches no computed frequency bin returns all-NaN (a mean of
        an empty selection) rather than raising -- the honest signal that
        the request didn't match anything, not a silent zero.
    """
    psds, freqs = epochs.compute_psd(method="welch").get_data(return_freqs=True)
    result = {}
    for band, (fmin, fmax) in bands.items():
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        result[band] = psds[:, :, idx].mean(axis=-1)
    return result


def compute_erp(epochs: mne.Epochs) -> mne.Evoked:
    """Average all trials to produce the event-related potential (ERP).

    Looks trivial but isn't: the true evoked response is time-locked to the
    event on every trial, so it survives this average intact, while
    background noise is uncorrelated trial-to-trial and shrinks toward zero
    (~1/sqrt(N)) -- the "root-N argument." Averaging can't undo contamination
    that survived filtering/rejection/ICA, though -- garbage trials just get
    averaged in alongside good ones.

    Args:
        epochs: Cleaned epochs (post-ICA) to average.

    Returns:
        An Evoked object -- the trial-averaged waveform.
    """
    return epochs.average()


def compare_conditions(epochs: mne.Epochs, conditions: list) -> dict:
    """Average each condition separately rather than collapsing to one grand average.

    A grand average would blend conditions together and could cancel out
    real differences that point in opposite directions between them.

    Args:
        epochs: Cleaned epochs whose event_id tags include every name in
            conditions.
        conditions: Condition tags to average separately, e.g.
            ["auditory/left", "auditory/right"] -- each must match at least
            one epoch.

    Returns:
        Dict mapping each condition tag to its own Evoked average.

    Raises:
        KeyError: if a condition tag matches zero epochs -- this fails
            loudly rather than silently returning an empty/garbage average,
            since a typo'd tag is a common mistake here.
    """
    result = {}
    for condition in conditions:
        result[condition] = epochs[condition].average()  # epochs[tag] selects matching trials
    return result


def compute_tfr(epochs: mne.Epochs, freqs: np.ndarray, n_cycles: int = 7) -> mne.time_frequency.AverageTFR:
    """Compute time-frequency power via Morlet wavelet convolution.

    Morlet wavelets (not a single FFT) because they use a frequency-
    dependent window -- short at high frequencies, longer at low ones -- so
    power can be resolved in time as well as frequency instead of collapsing
    the whole epoch into one spectrum. This catches effects like alpha
    suppression that are time-locked but not phase-locked, which ERP
    averaging alone would cancel out as noise.

    Args:
        epochs: Cleaned epochs to analyze.
        freqs: Frequencies (Hz) to compute power at.
        n_cycles: Wavelet cycles per frequency -- the time/frequency
            tradeoff knob (more cycles sharpen frequency resolution but
            blur timing). Defaults to a fixed 7, a common middle ground,
            but a fixed value needs a long enough epoch window at low
            frequencies -- pass a frequency-scaled array (e.g. freqs / 2.0)
            for short epochs, matching what this pipeline's callers do.

    Returns:
        An AverageTFR of power (not ITC -- see compute_itc for phase
        consistency, a separate question computed via its own call).
    """
    return tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False)


def compute_itc(epochs: mne.Epochs, freqs: np.ndarray = None, n_cycles=None) -> mne.time_frequency.AverageTFR:
    """Compute inter-trial phase coherence via Morlet wavelet convolution.

    ITC has to be computed from epochs, not an averaged Evoked: averaging
    collapses the trial dimension, and per-trial phase is exactly what's
    being compared here -- once the trials are gone there's nothing left to
    check for agreement. tfr_morlet tracks the full complex wavelet
    coefficient per trial, then averages unit-length phase vectors across
    trials at each time-frequency point; the result is bounded in [0, 1]
    (1 = every trial had the same phase, 0 = phase was uniformly random).

    Args:
        epochs: Cleaned epochs to analyze.
        freqs: Frequencies (Hz) to compute ITC at. Defaults to 4-40 Hz in
            1 Hz steps -- the same range compute_tfr()'s callers default to,
            so power and ITC results line up cell-for-cell.
        n_cycles: Wavelet cycles per frequency. Defaults to a fixed 7 (same
            as compute_tfr()'s default) so both stay at identical
            time/frequency resolution when neither caller overrides it. A
            fixed 7 needs a 1.75s wavelet window at 4 Hz, longer than some
            epochs -- pass a frequency-scaled n_cycles (e.g. freqs / 2.0)
            explicitly when the epoch window is short.

    Returns:
        An AverageTFR of ITC values.
    """
    if freqs is None:
        freqs = np.arange(4, 40, 1)  # same range compute_tfr() defaults its callers to, so power and ITC line up cell-for-cell

    # n_cycles defaults to 7, same as compute_tfr()'s default, keeping both at
    # identical time/frequency resolution when neither caller overrides it. A
    # fixed 7 needs a 1.75s wavelet window at 4 Hz, longer than some epochs --
    # pass a frequency-scaled n_cycles (e.g. freqs / 2.0, as compute_tfr's
    # callers do) explicitly when the epoch window is short.
    if n_cycles is None:
        n_cycles = 7

    # tfr_morlet tracks the full complex wavelet coefficient per trial when
    # return_itc=True, then averages unit-length phase vectors across
    # trials at each time-frequency point (see compute_tfr's power path for
    # the same wavelets used without phase tracking). The result is bounded
    # in [0, 1]: 1 means every trial had the same phase at that point, 0
    # means phase was uniformly random across trials.
    power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, return_itc=True)
    return itc
