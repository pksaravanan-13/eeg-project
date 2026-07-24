import numpy as np
import mne
import pytest

from src.analysis.features import band_power, compute_erp, compare_conditions, compute_tfr, compute_itc


def _n_data_channels(epochs):
    # compute_psd/average() only operate on picked data channels (EEG here),
    # not the raw STI channel that's still listed in epochs.ch_names.
    return len(mne.pick_types(epochs.info, eeg=True))


def _make_phase_epochs(freq: float = 25.0, sfreq: float = 250.0, n_epochs: int = 20, jitter: bool = False, seed: int = 0) -> mne.EpochsArray:
    # Built directly (not via make_epochs) so each trial's phase at `freq`
    # is fully controlled: jitter=False puts every trial at the same
    # starting phase (0 rad), jitter=True gives each trial an independent
    # random phase -- the two ends of the ITC scale compute_itc should report.
    # 1s duration comfortably fits a 25 Hz, n_cycles=7 wavelet (compute_itc's
    # fixed n_cycles), unlike the shorter synthetic_epochs fixture used elsewhere
    # in this file, which is too short for anything near compute_itc's 4 Hz floor.
    duration = 1.0
    n_samples = int(sfreq * duration)
    times = np.arange(n_samples) / sfreq
    rng = np.random.RandomState(seed)
    data = np.zeros((n_epochs, 1, n_samples))
    for i in range(n_epochs):
        phase = rng.uniform(0, 2 * np.pi) if jitter else 0.0
        data[i, 0, :] = np.sin(2 * np.pi * freq * times + phase)
    info = mne.create_info(["EEG1"], sfreq=sfreq, ch_types="eeg")
    return mne.EpochsArray(data, info, tmin=-duration / 2, verbose=False)


def test_band_power_returns_one_array_per_band(synthetic_epochs):
    bands = {"alpha": (8, 13), "beta": (13, 30)}

    powers = band_power(synthetic_epochs, bands)

    assert set(powers.keys()) == set(bands.keys())
    n_channels = _n_data_channels(synthetic_epochs)
    for arr in powers.values():
        assert arr.shape == (len(synthetic_epochs), n_channels)


def test_band_power_empty_band_range_gives_zero_columns(synthetic_epochs):
    # A band whose (fmin, fmax) falls entirely outside any PSD bin selects
    # zero frequency columns -- band_power should return an all-NaN mean
    # rather than raising (mean of an empty slice), which is the honest
    # signal that this band request didn't match any computed frequency.
    bands = {"nonexistent": (500, 600)}

    powers = band_power(synthetic_epochs, bands)

    assert powers["nonexistent"].shape == (len(synthetic_epochs), _n_data_channels(synthetic_epochs))
    assert np.all(np.isnan(powers["nonexistent"]))


def test_compute_erp_averages_all_epochs(synthetic_epochs):
    erp = compute_erp(synthetic_epochs)

    assert isinstance(erp, mne.Evoked)
    assert erp.nave == len(synthetic_epochs)
    assert erp.data.shape == (_n_data_channels(synthetic_epochs), len(synthetic_epochs.times))


def test_compare_conditions_returns_one_evoked_per_condition(synthetic_epochs):
    result = compare_conditions(synthetic_epochs, ["auditory/left", "auditory/right"])

    assert set(result.keys()) == {"auditory/left", "auditory/right"}
    for evoked in result.values():
        assert isinstance(evoked, mne.Evoked)
        assert evoked.nave == 2  # synthetic_raw (conftest) cycles 2x through each trigger code


def test_compare_conditions_zero_matching_epochs_raises(synthetic_epochs):
    # A condition tag that matches no epochs (typo, or a trigger never
    # present in this recording) must fail loudly rather than silently
    # return an empty/garbage average.
    with pytest.raises(KeyError):
        compare_conditions(synthetic_epochs, ["not_a_real_condition"])


def test_compute_tfr_returns_power_for_frequencies_that_fit(synthetic_epochs):
    # synthetic_epochs (conftest) spans tmin=-0.2/tmax=0.4 -> 0.6s. At
    # n_cycles=3, a 15-30 Hz wavelet comfortably fits inside that window.
    freqs = np.arange(15, 30, 5)

    power = compute_tfr(synthetic_epochs, freqs=freqs, n_cycles=3)

    assert isinstance(power, mne.time_frequency.AverageTFR)
    assert power.data.shape == (_n_data_channels(synthetic_epochs), len(freqs), len(synthetic_epochs.times))


def test_compute_tfr_wavelet_longer_than_epoch_raises(synthetic_epochs):
    # Real failure mode hit while building M7_time_frequency.ipynb: at
    # n_cycles=7, a 4 Hz wavelet needs ~1.75s of signal, far longer than
    # this fixture's 0.6s epoch, so MNE must raise rather than silently
    # truncate the wavelet.
    freqs = np.arange(4, 10, 1)

    with pytest.raises(ValueError):
        compute_tfr(synthetic_epochs, freqs=freqs, n_cycles=7)


def test_compute_itc_returns_values_bounded_in_unit_interval(synthetic_epochs):
    # compute_itc() hardcodes n_cycles=7 internally (same as compute_tfr's
    # default), so -- like test_compute_tfr_wavelet_longer_than_epoch_raises
    # above -- the frequency needs to be well above synthetic_epochs' fixture
    # floor (4 Hz would need a longer window than this 0.6s epoch has).
    freqs = np.arange(20, 30, 5)

    itc = compute_itc(synthetic_epochs, freqs=freqs)

    assert isinstance(itc, mne.time_frequency.AverageTFR)
    assert itc.data.shape == (_n_data_channels(synthetic_epochs), len(freqs), len(synthetic_epochs.times))
    # ITC is a resultant vector length averaged over unit-length arrows, so
    # it's mathematically bounded in [0, 1] -- allow a hair of floating-point
    # slack rather than an exact 0.0/1.0 bound.
    assert itc.data.min() >= -1e-9
    assert itc.data.max() <= 1.0 + 1e-9


def test_compute_itc_identical_phase_across_trials_gives_near_one_itc():
    # Every trial has the exact same phase at the analyzed frequency, so
    # the per-trial phase "arrows" all point the same direction and the
    # averaged resultant vector should be almost exactly length 1.
    freq = 25.0
    epochs = _make_phase_epochs(freq=freq, jitter=False)

    itc = compute_itc(epochs, freqs=np.array([freq]))

    mid = itc.data.shape[-1] // 2  # epoch center -- away from wavelet edge effects
    assert itc.data[0, 0, mid] > 0.95


def test_compute_itc_random_phase_across_trials_gives_much_lower_itc():
    # Each trial gets an independent random phase at the analyzed frequency,
    # so the per-trial arrows scatter around the circle and mostly cancel
    # when averaged -- ITC should land well below the identical-phase case.
    freq = 25.0
    locked = _make_phase_epochs(freq=freq, jitter=False)
    randomized = _make_phase_epochs(freq=freq, jitter=True)

    itc_locked = compute_itc(locked, freqs=np.array([freq]))
    itc_random = compute_itc(randomized, freqs=np.array([freq]))

    mid = itc_locked.data.shape[-1] // 2
    assert itc_random.data[0, 0, mid] < 0.5
    assert itc_random.data[0, 0, mid] < itc_locked.data[0, 0, mid]
