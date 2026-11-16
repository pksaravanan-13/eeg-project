import mne
import numpy as np
import pytest

import pipeline


def _add_correlated_eog(raw, seed=1, scale=3.0):
    # ICA's auto_detect_eog needs a real EOG channel to correlate against
    # (see tests/preprocessing/test_ica.py's identical helper) -- synthetic_raw
    # (conftest) has none, so add one here rather than in the shared fixture.
    rng = np.random.RandomState(seed)
    eeg_data = raw.get_data(picks=[0])
    eog_data = eeg_data * scale + rng.normal(0, 5e-6, size=eeg_data.shape)
    eog_info = mne.create_info(["EOG 001"], sfreq=raw.info["sfreq"], ch_types=["eog"])
    eog_raw = mne.io.RawArray(eog_data, eog_info, verbose=False)
    return raw.copy().add_channels([eog_raw], force_update_info=True)


def _cfg_with_tmp_paths(tmp_path):
    cfg = pipeline.load_config("config.yaml")
    cfg["paths"] = {
        "raw_data": str(tmp_path / "raw"),
        "processed_data": str(tmp_path / "processed"),
        "results": str(tmp_path / "results"),
        "figures": str(tmp_path / "results" / "figures"),
        "reports": str(tmp_path / "results" / "reports"),
    }
    # The real config names a sample-dataset channel; synthetic data (conftest) has C3.
    cfg["analysis"]["tfr_plot_channel"] = "C3"
    return cfg


def test_load_config_parses_real_config():
    cfg = pipeline.load_config("config.yaml")
    assert cfg["preprocessing"]["bandpass"]["l_freq"] == 1.0
    assert cfg["preprocessing"]["notch_freq"] == 60.0
    assert "subjects" in cfg
    assert "recording" not in cfg


@pytest.mark.slow
def test_pipeline_run_end_to_end(tmp_path):
    import mne

    sample_path = mne.datasets.sample.data_path()
    raw_file = str(sample_path / "MEG" / "sample" / "sample_audvis_raw.fif")

    cfg = _cfg_with_tmp_paths(tmp_path)
    powers = pipeline.run("sub-e2e-test", raw_file, cfg, bad_channels=["EEG 053"])

    assert (tmp_path / "processed" / "sub-e2e-test-epo.fif").exists()
    assert (tmp_path / "results" / "figures" / "sub-e2e-test_erp.png").exists()
    assert "alpha" in powers


def test_run_second_call_skips_preprocessing(tmp_path, monkeypatch, synthetic_raw):
    raw_path = tmp_path / "raw" / "sub-resume_raw.fif"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw = _add_correlated_eog(synthetic_raw)
    # classifier stage needs more than one trial per class in every CV fold's
    # training split; conftest's synthetic_raw only cycles each event code
    # twice, so stack 3 copies end-to-end to get 6 trials/class.
    raw = mne.concatenate_raws([raw.copy() for _ in range(3)])
    raw.save(str(raw_path), overwrite=True)

    cfg = _cfg_with_tmp_paths(tmp_path)
    cfg["preprocessing"]["ica"]["n_components"] = 4  # fixed count for a deterministic/fast fit on tiny synthetic data
    cfg["classifier"]["cv_folds"] = 2  # keep folds small even with the tripled trial count above

    pipeline.run("sub-resume", str(raw_path), cfg)

    call_count = {"n": 0}
    original_apply_filters = pipeline.apply_filters

    def counting_apply_filters(*args, **kwargs):
        call_count["n"] += 1
        return original_apply_filters(*args, **kwargs)

    monkeypatch.setattr(pipeline, "apply_filters", counting_apply_filters)

    pipeline.run("sub-resume", str(raw_path), cfg)

    assert call_count["n"] == 0


def test_real_config_defines_tfr_plot_channel_as_nonempty_string():
    cfg = pipeline.load_config("config.yaml")
    channel = cfg["analysis"].get("tfr_plot_channel")
    assert isinstance(channel, str) and channel.strip(), f"analysis.tfr_plot_channel is {channel!r}"


def test_run_passes_configured_tfr_channel_to_plot_tfr(tmp_path, monkeypatch, synthetic_raw):
    raw_path = tmp_path / "raw" / "sub-chan_raw.fif"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw = _add_correlated_eog(synthetic_raw)
    raw = mne.concatenate_raws([raw.copy() for _ in range(3)])  # see test_run_second_call_skips_preprocessing
    raw.save(str(raw_path), overwrite=True)

    cfg = _cfg_with_tmp_paths(tmp_path)
    cfg["preprocessing"]["ica"]["n_components"] = 4
    cfg["classifier"]["cv_folds"] = 2
    cfg["analysis"]["tfr_plot_channel"] = "C3"

    calls = []
    monkeypatch.setattr(pipeline, "plot_tfr", lambda *args, **kwargs: calls.append(kwargs))

    pipeline.run("sub-chan", str(raw_path), cfg)

    assert len(calls) == 1
    assert calls[0].get("channel") == "C3"


def test_plot_tfr_passes_channel_to_power_plot_as_single_pick(tmp_path, monkeypatch):
    import inspect

    import matplotlib.pyplot as plt

    from src.visualization.plot import plot_tfr

    assert "channel" in inspect.signature(plot_tfr).parameters, "plot_tfr has no channel parameter"

    rng = np.random.default_rng(0)
    info = mne.create_info(["Fp1", "Fp2", "F3", "F4", "C3"], sfreq=100.0, ch_types="eeg")
    freqs = np.array([6.0, 10.0, 14.0])
    times = np.linspace(-0.2, 0.5, 36)
    power = mne.time_frequency.AverageTFRArray(
        info=info, data=rng.uniform(0.5, 2.0, size=(5, freqs.size, times.size)), times=times, freqs=freqs, nave=10
    )

    seen = []
    original_plot = power.plot

    def spy_plot(*args, **kwargs):
        seen.append(kwargs)
        return original_plot(*args, **kwargs)

    monkeypatch.setattr(power, "plot", spy_plot)

    plot_tfr(power, baseline=(None, 0), channel="C3", out_path=str(tmp_path / "tfr.png"))
    plt.close("all")

    assert len(seen) == 1
    assert seen[0].get("picks") == ["C3"]
