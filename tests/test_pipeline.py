import pytest

import pipeline


def _cfg_with_tmp_paths(tmp_path):
    cfg = pipeline.load_config("config.yaml")
    cfg["paths"] = {
        "raw_data": str(tmp_path / "raw"),
        "processed_data": str(tmp_path / "processed"),
        "results": str(tmp_path / "results"),
        "figures": str(tmp_path / "results" / "figures"),
        "reports": str(tmp_path / "results" / "reports"),
    }
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
    synthetic_raw.save(str(raw_path), overwrite=True)

    cfg = _cfg_with_tmp_paths(tmp_path)

    pipeline.run("sub-resume", str(raw_path), cfg)

    call_count = {"n": 0}
    original_apply_filters = pipeline.apply_filters

    def counting_apply_filters(*args, **kwargs):
        call_count["n"] += 1
        return original_apply_filters(*args, **kwargs)

    monkeypatch.setattr(pipeline, "apply_filters", counting_apply_filters)

    pipeline.run("sub-resume", str(raw_path), cfg)

    assert call_count["n"] == 0
