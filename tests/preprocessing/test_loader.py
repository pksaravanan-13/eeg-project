import pytest

from src.preprocessing.loader import load_raw, inspect_raw


def test_load_raw_missing_file_raises(tmp_path):
    missing = tmp_path / "nope.fif"
    with pytest.raises(FileNotFoundError):
        load_raw(str(missing))


def test_load_raw_unsupported_suffix_raises(tmp_path):
    bad_file = tmp_path / "data.txt"
    bad_file.write_text("not eeg data")
    with pytest.raises(ValueError):
        load_raw(str(bad_file))


def test_load_raw_roundtrip_fif(tmp_path, synthetic_raw):
    saved_path = tmp_path / "roundtrip_raw.fif"
    synthetic_raw.save(str(saved_path), overwrite=True)

    loaded = load_raw(str(saved_path))

    assert loaded.ch_names == synthetic_raw.ch_names
    assert loaded.info["sfreq"] == synthetic_raw.info["sfreq"]


def test_inspect_raw_smoke(synthetic_raw, capsys):
    inspect_raw(synthetic_raw)
    captured = capsys.readouterr()
    assert "Channels" in captured.out
