import numpy as np
import pytest

from group_decode_study import paired_test, main


def test_paired_test_reports_higher_ica_mean_correctly():
    acc_no_rejection = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    acc_ica_cleaned = np.array([0.7, 0.7, 0.7, 0.7, 0.7])

    result = paired_test(acc_no_rejection, acc_ica_cleaned)

    assert result["n_subjects"] == 5
    assert result["mean_no_cleaning"] == pytest.approx(0.5)
    assert result["mean_ica_cleaned"] == pytest.approx(0.7)
    assert 0.0 <= result["p_value"] <= 1.0


def test_paired_test_near_identical_arms_gives_high_p_value():
    # Wilcoxon requires at least one non-zero paired difference, so use
    # near-identical (not exactly identical) arms for this edge case.
    acc_no_rejection = np.array([0.60, 0.55, 0.62, 0.58])
    acc_ica_cleaned = np.array([0.60, 0.55, 0.62, 0.581])

    result = paired_test(acc_no_rejection, acc_ica_cleaned)

    assert result["p_value"] > 0.5


@pytest.mark.slow
def test_main_runs_end_to_end_on_two_subjects():
    # Requires network access to fetch PhysioNet EEGBCI runs on first use.
    result = main(subjects=[1, 2])

    assert result["n_subjects"] == 2
    assert 0.0 <= result["mean_no_cleaning"] <= 1.0
    assert 0.0 <= result["mean_ica_cleaned"] <= 1.0
    assert 0.0 <= result["p_value"] <= 1.0
