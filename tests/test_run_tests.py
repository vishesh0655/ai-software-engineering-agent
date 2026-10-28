from backend.tools.run_tests import _pytest_summary_counts


def test_pytest_summary_counts_include_passed_failed_and_skipped():
    counts = _pytest_summary_counts("3 passed, 2 failed, 1 skipped in 1.25s")
    assert counts == {"passed_count": 3, "failed_count": 2, "skipped_count": 1}


def test_pytest_summary_counts_default_to_zero_when_absent():
    counts = _pytest_summary_counts("No pytest-style test files were found")
    assert counts == {"passed_count": 0, "failed_count": 0, "skipped_count": 0}
