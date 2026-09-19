"""
test_data_pipeline.py
----------------------
Tests for the data loading and preparation workflow (src/data_loader.py
and src/prepare_data.py).

These tests use small synthetic fixtures rather than the real 64,000-row
Hillstrom dataset, and never hit the network -- both intentional, since
unit tests shouldn't depend on an external host being reachable (see the
known upstream-reliability limitation documented in data/MANIFEST.md,
which these tests exist partly to guard against).

Run with:
    pytest tests/test_data_pipeline.py -v
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_loader  # noqa: E402
import prepare_data  # noqa: E402


def make_valid_fixture(n=50) -> pd.DataFrame:
    """A minimal DataFrame with every expected column, valid dtypes, no
    missing values -- small on purpose, since these tests check structure
    and behavior, not statistical properties of the real dataset."""
    return pd.DataFrame(
        {
            "recency": [1] * n,
            "history_segment": ["1) $0 - $100"] * n,
            "history": [29.99] * n,
            "mens": [0, 1] * (n // 2),
            "womens": [1, 0] * (n // 2),
            "zip_code": ["Urban"] * n,
            "newbie": [0] * n,
            "channel": ["Web"] * n,
            "segment": (["Mens E-Mail", "No E-Mail"] * (n // 2)),
            "visit": [0, 1] * (n // 2),
            "conversion": [0] * n,
            "spend": [0.0] * n,
        }
    )


# ---------------------------------------------------------------------
# 1. Successful loading
# ---------------------------------------------------------------------
class TestSuccessfulLoading:
    def test_loads_valid_local_csv(self, tmp_path, monkeypatch):
        csv_path = tmp_path / "hillstrom.csv"
        make_valid_fixture().to_csv(csv_path, index=False)
        monkeypatch.setattr(data_loader, "LOCAL_CSV_PATH", str(csv_path))

        df = data_loader.load_hillstrom(prefer_sklift=False)

        assert not df.empty
        assert list(df.columns) == data_loader.EXPECTED_COLUMNS

    def test_prepare_data_validate_columns_accepts_valid_frame(self):
        df = make_valid_fixture()
        # Should not raise.
        prepare_data.validate_columns(df)

    def test_prepare_data_report_runs_without_error(self, capsys):
        df = make_valid_fixture()
        prepare_data.report(df)
        captured = capsys.readouterr()
        assert "Shape:" in captured.out
        assert "Treatment-group counts" in captured.out


# ---------------------------------------------------------------------
# 2. Missing-column validation
# ---------------------------------------------------------------------
class TestMissingColumnValidation:
    def test_data_loader_raises_on_missing_columns(self, tmp_path, monkeypatch):
        incomplete = make_valid_fixture().drop(columns=["spend", "conversion"])
        csv_path = tmp_path / "hillstrom.csv"
        incomplete.to_csv(csv_path, index=False)
        monkeypatch.setattr(data_loader, "LOCAL_CSV_PATH", str(csv_path))

        with pytest.raises(ValueError, match="missing expected columns"):
            data_loader.load_hillstrom(prefer_sklift=False)

    def test_prepare_data_validate_columns_raises_on_missing_columns(self):
        incomplete = make_valid_fixture().drop(columns=["segment"])
        with pytest.raises(ValueError, match="missing expected columns"):
            prepare_data.validate_columns(incomplete)

    def test_error_message_names_the_missing_column(self):
        incomplete = make_valid_fixture().drop(columns=["zip_code"])
        with pytest.raises(ValueError, match="zip_code"):
            prepare_data.validate_columns(incomplete)


# ---------------------------------------------------------------------
# 3. Missing-file behavior
# ---------------------------------------------------------------------
class TestMissingFileBehavior:
    def test_data_loader_raises_file_not_found_when_no_local_cache(
        self, tmp_path, monkeypatch
    ):
        nonexistent_path = tmp_path / "does_not_exist.csv"
        monkeypatch.setattr(data_loader, "LOCAL_CSV_PATH", str(nonexistent_path))

        with pytest.raises(FileNotFoundError):
            data_loader.load_hillstrom(prefer_sklift=False)

    def test_prepare_data_exits_cleanly_when_download_fails_and_no_cache(
        self, tmp_path, monkeypatch
    ):
        nonexistent_path = tmp_path / "does_not_exist.csv"
        monkeypatch.setattr(prepare_data, "LOCAL_CSV_PATH", str(nonexistent_path))
        monkeypatch.setattr(prepare_data, "RAW_DATA_DIR", str(tmp_path))

        def fail_download():
            raise ConnectionError("simulated network failure")

        monkeypatch.setattr(prepare_data, "download_raw_dataset", fail_download)

        with pytest.raises(SystemExit) as exc_info:
            prepare_data.main()
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------
# 4. Expected schema
# ---------------------------------------------------------------------
class TestExpectedSchema:
    def test_expected_columns_count(self):
        assert len(data_loader.EXPECTED_COLUMNS) == 12

    def test_expected_columns_match_manifest(self):
        expected = [
            "recency", "history_segment", "history", "mens", "womens",
            "zip_code", "newbie", "channel", "segment", "visit",
            "conversion", "spend",
        ]
        assert data_loader.EXPECTED_COLUMNS == expected

    def test_prepare_data_reuses_data_loader_schema(self):
        # prepare_data.py must not define its own separate column list --
        # this test guards against the two files silently drifting apart.
        assert prepare_data.EXPECTED_COLUMNS is data_loader.EXPECTED_COLUMNS

    def test_treatment_binarization_matches_notebook_convention(self):
        df = make_valid_fixture()
        treatment = (df["segment"] != "No E-Mail").astype(int)
        assert set(treatment.unique()) <= {0, 1}