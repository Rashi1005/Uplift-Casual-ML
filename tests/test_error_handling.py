"""
test_error_handling.py
------------------------
Dedicated tests for how the pipeline behaves on malformed input — distinct
from ``tests/test_data_pipeline.py``'s missing-file tests, which cover
files that do not exist at all.  These cover files that **do** exist but
are corrupted, empty, wrongly-typed, or structurally inconsistent with
what a downstream function expects — the more common real-world failure
mode once a pipeline has been run a few times (a partially written CSV
from an interrupted run, a hand-edited file, a schema change upstream)
than a simple missing file.

Every test here asserts a specific, informative failure (a particular
exception type, and where practical a message that would actually help
someone debug it) rather than merely "does not silently produce wrong
output" — silently returning bad data is arguably worse than crashing,
and several tests here specifically guard against that.
"""

import numpy as np
import pandas as pd
import pytest

import baseline_model
import preprocessing
import uplift_models
from data_loader import EXPECTED_COLUMNS, load_hillstrom


# ---------------------------------------------------------------------
# Malformed raw CSV (data_loader.py)
# ---------------------------------------------------------------------
class TestMalformedRawFile:
    def test_empty_file_raises(self, tmp_path):
        empty_path = tmp_path / "empty.csv"
        empty_path.write_text("")
        with pytest.raises(pd.errors.EmptyDataError):
            load_hillstrom(prefer_sklift=False, local_path=str(empty_path))

    def test_truncated_csv_with_missing_columns_raises_value_error(self, tmp_path):
        """A CSV that parses fine as a DataFrame, but is missing columns
        this project's downstream code depends on (e.g. a partially
        written file from an interrupted prepare-data run).
        """
        bad_path = tmp_path / "truncated.csv"
        bad_path.write_text("recency,history\n1,29.99\n2,40.00\n")
        with pytest.raises(ValueError, match="missing expected columns"):
            load_hillstrom(prefer_sklift=False, local_path=str(bad_path))

    def test_wrong_delimiter_produces_missing_columns_error(self, tmp_path):
        """A semicolon- or tab-delimited file (e.g. from a different
        regional Excel export) parses as one giant malformed column under
        pandas' default comma delimiter — this should be caught by the
        same missing-columns check, not silently accepted.
        """
        bad_path = tmp_path / "wrong_delimiter.csv"
        header = ";".join(EXPECTED_COLUMNS)
        bad_path.write_text(f"{header}\n1;seg;2;0;1;Urban;0;Web;Mens;1;0;5.0\n")
        with pytest.raises(ValueError, match="missing expected columns"):
            load_hillstrom(prefer_sklift=False, local_path=str(bad_path))


# ---------------------------------------------------------------------
# Malformed processed splits (preprocessing.py / baseline_model.py)
# ---------------------------------------------------------------------
class TestMalformedProcessedData:
    def test_encode_features_raises_on_missing_categorical_column(self):
        df = pd.DataFrame({"num": [1, 2, 3], "cat": ["A", "B", "A"]})
        with pytest.raises(KeyError):
            preprocessing.encode_features(
                df,
                numeric_features=["num"],
                categorical_features=["cat", "does_not_exist"],
            )

    def test_encode_features_raises_on_missing_numeric_column(self):
        df = pd.DataFrame({"cat": ["A", "B"]})
        with pytest.raises(KeyError):
            preprocessing.encode_features(
                df,
                numeric_features=["missing_num"],
                categorical_features=["cat"],
            )

    def test_binarize_treatment_raises_on_missing_raw_column(self):
        df = pd.DataFrame({"unrelated": [1, 2, 3]})
        with pytest.raises(KeyError):
            preprocessing.binarize_treatment(df, raw_col="segment")

    def test_train_baseline_model_raises_on_mismatched_lengths(self):
        X_train = pd.DataFrame({"x1": [1, 2, 3, 4]})
        # 3 rows vs. X's 4 — should not silently truncate/pad.
        y_train_wrong_length = pd.Series([0, 1, 0])
        # LightGBM itself catches this (via its own LightGBMError, not a
        # plain ValueError) — confirmed by actually running this test
        # and inspecting the real exception, rather than assuming.
        import lightgbm

        with pytest.raises(lightgbm.basic.LightGBMError, match="differs"):
            baseline_model.train_baseline_model(X_train, y_train_wrong_length, random_state=42)

    def test_train_baseline_model_raises_on_all_nan_target(self):
        X_train = pd.DataFrame({"x1": [1.0, 2.0, 3.0, 4.0]})
        y_train_all_nan = pd.Series([np.nan, np.nan, np.nan, np.nan])
        with pytest.raises((ValueError, TypeError)):
            baseline_model.train_baseline_model(X_train, y_train_all_nan, random_state=42)

    def test_stratified_split_raises_on_length_mismatch(self):
        X = pd.DataFrame({"x1": range(10)})
        treatment = pd.Series([0, 1] * 4)  # length 8, not 10 — mismatched with X
        y = pd.DataFrame({"conversion": [0] * 10, "visit": [0] * 10})
        with pytest.raises(ValueError):
            preprocessing.stratified_split(X, treatment, y, random_state=42)


# ---------------------------------------------------------------------
# Malformed combined rankings (uplift_models.py)
# ---------------------------------------------------------------------
class TestMalformedCombinedRankings:
    def test_combine_rankings_raises_on_length_mismatch(self):
        n = 20
        X_test = pd.DataFrame({"x1": range(n)})
        treatment_test = pd.Series([0, 1] * (n // 2))
        y_test = pd.DataFrame({"visit": [0] * n, "conversion": [0] * n})
        baseline_ranking = pd.DataFrame(
            {
                "row_index": range(n),
                "predicted_prob": np.zeros(n),
            }
        )
        wrong_length_scores = np.zeros(n - 5)  # deliberately too short
        with pytest.raises(ValueError):
            uplift_models.combine_rankings(
                X_test,
                treatment_test,
                y_test,
                wrong_length_scores,
                np.zeros(n),
                np.zeros(n),
                baseline_ranking,
            )

    def test_combine_rankings_raises_on_duplicate_row_index_in_baseline_ranking(self):
        n = 10
        X_test = pd.DataFrame({"x1": range(n)})
        treatment_test = pd.Series([0, 1] * (n // 2))
        y_test = pd.DataFrame({"visit": [0] * n, "conversion": [0] * n})
        # Duplicate row_index values -> a merge would fan out to more
        # rows than expected, which combine_rankings should not allow
        # to pass silently.
        baseline_ranking = pd.DataFrame(
            {
                "row_index": [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
                "predicted_prob": np.zeros(n),
            }
        )
        with pytest.raises(ValueError):
            result = uplift_models.combine_rankings(
                X_test,
                treatment_test,
                y_test,
                np.zeros(n),
                np.zeros(n),
                np.zeros(n),
                baseline_ranking,
            )
            # If no exception, the fan-out itself is the bug — fail explicitly.
            assert len(result) == n, (
                f"Expected {n} rows but got {len(result)} — duplicate row_index "
                "values in baseline_ranking caused a silent row fan-out."
            )
