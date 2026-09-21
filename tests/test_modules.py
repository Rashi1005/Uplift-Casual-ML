"""
test_modules.py
----------------
Tests for the reusable modules extracted from notebooks 01-06:
src/preprocessing.py, src/baseline_model.py, src/uplift_models.py,
src/evaluation.py, src/business_simulation.py, and src/utils.py.

Uses small synthetic fixtures, not the real 64,000-row dataset -- these
test each function's logic and contracts, not statistical properties of
the real data (that verification was done separately, by running the
actual notebooks against the real committed data/processed/ files).

Run with:
    pytest tests/test_modules.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import utils  # noqa: E402
import preprocessing  # noqa: E402
import baseline_model  # noqa: E402
import uplift_models  # noqa: E402
import evaluation  # noqa: E402
import business_simulation  # noqa: E402


# ---------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------
@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_split(rng):
    """A small, synthetic train/test split with the same shape/roles as
    the real pipeline: a feature matrix, binary treatment, and a binary
    outcome, with treatment carrying real (if modest) signal."""
    n_train, n_test = 400, 100
    n = n_train + n_test

    treatment = pd.Series(rng.integers(0, 2, n))
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "cat_A": rng.integers(0, 2, n), "cat_B": rng.integers(0, 2, n)})

    # outcome depends a bit on x1 and on treatment, so models have something
    # real, if weak, to learn.
    logit = -1.0 + 0.5 * x1 + 0.8 * treatment.values
    prob = 1 / (1 + np.exp(-logit))
    y = pd.Series((rng.random(n) < prob).astype(int))
    y_df = pd.DataFrame({"visit": y, "conversion": pd.Series(rng.integers(0, 2, n))})

    X_train, X_test = X.iloc[:n_train].reset_index(drop=True), X.iloc[n_train:].reset_index(drop=True)
    treatment_train, treatment_test = treatment.iloc[:n_train].reset_index(drop=True), treatment.iloc[n_train:].reset_index(drop=True)
    y_train, y_test = y_df.iloc[:n_train].reset_index(drop=True), y_df.iloc[n_train:].reset_index(drop=True)

    return X_train, X_test, treatment_train, treatment_test, y_train, y_test


# ---------------------------------------------------------------------
# utils.py
# ---------------------------------------------------------------------
class TestUtils:
    def test_sanitize_columns_replaces_special_characters(self):
        cols = ["history_segment_7) $1,000 +", "plain_col", "a/b(c)"]
        sanitized = utils.sanitize_columns(cols)
        assert all(c.replace("_", "").isalnum() for c in sanitized)
        assert len(sanitized) == len(cols)

    def test_compute_scale_pos_weight(self):
        y = pd.Series([0] * 80 + [1] * 20)
        assert utils.compute_scale_pos_weight(y) == pytest.approx(4.0)

    def test_top_k_split_sizes(self):
        df = pd.DataFrame({"score": np.arange(100)})
        top, rest = utils.top_k_split(df, "score", 0.10)
        assert len(top) == 10
        assert len(rest) == 90
        # top should be the highest-scoring rows
        assert top["score"].min() >= rest["score"].max()

    def test_actual_uplift_computes_real_gap(self):
        df = pd.DataFrame({
            "actual_treatment": [1, 1, 1, 0, 0, 0],
            "actual_visit": [1, 1, 0, 0, 0, 0],
        })
        gap, n_treat, n_control = utils.actual_uplift(df)
        assert gap == pytest.approx(2 / 3)
        assert n_treat == 3 and n_control == 3

    def test_actual_uplift_nan_when_arm_empty(self):
        df = pd.DataFrame({"actual_treatment": [1, 1], "actual_visit": [1, 0]})
        gap, _, _ = utils.actual_uplift(df)
        assert np.isnan(gap)


# ---------------------------------------------------------------------
# preprocessing.py
# ---------------------------------------------------------------------
class TestPreprocessing:
    def test_binarize_treatment(self):
        df = pd.DataFrame({"segment": ["No E-Mail", "Mens E-Mail", "Womens E-Mail"]})
        out = preprocessing.binarize_treatment(df)
        assert out["treatment"].tolist() == [0, 1, 1]

    def test_run_randomization_check_passes_on_balanced_data(self, rng):
        n = 2000
        treatment = rng.integers(0, 2, n)
        df = pd.DataFrame({
            "treatment": treatment,
            "numeric_feat": rng.normal(0, 1, n),  # independent of treatment
            "cat_feat": rng.choice(["A", "B"], n),  # independent of treatment
        })
        _, _, passed, flagged = preprocessing.run_randomization_check(
            df, ["numeric_feat"], ["cat_feat"], treat_col="treatment",
        )
        assert passed
        assert flagged == []

    def test_run_randomization_check_fails_on_imbalanced_data(self, rng):
        n = 2000
        treatment = np.array([1] * 1000 + [0] * 1000)
        # numeric_feat is now strongly dependent on treatment -> should fail
        numeric_feat = np.where(treatment == 1, rng.normal(5, 1, n), rng.normal(0, 1, n))
        df = pd.DataFrame({"treatment": treatment, "numeric_feat": numeric_feat})
        _, _, passed, flagged = preprocessing.run_randomization_check(
            df, ["numeric_feat"], [], treat_col="treatment",
        )
        assert not passed
        assert "numeric_feat" in flagged

    def test_encode_features_one_hot(self):
        df = pd.DataFrame({"num": [1, 2, 3], "cat": ["A", "B", "A"]})
        X = preprocessing.encode_features(df, numeric_features=["num"], categorical_features=["cat"])
        assert "num" in X.columns
        assert "cat_A" in X.columns and "cat_B" in X.columns
        assert X["cat_A"].tolist() == [1, 0, 1]

    def test_stratified_split_respects_test_size(self, synthetic_split):
        X_train, X_test, treatment_train, treatment_test, y_train, y_test = synthetic_split
        X = pd.concat([X_train, X_test], ignore_index=True)
        treatment = pd.concat([treatment_train, treatment_test], ignore_index=True)
        y = pd.concat([y_train, y_test], ignore_index=True)

        Xtr, Xte, ttr, tte, ytr, yte = preprocessing.stratified_split(
            X, treatment, y, test_size=0.25, random_state=42, strat_outcome_col="conversion",
        )
        assert len(Xte) == pytest.approx(len(X) * 0.25, abs=1)

    def test_stratified_split_is_reproducible_with_same_seed(self, synthetic_split):
        X_train, X_test, treatment_train, treatment_test, y_train, y_test = synthetic_split
        X = pd.concat([X_train, X_test], ignore_index=True)
        treatment = pd.concat([treatment_train, treatment_test], ignore_index=True)
        y = pd.concat([y_train, y_test], ignore_index=True)

        result_a = preprocessing.stratified_split(X, treatment, y, random_state=7, strat_outcome_col="conversion")
        result_b = preprocessing.stratified_split(X, treatment, y, random_state=7, strat_outcome_col="conversion")
        pd.testing.assert_frame_equal(result_a[0], result_b[0])


# ---------------------------------------------------------------------
# baseline_model.py
# ---------------------------------------------------------------------
class TestBaselineModel:
    def test_train_and_evaluate_baseline_model(self, synthetic_split):
        X_train, X_test, treatment_train, treatment_test, y_train, y_test = synthetic_split
        model = baseline_model.train_baseline_model(X_train, y_train["visit"], random_state=42)
        results = baseline_model.evaluate_baseline_model(model, X_test, y_test["visit"])

        assert 0.0 <= results["auc_roc"] <= 1.0
        assert len(results["test_probs"]) == len(X_test)

    def test_build_ranking_sorted_descending(self, synthetic_split):
        X_train, X_test, treatment_train, treatment_test, y_train, y_test = synthetic_split
        probs = np.linspace(1, 0, len(X_test))  # deliberately already descending
        ranking = baseline_model.build_ranking(X_test, probs, treatment_test, y_test)
        assert ranking["predicted_prob"].is_monotonic_decreasing

    def test_signal_check_structure(self, synthetic_split):
        X_train, X_test, treatment_train, treatment_test, y_train, y_test = synthetic_split
        probs = np.linspace(1, 0, len(X_test))
        ranking = baseline_model.build_ranking(X_test, probs, treatment_test, y_test)
        result = baseline_model.signal_check(ranking, "visit", top_fraction=0.10)
        assert set(result.keys()) == {"overall_rate", "top_rate", "rest_rate", "gap", "lift_ratio", "passed"}


# ---------------------------------------------------------------------
# uplift_models.py
# ---------------------------------------------------------------------
class TestUpliftModels:
    def test_two_model_approach_output_shape(self, synthetic_split):
        X_train, X_test, treatment_train, treatment_test, y_train, y_test = synthetic_split
        uplift, treated_model, control_model = uplift_models.two_model_approach(
            X_train, y_train["visit"], treatment_train, X_test, random_state=42,
        )
        assert len(uplift) == len(X_test)
        assert treated_model is not None and control_model is not None

    def test_class_transformation_output_range(self, synthetic_split):
        X_train, X_test, treatment_train, treatment_test, y_train, y_test = synthetic_split
        uplift, model = uplift_models.class_transformation(
            X_train, y_train["visit"], treatment_train, X_test, random_state=42,
        )
        assert len(uplift) == len(X_test)
        # 2*P(Z=1)-1 must land in [-1, 1]
        assert uplift.min() >= -1.0001 and uplift.max() <= 1.0001

    def test_two_model_approach_reproducible_with_same_seed(self, synthetic_split):
        X_train, X_test, treatment_train, treatment_test, y_train, y_test = synthetic_split
        uplift_a, _, _ = uplift_models.two_model_approach(X_train, y_train["visit"], treatment_train, X_test, random_state=7)
        uplift_b, _, _ = uplift_models.two_model_approach(X_train, y_train["visit"], treatment_train, X_test, random_state=7)
        np.testing.assert_array_equal(uplift_a, uplift_b)

    def test_signal_check_uplift_structure(self, synthetic_split):
        X_train, X_test, treatment_train, treatment_test, y_train, y_test = synthetic_split
        fake_scores = np.linspace(1, -1, len(X_test))
        result = uplift_models.signal_check_uplift(fake_scores, treatment_test, y_test["visit"])
        assert set(result.keys()) == {"top_decile_uplift", "bottom_90pct_uplift", "gap", "passed"}

    def test_combine_rankings_merges_correctly(self, synthetic_split):
        X_train, X_test, treatment_train, treatment_test, y_train, y_test = synthetic_split
        n = len(X_test)
        baseline_ranking = pd.DataFrame({"row_index": X_test.index, "predicted_prob": np.linspace(1, 0, n)})

        combined = uplift_models.combine_rankings(
            X_test, treatment_test, y_test,
            np.zeros(n), np.zeros(n), np.zeros(n),
            baseline_ranking,
        )
        assert len(combined) == n
        assert combined["baseline_predicted_prob"].isna().sum() == 0

    def test_combine_rankings_raises_on_row_index_mismatch(self, synthetic_split):
        X_train, X_test, treatment_train, treatment_test, y_train, y_test = synthetic_split
        n = len(X_test)
        # baseline_ranking with mismatched row_index values -> should fail to join cleanly
        bad_ranking = pd.DataFrame({"row_index": np.arange(9000, 9000 + n), "predicted_prob": np.zeros(n)})
        with pytest.raises(ValueError, match="failed to join"):
            uplift_models.combine_rankings(
                X_test, treatment_test, y_test, np.zeros(n), np.zeros(n), np.zeros(n), bad_ranking,
            )


# ---------------------------------------------------------------------
# evaluation.py
# ---------------------------------------------------------------------
class TestEvaluation:
    @pytest.fixture
    def qini_inputs(self, rng):
        n = 500
        y_true = rng.integers(0, 2, n)
        treatment = rng.integers(0, 2, n)
        rankings = {
            "model_a": rng.normal(0, 1, n),
            "model_b": rng.normal(0, 1, n),
        }
        return y_true, treatment, rankings

    def test_compute_qini_metrics_returns_all_models(self, qini_inputs):
        y_true, treatment, rankings = qini_inputs
        qini_curves, qini_scores, uplift_at_k_scores = evaluation.compute_qini_metrics(y_true, treatment, rankings)
        assert set(qini_scores.keys()) == set(rankings.keys())
        assert set(qini_curves.keys()) == set(rankings.keys())

    def test_bootstrap_qini_ci_structure(self, qini_inputs):
        y_true, treatment, rankings = qini_inputs
        _, qini_scores, _ = evaluation.compute_qini_metrics(y_true, treatment, rankings)
        ci_results = evaluation.bootstrap_qini_ci(y_true, treatment, rankings, qini_scores, n_bootstrap=30, random_state=42)
        for name in rankings:
            assert ci_results[name]["ci_lower_95"] <= ci_results[name]["ci_upper_95"]

    def test_bootstrap_qini_ci_reproducible_with_same_seed(self, qini_inputs):
        y_true, treatment, rankings = qini_inputs
        _, qini_scores, _ = evaluation.compute_qini_metrics(y_true, treatment, rankings)
        ci_a = evaluation.bootstrap_qini_ci(y_true, treatment, rankings, qini_scores, n_bootstrap=30, random_state=5)
        ci_b = evaluation.bootstrap_qini_ci(y_true, treatment, rankings, qini_scores, n_bootstrap=30, random_state=5)
        assert ci_a["model_a"]["ci_lower_95"] == ci_b["model_a"]["ci_lower_95"]

    def test_pairwise_significance_detects_overlap(self):
        ci_results = {
            "A": {"qini_coefficient": 0.05, "ci_lower_95": 0.01, "ci_upper_95": 0.09},
            "B": {"qini_coefficient": 0.06, "ci_lower_95": 0.02, "ci_upper_95": 0.10},
            "C": {"qini_coefficient": 0.50, "ci_lower_95": 0.45, "ci_upper_95": 0.55},
        }
        table = evaluation.pairwise_significance(ci_results)
        ab_row = table[(table["model_a"] == "A") & (table["model_b"] == "B")].iloc[0]
        ac_row = table[(table["model_a"] == "A") & (table["model_b"] == "C")].iloc[0]
        assert ab_row["cis_overlap"] == True  # noqa: E712
        assert ac_row["cis_overlap"] == False  # noqa: E712

    def test_build_final_verdict_returns_string(self):
        qini_scores = {"A": 0.05, "B": 0.03}
        ci_results = {
            "A": {"qini_coefficient": 0.05, "ci_lower_95": -0.01, "ci_upper_95": 0.11},
            "B": {"qini_coefficient": 0.03, "ci_lower_95": -0.02, "ci_upper_95": 0.08},
        }
        significance_table = evaluation.pairwise_significance(ci_results)
        verdict = evaluation.build_final_verdict(qini_scores, ci_results, significance_table, reference_model="B")
        assert isinstance(verdict, str)
        assert "A" in verdict


# ---------------------------------------------------------------------
# business_simulation.py
# ---------------------------------------------------------------------
class TestBusinessSimulation:
    @pytest.fixture
    def combined_fixture(self, rng):
        n = 500
        return pd.DataFrame({
            "actual_treatment": rng.integers(0, 2, n),
            "actual_visit": rng.integers(0, 2, n),
            "score_a": rng.normal(0, 1, n),
            "score_b": rng.normal(0, 1, n),
        })

    def test_simulate_budget_strategies_includes_random(self, combined_fixture):
        summary = business_simulation.simulate_budget_strategies(
            combined_fixture, {"A": "score_a", "B": "score_b"}, budget_levels=(0.10,), random_state=42,
        )
        assert "Random Selection" in summary["strategy"].values
        assert set(summary["strategy"]) == {"A", "B", "Random Selection"}

    def test_simulate_budget_strategies_customer_counts(self, combined_fixture):
        summary = business_simulation.simulate_budget_strategies(
            combined_fixture, {"A": "score_a"}, budget_levels=(0.20,), random_state=42,
        )
        expected_k = int(np.ceil(len(combined_fixture) * 0.20))
        assert (summary["customers_contacted"] == expected_k).all()

    def test_check_beat_random_structure(self, combined_fixture):
        summary = business_simulation.simulate_budget_strategies(
            combined_fixture, {"A": "score_a"}, budget_levels=(0.10, 0.20), random_state=42,
        )
        beat_random_df, all_beat_random = business_simulation.check_beat_random(
            summary, ["A", "Random Selection"], ["10%", "20%"],
        )
        assert isinstance(all_beat_random, (bool, np.bool_))
        assert len(beat_random_df) == 2  # 1 non-random strategy x 2 budgets