"""
baseline_model.py
------------------
Reusable functions behind notebooks/03_baseline_model.ipynb: training the
naive (non-causal) LightGBM classifier, evaluating it, building the
"naive targeting" ranking, and the top-decile signal check.

This module does NOT decide what target column to use -- the caller
passes `target_col` explicitly (notebooks/03_baseline_model.ipynb passes
"visit", the corrected target documented in that notebook; see the README
for why "conversion" was abandoned). This module never silently changes
that choice.
"""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from src.utils import compute_scale_pos_weight, top_k_split
except ImportError:
    from utils import compute_scale_pos_weight, top_k_split


def train_baseline_model(
    X_train: pd.DataFrame,
    y_train_target: pd.Series,
    random_state: int = 42,
    n_estimators: int = 200,
) -> LGBMClassifier:
    """Train a plain LightGBM classifier on the feature matrix to predict
    `y_train_target`. Treatment must already be excluded from `X_train`
    by the caller -- this function does not check for or drop it.

    `scale_pos_weight` is computed from the target's own class balance
    (see utils.compute_scale_pos_weight), exactly as the original
    notebook did.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix (treatment must not be a column).
    y_train_target : pd.Series
        Binary target to predict.
    random_state : int
        Seed, configurable rather than hardcoded.
    n_estimators : int
        Number of boosting rounds (200 in the original notebook).

    Returns
    -------
    LGBMClassifier
        Fitted model.
    """
    scale_pos_weight = compute_scale_pos_weight(y_train_target)
    model = LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_estimators=n_estimators,
        verbosity=-1,
    )
    model.fit(X_train, y_train_target)
    return model


def evaluate_baseline_model(
    model: LGBMClassifier,
    X_test: pd.DataFrame,
    y_test_target: pd.Series,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Predict on `X_test` and report AUC-ROC, precision/recall (at
    `threshold`), and average precision -- the same metrics notebook 03
    reports. Accuracy is intentionally not included (see the notebook's
    markdown for why: it's misleading on this class-imbalanced target).

    Returns
    -------
    dict
        Keys: "auc_roc", "precision", "recall", "average_precision",
        plus "test_probs" (the raw predicted probabilities, needed by
        the ranking/signal-check steps that follow).
    """
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= threshold).astype(int)

    return {
        "auc_roc": roc_auc_score(y_test_target, test_probs),
        "precision": precision_score(y_test_target, test_preds, zero_division=0),
        "recall": recall_score(y_test_target, test_preds, zero_division=0),
        "average_precision": average_precision_score(y_test_target, test_probs),
        "test_probs": test_probs,
    }


def build_ranking(
    X_test: pd.DataFrame,
    test_probs: np.ndarray,
    treatment_test: pd.Series,
    y_test: pd.DataFrame,
) -> pd.DataFrame:
    """Build the "naive targeting" ranking: every test customer, their
    predicted probability, and their actual treatment/outcomes, sorted
    descending by predicted probability.

    Parameters
    ----------
    X_test : pd.DataFrame
        Used only for its index (`row_index` in the output).
    test_probs : np.ndarray
        Predicted probabilities from evaluate_baseline_model().
    treatment_test, y_test : the actual treatment and outcome columns
        for the same rows, in the same order as X_test.

    Returns
    -------
    pd.DataFrame
        Columns: row_index, predicted_prob, actual_treatment,
        actual_visit, actual_conversion -- sorted by predicted_prob
        descending.
    """
    return pd.DataFrame({
        "row_index": X_test.index,
        "predicted_prob": test_probs,
        "actual_treatment": treatment_test.values,
        "actual_visit": y_test["visit"].values,
        "actual_conversion": y_test["conversion"].values,
    }).sort_values("predicted_prob", ascending=False).reset_index(drop=True)


def signal_check(
    ranking: pd.DataFrame,
    target_col: str,
    top_fraction: float = 0.10,
    min_meaningful_gap_pp: float = 0.02,
) -> Dict[str, float]:
    """Compute the actual `target_col` rate in the top `top_fraction` of
    the ranking vs. the rest, and flag whether the gap is meaningful.

    This is the check that originally caught the near-random `conversion`
    model (see notebooks/03_baseline_model.ipynb, Section 2) -- kept as a
    permanent, reusable safeguard rather than a one-off notebook cell.

    Parameters
    ----------
    ranking : pd.DataFrame
        Output of build_ranking(), must contain an `actual_{target_col}`
        column.
    target_col : str
        Name of the target the ranking was built for (e.g. "visit").
    top_fraction : float
        Size of the "top" slice to compare (0.10 = top decile, matching
        the original notebook).
    min_meaningful_gap_pp : float
        Minimum gap (as a fraction, e.g. 0.02 = 2 percentage points) to
        consider the ranking as carrying meaningful signal.

    Returns
    -------
    dict
        Keys: overall_rate, top_rate, rest_rate, gap, lift_ratio, passed.
    """
    outcome_col = f"actual_{target_col}"
    top, rest = top_k_split(ranking, "predicted_prob", top_fraction)

    overall_rate = ranking[outcome_col].mean()
    top_rate = top[outcome_col].mean()
    rest_rate = rest[outcome_col].mean()
    gap = top_rate - rest_rate
    lift_ratio = (top_rate / overall_rate) if overall_rate > 0 else np.nan

    return {
        "overall_rate": overall_rate,
        "top_rate": top_rate,
        "rest_rate": rest_rate,
        "gap": gap,
        "lift_ratio": lift_ratio,
        "passed": gap >= min_meaningful_gap_pp,
    }