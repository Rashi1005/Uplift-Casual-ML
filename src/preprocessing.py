"""
preprocessing.py
-----------------
Reusable functions behind ``notebooks/01_eda.ipynb`` and
``notebooks/02_preprocessing.ipynb``: deriving the binary treatment
column, the randomisation balance checks, one-hot encoding, and the
stratified train/test split with its post-split balance verification.

Random seeds and file paths are parameters, not hardcoded, so callers
(notebooks or other scripts) control reproducibility and output location.

Public API
----------
binarize_treatment(df, raw_col, treatment_col)
    Add a binary ``treatment`` column derived from the raw 3-arm segment.
outcome_rate_by_group(data, outcome, treat_col)
    Mean outcome split by treatment group.
standardized_mean_diff(data, col, treat_col)
    Standardised mean difference of a numeric feature between groups.
check_numeric_balance(df, numeric_features, treat_col, smd_threshold)
    SMD table for numeric features, flagged if |SMD| > threshold.
check_categorical_balance(df, categorical_features, treat_col, p_threshold)
    Chi-square balance table for categorical features.
run_randomization_check(df, numeric_features, categorical_features, ...)
    Run both balance checks and return a combined PASS/FAIL verdict.
encode_features(df, numeric_features, categorical_features)
    One-hot encode categoricals and concatenate with numerics.
stratified_split(X, treatment, y, test_size, random_state, strat_outcome_col)
    Stratified 80/20 train/test split.
verify_split_balance(treatment_train, y_train, treatment_test, y_test, ...)
    Compare split rates against Phase 1 reference numbers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    pass

# Phase 1's full-dataset reference numbers, used by verify_split_balance()
# as its default comparison target.  Exposed as a module-level constant
# (not buried inside the function) so a caller can inspect or override it
# without editing this file.
PHASE1_REFERENCE_RATES: dict[tuple[str, int], float] = {
    ("conversion", 0): 0.0057,
    ("conversion", 1): 0.0107,
    ("visit", 0): 0.1062,
    ("visit", 1): 0.1670,
}
PHASE1_REFERENCE_CONTROL_PCT: float = 33.29
PHASE1_REFERENCE_TREATMENT_PCT: float = 66.71


# ---------------------------------------------------------------------------
# Treatment derivation
# ---------------------------------------------------------------------------


def binarize_treatment(
    df: pd.DataFrame,
    raw_col: str = "segment",
    treatment_col: str = "treatment",
) -> pd.DataFrame:
    """Add a binary ``treatment_col`` derived from the raw 3-arm ``raw_col``.

    1 = received either email arm; 0 = "No E-Mail" (control).  This is
    the exact binarisation notebooks 01 and 02 each performed
    independently; centralising it here means every caller uses the
    identical definition.

    Parameters
    ----------
    df:
        Must contain ``raw_col``.
    raw_col:
        Name of the raw 3-arm segment column.
    treatment_col:
        Name of the binary column to create.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` with ``treatment_col`` added.
    """
    out = df.copy()
    out[treatment_col] = (out[raw_col] != "No E-Mail").astype(int)
    return out


# ---------------------------------------------------------------------------
# Randomisation / balance checks (notebooks/01_eda.ipynb)
# ---------------------------------------------------------------------------


def outcome_rate_by_group(
    data: pd.DataFrame,
    outcome: str,
    treat_col: str = "treatment",
) -> tuple[float, float]:
    """Mean of ``outcome``, split by ``treat_col``.

    Parameters
    ----------
    data:
        DataFrame containing both ``outcome`` and ``treat_col``.
    outcome:
        Name of the binary outcome column.
    treat_col:
        Name of the binary treatment column.

    Returns
    -------
    tuple[float, float]
        ``(control_rate, treat_rate)``
    """
    rates = data.groupby(treat_col)[outcome].mean()
    return rates.get(0, np.nan), rates.get(1, np.nan)


def standardized_mean_diff(
    data: pd.DataFrame,
    col: str,
    treat_col: str = "treatment",
) -> tuple[float, float, float]:
    """Standardised mean difference (SMD) of a numeric feature between
    the treatment and control groups.

    Parameters
    ----------
    data:
        DataFrame containing ``col`` and ``treat_col``.
    col:
        Numeric feature column to compare.
    treat_col:
        Name of the binary treatment column.

    Returns
    -------
    tuple[float, float, float]
        ``(treat_mean, control_mean, smd)``
    """
    t = data.loc[data[treat_col] == 1, col]
    c = data.loc[data[treat_col] == 0, col]
    pooled_std = np.sqrt((t.std() ** 2 + c.std() ** 2) / 2)
    smd = (t.mean() - c.mean()) / pooled_std if pooled_std > 0 else 0.0
    return t.mean(), c.mean(), smd


def check_numeric_balance(
    df: pd.DataFrame,
    numeric_features: list[str],
    treat_col: str = "treatment",
    smd_threshold: float = 0.1,
) -> pd.DataFrame:
    """Standardised mean difference for each numeric feature, flagged
    ``IMBALANCED`` if ``|SMD|`` exceeds ``smd_threshold``.

    Parameters
    ----------
    df:
        DataFrame with treatment and feature columns.
    numeric_features:
        Feature columns to check.
    treat_col:
        Name of the binary treatment column.
    smd_threshold:
        Absolute SMD above which a feature is flagged as imbalanced.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``control_mean``, ``treatment_mean``,
        ``SMD``, ``flag``.
    """
    rows = []
    for col in numeric_features:
        treat_mean, control_mean, smd = standardized_mean_diff(df, col, treat_col)
        rows.append(
            {
                "feature": col,
                "control_mean": round(control_mean, 4),
                "treatment_mean": round(treat_mean, 4),
                "SMD": round(smd, 4),
                "flag": "IMBALANCED" if abs(smd) > smd_threshold else "ok",
            }
        )
    if not rows:
        return pd.DataFrame(columns=["feature", "control_mean", "treatment_mean", "SMD", "flag"])
    return pd.DataFrame(rows)


def check_categorical_balance(
    df: pd.DataFrame,
    categorical_features: list[str],
    treat_col: str = "treatment",
    p_threshold: float = 0.05,
) -> pd.DataFrame:
    """Chi-square test of independence between each categorical feature
    and treatment assignment, flagged ``IMBALANCED`` if p < ``p_threshold``.

    Parameters
    ----------
    df:
        DataFrame with treatment and feature columns.
    categorical_features:
        Feature columns to check.
    treat_col:
        Name of the binary treatment column.
    p_threshold:
        p-value below which a feature is flagged as imbalanced.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``chi2``, ``p_value``, ``flag``.
    """
    rows = []
    for col in categorical_features:
        contingency = pd.crosstab(df[col], df[treat_col])
        chi2, p_value, _dof, _expected = stats.chi2_contingency(contingency)
        rows.append(
            {
                "feature": col,
                "chi2": round(chi2, 3),
                "p_value": round(p_value, 4),
                "flag": "IMBALANCED" if p_value < p_threshold else "ok",
            }
        )
    if not rows:
        return pd.DataFrame(columns=["feature", "chi2", "p_value", "flag"])
    return pd.DataFrame(rows)


def run_randomization_check(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    treat_col: str = "treatment",
    smd_threshold: float = 0.1,
    p_threshold: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame, bool, list[str]]:
    """Run both balance checks and combine them into one PASS/FAIL verdict.

    Parameters
    ----------
    df:
        Full dataset with treatment and all feature columns.
    numeric_features:
        Numeric feature columns to check.
    categorical_features:
        Categorical feature columns to check.
    treat_col:
        Name of the binary treatment column.
    smd_threshold:
        Absolute SMD threshold (see ``check_numeric_balance``).
    p_threshold:
        p-value threshold (see ``check_categorical_balance``).

    Returns
    -------
    tuple
        ``(numeric_balance_df, categorical_balance_df, passed, flagged_features)``
    """
    numeric_balance_df = check_numeric_balance(df, numeric_features, treat_col, smd_threshold)
    categorical_balance_df = check_categorical_balance(
        df, categorical_features, treat_col, p_threshold
    )

    flagged = (
        numeric_balance_df.loc[numeric_balance_df["flag"] == "IMBALANCED", "feature"].tolist()
        + categorical_balance_df.loc[
            categorical_balance_df["flag"] == "IMBALANCED", "feature"
        ].tolist()
    )
    passed = len(flagged) == 0
    return numeric_balance_df, categorical_balance_df, passed, flagged


# ---------------------------------------------------------------------------
# Feature encoding and splitting (notebooks/02_preprocessing.ipynb)
# ---------------------------------------------------------------------------


def encode_features(
    df: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    """One-hot encode ``categorical_features`` and concatenate with
    ``numeric_features``, unchanged.

    ``drop_first=False`` and ``dtype=int`` match the original notebook's
    encoding exactly (dummy columns as 0/1 integers, not booleans; no
    reference category dropped, since tree-based models do not need it).

    Parameters
    ----------
    df:
        Source DataFrame containing all listed feature columns.
    numeric_features:
        Numeric columns to pass through unchanged.
    categorical_features:
        Categorical columns to one-hot encode.

    Returns
    -------
    pd.DataFrame
        Feature matrix X: numeric_features (untouched) + one-hot encoded
        categorical_features.
    """
    encoded_categoricals = pd.get_dummies(
        df[categorical_features],
        columns=categorical_features,
        drop_first=False,
        dtype=int,
    )
    return pd.concat([df[numeric_features], encoded_categoricals], axis=1)


def stratified_split(
    X: pd.DataFrame,
    treatment: pd.Series,
    y: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    strat_outcome_col: str = "conversion",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """80/20 (by default) train/test split, stratified on a combined
    treatment + ``strat_outcome_col`` key, so all four
    (treated/control x outcome-positive/negative) combinations are
    proportionally represented in both sets.

    Parameters
    ----------
    X:
        Feature matrix.
    treatment:
        Binary treatment series; must share index/order with ``X`` and ``y``.
    y:
        Outcome DataFrame; must contain ``strat_outcome_col``.
    test_size:
        Fraction of rows to place in the test set.
    random_state:
        Seed for reproducibility -- configurable, not hardcoded.
    strat_outcome_col:
        Which column of ``y`` to combine with ``treatment`` for the
        stratification key (the original notebook used ``"conversion"``).

    Returns
    -------
    tuple
        ``(X_train, X_test, treatment_train, treatment_test, y_train, y_test)``
    """
    strat_key = treatment.astype(str) + "_" + y[strat_outcome_col].astype(str)
    return train_test_split(
        X,
        treatment,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=strat_key,
    )


def verify_split_balance(
    treatment_train: pd.Series,
    y_train: pd.DataFrame,
    treatment_test: pd.Series,
    y_test: pd.DataFrame,
    reference_rates: dict[tuple[str, int], float] | None = None,
    reference_control_pct: float = PHASE1_REFERENCE_CONTROL_PCT,
    reference_treatment_pct: float = PHASE1_REFERENCE_TREATMENT_PCT,
    tolerance: float = 0.01,
    outcomes_to_check: tuple[str, ...] = ("visit", "conversion"),
) -> tuple[pd.DataFrame, bool]:
    """Compare the train and test splits' treatment ratio and outcome
    rates against a reference (by default, Phase 1's full-dataset
    numbers), flagging any deviation beyond ``tolerance``.

    Parameters
    ----------
    treatment_train:
        Binary treatment series for the training set.
    y_train:
        Outcome DataFrame for the training set.
    treatment_test:
        Binary treatment series for the test set.
    y_test:
        Outcome DataFrame for the test set.
    reference_rates:
        ``{(outcome_col, treatment_value): rate}`` reference rates.
        Defaults to ``PHASE1_REFERENCE_RATES``.
    reference_control_pct:
        Expected control-group percentage.
    reference_treatment_pct:
        Expected treatment-group percentage.
    tolerance:
        Absolute deviation (as a fraction) above which a rate is flagged.
    outcomes_to_check:
        Outcome columns to verify.

    Returns
    -------
    tuple[pd.DataFrame, bool]
        ``(verification_df, any_deviation)``
    """
    if reference_rates is None:
        reference_rates = PHASE1_REFERENCE_RATES

    def treatment_ratio(treat_series: pd.Series) -> tuple[int, int, float, float]:
        n = len(treat_series)
        n_treat = int(treat_series.sum())
        return n_treat, n - n_treat, n_treat / n * 100, (n - n_treat) / n * 100

    rows = []
    for split_name, treat_series, y_series in [
        ("train", treatment_train, y_train),
        ("test", treatment_test, y_test),
    ]:
        n_treat, n_control, pct_treat, pct_control = treatment_ratio(treat_series)

        control_pct_dev = abs(pct_control - reference_control_pct)
        treat_pct_dev = abs(pct_treat - reference_treatment_pct)
        split_flag = (
            "DEVIATES"
            if (control_pct_dev > tolerance * 100 or treat_pct_dev > tolerance * 100)
            else "ok"
        )

        rows.append(
            {
                "split": split_name,
                "metric": "treatment/control split",
                "control": f"{n_control:,} ({pct_control:.2f}%, ref {reference_control_pct:.2f}%)",
                "treatment": (
                    f"{n_treat:,} ({pct_treat:.2f}%, ref {reference_treatment_pct:.2f}%)"
                ),
                "flag": split_flag,
            }
        )

        for outcome in outcomes_to_check:
            rates = y_series.groupby(treat_series)[outcome].mean()
            control_rate = rates.get(0, np.nan)
            treat_rate = rates.get(1, np.nan)

            ref_control = reference_rates[(outcome, 0)]
            ref_treat = reference_rates[(outcome, 1)]
            control_dev = abs(control_rate - ref_control)
            treat_dev = abs(treat_rate - ref_treat)
            flag = "DEVIATES" if (control_dev > tolerance or treat_dev > tolerance) else "ok"

            rows.append(
                {
                    "split": split_name,
                    "metric": f"{outcome} rate",
                    "control": f"{control_rate * 100:.2f}% (ref {ref_control * 100:.2f}%)",
                    "treatment": f"{treat_rate * 100:.2f}% (ref {ref_treat * 100:.2f}%)",
                    "flag": flag,
                }
            )

    verification_df = pd.DataFrame(rows)
    any_deviation = bool((verification_df["flag"] == "DEVIATES").any())
    return verification_df, any_deviation
