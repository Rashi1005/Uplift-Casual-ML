"""
utils.py
--------
Small, shared helper functions used by more than one module in this
project. Extracted here specifically to avoid duplicating the same logic
in both src/baseline_model.py and src/uplift_models.py (they used
identical column-sanitization and class-weighting code in the original
notebooks).
"""

import re
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd


def sanitize_columns(columns: Iterable[str]) -> List[str]:
    """Replace any character that isn't alphanumeric or underscore with
    an underscore, in every column name.

    LightGBM rejects feature names containing special JSON characters
    (e.g. the one-hot columns produced in preprocessing, such as
    "history_segment_7) $1,000 +", contain '$', ',', '(', ')'). This does
    not change which features exist or what they mean -- only how their
    names are spelled for LightGBM's benefit. Identical logic to what
    notebooks 03 and 04 each used to run separately.

    Parameters
    ----------
    columns : Iterable[str]
        Original column names.

    Returns
    -------
    List[str]
        Sanitized column names, same order and length as the input.
    """
    return [re.sub(r"[^A-Za-z0-9_]+", "_", str(col)) for col in columns]


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute LightGBM's `scale_pos_weight` from a binary target's class
    counts (negative count / positive count).

    Used identically for the Phase 3 baseline model and for each of the
    two per-arm submodels in the Phase 4 Two-Model Approach.

    Parameters
    ----------
    y : pd.Series
        Binary (0/1) target.

    Returns
    -------
    float
        n_negative / n_positive.
    """
    n_negative = (y == 0).sum()
    n_positive = (y == 1).sum()
    return n_negative / n_positive


def top_k_split(df: pd.DataFrame, score_col: str, k_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into its top-k_fraction rows by `score_col`
    (descending) and everything else.

    Shared logic behind the Phase 3 signal check (top decile vs. bottom
    90% by predicted probability), the Phase 4 signal check (top decile
    vs. bottom 90% by uplift score), and the Phase 6 budget simulation's
    "top-K% by predicted score" selection -- all three used this same
    "rank descending, slice the top k_fraction" operation on different
    columns.

    Parameters
    ----------
    df : pd.DataFrame
        Rows to split (each row already carries its score column).
    score_col : str
        Column to rank by, descending.
    k_fraction : float
        Fraction of rows (0 < k_fraction < 1) to place in the "top" split.

    Returns
    -------
    (top, rest) : Tuple[pd.DataFrame, pd.DataFrame]
        top has `ceil(len(df) * k_fraction)` rows; rest has the remainder.
        Both are freshly indexed (0..n-1).
    """
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    n = len(ranked)
    top_n = int(np.ceil(n * k_fraction))
    return ranked.iloc[:top_n].reset_index(drop=True), ranked.iloc[top_n:].reset_index(drop=True)


def actual_uplift(df: pd.DataFrame, treat_col: str = "actual_treatment", outcome_col: str = "actual_visit") -> Tuple[float, int, int]:
    """Real (treated rate - control rate) within a subgroup of rows that
    already carry actual treatment/outcome columns.

    Identical logic used in Phase 4 (signal check on uplift-model
    rankings) and Phase 6 (business simulation's incremental-rate
    calculation per selected group) -- both computed this same quantity
    on different subgroups.

    Parameters
    ----------
    df : pd.DataFrame
        Subgroup of test rows, must contain `treat_col` and `outcome_col`.
    treat_col : str
        Column holding the actual (real, not predicted) treatment
        indicator (1 = treated, 0 = control).
    outcome_col : str
        Column holding the actual (real, not predicted) outcome.

    Returns
    -------
    (uplift, n_treated, n_control) : Tuple[float, int, int]
        uplift is NaN if either arm is empty within this subgroup.
    """
    treated = df.loc[df[treat_col] == 1, outcome_col]
    control = df.loc[df[treat_col] == 0, outcome_col]
    if len(treated) == 0 or len(control) == 0:
        return np.nan, len(treated), len(control)
    return treated.mean() - control.mean(), len(treated), len(control)