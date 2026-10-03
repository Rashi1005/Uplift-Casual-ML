"""
utils.py
--------
Small, shared helper functions used by more than one module in this
project.  Extracted here specifically to avoid duplicating the same logic
in both ``src/baseline_model.py`` and ``src/uplift_models.py`` (they used
identical column-sanitisation and class-weighting code in the original
notebooks).

Public API
----------
sanitize_columns(columns)
    Replace special characters in column names so LightGBM accepts them.
compute_scale_pos_weight(y)
    LightGBM ``scale_pos_weight`` from a binary target's class counts.
top_k_split(df, score_col, k_fraction)
    Split a DataFrame into the top-k and the rest by a score column.
actual_uplift(df, treat_col, outcome_col)
    Real treated-vs-control outcome-rate gap within a subgroup.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


def sanitize_columns(columns: Iterable[str]) -> list[str]:
    """Replace every character that is not alphanumeric or underscore with
    an underscore, in every column name.

    LightGBM rejects feature names that contain special JSON characters
    (e.g. the one-hot columns produced in preprocessing, such as
    ``"history_segment_7) $1,000 +"``, contain ``$``, ``,``, ``(``, ``)``).
    This does not change which features exist or what they mean -- only how
    their names are spelled for LightGBM's benefit.  Identical logic to
    what notebooks 03 and 04 each used to run separately.

    Parameters
    ----------
    columns:
        Original column names.

    Returns
    -------
    list[str]
        Sanitised column names, same order and length as the input.
    """
    return [re.sub(r"[^A-Za-z0-9_]+", "_", str(col)) for col in columns]


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Compute LightGBM's ``scale_pos_weight`` from a binary target's class
    counts (negative count / positive count).

    Used identically for the Phase 3 baseline model and for each of the
    two per-arm submodels in the Phase 4 Two-Model Approach.

    Parameters
    ----------
    y:
        Binary (0/1) target.

    Returns
    -------
    float
        ``n_negative / n_positive``.
    """
    n_negative = int((y == 0).sum())
    n_positive = int((y == 1).sum())
    if n_positive == 0:
        raise ValueError(
            f"compute_scale_pos_weight: target has no positive examples "
            f"(n_positive=0, n_negative={n_negative}). "
            "Check that the target column is binary (0/1) with at least one positive."
        )
    return n_negative / n_positive


def top_k_split(
    df: pd.DataFrame,
    score_col: str,
    k_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into its top-``k_fraction`` rows by ``score_col``
    (descending) and everything else.

    Shared logic behind the Phase 3 signal check (top decile vs. bottom
    90% by predicted probability), the Phase 4 signal check (top decile
    vs. bottom 90% by uplift score), and the Phase 6 budget simulation's
    "top-K% by predicted score" selection -- all three used this same
    "rank descending, slice the top k_fraction" operation on different
    columns.

    Parameters
    ----------
    df:
        Rows to split (each row already carries its score column).
    score_col:
        Column to rank by, descending.
    k_fraction:
        Fraction of rows (0 < k_fraction < 1) to place in the "top" split.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(top, rest)`` -- ``top`` has ``ceil(len(df) * k_fraction)`` rows;
        ``rest`` has the remainder.  Both are freshly indexed (0..n-1).
    """
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)
    n = len(ranked)
    top_n = int(np.ceil(n * k_fraction))
    return ranked.iloc[:top_n].reset_index(drop=True), ranked.iloc[top_n:].reset_index(drop=True)


def actual_uplift(
    df: pd.DataFrame,
    treat_col: str = "actual_treatment",
    outcome_col: str = "actual_visit",
) -> tuple[float, int, int]:
    """Real (treated rate - control rate) within a subgroup of rows that
    already carry actual treatment/outcome columns.

    Identical logic used in Phase 4 (signal check on uplift-model
    rankings) and Phase 6 (business simulation's incremental-rate
    calculation per selected group) -- both computed this same quantity
    on different subgroups.

    Parameters
    ----------
    df:
        Subgroup of test rows; must contain ``treat_col`` and
        ``outcome_col``.
    treat_col:
        Column holding the actual (real, not predicted) treatment
        indicator (1 = treated, 0 = control).
    outcome_col:
        Column holding the actual (real, not predicted) outcome.

    Returns
    -------
    tuple[float, int, int]
        ``(uplift, n_treated, n_control)`` -- ``uplift`` is ``nan`` if
        either arm is empty within this subgroup.
    """
    treated = df.loc[df[treat_col] == 1, outcome_col]
    control = df.loc[df[treat_col] == 0, outcome_col]
    if len(treated) == 0 or len(control) == 0:
        return np.nan, len(treated), len(control)
    return treated.mean() - control.mean(), len(treated), len(control)
