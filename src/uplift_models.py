"""
uplift_models.py
-----------------
Reusable functions behind ``notebooks/04_uplift_models.ipynb``: the
Two-Model Approach, Class Transformation, and Causal Forest, plus the
top-decile signal check and the combined-rankings merge.

This module preserves the original notebook's exact modelling choices:

- **Two-Model Approach**: separate ``LGBMClassifier`` instances on
  treated-only and control-only rows; uplift = difference in predictions.
- **Class Transformation**: the standard revert-label formula
  (Gutierrez & Gerardy, 2017); uplift = 2 x P(Z=1) - 1.  This module
  does **not** include the propensity-weighted correction explored later
  for the Criteo scale-up work -- that was a separate addendum, not part
  of the original six-notebook methodology this refactor preserves.
- **Causal Forest**: econml's ``CausalForestDML`` (Athey, Tibshirani &
  Wager, 2019), with a fallback to causalml's
  ``UpliftRandomForestClassifier`` if econml is unavailable.

Public API
----------
two_model_approach(X_train, y_train_target, treatment_train, X_test, ...)
    Two-model uplift estimator.
class_transformation(X_train, y_train_target, treatment_train, X_test, ...)
    Class-transformation (revert-label) uplift estimator.
causal_forest_model(X_train, y_train_target, treatment_train, X_test, ...)
    Causal Forest treatment-effect estimator (econml / causalml).
signal_check_uplift(uplift_scores, treatment_test, y_test_target, ...)
    Top-decile actual-uplift signal check for an uplift model.
combine_rankings(X_test, treatment_test, y_test, ...)
    Merge all model scores into one combined-rankings DataFrame.
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

try:
    from src.utils import actual_uplift, compute_scale_pos_weight, top_k_split
except ImportError:
    from utils import actual_uplift, compute_scale_pos_weight, top_k_split


# ---------------------------------------------------------------------------
# Two-Model Approach
# ---------------------------------------------------------------------------


def two_model_approach(
    X_train: pd.DataFrame,
    y_train_target: pd.Series,
    treatment_train: pd.Series,
    X_test: pd.DataFrame,
    random_state: int = 42,
    n_estimators: int = 200,
) -> tuple[np.ndarray, LGBMClassifier, LGBMClassifier]:
    """Train separate classifiers on the treated-only and control-only
    training rows; uplift = P(outcome | treated model) - P(outcome |
    control model), evaluated on the same ``X_test`` for both.

    Parameters
    ----------
    X_train:
        Feature matrix for the training set (treatment excluded).
    y_train_target:
        Binary outcome for the training set.
    treatment_train:
        Binary treatment indicator for the training set.
    X_test:
        Feature matrix for the test set.
    random_state:
        Seed for reproducibility.
    n_estimators:
        Number of boosting rounds for each sub-model.

    Returns
    -------
    tuple[np.ndarray, LGBMClassifier, LGBMClassifier]
        ``(uplift_scores, treated_model, control_model)``
    """
    train_treated_mask = treatment_train == 1
    train_control_mask = treatment_train == 0

    X_train_treated = X_train[train_treated_mask]
    y_train_treated = y_train_target[train_treated_mask]
    X_train_control = X_train[train_control_mask]
    y_train_control = y_train_target[train_control_mask]

    treated_model = LGBMClassifier(
        scale_pos_weight=compute_scale_pos_weight(y_train_treated),
        random_state=random_state,
        n_estimators=n_estimators,
        verbosity=-1,
    )
    control_model = LGBMClassifier(
        scale_pos_weight=compute_scale_pos_weight(y_train_control),
        random_state=random_state,
        n_estimators=n_estimators,
        verbosity=-1,
    )

    treated_model.fit(X_train_treated, y_train_treated)
    control_model.fit(X_train_control, y_train_control)

    treated_pred = treated_model.predict_proba(X_test)[:, 1]
    control_pred = control_model.predict_proba(X_test)[:, 1]
    uplift_scores = treated_pred - control_pred

    return uplift_scores, treated_model, control_model


# ---------------------------------------------------------------------------
# Class Transformation
# ---------------------------------------------------------------------------


def class_transformation(
    X_train: pd.DataFrame,
    y_train_target: pd.Series,
    treatment_train: pd.Series,
    X_test: pd.DataFrame,
    random_state: int = 42,
    n_estimators: int = 200,
) -> tuple[np.ndarray, LGBMClassifier]:
    """Standard class-transformation ("revert label") uplift method.

    ``Z = 1`` if (treatment and outcome) agree (both 1, or both 0),
    else ``Z = 0``.  A single classifier is trained on the full training
    set to predict Z; uplift = 2 x P(Z=1) - 1.

    .. note::
       This formula assumes ~50/50 treatment randomisation.  It is
       reproduced here exactly as the original notebook implemented it
       (Hillstrom's actual split is 66.71%/33.29%, so the resulting
       uplift scores carry a documented calibration bias -- see the
       project's Criteo addendum for the propensity-weighted correction
       explored separately; that correction is intentionally **not**
       included in this module, to preserve the original methodology this
       refactor covers).

    Parameters
    ----------
    X_train:
        Feature matrix for the training set.
    y_train_target:
        Binary outcome for the training set.
    treatment_train:
        Binary treatment indicator for the training set.
    X_test:
        Feature matrix for the test set.
    random_state:
        Seed for reproducibility.
    n_estimators:
        Number of boosting rounds.

    Returns
    -------
    tuple[np.ndarray, LGBMClassifier]
        ``(uplift_scores, model)``
    """
    z_train = (
        ((treatment_train == 1) & (y_train_target == 1))
        | ((treatment_train == 0) & (y_train_target == 0))
    ).astype(int)

    model = LGBMClassifier(
        random_state=random_state,
        n_estimators=n_estimators,
        verbosity=-1,
    )
    model.fit(X_train, z_train)

    p_z1 = model.predict_proba(X_test)[:, 1]
    uplift_scores = 2 * p_z1 - 1
    return uplift_scores, model


# ---------------------------------------------------------------------------
# Causal Forest
# ---------------------------------------------------------------------------


def causal_forest_model(
    X_train: pd.DataFrame,
    y_train_target: pd.Series,
    treatment_train: pd.Series,
    X_test: pd.DataFrame,
    random_state: int = 42,
    n_estimators: int = 200,
    nuisance_n_estimators: int = 100,
    cv: int = 2,
    auto_install: bool = True,
) -> tuple[np.ndarray, Any, str]:
    """Fit a Causal Forest and return per-customer treatment-effect
    estimates on ``X_test``.

    Prefers econml's ``CausalForestDML`` (Athey, Tibshirani & Wager,
    2019).  If unavailable and ``auto_install`` is ``True``, attempts
    ``pip install econml`` at runtime; if that still fails, falls back to
    causalml's ``UpliftRandomForestClassifier``.

    Parameters
    ----------
    X_train:
        Feature matrix for the training set.
    y_train_target:
        Continuous or binary outcome for the training set.
    treatment_train:
        Binary treatment indicator for the training set.
    X_test:
        Feature matrix for the test set.
    random_state:
        Seed for reproducibility.
    n_estimators:
        Trees in the Causal Forest itself.
    nuisance_n_estimators:
        Trees used by econml's internal ``model_y``/``model_t`` nuisance
        estimators (100 in the original notebook, distinct from the
        causal forest's own ``n_estimators``).
    cv:
        Cross-fitting folds for econml's DML residualisation.
    auto_install:
        If ``False``, never attempt a runtime pip install -- just try the
        import and fall back to causalml immediately if it fails.  Useful
        for environments where runtime installs are undesirable (e.g. CI).

    Returns
    -------
    tuple[np.ndarray, Any, str]
        ``(uplift_scores, model, library_used)`` where ``library_used``
        is ``"econml"`` or ``"causalml"``.

    Raises
    ------
    ImportError
        If neither econml nor causalml can be imported (and
        ``auto_install=False`` or the install attempt failed).
    """
    causal_library: str | None = None
    causal_forest_dml_cls: Any = None
    uplift_rf_cls: Any = None

    try:
        from econml.dml import CausalForestDML

        causal_forest_dml_cls = CausalForestDML
        causal_library = "econml"
    except ImportError:
        if auto_install:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "econml", "-q"])
                from econml.dml import CausalForestDML

                causal_forest_dml_cls = CausalForestDML
                causal_library = "econml"
            except Exception:
                causal_library = None
        if causal_library is None:
            try:
                from causalml.inference.tree import UpliftRandomForestClassifier

                uplift_rf_cls = UpliftRandomForestClassifier
                causal_library = "causalml"
            except ImportError as causalml_error:
                raise ImportError(
                    "Neither econml nor causalml could be imported.  Install one "
                    "of them manually (`pip install econml` or `pip install "
                    "causalml`) and retry."
                ) from causalml_error

    if causal_library == "econml":
        model = causal_forest_dml_cls(
            model_y=LGBMRegressor(
                n_estimators=nuisance_n_estimators,
                verbosity=-1,
                random_state=random_state,
            ),
            model_t=LGBMClassifier(
                n_estimators=nuisance_n_estimators,
                verbosity=-1,
                random_state=random_state,
            ),
            discrete_treatment=True,
            n_estimators=n_estimators,
            random_state=random_state,
            cv=cv,
            n_jobs=-1,
        )
        model.fit(y_train_target, treatment_train, X=X_train)
        uplift_scores = model.effect(X_test)
    else:  # causalml fallback
        model = uplift_rf_cls(
            n_estimators=n_estimators,
            control_name="0",
            random_state=random_state,
        )
        model.fit(
            X_train.values,
            treatment_train.astype(str).values,
            y_train_target.values,
        )
        uplift_scores = model.predict(X_test.values).flatten()

    return uplift_scores, model, causal_library


# ---------------------------------------------------------------------------
# Signal check (top-decile actual uplift vs. rest)
# ---------------------------------------------------------------------------


def signal_check_uplift(
    uplift_scores: np.ndarray,
    treatment_test: pd.Series,
    y_test_target: pd.Series,
    top_fraction: float = 0.10,
    min_meaningful_gap_pp: float = 0.01,
) -> dict[str, float]:
    """For a single uplift model's scores, rank test customers
    descending, then compute the ACTUAL uplift (real treated-vs-control
    outcome-rate gap) within the top ``top_fraction`` vs. the rest.

    Parameters
    ----------
    uplift_scores:
        Predicted uplift scores for each test customer, same order as
        ``treatment_test`` and ``y_test_target``.
    treatment_test:
        Actual treatment indicator for the test set.
    y_test_target:
        Actual binary outcome for the test set.
    top_fraction:
        Fraction of customers to place in the "top" group.
    min_meaningful_gap_pp:
        Minimum top-vs-rest uplift gap to consider the model as passing
        the signal check.

    Returns
    -------
    dict
        Keys: ``top_decile_uplift``, ``bottom_90pct_uplift``, ``gap``,
        ``passed``.
    """
    ranked = pd.DataFrame(
        {
            "actual_treatment": treatment_test.values,
            "actual_visit": y_test_target.values,
            "score": uplift_scores,
        }
    )
    top, rest = top_k_split(ranked, "score", top_fraction)

    top_uplift, _, _ = actual_uplift(top)
    rest_uplift, _, _ = actual_uplift(rest)
    gap = top_uplift - rest_uplift

    return {
        "top_decile_uplift": top_uplift,
        "bottom_90pct_uplift": rest_uplift,
        "gap": gap,
        "passed": gap >= min_meaningful_gap_pp,
    }


# ---------------------------------------------------------------------------
# Combine all rankings (baseline + 3 uplift models) into one DataFrame
# ---------------------------------------------------------------------------


def combine_rankings(
    X_test: pd.DataFrame,
    treatment_test: pd.Series,
    y_test: pd.DataFrame,
    uplift_two_model: np.ndarray,
    uplift_class_transform: np.ndarray,
    uplift_causal_forest: np.ndarray,
    baseline_ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the three uplift models' scores with the Phase 3 baseline's
    predicted probability, joined on ``row_index``, into one DataFrame.

    Parameters
    ----------
    X_test:
        Test-set feature matrix; its index is used as ``row_index``.
    treatment_test:
        Actual treatment indicator for each test row.
    y_test:
        Actual outcome DataFrame; must contain ``"visit"`` and
        ``"conversion"``.
    uplift_two_model:
        Uplift scores from ``two_model_approach()``.
    uplift_class_transform:
        Uplift scores from ``class_transformation()``.
    uplift_causal_forest:
        Uplift scores from ``causal_forest_model()``.
    baseline_ranking:
        Output of ``baseline_model.build_ranking()``; must contain
        ``row_index`` and ``predicted_prob``.

    Returns
    -------
    pd.DataFrame
        Columns: ``row_index``, ``actual_treatment``, ``actual_visit``,
        ``actual_conversion``, ``uplift_two_model``,
        ``uplift_class_transform``, ``uplift_causal_forest``,
        ``baseline_predicted_prob``.

    Raises
    ------
    ValueError
        If ``uplift_two_model``, ``uplift_class_transform``, or
        ``uplift_causal_forest`` have a different length from ``X_test``.
    ValueError
        If the join with ``baseline_ranking`` leaves any row without a
        matched baseline probability (``row_index`` mismatch).
    """
    n = len(X_test)
    for name, arr in [
        ("uplift_two_model", uplift_two_model),
        ("uplift_class_transform", uplift_class_transform),
        ("uplift_causal_forest", uplift_causal_forest),
    ]:
        if len(arr) != n:
            raise ValueError(f"Length mismatch: X_test has {n} rows but {name} has {len(arr)}.")

    combined = pd.DataFrame(
        {
            "row_index": X_test.index,
            "actual_treatment": treatment_test.values,
            "actual_visit": y_test["visit"].values,
            "actual_conversion": y_test["conversion"].values,
            "uplift_two_model": uplift_two_model,
            "uplift_class_transform": uplift_class_transform,
            "uplift_causal_forest": uplift_causal_forest,
        }
    )

    baseline_lookup = baseline_ranking[["row_index", "predicted_prob"]].rename(
        columns={"predicted_prob": "baseline_predicted_prob"}
    )
    combined = combined.merge(baseline_lookup, on="row_index", how="left")

    n_unmatched = combined["baseline_predicted_prob"].isna().sum()
    if n_unmatched != 0:
        raise ValueError(
            f"{n_unmatched} row(s) failed to join with the baseline ranking -- row_index mismatch."
        )

    return combined
