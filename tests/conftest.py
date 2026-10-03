"""
conftest.py
-----------
Shared pytest configuration and fixtures for the whole ``tests/`` suite.

Adds ``src/`` to ``sys.path`` once, here, so individual test files do not
each need their own ``sys.path.insert`` boilerplate, and defines the
fixtures used by more than one test file (``test_modules.py``,
``test_uplift_causal_forest.py``, ``test_error_handling.py``) so the
same synthetic dataset-generation logic is not duplicated across files.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def rng():
    """A seeded RNG — deterministic across test runs.

    Every test that generates synthetic data goes through this (directly
    or via a fixture built on it), so re-running the suite always produces
    the same fixtures and the same assertions either pass or fail
    consistently, never flakily.
    """
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_split(rng):
    """A small, synthetic train/test split with the same shape/roles as
    the real pipeline: a feature matrix, binary treatment, and a binary
    outcome, with treatment carrying real (if modest) signal.

    Used by baseline-model, uplift-model, and causal-forest tests alike,
    so all three exercise a consistent, comparable dataset shape.
    """
    n_train, n_test = 400, 100
    n = n_train + n_test

    treatment = pd.Series(rng.integers(0, 2, n))
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    X = pd.DataFrame(
        {"x1": x1, "x2": x2, "cat_A": rng.integers(0, 2, n), "cat_B": rng.integers(0, 2, n)}
    )

    # outcome depends a bit on x1 and on treatment, so models have something
    # real, if weak, to learn.
    logit = -1.0 + 0.5 * x1 + 0.8 * treatment.values
    prob = 1 / (1 + np.exp(-logit))
    y = pd.Series((rng.random(n) < prob).astype(int))
    y_df = pd.DataFrame({"visit": y, "conversion": pd.Series(rng.integers(0, 2, n))})

    X_train, X_test = (
        X.iloc[:n_train].reset_index(drop=True),
        X.iloc[n_train:].reset_index(drop=True),
    )
    treatment_train, treatment_test = (
        treatment.iloc[:n_train].reset_index(drop=True),
        treatment.iloc[n_train:].reset_index(drop=True),
    )
    y_train, y_test = (
        y_df.iloc[:n_train].reset_index(drop=True),
        y_df.iloc[n_train:].reset_index(drop=True),
    )

    return X_train, X_test, treatment_train, treatment_test, y_train, y_test


@pytest.fixture
def tiny_split(rng):
    """An even smaller split than ``synthetic_split``, purpose-built for the
    Causal Forest test — big enough for econml's cross-fitting to run
    without degenerate folds, small enough to keep the (inherently
    slower) Causal Forest fit under a few seconds.
    """
    n_train, n_test = 150, 40
    n = n_train + n_test

    treatment = pd.Series(rng.integers(0, 2, n))
    x1 = rng.normal(0, 1, n)
    X = pd.DataFrame({"x1": x1, "x2": rng.normal(0, 1, n)})

    logit = -0.5 + 0.4 * x1 + 0.6 * treatment.values
    prob = 1 / (1 + np.exp(-logit))
    y = pd.Series((rng.random(n) < prob).astype(int))

    X_train, X_test = (
        X.iloc[:n_train].reset_index(drop=True),
        X.iloc[n_train:].reset_index(drop=True),
    )
    treatment_train, treatment_test = (
        treatment.iloc[:n_train].reset_index(drop=True),
        treatment.iloc[n_train:].reset_index(drop=True),
    )
    y_train, y_test = (
        y.iloc[:n_train].reset_index(drop=True),
        y.iloc[n_train:].reset_index(drop=True),
    )

    return X_train, X_test, treatment_train, treatment_test, y_train, y_test
