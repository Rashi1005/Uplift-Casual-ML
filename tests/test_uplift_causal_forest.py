"""
test_uplift_causal_forest.py
------------------------------
Dedicated tests for ``src/uplift_models.py``'s ``causal_forest_model()``
— split out from ``test_modules.py`` because, unlike every other function
in this project's test suite, this one cannot be made to run in
milliseconds: even on a tiny synthetic dataset, fitting a cross-fitted
Causal Forest (econml's ``CausalForestDML``) takes several seconds, since
the cross-fitting/nuisance-model machinery has real, fixed overhead
regardless of dataset size.

All tests here are marked ``@pytest.mark.slow``.  Run the fast suite only::

    pytest -m "not slow"

Run everything, including these::

    pytest

Run only these::

    pytest -m slow tests/test_uplift_causal_forest.py -v
"""

import numpy as np
import pytest

import uplift_models

pytestmark = pytest.mark.slow


class TestCausalForestModel:
    def test_output_shape_matches_test_set(self, tiny_split):
        X_train, X_test, treatment_train, treatment_test, y_train, y_test = tiny_split
        uplift, model, library = uplift_models.causal_forest_model(
            X_train,
            y_train,
            treatment_train,
            X_test,
            random_state=42,
            n_estimators=20,
            nuisance_n_estimators=10,
            cv=2,
            auto_install=False,
        )
        assert len(uplift) == len(X_test)
        assert model is not None

    def test_output_is_finite(self, tiny_split):
        X_train, X_test, treatment_train, treatment_test, y_train, y_test = tiny_split
        uplift, _, _ = uplift_models.causal_forest_model(
            X_train,
            y_train,
            treatment_train,
            X_test,
            random_state=42,
            n_estimators=20,
            nuisance_n_estimators=10,
            cv=2,
            auto_install=False,
        )
        assert np.all(np.isfinite(uplift)), (
            "Causal Forest produced NaN or infinite treatment effect estimates."
        )

    def test_reports_library_used(self, tiny_split):
        X_train, X_test, treatment_train, treatment_test, y_train, y_test = tiny_split
        _, _, library = uplift_models.causal_forest_model(
            X_train,
            y_train,
            treatment_train,
            X_test,
            random_state=42,
            n_estimators=20,
            nuisance_n_estimators=10,
            cv=2,
            auto_install=False,
        )
        assert library in ("econml", "causalml")

    def test_reproducible_with_same_seed(self, tiny_split):
        X_train, X_test, treatment_train, treatment_test, y_train, y_test = tiny_split
        uplift_a, _, _ = uplift_models.causal_forest_model(
            X_train,
            y_train,
            treatment_train,
            X_test,
            random_state=11,
            n_estimators=20,
            nuisance_n_estimators=10,
            cv=2,
            auto_install=False,
        )
        uplift_b, _, _ = uplift_models.causal_forest_model(
            X_train,
            y_train,
            treatment_train,
            X_test,
            random_state=11,
            n_estimators=20,
            nuisance_n_estimators=10,
            cv=2,
            auto_install=False,
        )
        np.testing.assert_allclose(uplift_a, uplift_b)

    def test_raises_when_neither_library_available(self, tiny_split, monkeypatch):
        """If a caller disables auto-install AND neither econml nor
        causalml can be imported, this should fail loudly with a clear,
        actionable ``ImportError`` — not hang, not silently return garbage.
        """
        X_train, X_test, treatment_train, treatment_test, y_train, y_test = tiny_split

        import builtins

        real_import = builtins.__import__

        def blocking_import(name, *args, **kwargs):
            if name in ("econml", "econml.dml", "causalml", "causalml.inference.tree"):
                raise ImportError(f"simulated: {name} not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocking_import)

        with pytest.raises(ImportError, match="Neither econml nor causalml"):
            uplift_models.causal_forest_model(
                X_train,
                y_train,
                treatment_train,
                X_test,
                random_state=42,
                n_estimators=20,
                nuisance_n_estimators=10,
                cv=2,
                auto_install=False,
            )
