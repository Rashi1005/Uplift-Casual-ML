"""
pipeline_meta.py
----------------
Writes a JSON metadata record after every successful CLI pipeline run.

Two files are maintained:

``run_metadata.json``
    The complete metadata for the **most recent** run.  Overwritten on every
    invocation so it always reflects the current state of ``data/processed/``.

``run_log.jsonl``
    Append-only log with one JSON object per line, oldest first.  Every run
    appends one record so the full history is preserved without duplication.

Metadata schema (all fields are optional; callers pass what they know)::

    {
      "run_id":        "f3a1b2c4-...",           # UUID4
      "timestamp":     "2026-10-03T18:31:04+05:30",
      "command":       "evaluate",
      "random_seed":   42,
      "processed_dir": "data/processed",
      "dataset": {
        "n_train":           51200,
        "n_test":            12800,
        "n_features":        17,
        "feature_columns":   ["recency", "history", ...],
        "treatment_column":  "treatment",
        "outcome_column":    "visit"
      },
      "model_params": {"n_estimators": 200, "learning_rate": 0.05},
      "evaluation_metrics": {"Two-Model Approach": {"qini_coefficient": 0.0234}},
      "artifacts_written": ["data/processed/phase5_results.csv", ...],
      "software_versions": {
        "python":        "3.12.5",
        "numpy":         "2.4.6",
        "pandas":        "3.0.5",
        "scikit_learn":  "1.9.0",
        "lightgbm":      "4.7.0",
        "econml":        "0.17.0",
        "scikit_uplift": "0.5.1"
      }
    }
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from typing import Any

# ---------------------------------------------------------------------------
# Software version helpers
# ---------------------------------------------------------------------------

_TRACKED_PACKAGES = [
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "lightgbm",
    "econml",
    "scikit-uplift",
    "joblib",
    "matplotlib",
]


def _pkg_version(name: str) -> str:
    """Return the installed version of *name*, or ``"not installed"``."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def collect_software_versions() -> dict[str, str]:
    """Return a dict of package-name → installed version for all tracked packages.

    The Python interpreter version is included under the key ``"python"``.
    Package names with hyphens are normalised to underscores for JSON
    compatibility (e.g. ``"scikit-learn"`` → ``"scikit_learn"``).
    """
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
    }
    for pkg in _TRACKED_PACKAGES:
        key = pkg.replace("-", "_")
        versions[key] = _pkg_version(pkg)
    return versions


# ---------------------------------------------------------------------------
# Core metadata writer
# ---------------------------------------------------------------------------


def build_metadata(
    *,
    command: str,
    processed_dir: str,
    random_seed: int,
    dataset: dict[str, Any] | None = None,
    model_params: dict[str, Any] | None = None,
    evaluation_metrics: dict[str, Any] | None = None,
    artifacts_written: list[str] | None = None,
) -> dict[str, Any]:
    """Build and return a metadata dict for one pipeline run.

    Parameters
    ----------
    command:
        The CLI subcommand that was run (e.g. ``"train-baseline"``).
    processed_dir:
        Path to the directory where artifacts were written.
    random_seed:
        The random seed used for the run.
    dataset:
        Optional dataset statistics (n_train, n_test, feature_columns, etc.).
    model_params:
        Optional dict of model hyperparameters for this command.
    evaluation_metrics:
        Optional dict of metric name → value (populated by the ``evaluate`` command).
    artifacts_written:
        List of file paths that were written during this run.

    Returns
    -------
    dict[str, Any]
        A fully-populated metadata record ready to be serialised as JSON.
    """
    return {
        "run_id": str(uuid.uuid4()),
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "command": command,
        "random_seed": random_seed,
        "processed_dir": processed_dir,
        "dataset": dataset or {},
        "model_params": model_params or {},
        "evaluation_metrics": evaluation_metrics or {},
        "artifacts_written": artifacts_written or [],
        "software_versions": collect_software_versions(),
    }


def write_run_metadata(
    *,
    command: str,
    processed_dir: str,
    random_seed: int,
    dataset: dict[str, Any] | None = None,
    model_params: dict[str, Any] | None = None,
    evaluation_metrics: dict[str, Any] | None = None,
    artifacts_written: list[str] | None = None,
) -> dict[str, Any]:
    """Write run metadata to ``run_metadata.json`` and append to ``run_log.jsonl``.

    Both files are written to *processed_dir*.  If *processed_dir* does not
    exist it is created automatically.

    Parameters
    ----------
    command:
        The CLI subcommand that was run.
    processed_dir:
        Directory where the metadata files will be written.
    random_seed:
        Random seed used for the run.
    dataset:
        Optional dataset statistics dict.
    model_params:
        Optional model hyperparameters dict.
    evaluation_metrics:
        Optional evaluation metrics dict (from the ``evaluate`` command).
    artifacts_written:
        List of artifact paths written during the run.

    Returns
    -------
    dict[str, Any]
        The metadata record that was written.
    """
    os.makedirs(processed_dir, exist_ok=True)

    meta = build_metadata(
        command=command,
        processed_dir=processed_dir,
        random_seed=random_seed,
        dataset=dataset,
        model_params=model_params,
        evaluation_metrics=evaluation_metrics,
        artifacts_written=artifacts_written,
    )

    # Overwrite latest-run snapshot.
    latest_path = os.path.join(processed_dir, "run_metadata.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    # Append one line to the cumulative log.
    log_path = os.path.join(processed_dir, "run_log.jsonl")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, default=str) + "\n")

    return meta


# ---------------------------------------------------------------------------
# Safe model loading
# ---------------------------------------------------------------------------


def safe_load_model(path: str) -> Any:
    """Load a joblib-serialised model from *path* with informative error messages.

    Raises :class:`FileNotFoundError` if the file does not exist, or
    :class:`RuntimeError` if loading fails (e.g. due to a library version
    mismatch after upgrading econml or scikit-learn).

    Parameters
    ----------
    path:
        Absolute or relative path to the ``.pkl`` file.

    Returns
    -------
    Any
        The deserialised model object.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    RuntimeError
        If loading fails for any reason other than the file being missing.
    """
    import joblib  # imported here so the module is importable without joblib

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            f"Regenerate it by running the appropriate CLI command:\n"
            f"  baseline_model.pkl    -> python -m src.cli train-baseline\n"
            f"  causal_forest_model.pkl -> python -m src.cli train-uplift"
        )
    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load model from {path}.\n"
            f"This often happens after upgrading econml, scikit-learn, or lightgbm,\n"
            f"because serialised models are version-sensitive.\n"
            f"Regenerate the model by running the appropriate CLI command again.\n"
            f"Original error: {exc.__class__.__name__}: {exc}"
        ) from exc
