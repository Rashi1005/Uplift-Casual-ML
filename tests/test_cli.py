"""
test_cli.py
-----------
Smoke tests for src/cli.py -- confirms the CLI's argument parsing,
progress messages, exit codes, and configurable directories behave as
documented, using small synthetic fixtures (not the real dataset, so
these run without network access, consistent with the other test files
in this project).

These are smoke tests, not full pipeline re-verification: they check
that each command runs, writes to the directory it was told to, and
fails loudly (nonzero exit code) when its inputs are missing -- they do
not re-check the modeling numbers themselves (that's covered by
tests/test_modules.py and by actually running the notebooks/CLI against
the real data/processed/ files, as documented in the README).

Run with:
    pytest tests/test_cli.py -v
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run_cli(args, cwd=REPO_ROOT):
    """Run `python -m src.cli <args>` as a real subprocess (not an
    in-process function call) so this test also exercises argument
    parsing and the process-level exit code exactly as a user would
    experience them."""
    return subprocess.run(
        [sys.executable, "-m", "src.cli"] + args,
        cwd=cwd, capture_output=True, text=True, timeout=120,
    )


@pytest.fixture
def small_processed_dir(tmp_path):
    """A tiny, synthetic already-preprocessed split -- enough to run
    train-baseline/train-uplift/evaluate/simulate against, without
    needing the real 64,000-row dataset or a network connection."""
    rng = np.random.default_rng(42)
    n_train, n_test = 200, 60
    n = n_train + n_test

    treatment = rng.integers(0, 2, n)
    x1 = rng.normal(0, 1, n)
    logit = -1.0 + 0.5 * x1 + 0.8 * treatment
    prob = 1 / (1 + np.exp(-logit))
    visit = (rng.random(n) < prob).astype(int)
    conversion = rng.integers(0, 2, n)

    X = pd.DataFrame({"x1": x1, "x2": rng.normal(0, 1, n), "cat_A": rng.integers(0, 2, n)})
    y = pd.DataFrame({"visit": visit, "conversion": conversion, "spend": rng.uniform(0, 50, n)})
    treatment_s = pd.Series(treatment)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir(parents=True)

    X.iloc[:n_train].to_csv(processed_dir / "X_train.csv", index=False)
    X.iloc[n_train:].to_csv(processed_dir / "X_test.csv", index=False)
    treatment_s.iloc[:n_train].to_frame("treatment").to_csv(processed_dir / "treatment_train.csv", index=False)
    treatment_s.iloc[n_train:].to_frame("treatment").to_csv(processed_dir / "treatment_test.csv", index=False)
    y.iloc[:n_train].to_csv(processed_dir / "y_train.csv", index=False)
    y.iloc[n_train:].to_csv(processed_dir / "y_test.csv", index=False)

    return str(processed_dir)


# ---------------------------------------------------------------------
# Argument parsing / help text
# ---------------------------------------------------------------------
class TestCliHelp:
    def test_top_level_help_lists_all_seven_commands(self):
        result = run_cli(["--help"])
        assert result.returncode == 0
        for command in ["prepare-data", "preprocess", "train-baseline", "train-uplift", "evaluate", "simulate", "run-all"]:
            assert command in result.stdout

    def test_no_command_exits_nonzero(self):
        result = run_cli([])
        assert result.returncode != 0

    def test_unknown_command_exits_nonzero(self):
        result = run_cli(["not-a-real-command"])
        assert result.returncode != 0


# ---------------------------------------------------------------------
# Progress messages and configurable directories
# ---------------------------------------------------------------------
class TestCliBasicRun:
    def test_train_baseline_writes_to_configured_processed_dir(self, small_processed_dir):
        result = run_cli(["train-baseline", "--processed-dir", small_processed_dir, "--seed", "1"])
        assert result.returncode == 0
        assert "[uplift-cli]" in result.stdout  # progress messages present
        assert os.path.exists(os.path.join(small_processed_dir, "baseline_model.pkl"))
        assert os.path.exists(os.path.join(small_processed_dir, "baseline_ranking.csv"))

    def test_train_baseline_skip_if_exists(self, small_processed_dir):
        first = run_cli(["train-baseline", "--processed-dir", small_processed_dir, "--seed", "1"])
        assert first.returncode == 0

        second = run_cli(["train-baseline", "--processed-dir", small_processed_dir, "--seed", "1", "--skip-if-exists"])
        assert second.returncode == 0
        assert "SKIPPED" in second.stdout

    def test_seed_is_reported_in_output(self, small_processed_dir):
        result = run_cli(["train-baseline", "--processed-dir", small_processed_dir, "--seed", "99"])
        assert "seed=99" in result.stdout or "random_state=99" in result.stdout


# ---------------------------------------------------------------------
# Failure behavior (nonzero exit codes)
# ---------------------------------------------------------------------
class TestCliFailureExitCodes:
    def test_train_baseline_fails_loudly_on_missing_processed_dir(self, tmp_path):
        missing_dir = str(tmp_path / "does_not_exist")
        result = run_cli(["train-baseline", "--processed-dir", missing_dir])
        assert result.returncode != 0
        assert "ERROR" in result.stdout

    def test_train_uplift_fails_without_baseline_ranking(self, small_processed_dir):
        # baseline_ranking.csv doesn't exist yet in this fixture -- train-uplift
        # should fail with a clear message, not a raw traceback with no context.
        result = run_cli(["train-uplift", "--processed-dir", small_processed_dir])
        assert result.returncode != 0
        assert "train-baseline" in result.stdout  # tells the user what to run first

    def test_evaluate_fails_without_combined_rankings(self, small_processed_dir):
        result = run_cli(["evaluate", "--processed-dir", small_processed_dir, "--reports-dir", str(small_processed_dir)])
        assert result.returncode != 0
        assert "train-uplift" in result.stdout


# ---------------------------------------------------------------------
# Full small chain: train-baseline -> train-uplift -> evaluate -> simulate
# ---------------------------------------------------------------------
class TestCliSmallChain:
    def test_full_chain_on_tiny_synthetic_data(self, small_processed_dir, tmp_path):
        """Runs the 4 downstream commands back-to-back on the tiny
        synthetic fixture (skipping prepare-data/preprocess, which need
        either network access or a full raw dataset) -- confirms the
        commands compose correctly end-to-end, each reading what the
        previous one wrote."""
        reports_dir = str(tmp_path / "reports")

        r1 = run_cli(["train-baseline", "--processed-dir", small_processed_dir, "--seed", "7"])
        assert r1.returncode == 0

        r2 = run_cli(["train-uplift", "--processed-dir", small_processed_dir, "--seed", "7", "--no-auto-install"])
        # Allow either a clean pass, or a clear, non-crashing failure if
        # neither econml nor causalml is installed in this environment --
        # either way the exit code must be meaningful, not a silent hang.
        assert r2.returncode in (0, 1)

        if r2.returncode == 0:
            assert os.path.exists(os.path.join(small_processed_dir, "uplift_scores_combined.csv"))

            r3 = run_cli(["evaluate", "--processed-dir", small_processed_dir, "--reports-dir", reports_dir, "--seed", "7"])
            assert r3.returncode == 0
            assert os.path.exists(os.path.join(small_processed_dir, "phase5_results.csv"))

            r4 = run_cli(["simulate", "--processed-dir", small_processed_dir, "--reports-dir", reports_dir, "--seed", "7"])
            assert r4.returncode == 0
            assert os.path.exists(os.path.join(small_processed_dir, "phase6_business_impact.csv"))