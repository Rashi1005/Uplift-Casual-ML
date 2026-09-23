"""
cli.py
------
Command-line interface for running the uplift modeling pipeline without
opening notebooks. Each notebook (01 through 06) has a matching
subcommand that calls the same src/ modules the notebook itself calls --
this CLI does not reimplement any modeling logic, it only orchestrates
the existing functions in src/preprocessing.py, src/baseline_model.py,
src/uplift_models.py, src/evaluation.py, and src/business_simulation.py.

Usage
-----
    python -m src.cli prepare-data
    python -m src.cli preprocess
    python -m src.cli train-baseline
    python -m src.cli train-uplift
    python -m src.cli evaluate
    python -m src.cli simulate
    python -m src.cli run-all

Run `python -m src.cli <command> --help` for each command's options.
See README.md's "Running the pipeline from the command line" section
for full usage examples and what each command reads/writes.
"""

import argparse
import os
import sys
import traceback
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# Support running both as `python -m src.cli` (repo root on sys.path,
# package-style imports) and `python src/cli.py` directly (this file's
# own directory on sys.path, flat imports) -- same dual pattern already
# used inside the src/ modules themselves.
try:
    from src import baseline_model, business_simulation, evaluation, preprocessing, uplift_models, utils
    from src.data_loader import EXPECTED_COLUMNS, load_hillstrom
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    import baseline_model, business_simulation, evaluation, preprocessing, uplift_models, utils  # noqa: E401
    from data_loader import EXPECTED_COLUMNS, load_hillstrom


# ---------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------
def log(message: str) -> None:
    """Print a clear, consistently-formatted progress message."""
    print(f"[uplift-cli] {message}")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def artifacts_exist(paths) -> bool:
    return all(os.path.exists(p) for p in paths)


def load_processed_split(processed_dir: str):
    """Load X_train/X_test/treatment_train/treatment_test/y_train/y_test
    from `processed_dir`, sanitizing column names for LightGBM the same
    way notebooks 03 and 04 each do."""
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    treatment_train = pd.read_csv(os.path.join(processed_dir, "treatment_train.csv"))["treatment"]
    treatment_test = pd.read_csv(os.path.join(processed_dir, "treatment_test.csv"))["treatment"]
    y_train = pd.read_csv(os.path.join(processed_dir, "y_train.csv"))
    y_test = pd.read_csv(os.path.join(processed_dir, "y_test.csv"))

    X_train.columns = utils.sanitize_columns(X_train.columns)
    X_test.columns = utils.sanitize_columns(X_test.columns)

    return X_train, X_test, treatment_train, treatment_test, y_train, y_test


# ---------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------
def cmd_prepare_data(args: argparse.Namespace) -> None:
    """Data preparation: download/validate the raw Hillstrom dataset.

    Reads : (network) the Hillstrom dataset via scikit-uplift
    Writes: {data_dir}/hillstrom.csv
    """
    ensure_dir(args.data_dir)
    raw_path = os.path.join(args.data_dir, "hillstrom.csv")

    if args.skip_if_exists and os.path.exists(raw_path):
        log(f"SKIPPED (--skip-if-exists): {raw_path} already exists.")
        return

    log("Downloading Hillstrom Email Marketing dataset via scikit-uplift ...")
    from sklift.datasets import fetch_hillstrom

    bunch = fetch_hillstrom(target_col="all")
    df = bunch.data.copy()
    df = pd.concat([df, bunch.target], axis=1)
    df["segment"] = bunch.treatment
    df = df[EXPECTED_COLUMNS]

    df.to_csv(raw_path, index=False)
    log(f"Saved raw dataset -> {raw_path}  ({df.shape[0]:,} rows x {df.shape[1]} columns)")


def cmd_preprocess(args: argparse.Namespace) -> None:
    """Preprocessing: binarize treatment, encode features, stratified
    train/test split, and the post-split balance verification.

    Reads : {data_dir}/hillstrom.csv
    Writes: {processed_dir}/{X_train,X_test,treatment_train,
            treatment_test,y_train,y_test}.csv
    """
    ensure_dir(args.processed_dir)
    output_paths = [
        os.path.join(args.processed_dir, f) for f in
        ["X_train.csv", "X_test.csv", "treatment_train.csv", "treatment_test.csv", "y_train.csv", "y_test.csv"]
    ]
    if args.skip_if_exists and artifacts_exist(output_paths):
        log("SKIPPED (--skip-if-exists): processed train/test split already exists.")
        return

    log(f"Loading raw dataset (seed={args.seed} used for the split below) ...")
    raw_path = os.path.join(args.data_dir, "hillstrom.csv")
    df = load_hillstrom(local_path=raw_path)
    df = preprocessing.binarize_treatment(df)

    numeric_features = ["recency", "history", "mens", "womens", "newbie"]
    categorical_features = ["history_segment", "zip_code", "channel"]
    outcome_cols = ["visit", "conversion", "spend"]

    log("One-hot encoding categorical features ...")
    X = preprocessing.encode_features(df, numeric_features, categorical_features)
    treatment = df["treatment"].copy()
    y = df[outcome_cols].copy()

    log(f"Splitting 80/20, stratified by treatment + conversion, random_state={args.seed} ...")
    X_train, X_test, treatment_train, treatment_test, y_train, y_test = preprocessing.stratified_split(
        X, treatment, y, test_size=0.20, random_state=args.seed, strat_outcome_col="conversion",
    )

    verification_df, any_deviation = preprocessing.verify_split_balance(
        treatment_train, y_train, treatment_test, y_test,
    )
    if any_deviation:
        log("WARNING: the split deviates from the Phase 1 reference rates beyond tolerance:")
        print(verification_df.to_string(index=False))
    else:
        log("Split balance verified: all rates within tolerance of the Phase 1 reference.")

    X_train.to_csv(output_paths[0], index=False)
    X_test.to_csv(output_paths[1], index=False)
    treatment_train.to_frame("treatment").to_csv(output_paths[2], index=False)
    treatment_test.to_frame("treatment").to_csv(output_paths[3], index=False)
    y_train.to_csv(output_paths[4], index=False)
    y_test.to_csv(output_paths[5], index=False)

    log(f"Saved train ({len(X_train):,} rows) and test ({len(X_test):,} rows) splits -> {args.processed_dir}/")


def cmd_train_baseline(args: argparse.Namespace) -> None:
    """Baseline model training: naive LightGBM classifier, evaluation,
    ranking, and the top-decile signal check.

    Reads : {processed_dir}/{X_train,X_test,treatment_test,y_train,y_test}.csv
    Writes: {processed_dir}/baseline_model.pkl, baseline_ranking.csv
    """
    ensure_dir(args.processed_dir)
    model_path = os.path.join(args.processed_dir, "baseline_model.pkl")
    ranking_path = os.path.join(args.processed_dir, "baseline_ranking.csv")

    if args.skip_if_exists and artifacts_exist([model_path, ranking_path]):
        log(f"SKIPPED (--skip-if-exists): {model_path} and {ranking_path} already exist.")
        return

    X_train, X_test, treatment_train, treatment_test, y_train, y_test = load_processed_split(args.processed_dir)
    y_train_target, y_test_target = y_train["visit"], y_test["visit"]

    log(f"Training baseline LightGBM classifier on target 'visit', random_state={args.seed} ...")
    model = baseline_model.train_baseline_model(X_train, y_train_target, random_state=args.seed, n_estimators=200)

    results = baseline_model.evaluate_baseline_model(model, X_test, y_test_target)
    log(f"AUC-ROC={results['auc_roc']:.4f}  Precision={results['precision']:.4f}  "
        f"Recall={results['recall']:.4f}  AvgPrecision={results['average_precision']:.4f}")

    ranking = baseline_model.build_ranking(X_test, results["test_probs"], treatment_test, y_test)
    sc = baseline_model.signal_check(ranking, "visit")
    status = "PASSED" if sc["passed"] else "FAILED"
    log(f"Signal check {status}: top-decile rate {sc['top_rate']*100:.2f}% vs. "
        f"rest {sc['rest_rate']*100:.2f}% (gap {sc['gap']*100:+.2f} pp)")

    joblib.dump(model, model_path)
    ranking.to_csv(ranking_path, index=False)
    log(f"Saved model -> {model_path}")
    log(f"Saved ranking -> {ranking_path}")


def cmd_train_uplift(args: argparse.Namespace) -> None:
    """Uplift model training: Two-Model Approach, Class Transformation,
    and Causal Forest, plus their signal checks and combined rankings.

    This is the most computationally expensive command (the Causal
    Forest fit) -- use --skip-if-exists on repeated runs once it has
    completed successfully once.

    Reads : {processed_dir}/{X_train,X_test,treatment_train,
            treatment_test,y_train,y_test,baseline_ranking}.csv
    Writes: {processed_dir}/causal_forest_model.pkl,
            uplift_scores_combined.csv
    """
    ensure_dir(args.processed_dir)
    combined_path = os.path.join(args.processed_dir, "uplift_scores_combined.csv")
    causal_forest_path = os.path.join(args.processed_dir, "causal_forest_model.pkl")

    if args.skip_if_exists and artifacts_exist([combined_path]):
        log(f"SKIPPED (--skip-if-exists): {combined_path} already exists.")
        return

    baseline_ranking_path = os.path.join(args.processed_dir, "baseline_ranking.csv")
    if not os.path.exists(baseline_ranking_path):
        raise FileNotFoundError(
            f"{baseline_ranking_path} not found -- run 'train-baseline' before 'train-uplift'."
        )

    X_train, X_test, treatment_train, treatment_test, y_train, y_test = load_processed_split(args.processed_dir)
    y_train_target, y_test_target = y_train["visit"], y_test["visit"]
    baseline_ranking = pd.read_csv(baseline_ranking_path)

    log(f"Training Two-Model Approach, random_state={args.seed} ...")
    uplift_two_model, _, _ = uplift_models.two_model_approach(
        X_train, y_train_target, treatment_train, X_test, random_state=args.seed,
    )

    log(f"Training Class Transformation, random_state={args.seed} ...")
    uplift_class_transform, _ = uplift_models.class_transformation(
        X_train, y_train_target, treatment_train, X_test, random_state=args.seed,
    )

    log(f"Training Causal Forest, random_state={args.seed} (this is the slow step -- "
        "cross-fitted DML, can take several minutes) ...")
    uplift_causal_forest, causal_model, library_used = uplift_models.causal_forest_model(
        X_train, y_train_target, treatment_train, X_test, random_state=args.seed,
        auto_install=not args.no_auto_install,
    )
    log(f"Causal Forest fit complete (library: {library_used}).")

    for name, scores in [
        ("Two-Model Approach", uplift_two_model),
        ("Class Transformation", uplift_class_transform),
        ("Causal Forest", uplift_causal_forest),
    ]:
        sc = uplift_models.signal_check_uplift(scores, treatment_test, y_test_target)
        status = "PASSED" if sc["passed"] else "FAILED"
        log(f"  {name}: signal check {status} (gap {sc['gap']*100:+.2f} pp)")

    combined = uplift_models.combine_rankings(
        X_test, treatment_test, y_test,
        uplift_two_model, uplift_class_transform, uplift_causal_forest,
        baseline_ranking,
    )

    joblib.dump(causal_model, causal_forest_path)
    combined.to_csv(combined_path, index=False)
    log(f"Saved causal forest model -> {causal_forest_path}")
    log(f"Saved combined rankings -> {combined_path}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluation: Qini curves/coefficients, bootstrapped 95% confidence
    intervals, pairwise significance, and the final verdict.

    Reads : {processed_dir}/uplift_scores_combined.csv
    Writes: {processed_dir}/phase5_results.csv, {reports_dir}/qini_comparison.png
    """
    ensure_dir(args.reports_dir)
    results_path = os.path.join(args.processed_dir, "phase5_results.csv")
    plot_path = os.path.join(args.reports_dir, "qini_comparison.png")

    if args.skip_if_exists and artifacts_exist([results_path]):
        log(f"SKIPPED (--skip-if-exists): {results_path} already exists.")
        return

    combined_path = os.path.join(args.processed_dir, "uplift_scores_combined.csv")
    if not os.path.exists(combined_path):
        raise FileNotFoundError(f"{combined_path} not found -- run 'train-uplift' before 'evaluate'.")

    combined = pd.read_csv(combined_path)
    y_true = combined["actual_visit"].values
    treatment = combined["actual_treatment"].values
    rankings = {
        "Baseline (naive, non-causal)": combined["baseline_predicted_prob"].values,
        "Two-Model Approach": combined["uplift_two_model"].values,
        "Class Transformation": combined["uplift_class_transform"].values,
        "Causal Forest": combined["uplift_causal_forest"].values,
    }

    log("Computing Qini curves and coefficients ...")
    qini_curves, qini_scores, _ = evaluation.compute_qini_metrics(y_true, treatment, rankings)
    for name, score in sorted(qini_scores.items(), key=lambda kv: kv[1], reverse=True):
        log(f"  {name}: Qini AUC {score:+.4f}")

    evaluation.plot_qini_curves(qini_curves, qini_scores, n_total=len(y_true), save_path=plot_path)
    log(f"Saved plot -> {plot_path}")

    log(f"Bootstrapping 95% confidence intervals (500 resamples, random_state={args.seed}) ...")
    ci_results = evaluation.bootstrap_qini_ci(y_true, treatment, rankings, qini_scores, n_bootstrap=500, random_state=args.seed)

    significance_table = evaluation.pairwise_significance(ci_results)
    n_distinguishable = (~significance_table["cis_overlap"]).sum()
    log(f"{n_distinguishable} of {len(significance_table)} model pairs are statistically distinguishable (95% CI).")

    results_df = pd.DataFrame(ci_results).T.reset_index().rename(columns={"index": "model"})
    results_df = results_df[["model", "qini_coefficient", "ci_lower_95", "ci_upper_95"]]
    results_df.to_csv(results_path, index=False)
    log(f"Saved results -> {results_path}")


def cmd_simulate(args: argparse.Namespace) -> None:
    """Business-impact simulation: budget-constrained targeting
    (10%/20%) vs. random selection.

    Reads : {processed_dir}/uplift_scores_combined.csv
    Writes: {processed_dir}/phase6_business_impact.csv,
            {reports_dir}/business_impact_comparison.png
    """
    ensure_dir(args.reports_dir)
    output_path = os.path.join(args.processed_dir, "phase6_business_impact.csv")
    plot_path = os.path.join(args.reports_dir, "business_impact_comparison.png")

    if args.skip_if_exists and artifacts_exist([output_path]):
        log(f"SKIPPED (--skip-if-exists): {output_path} already exists.")
        return

    combined_path = os.path.join(args.processed_dir, "uplift_scores_combined.csv")
    if not os.path.exists(combined_path):
        raise FileNotFoundError(f"{combined_path} not found -- run 'train-uplift' before 'simulate'.")

    combined = pd.read_csv(combined_path)
    strategy_columns = {
        "Baseline (naive)": "baseline_predicted_prob",
        "Two-Model": "uplift_two_model",
        "Class Transformation": "uplift_class_transform",
        "Causal Forest": "uplift_causal_forest",
    }

    log(f"Simulating budget-constrained targeting (10%/20%), random_state={args.seed} ...")
    summary_table = business_simulation.simulate_budget_strategies(
        combined, strategy_columns, budget_levels=(0.10, 0.20), random_state=args.seed,
    )

    strategy_order = list(strategy_columns.keys()) + ["Random Selection"]
    business_simulation.plot_business_impact(summary_table, strategy_order, save_path=plot_path)
    log(f"Saved plot -> {plot_path}")

    beat_random_df, all_beat_random = business_simulation.check_beat_random(summary_table, strategy_order)
    log(f"All strategies beat Random Selection (point estimates): {all_beat_random}")

    summary_table.to_csv(output_path, index=False)
    log(f"Saved results -> {output_path}")


def cmd_run_all(args: argparse.Namespace) -> None:
    """Run the complete pipeline: prepare-data -> preprocess ->
    train-baseline -> train-uplift -> evaluate -> simulate, in order.

    Stops immediately (nonzero exit) if any step fails.
    """
    steps = [
        ("prepare-data", cmd_prepare_data),
        ("preprocess", cmd_preprocess),
        ("train-baseline", cmd_train_baseline),
        ("train-uplift", cmd_train_uplift),
        ("evaluate", cmd_evaluate),
        ("simulate", cmd_simulate),
    ]
    for i, (name, func) in enumerate(steps, start=1):
        log(f"===== Step {i}/{len(steps)}: {name} =====")
        func(args)
    log("===== Pipeline complete =====")


# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="uplift-cli",
        description="Run the uplift modeling pipeline (data prep through business "
                     "simulation) from the command line, without opening notebooks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--data-dir", default="data",
                         help="Directory for raw data (default: data)")
        sp.add_argument("--processed-dir", default=os.path.join("data", "processed"),
                         help="Directory for processed splits and model artifacts (default: data/processed)")
        sp.add_argument("--reports-dir", default="reports",
                         help="Directory for saved plots (default: reports)")
        sp.add_argument("--seed", type=int, default=42,
                         help="Random seed used for splitting, model training, and bootstrapping (default: 42)")
        sp.add_argument("--skip-if-exists", action="store_true",
                         help="Skip this step if its output artifacts already exist "
                              "(most useful for train-uplift, the expensive Causal Forest step)")

    p_prepare = subparsers.add_parser("prepare-data", help="Download/validate the raw Hillstrom dataset")
    add_common_args(p_prepare)
    p_prepare.set_defaults(func=cmd_prepare_data)

    p_preprocess = subparsers.add_parser("preprocess", help="Encode features and create the stratified train/test split")
    add_common_args(p_preprocess)
    p_preprocess.set_defaults(func=cmd_preprocess)

    p_baseline = subparsers.add_parser("train-baseline", help="Train and evaluate the naive baseline model")
    add_common_args(p_baseline)
    p_baseline.set_defaults(func=cmd_train_baseline)

    p_uplift = subparsers.add_parser("train-uplift", help="Train the three uplift models (Two-Model, Class Transform, Causal Forest)")
    add_common_args(p_uplift)
    p_uplift.add_argument("--no-auto-install", action="store_true",
                           help="Do not attempt a runtime `pip install econml` if econml is missing "
                                "-- fall back to causalml immediately instead")
    p_uplift.set_defaults(func=cmd_train_uplift)

    p_eval = subparsers.add_parser("evaluate", help="Compute Qini coefficients and bootstrapped confidence intervals")
    add_common_args(p_eval)
    p_eval.set_defaults(func=cmd_evaluate)

    p_sim = subparsers.add_parser("simulate", help="Run the budget-constrained business impact simulation")
    add_common_args(p_sim)
    p_sim.set_defaults(func=cmd_simulate)

    p_all = subparsers.add_parser("run-all", help="Run the complete pipeline, in order")
    add_common_args(p_all)
    p_all.add_argument("--no-auto-install", action="store_true",
                        help="Do not attempt a runtime `pip install econml` if econml is missing")
    p_all.set_defaults(func=cmd_run_all)

    return parser


def main(argv: Optional[list] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "no_auto_install"):
        args.no_auto_install = False

    try:
        args.func(args)
        return 0
    except Exception as exc:
        log(f"ERROR: {exc.__class__.__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())