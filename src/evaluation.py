"""
evaluation.py
--------------
Reusable functions behind notebooks/05_evaluation_qini.ipynb: Qini
curves/coefficients, bootstrapped confidence intervals, pairwise
statistical-significance checks, and the final-verdict summary.

Uses scikit-uplift's qini_curve / qini_auc_score / uplift_at_k
throughout -- the same standard, peer-reviewed-consistent metric
implementations the original notebook used, not a hand-rolled
computation.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# sklift's qini_curve internally calls a deprecated sklearn utility and
# emits a FutureWarning on every call -- an upstream library detail, not
# an issue with this code. Suppressed once here rather than per-notebook.
warnings.filterwarnings("ignore", category=FutureWarning)


def compute_qini_metrics(
    y_true: np.ndarray,
    treatment: np.ndarray,
    rankings: Dict[str, np.ndarray],
    k_values: Tuple[float, ...] = (0.1, 0.2, 0.3),
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, float], Dict[str, Dict[float, float]]]:
    """Compute the Qini curve, Qini AUC (coefficient), and uplift_at_k
    for each ranking in `rankings`.

    Parameters
    ----------
    y_true : np.ndarray
        Actual outcome (e.g. visit) for each test row.
    treatment : np.ndarray
        Actual treatment indicator for each test row.
    rankings : Dict[str, np.ndarray]
        Model name -> uplift/predicted-probability scores, same order
        and length as y_true/treatment.
    k_values : Tuple[float, ...]
        Fractions to compute uplift_at_k for.

    Returns
    -------
    (qini_curves, qini_scores, uplift_at_k_scores)
        qini_curves: name -> (x, y) points for plotting.
        qini_scores: name -> Qini AUC.
        uplift_at_k_scores: name -> {k: uplift_at_k value}.
    """
    from sklift.metrics import qini_auc_score, qini_curve, uplift_at_k

    qini_curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    qini_scores: Dict[str, float] = {}
    uplift_at_k_scores: Dict[str, Dict[float, float]] = {}

    for name, uplift in rankings.items():
        x, y = qini_curve(y_true, uplift, treatment)
        qini_curves[name] = (x, y)
        qini_scores[name] = qini_auc_score(y_true, uplift, treatment)
        uplift_at_k_scores[name] = {
            k: uplift_at_k(y_true, uplift, treatment, strategy="overall", k=k)
            for k in k_values
        }

    return qini_curves, qini_scores, uplift_at_k_scores


def plot_qini_curves(
    qini_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    qini_scores: Dict[str, float],
    n_total: int,
    colors: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
):
    """Plot all Qini curves on one chart with the random-targeting
    diagonal reference line. Optionally saves to `save_path`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    default_colors = {
        "Baseline (naive, non-causal)": "#8C8C8C",
        "Two-Model Approach": "#4C72B0",
        "Class Transformation": "#55A868",
        "Causal Forest": "#C44E52",
    }
    colors = colors or default_colors

    fig, ax = plt.subplots(figsize=(10, 7))

    for name, (x, y) in qini_curves.items():
        x_frac = x / n_total
        ax.plot(x_frac, y, label=f"{name} (Qini AUC={qini_scores[name]:+.4f})",
                color=colors.get(name), linewidth=2.2)

    any_curve_endpoint = next(iter(qini_curves.values()))[1][-1]
    ax.plot([0, 1], [0, any_curve_endpoint], color="black", linestyle="--", linewidth=1.5,
            label="Random targeting (reference)")

    ax.set_xlabel("Share of customers targeted", fontsize=12)
    ax.set_ylabel("Cumulative incremental visits", fontsize=12)
    ax.set_title("Qini Curve Comparison — Baseline vs. Uplift Models (Test Set)",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def bootstrap_qini_ci(
    y_true: np.ndarray,
    treatment: np.ndarray,
    rankings: Dict[str, np.ndarray],
    qini_scores: Dict[str, float],
    n_bootstrap: int = 500,
    random_state: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap (resample with replacement) the test set `n_bootstrap`
    times, recompute each ranking's Qini AUC on each resample, and
    report the 2.5th/97.5th percentile as a 95% confidence interval.

    Parameters
    ----------
    qini_scores : Dict[str, float]
        Point-estimate Qini AUC per model (from compute_qini_metrics),
        included in the output for convenience.
    n_bootstrap : int
        Number of resamples (500 in the original notebook).
    random_state : int
        Seed for the resampling RNG, configurable rather than hardcoded.

    Returns
    -------
    Dict[str, Dict[str, float]]
        name -> {"qini_coefficient", "ci_lower_95", "ci_upper_95",
        "n_successful_resamples"}.
    """
    from sklift.metrics import qini_auc_score

    rng = np.random.default_rng(random_state)
    n = len(y_true)

    bootstrap_scores = {name: [] for name in rankings}

    for _ in range(n_bootstrap):
        resample_idx = rng.integers(0, n, size=n)
        y_resampled = y_true[resample_idx]
        treatment_resampled = treatment[resample_idx]

        for name, uplift in rankings.items():
            uplift_resampled = uplift[resample_idx]
            try:
                score = qini_auc_score(y_resampled, uplift_resampled, treatment_resampled)
                bootstrap_scores[name].append(score)
            except Exception:
                # Extremely rare: a resample with (near-)zero customers in
                # one arm. Skipped rather than crashing the whole bootstrap.
                continue

    ci_results = {}
    for name, scores in bootstrap_scores.items():
        scores_arr = np.array(scores)
        ci_lower, ci_upper = np.percentile(scores_arr, [2.5, 97.5])
        ci_results[name] = {
            "qini_coefficient": qini_scores[name],
            "ci_lower_95": ci_lower,
            "ci_upper_95": ci_upper,
            "n_successful_resamples": len(scores_arr),
        }

    return ci_results


def pairwise_significance(ci_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """For every pair of models, check whether their 95% confidence
    intervals overlap. Overlapping intervals mean the apparent
    difference between them could plausibly be sampling noise.

    Returns
    -------
    pd.DataFrame
        Columns: model_a, model_b, qini_a, qini_b, cis_overlap, verdict.
    """
    def intervals_overlap(lower1, upper1, lower2, upper2):
        return lower1 <= upper2 and lower2 <= upper1

    model_names = list(ci_results.keys())
    rows = []
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a, name_b = model_names[i], model_names[j]
            a, b = ci_results[name_a], ci_results[name_b]
            overlap = intervals_overlap(a["ci_lower_95"], a["ci_upper_95"], b["ci_lower_95"], b["ci_upper_95"])
            rows.append({
                "model_a": name_a,
                "model_b": name_b,
                "qini_a": round(a["qini_coefficient"], 4),
                "qini_b": round(b["qini_coefficient"], 4),
                "cis_overlap": overlap,
                "verdict": "NOT distinguishable (CIs overlap)" if overlap else "Distinguishable (CIs do not overlap)",
            })

    return pd.DataFrame(rows)


def build_final_verdict(
    qini_scores: Dict[str, float],
    ci_results: Dict[str, Dict[str, float]],
    significance_table: pd.DataFrame,
    reference_model: str = "Causal Forest",
) -> str:
    """Assemble the same "Final Verdict" markdown the original notebook
    produced, computed dynamically from the actual results so it always
    reflects the real run's numbers rather than a hardcoded narrative.

    Parameters
    ----------
    reference_model : str
        A specific model to call out separately in the verdict (the
        original notebook singled out "Causal Forest" since it was the
        most methodologically sophisticated model and the one whose
        Phase 4 heuristic result needed confirming or overturning).

    Returns
    -------
    str
        Markdown-formatted verdict text.
    """
    best_model = max(qini_scores, key=qini_scores.get)
    best_score = qini_scores[best_model]
    best_ci = ci_results[best_model]

    indistinguishable_from_best = significance_table[
        ((significance_table["model_a"] == best_model) | (significance_table["model_b"] == best_model))
        & (significance_table["cis_overlap"])
    ]
    competitor_names = set(indistinguishable_from_best["model_a"]).union(
        indistinguishable_from_best["model_b"]
    ) - {best_model}

    n_distinguishable = (~significance_table["cis_overlap"]).sum()
    n_total_pairs = len(significance_table)

    if competitor_names:
        significance_note = (
            f"However, its 95% confidence interval overlaps with: {', '.join(sorted(competitor_names))} -- "
            "meaning we **cannot** confidently declare a single statistical winner among these; the "
            "apparent ranking could partly reflect sampling noise rather than a true difference in "
            "targeting quality."
        )
    else:
        significance_note = (
            f"Its 95% confidence interval does **not** overlap with any other model's, so "
            f"`{best_model}` can be considered a statistically distinguishable, genuine winner on this test set."
        )

    reference_note = ""
    if reference_model in qini_scores:
        ref_qini = qini_scores[reference_model]
        ref_rank = sorted(qini_scores.values(), reverse=True).index(ref_qini) + 1
        ref_worst = ref_rank == len(qini_scores)
        ref_ci = ci_results[reference_model]
        reference_note = (
            f"- **On {reference_model} specifically:** ranked **{ref_rank} of {len(qini_scores)}** models "
            f"by Qini coefficient ({ref_qini:+.4f}, 95% CI [{ref_ci['ci_lower_95']:+.4f}, "
            f"{ref_ci['ci_upper_95']:+.4f}]).\n"
        )

    return f"""### Final Verdict

- **Highest Qini coefficient: `{best_model}`** ({best_score:+.4f}, 95% CI
  [{best_ci['ci_lower_95']:+.4f}, {best_ci['ci_upper_95']:+.4f}]).
- **Statistical significance:** {significance_note}
{reference_note}- **Overall:** {n_distinguishable} of {n_total_pairs} model-pair comparisons were
  statistically distinguishable at the 95% confidence level. Where confidence intervals
  overlap, this is itself a valid, citable finding -- Diemert et al. (2018), in their
  large-scale Criteo uplift benchmark, similarly found several uplift methods statistically
  indistinguishable from each other on certain datasets.

**Bottom line:** {best_model} produced the highest point-estimate Qini coefficient on this
test set{" and is a statistically confirmed winner" if not competitor_names else ", but the difference from " + ", ".join(sorted(competitor_names)) + " is not statistically significant given the observed sampling variability"}.
"""