"""
business_simulation.py
-----------------------
Reusable functions behind notebooks/06_business_simulation.ipynb: the
budget-constrained targeting simulation, its comparison plot, and the
"did every real strategy beat random selection" check.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from src.utils import actual_uplift
except ImportError:
    from utils import actual_uplift


def simulate_budget_strategies(
    combined: pd.DataFrame,
    strategy_columns: Dict[str, str],
    budget_levels: Tuple[float, ...] = (0.10, 0.20),
    random_state: int = 42,
) -> pd.DataFrame:
    """For each strategy and each budget level, select the top-K% of
    customers by that strategy's score column (or a random K% sample for
    "Random Selection"), compute the ACTUAL treated-vs-control outcome
    rate within that selected group, and estimate total incremental
    visits.

    Parameters
    ----------
    combined : pd.DataFrame
        Output of uplift_models.combine_rankings() -- must contain
        `actual_treatment`, `actual_visit`, and every column named in
        `strategy_columns`.
    strategy_columns : Dict[str, str]
        Strategy display name -> score column to rank by, descending
        (e.g. {"Two-Model": "uplift_two_model"}). "Random Selection" is
        always added automatically and does not need to be included here.
    budget_levels : Tuple[float, ...]
        Fractions of the customer base to simulate contacting.
    random_state : int
        Seed for the Random Selection sampling, configurable rather than
        hardcoded.

    Returns
    -------
    pd.DataFrame
        Columns: strategy, budget_level, customers_contacted,
        actual_incremental_visit_rate, estimated_total_incremental_visits.
    """
    n_total = len(combined)
    rng = np.random.default_rng(random_state)

    rows = []
    for budget in budget_levels:
        k = int(np.ceil(n_total * budget))

        for strategy_name, score_col in strategy_columns.items():
            selected = combined.sort_values(score_col, ascending=False).head(k)
            rate, _, _ = actual_uplift(selected, "actual_treatment", "actual_visit")
            estimated_total = rate * len(selected) if not np.isnan(rate) else np.nan
            rows.append({
                "strategy": strategy_name,
                "budget_level": f"{int(budget*100)}%",
                "customers_contacted": len(selected),
                "actual_incremental_visit_rate": rate,
                "estimated_total_incremental_visits": estimated_total,
            })

        random_idx = rng.choice(combined.index, size=k, replace=False)
        random_selected = combined.loc[random_idx]
        rate, _, _ = actual_uplift(random_selected, "actual_treatment", "actual_visit")
        estimated_total = rate * len(random_selected) if not np.isnan(rate) else np.nan
        rows.append({
            "strategy": "Random Selection",
            "budget_level": f"{int(budget*100)}%",
            "customers_contacted": len(random_selected),
            "actual_incremental_visit_rate": rate,
            "estimated_total_incremental_visits": estimated_total,
        })

    summary_table = pd.DataFrame(rows)
    return summary_table.sort_values(
        ["budget_level", "estimated_total_incremental_visits"], ascending=[True, False]
    ).reset_index(drop=True)


def plot_business_impact(
    summary_table: pd.DataFrame,
    strategy_order: List[str],
    budget_labels: Optional[List[str]] = None,
    colors: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
):
    """Grouped bar chart: x-axis = strategy, grouped bars per budget
    level, y-axis = estimated total incremental visits.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if budget_labels is None:
        budget_labels = sorted(summary_table["budget_level"].unique(), key=lambda s: int(s.rstrip("%")))
    colors = colors or {"10%": "#4C72B0", "20%": "#DD8452"}

    fig, ax = plt.subplots(figsize=(11, 7))
    bar_width = 0.35
    x_positions = np.arange(len(strategy_order))

    for i, budget_label in enumerate(budget_labels):
        values = [
            summary_table.loc[
                (summary_table["strategy"] == s) & (summary_table["budget_level"] == budget_label),
                "estimated_total_incremental_visits"
            ].values[0]
            for s in strategy_order
        ]
        offset = (i - (len(budget_labels) - 1) / 2) * bar_width
        ax.bar(x_positions + offset, values, width=bar_width, label=f"{budget_label} budget",
               color=colors.get(budget_label))

    ax.set_xticks(x_positions)
    ax.set_xticklabels(strategy_order, rotation=15, ha="right")
    ax.set_ylabel("Estimated total incremental visits", fontsize=12)
    ax.set_xlabel("Targeting strategy", fontsize=12)
    ax.set_title("Business Impact Comparison — Estimated Incremental Visits by Strategy and Budget\n"
                 "(Point estimates only -- see caveat below)", fontsize=13, fontweight="bold")
    ax.legend(title="Budget level")
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.8)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def check_beat_random(
    summary_table: pd.DataFrame,
    strategy_order: List[str],
    budget_labels: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, bool]:
    """For each non-random strategy at each budget level, check whether
    it beat "Random Selection" in point-estimate terms.

    Returns
    -------
    (beat_random_df, all_beat_random) : Tuple[pd.DataFrame, bool]
    """
    if budget_labels is None:
        budget_labels = sorted(summary_table["budget_level"].unique(), key=lambda s: int(s.rstrip("%")))

    non_random_strategies = [s for s in strategy_order if s != "Random Selection"]

    rows = []
    for budget_label in budget_labels:
        random_value = summary_table.loc[
            (summary_table["strategy"] == "Random Selection") & (summary_table["budget_level"] == budget_label),
            "estimated_total_incremental_visits"
        ].values[0]
        for s in non_random_strategies:
            s_value = summary_table.loc[
                (summary_table["strategy"] == s) & (summary_table["budget_level"] == budget_label),
                "estimated_total_incremental_visits"
            ].values[0]
            rows.append({
                "budget_level": budget_label, "strategy": s,
                "beat_random_in_point_estimate": s_value > random_value,
            })

    beat_random_df = pd.DataFrame(rows)
    all_beat_random = beat_random_df["beat_random_in_point_estimate"].all()
    return beat_random_df, all_beat_random