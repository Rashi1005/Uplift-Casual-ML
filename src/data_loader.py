"""
data_loader.py
--------------
Loads the Hillstrom Email Marketing dataset for the uplift modeling project.
"""

import os
import pandas as pd

LOCAL_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "hillstrom.csv")

EXPECTED_COLUMNS = [
    "recency", "history_segment", "history", "mens", "womens", "zip_code",
    "newbie", "channel", "segment", "visit", "conversion", "spend",
]


def load_hillstrom(prefer_sklift: bool = True, local_path: str = None) -> pd.DataFrame:
    """Load the Hillstrom dataset.

    Parameters
    ----------
    prefer_sklift : bool
        If True (default), try the live scikit-uplift download first.
    local_path : str, optional
        Path to read the local cached CSV from, used either as the
        fallback when `prefer_sklift` download fails, or directly when
        `prefer_sklift=False`. Defaults to `LOCAL_CSV_PATH` (this
        module's original, unchanged behavior) when not given -- this
        parameter is purely additive, added so callers such as the CLI
        can point at a configurable data directory without changing
        behavior for any existing caller that doesn't pass it.
    """
    if local_path is None:
        local_path = LOCAL_CSV_PATH

    if prefer_sklift:
        try:
            from sklift.datasets import fetch_hillstrom

            # target_col="all" is the key fix: returns ALL THREE outcome
            # columns (visit, conversion, spend), plus the raw 3-arm
            # "segment" treatment column, instead of dropping two of them.
            bunch = fetch_hillstrom(target_col="all")

            df = bunch.data.copy()
            df = pd.concat([df, bunch.target], axis=1)   # + visit, conversion, spend
            df["segment"] = bunch.treatment                # + raw 3-arm segment

            print("Loaded dataset via sklift.datasets.fetch_hillstrom().")
            return df[EXPECTED_COLUMNS]
        except Exception as exc:
            print(f"sklift download unavailable ({exc.__class__.__name__}: {exc}).")
            print(f"Falling back to local cached copy at {local_path} ...")

    df = pd.read_csv(local_path)
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Local CSV is missing expected columns: {missing_cols}")

    print(f"Loaded dataset from local cache: {local_path}")
    return df[EXPECTED_COLUMNS]


if __name__ == "__main__":
    data = load_hillstrom()
    print(data.shape)
    print(data.head())
    print("\nMissing values per column:")
    print(data.isna().sum())