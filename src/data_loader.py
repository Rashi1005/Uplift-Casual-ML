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


def load_hillstrom(prefer_sklift: bool = True) -> pd.DataFrame:
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
            print("Falling back to local cached copy at data/hillstrom.csv ...")

    df = pd.read_csv(LOCAL_CSV_PATH)
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Local CSV is missing expected columns: {missing_cols}")

    print(f"Loaded dataset from local cache: {LOCAL_CSV_PATH}")
    return df[EXPECTED_COLUMNS]


if __name__ == "__main__":
    data = load_hillstrom()
    print(data.shape)
    print(data.head())
    print("\nMissing values per column:")
    print(data.isna().sum())