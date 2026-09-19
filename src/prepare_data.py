"""
prepare_data.py
----------------
Reproducible raw-data workflow for the Hillstrom Email Marketing dataset.

Run this once after a clean clone, before opening any notebook:

    python src/prepare_data.py

What it does:
  1. Downloads the Hillstrom dataset via scikit-uplift's fetch_hillstrom()
     (the same loader src/data_loader.py already uses).
  2. Saves the raw dataset to data/hillstrom.csv -- the exact path
     src/data_loader.py already falls back to when the live download is
     unavailable, so the two scripts stay consistent with each other.
  3. Validates that every expected column is present.
  4. Reports dataset shape, missing values per column, and treatment-group
     counts (binarized the same way notebooks/01_eda.ipynb does).

This script does not train or evaluate any model -- it only prepares the
raw input the pipeline's notebooks read from.
"""

import os
import sys

import pandas as pd

# Import the single source of truth for the expected schema, so this script
# and data_loader.py can never silently drift out of sync with each other.
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import EXPECTED_COLUMNS, LOCAL_CSV_PATH  # noqa: E402

RAW_DATA_DIR = os.path.dirname(LOCAL_CSV_PATH)


def download_raw_dataset() -> pd.DataFrame:
    """Fetch the raw Hillstrom dataset via scikit-uplift.

    Raises the underlying exception if the download fails (e.g. no network,
    or the upstream host is unreachable/blocked on a given network) -- the
    caller decides how to report that.
    """
    from sklift.datasets import fetch_hillstrom

    bunch = fetch_hillstrom(target_col="all")
    df = bunch.data.copy()
    df = pd.concat([df, bunch.target], axis=1)
    df["segment"] = bunch.treatment
    return df[EXPECTED_COLUMNS]


def validate_columns(df: pd.DataFrame) -> None:
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Downloaded dataset is missing expected columns: {missing}")


def report(df: pd.DataFrame) -> None:
    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\nMissing values per column:")
    missing = df.isna().sum()
    print(missing.to_string())
    if missing.sum() == 0:
        print("(none)")

    treatment = (df["segment"] != "No E-Mail").astype(int)
    print("\nTreatment-group counts (binarized: any email = 1, no email = 0):")
    counts = treatment.value_counts().sort_index()
    total = len(treatment)
    for value, count in counts.items():
        label = "Treatment (received an email)" if value == 1 else "Control (no email)"
        print(f"  {label}: {count:,} ({count / total * 100:.2f}%)")


def main() -> None:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    print("Downloading Hillstrom Email Marketing dataset via scikit-uplift ...")
    try:
        df = download_raw_dataset()
    except Exception as exc:
        print(f"\nDownload failed: {exc.__class__.__name__}: {exc}")
        print(
            "\nThis is a known limitation (see data/MANIFEST.md) -- the upstream "
            "host is occasionally unreachable from restricted networks (e.g. "
            "campus/corporate networks that block direct S3 access). If this "
            "keeps failing:\n"
            "  1. Try again from a different network, or\n"
            "  2. Obtain hillstrom.csv from a teammate who has already run this "
            "script successfully, and place it directly at:\n"
            f"     {LOCAL_CSV_PATH}\n"
            "  3. Then re-run this script -- it will validate the file you "
            "placed there instead of re-downloading."
        )
        if os.path.exists(LOCAL_CSV_PATH):
            print(f"\nFound an existing file at {LOCAL_CSV_PATH} -- validating it instead.")
            df = pd.read_csv(LOCAL_CSV_PATH)
        else:
            sys.exit(1)

    validate_columns(df)

    df.to_csv(LOCAL_CSV_PATH, index=False)
    print(f"\nSaved raw dataset to: {LOCAL_CSV_PATH}")

    report(df)

    print(
        "\nDone. You can now run notebooks/01_eda.ipynb, or import "
        "src.data_loader.load_hillstrom() directly."
    )


if __name__ == "__main__":
    main()