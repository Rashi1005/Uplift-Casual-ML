"""
data_loader.py
--------------
Loads the Hillstrom Email Marketing dataset for the uplift-modelling
pipeline.

The single source of truth for the expected column schema
(``EXPECTED_COLUMNS``) and the default local CSV path
(``LOCAL_CSV_PATH``) live here so that ``prepare_data.py``, ``cli.py``,
and all notebooks that import this module always use the same definition.

Public API
----------
load_hillstrom(prefer_sklift, local_path)
    Load the dataset from scikit-uplift's fetch_hillstrom() or from a
    local CSV fallback, returning a DataFrame with ``EXPECTED_COLUMNS``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Schema constants — imported by prepare_data.py and cli.py to avoid drift.
# ---------------------------------------------------------------------------
LOCAL_CSV_PATH = str(Path(__file__).parent.parent / "data" / "hillstrom.csv")

EXPECTED_COLUMNS: list[str] = [
    "recency",
    "history_segment",
    "history",
    "mens",
    "womens",
    "zip_code",
    "newbie",
    "channel",
    "segment",
    "visit",
    "conversion",
    "spend",
]


def load_hillstrom(
    prefer_sklift: bool = True,
    local_path: str | None = None,
) -> pd.DataFrame:
    """Load the Hillstrom Email Marketing dataset.

    Parameters
    ----------
    prefer_sklift:
        If ``True`` (default), try the live scikit-uplift download first.
        If the download fails the function falls back to ``local_path``
        and prints a warning.
    local_path:
        Path to read the local cached CSV from.  Used either as the
        fallback when the ``prefer_sklift`` download fails, or directly
        when ``prefer_sklift=False``.  Defaults to ``LOCAL_CSV_PATH``
        (this module's original, unchanged behaviour) when not given —
        this parameter is purely additive so callers such as the CLI can
        point at a configurable data directory without changing behaviour
        for any existing caller that does not pass it.

    Returns
    -------
    pd.DataFrame
        Exactly the columns listed in ``EXPECTED_COLUMNS``, in that order.

    Raises
    ------
    ValueError
        If the local CSV is missing any of the ``EXPECTED_COLUMNS``.
    FileNotFoundError
        If the local CSV does not exist and the live download failed (or
        ``prefer_sklift=False``).
    """
    if local_path is None:
        local_path = LOCAL_CSV_PATH

    if prefer_sklift:
        try:
            from sklift.datasets import fetch_hillstrom

            # target_col="all" returns ALL THREE outcome columns
            # (visit, conversion, spend), plus the raw 3-arm "segment"
            # treatment column, instead of dropping two of them.
            bunch = fetch_hillstrom(target_col="all")

            df = bunch.data.copy()
            df = pd.concat([df, bunch.target], axis=1)  # + visit, conversion, spend
            df["segment"] = bunch.treatment  # + raw 3-arm segment

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
