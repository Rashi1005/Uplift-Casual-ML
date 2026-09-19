# Data Manifest

## Dataset name
Hillstrom Email Marketing dataset (also known as the "MineThatData" email
marketing challenge dataset).

## Source
Distributed via [`scikit-uplift`](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_hillstrom.html)'s
`fetch_hillstrom()` loader, which downloads the original CSV from its
publicly hosted source. This is the same loader `src/data_loader.py` and
`src/prepare_data.py` both use.

## Expected row count
64,000 customers.

## Expected columns
12 columns, defined once in `src/data_loader.py` (`EXPECTED_COLUMNS`) and
reused by `src/prepare_data.py` so the two never drift out of sync:

| Column | Type | Description |
|---|---|---|
| `recency` | numeric | Months since last purchase |
| `history_segment` | categorical | Binned historical spend segment |
| `history` | numeric | Actual dollar value of historical purchases |
| `mens` | binary | Purchased menswear in the past year |
| `womens` | binary | Purchased womenswear in the past year |
| `zip_code` | categorical | Urban / Suburban / Rural |
| `newbie` | binary | New customer in the last 12 months |
| `channel` | categorical | Purchase channel (Phone / Web / Multichannel) |
| `segment` | categorical (3-arm) | Raw treatment: `Mens E-Mail`, `Womens E-Mail`, `No E-Mail` |
| `visit` | binary | Visited the site in the following two weeks |
| `conversion` | binary | Made a purchase in the following two weeks |
| `spend` | numeric | Actual dollars spent in the following two weeks |

`notebooks/01_eda.ipynb` binarizes `segment` into a single `treatment`
column (`1` = received either email, `0` = received no email) for the
randomization checks and all downstream modeling. `prepare_data.py` reports
group counts using this same binarization so its output is directly
comparable to the notebooks.

## Download / preparation instructions
From a clean clone:

```bash
pip install -r requirements.txt
python src/prepare_data.py
```

This downloads the raw dataset and saves it to `data/hillstrom.csv`, which
is the same path `src/data_loader.py` already reads from as its local
fallback. The script validates the schema and prints shape, missing-value,
and treatment-group counts before finishing.

`data/hillstrom.csv` is **not** committed to this repository (see
`.gitignore`) — it is downloaded fresh by `prepare_data.py` on each clean
clone rather than duplicated in version control.

## Known limitations
- **Upstream host reliability:** the underlying dataset is hosted on Amazon
  S3. This has been observed to return `403 Forbidden` from some restricted
  or sandboxed networks (e.g. certain campus/corporate networks, or CI
  environments without outbound access to S3), even though the same request
  succeeds from an unrestricted network. `prepare_data.py` detects this
  failure and prints manual fallback instructions rather than failing
  silently.
- **No official train/validation/test split is provided upstream** — the
  80/20 stratified split used by this project is created in
  `notebooks/02_preprocessing.ipynb`, not part of the raw dataset itself.
- **`conversion` is a very rare outcome** (~0.9% base rate) — this is a
  property of the real data, not a data-quality issue, but it is the reason
  `03_baseline_model.ipynb` documents switching its modeling target from
  `conversion` to `visit` (see the Pipeline section of the main README).