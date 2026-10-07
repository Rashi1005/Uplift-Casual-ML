# Reproducibility Guide

## Supported environment

The project is intended for:

- Python 3.11 or newer
- Windows, macOS, or Linux
- A virtual environment
- Internet access for the initial dataset download

## 1. Clone the repository

```bash
git clone https://github.com/Rashi1005/Uplift-Casual-ML.git
cd Uplift-Casual-ML
```

## 2. Create a virtual environment

### Windows PowerShell

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### Linux or macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

For development and testing:

```bash
pip install -r requirements-dev.txt
```

## 4. Download and validate the dataset

```bash
python src/prepare_data.py
```

Equivalent CLI command:

```bash
python -m src.cli prepare-data
```

The command downloads the Hillstrom dataset and writes:

```text
data/hillstrom.csv
```

It reports:

- Dataset shape
- Missing values
- Treatment/control counts
- Schema validation status

The raw dataset is intentionally not committed to the repository.

## 5. Run the full CLI pipeline

```bash
python -m src.cli run-all
```

The main stages are:

```text
prepare-data
preprocess
train-baseline
train-uplift
evaluate
simulate
```

For a separate output location:

```bash
python -m src.cli run-all \
  --seed 7 \
  --processed-dir data/processed_seed7 \
  --reports-dir reports_seed7
```

On Windows PowerShell, use one line or PowerShell backticks for line continuation.

## 6. Run individual stages

```bash
python -m src.cli prepare-data
python -m src.cli preprocess
python -m src.cli train-baseline
python -m src.cli train-uplift
python -m src.cli evaluate
python -m src.cli simulate
```

Use:

```bash
python -m src.cli <command> --help
```

to inspect command-specific options.

## 7. Run the notebooks

After preparing the data:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Run the notebooks in this order:

```text
01_eda.ipynb
02_preprocessing.ipynb
03_baseline_model.ipynb
04_uplift_models.ipynb
05_evaluation_qini.ipynb
06_business_simulation.ipynb
```

## 8. Run tests

Fast tests:

```bash
pytest -m "not slow" -v
```

Full suite:

```bash
pytest -v
```

Data-pipeline tests:

```bash
pytest tests/test_data_pipeline.py -v
```

Collect test count without running tests:

```bash
pytest --collect-only -q
```

## 9. Run the dashboard

Open directly:

```text
dashboard/index.html
```

Windows PowerShell:

```powershell
Start-Process dashboard/index.html
```

Alternatively:

```bash
python -m http.server 8000
```

Then visit:

```text
http://localhost:8000/dashboard/index.html
```

## 10. Expected outputs

Important generated outputs include:

```text
data/processed/X_train.csv
data/processed/X_test.csv
data/processed/y_train.csv
data/processed/y_test.csv
data/processed/treatment_train.csv
data/processed/treatment_test.csv
data/processed/baseline_model.pkl
data/processed/baseline_ranking.csv
data/processed/causal_forest_model.pkl
data/processed/uplift_scores_combined.csv
data/processed/phase5_results.csv
data/processed/phase6_business_impact.csv
reports/qini_comparison.png
reports/business_impact_comparison.png
```

Some generated artifacts may be excluded from version control because they are large or reproducible.

## 11. Reproducibility limitations

Exact output may vary because of:

- Package-version differences
- Operating-system differences
- LightGBM implementation details
- EconML or CausalML backend availability
- Random seed changes
- Dataset-host availability
- Differences between the live downloaded dataset and an older cached file

For comparable runs, use:

```text
Python 3.11 or 3.12
random seed 42
the same dependency versions
the same dataset source
```

## 12. Reproducibility checklist

Before considering a run complete, verify:

- [ ] The raw dataset downloaded successfully.
- [ ] The dataset has 64,000 rows.
- [ ] All expected columns are present.
- [ ] No unexpected missing values exist.
- [ ] Treatment counts are reported.
- [ ] Train/test files were generated.
- [ ] Baseline model completed.
- [ ] Uplift models completed.
- [ ] Qini results were generated.
- [ ] Business simulation completed.
- [ ] Tests passed.
- [ ] Dashboard reflects the current result files.