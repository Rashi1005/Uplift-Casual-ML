# Uplift Modeling for Targeted Interventions

### A Causal Machine Learning Approach

[![CI](https://github.com/Rashi1005/Uplift-Casual-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/Rashi1005/Uplift-Casual-ML/actions/workflows/ci.yml)

A phase-wise causal machine learning pipeline for answering a question that standard predictive machine learning cannot answer directly:

> Will sending this customer a marketing email cause them to visit or buy, compared with what they would have done without the email?

This project applies uplift modeling to a randomized email marketing experiment. It compares a naive response model with several uplift and causal models, evaluates their rankings using Qini metrics, and translates the results into budget-constrained targeting scenarios.

---

## Project status

The core analysis pipeline, reusable Python modules, command-line interface, automated tests, CI configuration, results dashboard, and project documentation are implemented.

The project is suitable for:

- Causal ML and uplift-modeling research
- Educational use
- Portfolio demonstration
- Experiment design
- Exploratory marketing-targeting analysis

It should not be treated as a production targeting policy without validation on a new randomized holdout experiment.

---

## What this project does

Given the Hillstrom Email Marketing dataset, this project:

1. Validates whether treatment assignment was sufficiently randomized.
2. Prepares customer features for modeling.
3. Trains a naive predictive baseline.
4. Trains three uplift or causal models.
5. Evaluates all four strategies using Qini coefficients.
6. Computes bootstrap 95% confidence intervals.
7. Compares model rankings at different targeting budgets.
8. Simulates estimated incremental visits under 10% and 20% contact budgets.
9. Provides a static interactive dashboard.
10. Documents the results and their limitations without overstating statistical evidence.

---

## Headline finding

**None of the four models can be shown to be statistically distinguishable from random targeting at the 95% confidence level for this test set.**

The Two-Model Approach achieved the highest point-estimate Qini coefficient:

```text
0.0234
```

However, its 95% confidence interval includes zero and overlaps substantially with the confidence intervals of the other models.

Therefore, the appropriate conclusion is:

> The Two-Model Approach produced the highest point estimate in this run, but the available evidence is insufficient to declare it a statistically confirmed winner.

This distinction between point estimates and statistical evidence is central to the project.

---

## Dataset

The project uses the **Hillstrom Email Marketing dataset**, which contains 64,000 customers from a randomized email marketing experiment.

The original dataset contains three treatment arms:

- `Mens E-Mail`
- `Womens E-Mail`
- `No E-Mail`

For the primary analysis, the two email arms are combined:

```text
Treatment: received either email
Control: received no email
```

### Dataset summary

| Property | Value |
|---|---:|
| Total customers | 64,000 |
| Treatment group | 42,694 customers, 66.71% |
| Control group | 21,306 customers, 33.29% |
| Training set | 51,200 customers |
| Test set | 12,800 customers |
| Split | 80/20, stratified by treatment and outcome |
| Primary outcome | `visit` |
| Additional outcomes | `conversion`, `spend` |

The primary target is `visit` because `conversion` is a rare outcome in this dataset. The `conversion` and `spend` columns are retained for future analysis.

Full schema, source information, preparation instructions, and known data limitations are documented in:

```text
data/MANIFEST.md
```

The raw dataset is not committed to the repository. It is downloaded locally using the preparation workflow.

---

## Research questions

The project investigates the following questions:

1. Was the original treatment assignment sufficiently balanced?
2. How well does a naive response model rank likely visitors?
3. Do uplift models produce better incremental-response rankings?
4. Are apparent model differences statistically meaningful?
5. Does the preferred targeting strategy change with the contact budget?
6. How should the results be interpreted from a business perspective?

---

## Pipeline

| Phase | Notebook | Reusable module | Description |
|---|---|---|---|
| 1 | `01_eda.ipynb` | `src/data_loader.py`, `src/preprocessing.py` | Load data, check treatment balance, inspect outcomes |
| 2 | `02_preprocessing.ipynb` | `src/preprocessing.py` | Encode features and create stratified train/test splits |
| 3 | `03_baseline_model.ipynb` | `src/baseline_model.py` | Train and evaluate a naive LightGBM response model |
| 4 | `04_uplift_models.ipynb` | `src/uplift_models.py` | Train Two-Model, Class Transformation, and Causal Forest models |
| 5 | `05_evaluation_qini.ipynb` | `src/evaluation.py` | Compute Qini metrics and bootstrap confidence intervals |
| 6 | `06_business_simulation.ipynb` | `src/business_simulation.py` | Simulate targeting under 10% and 20% budgets |

The notebooks are explanatory walkthroughs. The reusable computation is implemented in Python modules under `src/`.

---

## Code structure

### Data and preparation modules

| Module | Purpose |
|---|---|
| `src/data_loader.py` | Load the Hillstrom dataset from scikit-uplift or the local cache |
| `src/prepare_data.py` | Download, validate, save, and report on the raw dataset |
| `src/pipeline_meta.py` | Record run metadata and command execution information |

### Modeling modules

| Module | Purpose |
|---|---|
| `src/preprocessing.py` | Treatment conversion, balance checks, encoding, and splitting |
| `src/baseline_model.py` | Naive LightGBM response model |
| `src/uplift_models.py` | Two-Model, Class Transformation, and Causal Forest models |
| `src/business_simulation.py` | Budget-constrained targeting simulation |
| `src/evaluation.py` | Qini metrics, bootstrap confidence intervals, and verdict generation |
| `src/utils.py` | Shared helper functions |

### Command-line interface

```text
src/cli.py
```

The CLI runs the pipeline without requiring manual notebook execution.

Available commands:

```text
prepare-data
preprocess
train-baseline
train-uplift
evaluate
simulate
run-all
```

### Design principles

- Random seeds are configurable.
- Paths are configurable.
- The treatment definition is centralized.
- Notebook logic is implemented through reusable modules.
- Tests use small deterministic synthetic fixtures.
- Raw customer-level records are not exposed through the dashboard.
- Point estimates are clearly separated from statistical conclusions.

---

## Running the project from the command line

### Basic usage

```bash
python -m src.cli <command> [options]
```

### Available commands

| Command | Description | Main outputs |
|---|---|---|
| `prepare-data` | Download and validate the raw dataset | `data/hillstrom.csv` |
| `preprocess` | Encode features and create train/test splits | Processed split CSV files |
| `train-baseline` | Train the naive response model | Baseline model and ranking |
| `train-uplift` | Train the three uplift models | Uplift scores and causal model |
| `evaluate` | Compute Qini metrics and confidence intervals | Phase 5 result files and plot |
| `simulate` | Run business-impact simulation | Phase 6 result file and plot |
| `run-all` | Run the complete pipeline in order | All generated outputs |

### Common options

| Option | Default | Description |
|---|---|---|
| `--data-dir` | `data` | Directory containing raw data |
| `--processed-dir` | `data/processed` | Directory for processed data and model artifacts |
| `--reports-dir` | `reports` | Directory for generated plots |
| `--seed` | `42` | Random seed |
| `--skip-if-exists` | disabled | Skip a stage if its outputs already exist |

The `train-uplift` and `run-all` commands also support:

```text
--no-auto-install
```

This prevents runtime installation attempts for missing causal-model dependencies.

### Run the complete pipeline

```bash
python -m src.cli run-all
```

### Run with a different seed and output directory

```bash
python -m src.cli run-all \
  --seed 7 \
  --processed-dir data/processed_seed7 \
  --reports-dir reports_seed7
```

### Reuse existing artifacts

```bash
python -m src.cli run-all --skip-if-exists
```

### Run a single phase

```bash
python -m src.cli train-baseline --seed 42
```

Every command prints progress messages and returns a nonzero exit status when a required input is missing or a stage fails.

---

## Preparing the data

The raw dataset is intentionally not committed to Git.

Install the dependencies and run:

```bash
python src/prepare_data.py
```

Equivalent CLI command:

```bash
python -m src.cli prepare-data
```

The preparation script:

- Downloads the Hillstrom dataset through scikit-uplift.
- Saves the data to `data/hillstrom.csv`.
- Validates the expected columns.
- Reports the dataset shape.
- Reports missing values.
- Reports treatment/control counts.

If the download fails because of network restrictions, consult:

```text
data/MANIFEST.md
```

---

## Results

### Qini coefficients and 95% bootstrap confidence intervals

| Model | Qini coefficient | 95% confidence interval |
|---|---:|---:|
| Two-Model Approach | 0.0234 | [-0.0041, 0.0520] |
| Class Transformation | 0.0231 | [-0.0062, 0.0507] |
| Baseline | 0.0129 | [-0.0134, 0.0420] |
| Causal Forest | 0.0077 | [-0.0220, 0.0353] |

### Uplift at selected targeting percentages

| Model | 10% | 20% | 30% |
|---|---:|---:|---:|
| Baseline | 0.0821 | 0.0836 | 0.0634 |
| Two-Model Approach | 0.0858 | 0.0754 | 0.0740 |
| Class Transformation | 0.0712 | 0.0751 | 0.0654 |
| Causal Forest | 0.0476 | 0.0532 | 0.0596 |

These values are ranking metrics, not direct revenue estimates.

### Business impact simulation

#### 10% contact budget

| Strategy | Customers contacted | Estimated incremental visits |
|---|---:|---:|
| Two-Model | 1,280 | 109.9 |
| Baseline | 1,280 | 105.0 |
| Class Transformation | 1,280 | 91.1 |
| Causal Forest | 1,280 | 60.2 |
| Random Selection | 1,280 | 51.5 |

#### 20% contact budget

| Strategy | Customers contacted | Estimated incremental visits |
|---|---:|---:|
| Baseline | 2,560 | 213.9 |
| Two-Model | 2,560 | 193.6 |
| Class Transformation | 2,560 | 192.2 |
| Causal Forest | 2,560 | 136.2 |
| Random Selection | 2,560 | 117.2 |

All model-based strategies have higher point estimates than random selection at both budget levels.

However, these are point estimates from one held-out test set. They should not be interpreted as guaranteed future business gains.

Detailed results are documented in:

```text
docs/RESULTS.md
```

---

## Evaluation methodology

### Naive baseline

The naive baseline is a LightGBM classifier that predicts whether a customer will visit.

It is included because it represents a common business approach:

> Contact customers who are most likely to respond.

The baseline does not explicitly estimate treatment effects. It provides a reference point for evaluating whether uplift modeling produces a meaningfully different ranking.

### Two-Model Approach

Two separate models are trained:

1. A model using treated customers.
2. A model using control customers.

The uplift estimate is:

```text
predicted outcome under treatment
-
predicted outcome under control
```

### Class Transformation

The Class Transformation method creates a transformed treatment/outcome target and trains a single classifier.

The implementation follows the main methodology used in the project notebooks.

### Causal Forest

The project uses EconML's `CausalForestDML` when available.

The model estimates heterogeneous treatment effects using nuisance models, cross-fitting, and causal forest estimation.

A CausalML fallback is supported when EconML is unavailable and the fallback package is installed.

### Qini evaluation

Qini evaluation measures whether customers with larger estimated incremental effects are ranked near the top.

The project uses scikit-uplift implementations for:

- Qini curves
- Qini AUC
- Uplift at selected targeting percentages

### Bootstrap confidence intervals

The test set is resampled with replacement 500 times.

For each resample, the Qini coefficient is recalculated. The 2.5th and 97.5th percentiles are reported as the 95% confidence interval.

Because every interval includes zero, the project does not claim statistically confirmed positive uplift for any model.

---

## Interactive dashboard

The dashboard is a self-contained static HTML application:

```text
dashboard/index.html
```

It contains five views:

1. **Overview** — Qini coefficients and confidence intervals.
2. **Budget simulator** — estimated incremental visits at 10% and 20% budgets.
3. **Persuadables** — explanation of response types and uplift modeling.
4. **Customer-level explainer** — illustrative, non-identifying customer profiles.
5. **ROI view** — scenario-based value and cost calculation.

The dashboard uses values from:

```text
data/processed/phase5_results.csv
data/processed/phase6_business_impact.csv
```

The result values are embedded in the HTML intentionally. This allows the dashboard to work when opened directly from the filesystem without requiring a backend or CSV-fetch permission.

### Open directly

Windows PowerShell:

```powershell
Start-Process dashboard/index.html
```

Or open this file manually:

```text
dashboard/index.html
```

### Use a local server

From the repository root:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/dashboard/index.html
```

The customer-level profiles shown in the dashboard are illustrative educational examples. They are not predictions for real individuals.

If the pipeline is rerun and result values change, update the embedded data objects in:

```text
dashboard/index.html
```

---

## Artifact workflow

Generated artifacts are reproducible from a clean clone and are generally excluded from version control when they are large or machine-generated.

| Artifact category | Location | Generated by |
|---|---|---|
| Raw dataset | `data/hillstrom.csv` | `prepare-data` |
| Processed features | `data/processed/X_*.csv` | `preprocess` |
| Processed outcomes | `data/processed/y_*.csv` | `preprocess` |
| Treatment splits | `data/processed/treatment_*.csv` | `preprocess` |
| Trained models | `data/processed/*.pkl` | `train-baseline`, `train-uplift` |
| Predictions/rankings | `data/processed/*ranking.csv` | `train-baseline`, `train-uplift` |
| Uplift predictions | `data/processed/uplift_scores_*.csv` | `train-uplift` |
| Evaluation results | `data/processed/phase5_*.csv` | `evaluate` |
| Evaluation verdict | `data/processed/phase5_verdict.md` | `evaluate` |
| Business simulation | `data/processed/phase6_business_impact.csv` | `simulate` |
| Run metadata | `data/processed/run_metadata.json` | Pipeline commands |
| Run log | `data/processed/run_log.jsonl` | Pipeline commands |
| Qini plot | `reports/qini_comparison.png` | `evaluate` |
| Business-impact plot | `reports/business_impact_comparison.png` | `simulate` |

Artifact details are documented in:

```text
data/processed/ARTIFACTS.md
```

The raw dataset and generated model files should not be committed unnecessarily.

The Causal Forest model artifact may be large. If it is already tracked in Git, remove it from version control before pushing future changes:

```bash
git rm --cached data/processed/causal_forest_model.pkl
git commit -m "Remove generated causal forest artifact from Git tracking"
```

---

## Test suite

Tests are located in:

```text
tests/
```

| Test file | Coverage |
|---|---|
| `tests/conftest.py` | Shared deterministic fixtures |
| `tests/test_data_pipeline.py` | Data loading, schema validation, and missing-file behavior |
| `tests/test_modules.py` | Preprocessing, baseline, uplift, evaluation, and simulation modules |
| `tests/test_uplift_causal_forest.py` | Causal Forest behavior and fallback handling |
| `tests/test_error_handling.py` | Malformed and inconsistent input files |
| `tests/test_cli.py` | CLI parsing, failures, output paths, and integration behavior |

### Run all tests

```bash
pytest -v
```

### Run only fast tests

```bash
pytest -m "not slow" -v
```

### Run slow tests

```bash
pytest -m slow -v
```

### Collect the test count

```bash
pytest --collect-only -q
```

The slow tests include Causal Forest fitting and multi-command CLI integration checks.

Tests use deterministic synthetic fixtures and do not require downloading the full dataset.

---

## Code quality and formatting

The project uses Ruff for linting and formatting.

Run linting:

```bash
ruff check src/ tests/
```

Check formatting:

```bash
ruff format --check src/ tests/
```

Apply formatting:

```bash
ruff format src/ tests/
```

Project configuration is stored in:

```text
pyproject.toml
```

---

## Continuous integration

GitHub Actions configuration is located at:

```text
.github/workflows/ci.yml
```

CI checks include:

- Python setup
- Dependency installation
- Ruff linting
- Ruff formatting checks
- Fast pytest suite
- CLI smoke checks
- Full tests on the configured main-branch workflow

The CI workflow does not require private secrets.

---

## Repository structure

```text
Uplift-Casual-ML/
├── data/
│   ├── MANIFEST.md
│   ├── hillstrom.csv                  # downloaded locally; gitignored
│   └── processed/
│       ├── ARTIFACTS.md
│       └── generated artifacts
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_model.ipynb
│   ├── 04_uplift_models.ipynb
│   ├── 05_evaluation_qini.ipynb
│   └── 06_business_simulation.ipynb
├── src/
│   ├── data_loader.py
│   ├── prepare_data.py
│   ├── preprocessing.py
│   ├── baseline_model.py
│   ├── uplift_models.py
│   ├── evaluation.py
│   ├── business_simulation.py
│   ├── pipeline_meta.py
│   ├── utils.py
│   └── cli.py
├── tests/
│   ├── conftest.py
│   ├── test_data_pipeline.py
│   ├── test_modules.py
│   ├── test_uplift_causal_forest.py
│   ├── test_error_handling.py
│   └── test_cli.py
├── reports/
│   ├── qini_comparison.png
│   └── business_impact_comparison.png
├── dashboard/
│   └── index.html
├── docs/
│   ├── PROJECT_OVERVIEW.md
│   ├── METHODOLOGY.md
│   ├── ARCHITECTURE.md
│   ├── REPRODUCIBILITY.md
│   ├── LIMITATIONS.md
│   ├── RESULTS.md
│   └── FINAL_CHECKLIST.md
├── requirements.txt
├── requirements-dev.txt
├── requirements-ci.txt
├── pyproject.toml
├── pytest.ini
├── .gitignore
└── README.md
```

---

## Installation

### Prerequisites

| Requirement | Minimum | Tested |
|---|---:|---:|
| Python | 3.11 | 3.12 |
| pip | 23.0 | Recent pip version |
| Operating system | Windows, macOS, or Linux | Windows and Linux |

### Clone the repository

```bash
git clone https://github.com/Rashi1005/Uplift-Casual-ML.git
cd Uplift-Casual-ML
```

### Create a virtual environment

#### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Windows PowerShell

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

#### Windows Command Prompt

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### Install runtime dependencies

```bash
pip install -r requirements.txt
```

### Install development dependencies

```bash
pip install -r requirements-dev.txt
```

Or, if configured in `pyproject.toml`:

```bash
pip install -e ".[dev]"
```

The causal forest dependency may require additional compiled packages and can take longer to install than the other dependencies.

---

## Documentation

Detailed project documentation is available under `docs/`:

- [`PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md)
- [`METHODOLOGY.md`](docs/METHODOLOGY.md)
- [`ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- [`REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
- [`LIMITATIONS.md`](docs/LIMITATIONS.md)
- [`RESULTS.md`](docs/RESULTS.md)
- [`FINAL_CHECKLIST.md`](docs/FINAL_CHECKLIST.md)

The repository currently provides Markdown-based final documentation. Formal `Final_Report.docx` and `Presentation.pptx` files are not included.

---

## Limitations

The most important limitations are:

- The dataset may not represent a modern production population.
- The two email treatment arms are combined into one treatment group.
- The primary outcome is website visit, not revenue or profit.
- Conversion is a rare outcome and requires separate modeling.
- All model confidence intervals include zero.
- Individual treatment effects cannot be directly observed.
- Confidence-interval overlap is an informative diagnostic, not a complete paired hypothesis test.
- Business results are point estimates from one held-out test set.
- Exact outputs may vary across software versions and causal-model backends.
- A future randomized holdout is required before production use.

Additional details are documented in:

```text
docs/LIMITATIONS.md
```

---

## Future work

Potential future improvements include:

- Re-run the pipeline on the larger Criteo uplift dataset.
- Analyze the original three-arm treatment design separately.
- Extend modeling to the rare `conversion` outcome.
- Add formal paired bootstrap tests for model differences.
- Add repeated cross-validation and stability analysis.
- Add confidence intervals to the business simulation.
- Add real revenue and cost data to the ROI analysis.
- Add fairness and subgroup-treatment-effect analysis.
- Add automated dashboard data generation.
- Add model versioning and experiment tracking.
- Add a new randomized validation campaign.
- Add formal PDF report and presentation exports if required.

---

## References

- Athey, S., Tibshirani, J., & Wager, S. (2019). *Generalized Random Forests*. Annals of Statistics.
- Gutierrez, P., & Gerardy, J.-Y. (2017). *Causal Inference and Uplift Modelling: A Review of the Literature*. PMLR.
- Belbahri, M., Murua, A., Gandouet, O., & Partovi Nia, V. (2021). *Qini-based Uplift Regression*. Annals of Applied Statistics.
- Diemert, E., Betlei, A., Renaudin, C., & Amini, M.-R. (2018). *A Large Scale Benchmark for Uplift Modeling*. AdKDD & TargetAd Workshop.

---

## Final scientific conclusion

The project demonstrates the difference between predicting response and estimating incremental treatment effect.

The Two-Model Approach produced the highest Qini point estimate in the current run, but the confidence intervals are wide and overlap with the alternatives. Therefore, the evidence does not support declaring a statistically confirmed best model.

The most defensible next step is not to deploy the apparent winner immediately. It is to use the results to design a new randomized campaign with:

- A pre-specified targeting strategy
- A randomized holdout group
- Clearly defined business outcomes
- Revenue and cost measurement
- Confidence intervals for the final business impact