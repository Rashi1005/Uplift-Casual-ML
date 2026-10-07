# Architecture

## System overview

The project has four layers:

1. Data acquisition
2. Reusable modeling modules
3. Notebook and CLI orchestration
4. Results presentation

```text
Dataset source
      |
      v
src/data_loader.py
src/prepare_data.py
      |
      v
src/preprocessing.py
      |
      v
Train/test artifacts
      |
      +--> src/baseline_model.py
      |
      +--> src/uplift_models.py
      |
      v
src/evaluation.py
src/business_simulation.py
      |
      +--> data/processed/
      +--> reports/
      +--> dashboard/index.html
```

## Data layer

### `src/data_loader.py`

Responsibilities:

- Load the Hillstrom dataset through scikit-uplift
- Fall back to the local CSV cache
- Validate the expected schema
- Return the standardized column order

### `src/prepare_data.py`

Responsibilities:

- Download the raw dataset
- Save it to `data/hillstrom.csv`
- Validate required columns
- Report shape and missing values
- Report treatment/control counts

### `data/MANIFEST.md`

Documents:

- Dataset source
- Expected columns
- Treatment definition
- Download instructions
- Known limitations

## Preprocessing layer

### `src/preprocessing.py`

Responsibilities:

- Convert the original three-arm treatment into a binary treatment
- Check numeric balance
- Check categorical balance
- Encode categorical features
- Create stratified train/test splits
- Verify post-split treatment and outcome rates

Outputs include:

```text
X_train.csv
X_test.csv
y_train.csv
y_test.csv
treatment_train.csv
treatment_test.csv
```

## Modeling layer

### `src/baseline_model.py`

Implements:

- LightGBM baseline training
- Baseline evaluation
- Ranking generation
- Top-decile signal checks

### `src/uplift_models.py`

Implements:

- Two-Model Approach
- Class Transformation
- Causal Forest
- Uplift signal checks
- Combined model-ranking output

### `src/utils.py`

Contains shared utilities including:

- Column-name sanitization
- Positive-class weight calculation
- Top-k selection
- Actual treatment/control uplift calculation

## Evaluation layer

### `src/evaluation.py`

Implements:

- Qini curves
- Qini coefficients
- Uplift-at-k metrics
- Bootstrap confidence intervals
- Pairwise confidence-interval comparisons
- Dynamic final verdict generation

### `src/business_simulation.py`

Implements:

- Budget-constrained customer selection
- Random targeting comparison
- Estimated incremental visits
- Business-impact charts
- Point-estimate comparison against random selection

## Orchestration layer

### Notebooks

The notebooks are explanatory walkthroughs:

```text
01_eda.ipynb
02_preprocessing.ipynb
03_baseline_model.ipynb
04_uplift_models.ipynb
05_evaluation_qini.ipynb
06_business_simulation.ipynb
```

They document the research flow and call reusable code where available.

### CLI

`src/cli.py` provides:

```text
prepare-data
preprocess
train-baseline
train-uplift
evaluate
simulate
run-all
```

A complete command-line run is:

```bash
python -m src.cli run-all
```

## Storage layer

### Raw data

```text
data/hillstrom.csv
```

The raw dataset is downloaded locally and excluded from normal version control.

### Processed data

```text
data/processed/
```

Contains train/test splits, model artifacts, predictions, and evaluation results.

### Reports

```text
reports/
```

Contains generated plots such as:

```text
qini_comparison.png
business_impact_comparison.png
```

### Dashboard

```text
dashboard/index.html
```

The dashboard is a static HTML application. It embeds the current Phase 5 and Phase 6 result values so it can be opened without a backend.

## Testing architecture

The test suite uses small synthetic data for fast, deterministic checks.

Important test groups:

- Data pipeline tests
- Module-level tests
- Error-handling tests
- Causal Forest tests
- CLI tests

Slow tests are marked separately using pytest markers.

## Design principles

The project follows these principles:

- Scientific methodology should remain visible and documented.
- Random seeds should be configurable.
- Paths should be configurable.
- Unit tests should not depend on network access.
- Point estimates should be separated from statistical conclusions.
- Raw customer-level data should not be exposed through the dashboard.
- Notebooks should explain the analysis, while modules provide reusable implementation.