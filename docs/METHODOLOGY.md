# Methodology

## 1. Treatment design

The original dataset contains three treatment arms:

- `Mens E-Mail`
- `Womens E-Mail`
- `No E-Mail`

The project converts the original treatment into a binary variable:

```text
treatment = 1 if the customer received either email
treatment = 0 if the customer received no email
```

This allows the project to estimate the effect of receiving an email of either type compared with receiving no email.

The treatment definition is implemented in:

```text
src/preprocessing.py
```

## 2. Causal assumptions

The analysis relies on the structure of the original randomized experiment.

The main assumptions are:

### Randomized treatment assignment

Treatment assignment should be independent of customer characteristics before treatment.

The project checks this using:

- Standardized mean differences for numeric features
- Chi-square tests for categorical features
- Treatment-group proportions
- Outcome-rate comparisons after splitting

### Stable treatment definition

The treatment indicator must consistently represent whether a customer received an email.

The project treats both email arms as treatment and the no-email arm as control.

### Consistent outcome definition

The primary outcome is:

```text
visit
```

This indicates whether the customer visited the website during the follow-up period.

The pipeline also preserves:

```text
conversion
spend
```

for downstream analysis and future extensions.

### No interference assumption

The analysis assumes that one customer's treatment does not directly affect another customer's outcome.

This is a standard assumption for individual-level treatment-effect analysis, although it cannot be fully verified from this dataset.

## 3. Feature preparation

The raw data contains numeric and categorical variables.

Numeric variables include:

- `recency`
- `history`
- `mens`
- `womens`
- `newbie`

Categorical variables include:

- `history_segment`
- `zip_code`
- `channel`

Categorical variables are one-hot encoded using `pandas.get_dummies`.

The project intentionally keeps all categories instead of dropping the first category because the models are tree-based.

The treatment column is not included as a standard predictive feature for the naive baseline.

## 4. Train/test split

The project uses an 80/20 train/test split:

```text
Training rows: 51,200
Testing rows: 12,800
```

The split is stratified using a combined key based on:

```text
treatment + conversion
```

This preserves the treatment/outcome combinations in both partitions.

The split logic is implemented in:

```text
src/preprocessing.py
```

## 5. Naive baseline

The baseline is a LightGBM classifier trained to predict whether a customer will visit.

It is intentionally included because standard response modeling is a common business approach.

The baseline does not explicitly estimate treatment effect. It estimates response probability.

This provides an important comparison:

- Response modeling asks: who is likely to respond?
- Uplift modeling asks: who is likely to respond because of treatment?

The baseline uses:

```text
src/baseline_model.py
```

## 6. Two-Model Approach

The Two-Model Approach trains two separate outcome models:

1. A model using treated customers
2. A model using control customers

For each test customer, uplift is estimated as:

```text
predicted outcome under treatment
-
predicted outcome under control
```

This model is implemented in:

```text
src/uplift_models.py
```

## 7. Class Transformation

The Class Transformation approach constructs a transformed target based on treatment and outcome.

The transformed target is then modeled with a single classifier.

The implementation follows the original notebook methodology and uses the standard revert-label formulation.

Because the original treatment allocation is not exactly 50/50, this method may have calibration limitations. A propensity-weighted correction was explored separately but is not part of the main six-phase methodology.

## 8. Causal Forest

The project uses EconML's `CausalForestDML` when available.

The model estimates heterogeneous treatment effects using:

- Outcome nuisance modeling
- Treatment nuisance modeling
- Cross-fitting
- Causal forest estimation

If EconML is unavailable, the project can use a CausalML fallback when installed.

The implementation is in:

```text
src/uplift_models.py
```

## 9. Signal checks

Before final evaluation, the project performs ranking signal checks.

For each strategy, customers are ranked by the model score. The actual treatment/control outcome-rate difference is then compared between the top-ranked group and the rest.

These checks are diagnostic. They do not replace formal Qini evaluation.

## 10. Qini evaluation

The Qini coefficient evaluates whether a ranking places customers with larger incremental treatment effects near the top.

The project uses scikit-uplift implementations for:

- Qini curves
- Qini AUC
- Uplift at selected targeting percentages

The main evaluation code is in:

```text
src/evaluation.py
```

The project reports uplift at:

- 10%
- 20%
- 30%

## 11. Bootstrap confidence intervals

The test set is resampled with replacement multiple times.

For each bootstrap sample:

1. The model ranking is resampled.
2. The Qini coefficient is recalculated.
3. The resulting distribution is stored.

The 2.5th and 97.5th percentiles form the reported 95% confidence interval.

The project uses 500 bootstrap resamples for the main results.

Confidence intervals are important because point estimates alone do not show how uncertain the ranking is.

## 12. Pairwise model comparison

The project checks whether confidence intervals overlap between model pairs.

Overlapping confidence intervals are treated as evidence that a clear distinction cannot be established using this analysis.

This is an interpretive comparison, not a complete replacement for a formal paired statistical test.

## 13. Business simulation

The business simulation selects the top customers according to each model ranking under two budget levels:

- 10% of the test set
- 20% of the test set

The selected customers are evaluated using the observed treatment/control outcome gap.

The output reports estimated incremental visits.

The simulation is implemented in:

```text
src/business_simulation.py
```

The simulation produces point estimates only. It does not establish statistically significant business gains.

## 14. Reproducibility controls

Random seeds are exposed as parameters.

The default seed is:

```text
42
```

The command-line pipeline allows the seed to be changed:

```bash
python -m src.cli run-all --seed 7
```

The project also supports configurable data, processed-output, and report directories.