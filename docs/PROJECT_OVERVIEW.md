# Project Overview

## Project name

**Uplift Modeling for Targeted Interventions**

This project applies causal machine learning and uplift modeling to determine which customers should receive a marketing email.

The central question is not:

> Which customers are likely to visit?

It is:

> Which customers are more likely to visit because they received the email?

That distinction is important because a standard response model may prioritize customers who would have visited anyway.

## Business problem

Marketing budgets are limited. A company may not be able to contact every customer, so it must decide who should receive an intervention.

A standard predictive model ranks customers by their probability of responding. However, this ranking can include:

- Customers who would respond without treatment
- Customers who would not respond under either condition
- Customers who respond only because of treatment
- Customers who may react negatively to treatment

Uplift modeling attempts to rank customers by the expected incremental effect of treatment.

In this project, the intervention is an email marketing message and the primary outcome is whether the customer visited the website.

## Project objectives

The project has six objectives:

1. Validate that the original treatment assignment was sufficiently randomized.
2. Prepare customer features for modeling.
3. Train a naive predictive baseline.
4. Train three uplift or causal models.
5. Evaluate the models using Qini metrics and bootstrap confidence intervals.
6. Translate the results into budget-constrained targeting scenarios.

## Dataset

The project uses the Hillstrom Email Marketing dataset.

The dataset contains 64,000 customers from a randomized email marketing experiment with three original groups:

- `Mens E-Mail`
- `Womens E-Mail`
- `No E-Mail`

For the main analysis, the two email groups are combined into one treatment group:

- Treatment: received either email
- Control: received no email

The primary outcome is `visit`. The dataset also contains `conversion` and `spend`, although the main pipeline uses `visit` because `conversion` is a rare outcome.

Detailed dataset information is documented in:

```text
data/MANIFEST.md
```

## Main results

The Two-Model Approach had the highest point-estimate Qini coefficient:

| Model | Qini coefficient | 95% confidence interval |
|---|---:|---:|
| Two-Model Approach | 0.0234 | [-0.0041, 0.0520] |
| Class Transformation | 0.0231 | [-0.0062, 0.0507] |
| Baseline | 0.0129 | [-0.0134, 0.0420] |
| Causal Forest | 0.0077 | [-0.0220, 0.0353] |

However, every confidence interval includes zero and the intervals overlap substantially.

Therefore, the project does **not** establish that one model is statistically superior to the others.

The correct conclusion is:

> The Two-Model Approach produced the highest point estimate, but the available test-set evidence is insufficient to declare a statistically confirmed winner.

## Repository components

```text
data/
    MANIFEST.md
    processed/
        phase5_results.csv
        phase6_business_impact.csv
        generated model and split artifacts

notebooks/
    01_eda.ipynb
    02_preprocessing.ipynb
    03_baseline_model.ipynb
    04_uplift_models.ipynb
    05_evaluation_qini.ipynb
    06_business_simulation.ipynb

src/
    data_loader.py
    prepare_data.py
    preprocessing.py
    baseline_model.py
    uplift_models.py
    evaluation.py
    business_simulation.py
    utils.py
    cli.py

tests/
    test_data_pipeline.py
    test_modules.py
    test_error_handling.py
    test_uplift_causal_forest.py
    test_cli.py

dashboard/
    index.html

reports/
    qini_comparison.png
    business_impact_comparison.png
```

## Dashboard

The dashboard is a self-contained static HTML application.

It contains:

1. Model comparison
2. Budget simulation
3. Persuadables explanation
4. Illustrative customer-level explanation
5. Scenario-based ROI calculator

Open it directly:

```text
dashboard/index.html
```

Or serve the repository locally:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/dashboard/index.html
```

## Intended use

This is a research, educational, and portfolio project. It demonstrates:

- Causal reasoning
- Uplift modeling
- Experimental evaluation
- Statistical uncertainty
- Business-oriented interpretation
- Reproducible ML engineering

The results should not be treated as a production marketing policy without additional validation on a new randomized holdout experiment.