# Results

## Evaluation setup

The main evaluation uses:

- 12,800 held-out test customers
- `visit` as the primary outcome
- Binary treatment indicator
- Qini coefficient
- 500 bootstrap resamples
- 95% confidence intervals

The results are stored in:

```text
data/processed/phase5_results.csv
```

## Qini results

| Model | Qini coefficient | 95% CI | Uplift at 10% | Uplift at 20% | Uplift at 30% |
|---|---:|---:|---:|---:|---:|
| Two-Model Approach | 0.0234 | [-0.0041, 0.0520] | 0.0858 | 0.0754 | 0.0740 |
| Class Transformation | 0.0231 | [-0.0062, 0.0507] | 0.0712 | 0.0751 | 0.0654 |
| Baseline | 0.0129 | [-0.0134, 0.0420] | 0.0821 | 0.0836 | 0.0634 |
| Causal Forest | 0.0077 | [-0.0220, 0.0353] | 0.0476 | 0.0532 | 0.0596 |

## Interpretation of Qini results

The Two-Model Approach has the highest point estimate:

```text
0.0234
```

The Class Transformation model is very close:

```text
0.0231
```

However, all four confidence intervals include zero.

This means the analysis cannot confidently distinguish positive uplift from zero uplift for any model at the 95% confidence level.

The intervals also overlap substantially, so the project cannot establish that the Two-Model Approach is statistically better than the other models.

## Business simulation

The business simulation evaluates targeting at 10% and 20% budgets.

The source file is:

```text
data/processed/phase6_business_impact.csv
```

### Estimated incremental visits at 10% budget

| Strategy | Customers contacted | Estimated incremental visits |
|---|---:|---:|
| Two-Model | 1,280 | 109.9 |
| Baseline | 1,280 | 105.0 |
| Class Transformation | 1,280 | 91.1 |
| Causal Forest | 1,280 | 60.2 |
| Random Selection | 1,280 | 51.5 |

### Estimated incremental visits at 20% budget

| Strategy | Customers contacted | Estimated incremental visits |
|---|---:|---:|
| Baseline | 2,560 | 213.9 |
| Two-Model | 2,560 | 193.6 |
| Class Transformation | 2,560 | 192.2 |
| Causal Forest | 2,560 | 136.2 |
| Random Selection | 2,560 | 117.2 |

## Business interpretation

Every modeled strategy has a higher point estimate than random selection at both budget levels.

This is a useful descriptive finding, but it is not sufficient to establish a guaranteed business advantage.

The 20% budget ranking differs from the 10% ranking:

- Two-Model is highest at 10%.
- Baseline is highest at 20%.
- This demonstrates that the preferred strategy can depend on the targeting budget.

## What can be concluded

The project supports the following conclusions:

1. A response model and uplift models answer different questions.
2. The Two-Model Approach has the highest Qini point estimate in this run.
3. The Class Transformation model is close to the Two-Model Approach.
4. All model confidence intervals include zero.
5. No statistically confirmed model winner is established.
6. Budget level can change the point-estimate ranking.
7. A new randomized holdout would be needed before making a production targeting decision.

## What cannot be concluded

The project does not prove that:

- The Two-Model Approach is the best model in general.
- Any model will improve revenue.
- The reported incremental visits will occur in a future campaign.
- The results generalize to another population.
- Customer-level treatment effects are known exactly.
- The dashboard ROI scenario is a financial forecast.

## Recommended business use

The results are most appropriate for:

- Methodology demonstration
- Exploratory targeting research
- Portfolio presentation
- Designing a future randomized campaign
- Comparing model development approaches

A practical next step would be to select a candidate strategy, retain a randomized holdout group, and measure incremental outcomes in a fresh campaign.