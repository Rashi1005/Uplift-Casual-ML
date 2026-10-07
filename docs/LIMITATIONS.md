# Limitations

## 1. Dataset limitations

The project uses the Hillstrom Email Marketing dataset, which is a historical marketing experiment.

It may not represent:

- Current customer behavior
- Current marketing channels
- Current product offerings
- Current geographic or demographic populations
- The economics of a real production campaign

Results should not be generalized automatically to another business or campaign.

## 2. Treatment simplification

The original experiment contains two different email treatments:

- Mens email
- Womens email

The main project combines both email groups into one treatment group.

This simplifies the analysis but removes the ability to estimate separate effects for the two email types.

A future version could model the original three-arm treatment directly.

## 3. Outcome choice

The main modeling target is `visit`.

Conversion is much rarer and was not used as the primary modeling target.

This means the results primarily describe incremental visits, not purchases, revenue, or profit.

A model that improves visit uplift may not improve conversion uplift or financial outcomes.

## 4. Statistical power

The reported confidence intervals are wide.

Every model's Qini confidence interval includes zero:

```text
Two-Model Approach: [-0.0041, 0.0520]
Class Transformation: [-0.0062, 0.0507]
Baseline: [-0.0134, 0.0420]
Causal Forest: [-0.0220, 0.0353]
```

Therefore:

- The highest point estimate is not a statistically confirmed winner.
- Positive point estimates should not be interpreted as proof of positive population-level uplift.
- The apparent ranking may partly reflect sampling variation.

## 5. Confidence-interval interpretation

The project compares overlapping confidence intervals as an interpretive diagnostic.

Non-overlap or overlap is not a complete substitute for a formal paired hypothesis test.

Future work should consider:

- Paired bootstrap differences between model Qini scores
- Permutation tests
- Cross-fitting and repeated sample splits
- Multiple-comparison correction

## 6. Individual treatment effects are unobservable

For an individual customer, we can observe either:

- The outcome after treatment
- The outcome without treatment

We cannot observe both outcomes for the same customer at the same time.

Consequently, individual-level uplift estimates cannot be directly verified.

The project evaluates rankings at an aggregate level.

## 7. Treatment-effect heterogeneity

Uplift models estimate heterogeneous effects from finite data.

The models may:

- Overfit noise
- Produce unstable rankings
- Perform differently on another sample
- Fail to identify small subgroups reliably

A future campaign should validate rankings using a new randomized holdout.

## 8. Causal Forest fallback

The project prefers EconML's Causal Forest implementation and may use a CausalML fallback if EconML is unavailable.

Different libraries may produce different estimates.

The active backend should always be recorded when comparing runs.

## 9. Model assumptions

The methods rely on assumptions related to:

- Randomized treatment assignment
- Correct treatment coding
- Consistent outcome measurement
- No interference between customers
- Adequate overlap between treatment and control groups
- Reasonable model specification

The project checks some of these assumptions but cannot prove all of them.

## 10. Business simulation limitations

The business simulation reports estimated incremental visits.

It does not include:

- Email production cost beyond configurable dashboard assumptions
- Revenue per visit
- Profit margin
- Customer lifetime value
- Unsubscribe risk
- Brand effects
- Delivery constraints
- Contact-frequency limits
- Channel interaction
- Long-term retention

The dashboard ROI view is therefore a scenario calculator, not a financial forecast.

## 11. Dashboard limitations

The customer-level explainer uses illustrative profiles.

It does not expose or predict results for actual customers.

The dashboard embeds the current result files. If the pipeline is rerun, the embedded dashboard values must be updated manually.

## 12. Dataset availability

The raw dataset is downloaded from an external source.

Download availability may be affected by:

- Network restrictions
- Source-host changes
- S3 access limitations
- Package-version changes

The repository documents the fallback behavior, but it does not permanently guarantee external data availability.

## 13. Reproducibility limitations

Exact numerical results may differ across:

- Python versions
- Package versions
- Operating systems
- CPU environments
- Causal model backends
- Random seeds

The project exposes random seeds and documents the expected workflow, but bit-for-bit reproducibility is not guaranteed in every environment.

## 14. Production-readiness limitations

This repository is suitable for research, education, and portfolio demonstration.

It is not a complete production targeting system.

Before production use, the project would need:

- Privacy review
- Legal review
- Monitoring
- Model governance
- Fairness analysis
- Drift detection
- Campaign holdouts
- Data access controls
- Secure artifact storage
- Retraining procedures