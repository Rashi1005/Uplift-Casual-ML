# Final Completion Checklist

## Repository structure

- [ ] `README.md` accurately describes the current repository.
- [ ] `docs/` contains the project documentation.
- [ ] `data/MANIFEST.md` documents the raw dataset.
- [ ] `data/processed/ARTIFACTS.md` documents generated artifacts.
- [ ] `src/` contains reusable implementation modules.
- [ ] `tests/` contains automated tests.
- [ ] `dashboard/index.html` opens successfully.
- [ ] `reports/` contains generated visual outputs.

## Data workflow

- [ ] A clean clone can download the Hillstrom dataset.
- [ ] The raw dataset is saved to `data/hillstrom.csv`.
- [ ] The raw dataset is excluded from normal Git commits.
- [ ] The expected schema is validated.
- [ ] Missing values are reported.
- [ ] Treatment/control counts are reported.
- [ ] Dataset preparation instructions are documented.

## Modeling workflow

- [ ] The treatment definition is documented.
- [ ] Numeric and categorical features are documented.
- [ ] One-hot encoding is documented.
- [ ] Train/test splitting is documented.
- [ ] The naive baseline is explained.
- [ ] The Two-Model Approach is explained.
- [ ] Class Transformation is explained.
- [ ] Causal Forest is explained.
- [ ] The active causal-model backend is documented when applicable.

## Evaluation workflow

- [ ] Qini coefficient is explained.
- [ ] Uplift-at-k metrics are explained.
- [ ] Bootstrap confidence intervals are explained.
- [ ] Confidence-interval limitations are documented.
- [ ] Point estimates are clearly separated from statistical conclusions.
- [ ] Evaluation results are saved under `data/processed/`.
- [ ] Evaluation plots are saved under `reports/`.

## Business simulation

- [ ] 10% targeting budget is evaluated.
- [ ] 20% targeting budget is evaluated.
- [ ] Random selection is used as a comparison.
- [ ] Estimated incremental visits are reported.
- [ ] The point-estimate limitation is documented.
- [ ] Revenue and profit are not claimed without appropriate data.

## Dashboard

- [ ] Overview/model comparison view works.
- [ ] Confidence intervals are displayed.
- [ ] Budget simulator works at 10%.
- [ ] Budget simulator works at 20%.
- [ ] Persuadables explanation works.
- [ ] Customer-level view uses illustrative profiles only.
- [ ] ROI view clearly labels assumptions.
- [ ] Dashboard does not expose private customer information.
- [ ] Dashboard works from a clean clone.
- [ ] Dashboard works when opened directly or through a local server.
- [ ] Keyboard focus states are visible.
- [ ] Navigation is accessible.
- [ ] Mobile layout is usable.
- [ ] Reduced-motion preferences are respected.

## Tests and quality

- [ ] Data-pipeline tests pass.
- [ ] Module tests pass.
- [ ] Error-handling tests pass.
- [ ] CLI tests pass.
- [ ] Slow tests are clearly marked.
- [ ] Test results are documented.
- [ ] Linting passes.
- [ ] Formatting checks pass.
- [ ] CI workflow passes.
- [ ] No credentials or secrets are committed.
- [ ] Large generated artifacts are handled appropriately.

## Documentation accuracy

- [ ] README does not claim that missing files exist.
- [ ] `Final_Report.docx` is not referenced unless it exists.
- [ ] `Presentation.pptx` is not referenced unless it exists.
- [ ] Dashboard documentation matches the actual dashboard.
- [ ] CLI commands match the actual CLI.
- [ ] Result values match the committed result files.
- [ ] Known limitations are documented.
- [ ] Future work is separated from completed work.

## Final scientific review

- [ ] The model with the highest point estimate is not presented as a proven winner.
- [ ] Confidence intervals are reported.
- [ ] The fact that all intervals include zero is clearly stated.
- [ ] Business simulations are described as point estimates.
- [ ] Individual treatment effects are not presented as directly observed.
- [ ] Production deployment is not claimed without a new validation experiment.

## Missing final deliverables

The current repository does not contain:

```text
docs/Final_Report.docx
docs/Presentation.pptx
```

Equivalent Markdown documentation is provided in this `docs/` directory.

If formal report and presentation files are required later, create them from:

```text
docs/PROJECT_OVERVIEW.md
docs/METHODOLOGY.md
docs/ARCHITECTURE.md
docs/RESULTS.md
docs/LIMITATIONS.md
```