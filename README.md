# Uplift Modeling for Targeted Interventions
### A Causal Machine Learning Approach

A complete, phase-wise causal machine learning pipeline that answers a different question than standard predictive ML: not *"will this customer visit or buy?"* but *"will sending this customer a marketing email cause them to visit or buy, versus what they'd have done anyway?"*

This is the exact problem real companies solve when deciding who to target with a promotion — Netflix, Amazon, and Uber all apply causal inference in production, each to a different use case (see [Background](#background) below).

---

## What this project does

Given a dataset from a real randomized email marketing experiment, this project:

1. Validates that treatment assignment was genuinely randomized
2. Trains a naive predictive baseline (the common, causally-blind approach)
3. Builds three uplift models of increasing methodological rigor
4. Evaluates all four using the Qini coefficient with bootstrapped 95% confidence intervals — the field's standard metric, not accuracy or F1
5. Translates the results into a business-facing budget simulation
6. Reports findings honestly, including a statistically inconclusive result, rather than overclaiming a winner

---

## Headline finding

**None of the four models — including the naive baseline — can be shown to be statistically distinguishable from random targeting at this dataset's sample size.**

The Two-Model Approach had the highest point-estimate Qini coefficient (0.0234), but every model's 95% confidence interval included zero and overlapped substantially with every other model's. This mirrors a documented limitation in [Diemert et al. (2018)](http://papers.adkdd.org/2018/papers/adkdd18-diemert-large-scale.pdf), the paper behind the large-scale Criteo uplift benchmark, who found similar statistical indistinguishability at comparable sample sizes.

This is reported as a genuine, citable methodological finding — not hidden or adjusted to produce a cleaner story.

---

## Dataset

**Hillstrom Email Marketing dataset** — 64,000 customers from a real randomized controlled trial (RCT).

| | |
|---|---|
| Total customers | 64,000 |
| Treatment group | 66.71% (received a marketing email) |
| Control group | 33.29% (received nothing) |
| Train / test split | 51,200 / 12,800 (80/20, stratified by treatment + outcome) |
| Randomization check | Passed — max SMD 0.0071, all chi-square p > 0.05 |

---

## Pipeline

| Phase | Notebook | What it does |
|---|---|---|
| 1 | `01_eda.ipynb` | Load data, validate randomization, compute naive reference uplift |
| 2 | `02_preprocessing.ipynb` | Encode features, stratified train/test split, post-split verification |
| 3 | `03_baseline_model.ipynb` | Naive LightGBM classifier (target corrected from `conversion` → `visit` after the first attempt scored near-random — documented as a methodological finding, not hidden) |
| 4 | `04_uplift_models.ipynb` | Two-Model Approach, Class Transformation, Causal Forest (EconML) |
| 5 | `05_evaluation_qini.ipynb` | Qini coefficients + bootstrapped 95% confidence intervals |
| 6 | `06_business_simulation.ipynb` | Budget-constrained targeting simulation (10% / 20%) vs. random selection |

Each phase's output was independently verified against saved data files before moving to the next.

## Code structure: notebooks vs. modules

The core modeling logic behind each phase lives in reusable Python modules under `src/`, not only inline in the notebooks. Each notebook now reads as an explanatory walkthrough — the markdown commentary, the reasoning, the printed diagnostics — while the actual computation (encoding, training, evaluation, simulation) is a call into the matching module. This means the same functions can be reused outside a notebook (a script, an API, a future dashboard backend) without copy-pasting logic, and any bug fix only needs to happen in one place.

| Module | Used by | What it contains |
|---|---|---|
| `src/data_loader.py` | all notebooks | Loads the Hillstrom dataset (live download or local cache) |
| `src/prepare_data.py` | — (standalone script) | Raw-data download, validation, and reporting (see "Preparing the data" above) |
| `src/preprocessing.py` | `01_eda.ipynb`, `02_preprocessing.ipynb` | Treatment binarization, randomization/balance checks, one-hot encoding, stratified train/test split and its post-split verification |
| `src/baseline_model.py` | `03_baseline_model.ipynb` | Training and evaluating the naive (non-causal) classifier, building its ranking, and the top-decile signal check |
| `src/uplift_models.py` | `04_uplift_models.ipynb` | Two-Model Approach, Class Transformation, Causal Forest (with the econml → causalml fallback), the uplift signal check, and combining all rankings into one table |
| `src/evaluation.py` | `05_evaluation_qini.ipynb` | Qini curves/coefficients, the comparison plot, bootstrapped confidence intervals, pairwise significance, and the dynamically-generated final verdict |
| `src/business_simulation.py` | `06_business_simulation.ipynb` | The budget-constrained targeting simulation, its comparison plot, and the beat-random check |
| `src/utils.py` | `baseline_model.py`, `uplift_models.py`, `business_simulation.py` | Small helpers shared by more than one module (column-name sanitization, class-weight computation, top-k splitting, actual-uplift calculation) — extracted specifically to avoid the duplication that existed when this logic lived separately inside each notebook |

**Design choices carried through every module:**
- **Random seeds are always a parameter** (`random_state=...`), never hardcoded inside a function — every notebook still controls and prints the seed it used.
- **File paths are always a parameter or passed in by the caller** — no module hardcodes `data/processed/...`; that path lives in the notebook that calls it.
- **The target variable, treatment definition, and evaluation methodology are unchanged** from the original notebooks — this was a structural refactor, not a re-analysis. Every module was checked by re-running its notebook against the real, already-verified `data/processed/` files and confirming the output numbers matched exactly (e.g. the Phase 3 AUC-ROC of 0.6021, the Phase 5 Qini coefficients, and the Phase 4 signal-check results all reproduce identically).

### Running the module tests

```bash
pytest tests/test_modules.py -v
```

Covers the most important function in each module (treatment binarization, the randomization check's pass/fail behavior, encoding, the stratified split's reproducibility, model training and evaluation, both uplift methods, the signal checks, ranking combination, Qini computation and its bootstrap CI, pairwise significance, and the budget simulation) using small synthetic fixtures — not the real dataset, so these run without network access, same as `tests/test_data_pipeline.py`.
---

## Results

### Qini coefficients (95% bootstrap CI)

| Model | Qini AUC | 95% CI |
|---|---|---|
| Two-Model Approach | 0.0234 | [-0.0041, 0.0520] |
| Class Transformation | 0.0231 | [-0.0062, 0.0507] |
| Baseline (naive) | 0.0129 | [-0.0134, 0.0420] |
| Causal Forest | 0.0077 | [-0.0220, 0.0353] |

### Business impact simulation (estimated incremental visits)

| Strategy | 10% budget | 20% budget |
|---|---|---|
| Two-Model Approach | 109.9 | 193.6 |
| Baseline (naive) | 105.0 | 213.9 |
| Class Transformation | 91.1 | 192.2 |
| Causal Forest | 60.2 | 136.2 |
| Random Selection | 51.5 | 117.2 |

Every real strategy outperformed random selection in point-estimate terms at both budgets — a modest, defensible finding on its own, separate from the statistical significance question above.

---

## Interactive dashboard

`dashboard/index.html` — a working, self-contained prototype (open directly in any browser, no build step):

- **Model comparison** — click any model to isolate it and see its confidence interval
- **Budget simulator** — toggle between 10%/20% budget and watch the ranking shift
- **Persuadables quadrant** — an interactive explainer for why standard predictive models can't separate customers who respond *because of* an intervention from those who'd respond anyway

This covers 3 of 5 originally scoped dashboard screens (model comparison, budget simulator, persuadables quadrant); a customer-level explainer view and an ROI view are not yet built.

---

## Repository structure

```
Uplift-Casual-ML/
├── data/
│   ├── MANIFEST.md         # dataset name, source, schema, known limitations
│   ├── hillstrom.csv       # raw data (not committed -- created by prepare_data.py)
│   └── processed/          # cleaned splits, model outputs, saved scores
├── notebooks/               # 01 through 06, in pipeline order -- explanatory
│                             # walkthroughs that call the src/ modules below
├── src/
│   ├── data_loader.py
│   ├── prepare_data.py     # raw-data download, validation, and reporting
│   ├── preprocessing.py    # Phases 1-2: balance checks, encoding, splitting
│   ├── baseline_model.py   # Phase 3: naive classifier, ranking, signal check
│   ├── uplift_models.py    # Phase 4: the three uplift models + combination
│   ├── evaluation.py       # Phase 5: Qini metrics, bootstrap CIs, verdict
│   ├── business_simulation.py  # Phase 6: budget simulation
│   └── utils.py            # small helpers shared across the modules above
├── tests/
│   ├── test_data_pipeline.py   # src/data_loader.py, src/prepare_data.py
│   └── test_modules.py         # src/preprocessing.py through src/business_simulation.py
├── reports/                 # Qini and business impact chart images
├── dashboard/
│   └── index.html           # interactive results dashboard
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/Rashi1005/Uplift-Casual-ML.git
cd Uplift-Casual-ML
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Preparing the data

The raw dataset is not committed to this repository (see
`data/MANIFEST.md`) — download it once, right after installing
dependencies:

```bash
python src/prepare_data.py
```

This downloads the Hillstrom dataset, saves it to `data/hillstrom.csv`,
validates its schema, and prints shape, missing-value, and treatment-group
counts. `src/data_loader.py` (used throughout the notebooks) reads from
this same file as its local fallback, so the two stay in sync.

If the download fails (a known limitation — see `data/MANIFEST.md`),
`prepare_data.py` prints manual fallback instructions rather than failing
silently.

Once data preparation succeeds:

```bash
jupyter notebook notebooks/01_eda.ipynb
```

### Running the tests

```bash
pytest tests/test_data_pipeline.py -v
```

These cover successful loading, missing-column validation, missing-file
behavior, and schema consistency between `src/data_loader.py` and
`src/prepare_data.py` — using small synthetic fixtures, not the real
dataset, so they run without network access.

---

## Background

Causal inference and uplift modeling are used in production at several major technology companies, each applying the technique differently:

- **Netflix** uses observational causal inference (synthetic control methods) to measure impact when a feature is rolled out to an entire country and no control group remains.
- **Amazon (AWS)** contributed causal ML algorithms to DoWhy, used for root cause analysis — diagnosing *why* a system issue happened, not just predicting when.
- **Uber** open-sourced CausalML, purpose-built for uplift modeling and campaign targeting — the closest direct parallel to this project.

---

## References

- Athey, S., Tibshirani, J., & Wager, S. (2019). *Generalized Random Forests.* Annals of Statistics. [arxiv.org/abs/1610.01271](https://arxiv.org/abs/1610.01271)
- Gutierrez, P., & Gerardy, J.-Y. (2017). *Causal Inference and Uplift Modelling: A Review of the Literature.* PMLR. [proceedings.mlr.press/v67/gutierrez17a](http://proceedings.mlr.press/v67/gutierrez17a/gutierrez17a.pdf)
- Belbahri, M., Murua, A., Gandouet, O., & Partovi Nia, V. (2021). *Qini-based Uplift Regression.* Annals of Applied Statistics. [arxiv.org/pdf/1911.12474](https://arxiv.org/pdf/1911.12474)
- Diemert, E., Betlei, A., Renaudin, C., & Amini, M.-R. (2018). *A Large Scale Benchmark for Uplift Modeling.* AdKDD & TargetAd Workshop, KDD 2018. [papers.adkdd.org](http://papers.adkdd.org/2018/papers/adkdd18-diemert-large-scale.pdf)

---

## Limitations

- Causal conclusions depend on treatment having been genuinely randomly assigned — addressed here by using real RCT data and explicitly reporting balance checks.
- This dataset's size was insufficient to statistically distinguish between models — addressed by reporting bootstrap confidence intervals honestly rather than declaring a winner.
- Individual-level causal effects can never be directly verified (the fundamental problem of causal inference) — addressed by relying on aggregate, population-level evaluation (Qini) throughout.

## Future work

- Re-run the full pipeline on the larger Criteo Uplift Modeling dataset, where prior published work suggests methods become more statistically distinguishable at scale.
- Complete the remaining dashboard screens (customer-level explainer, ROI view).
- Extend evaluation to the `conversion` outcome using methods better suited to rare events.