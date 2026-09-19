# uplift-causal-ml

A 7-phase causal machine learning project on **uplift modeling**, using the
Hillstrom Email Marketing dataset — a public, randomized email-campaign experiment
(64,000 customers, treatment = email campaign type, outcomes = website visit / purchase
conversion / spend) — as the running example. The goal is to move beyond predicting
*who converts* to predicting *who converts because of the treatment*: estimating
individual/conditional treatment effects so that marketing spend (or any other
intervention) can be targeted at the customers it actually influences, rather than
everyone. **Phase 1** (this stage) covers data acquisition, environment setup, and
exploratory analysis, including a randomization/balance check to confirm the experiment
is safe to build on.

## Project structure

\`\`\`
uplift-causal-ml/
├── data/                # Cached copy of the Hillstrom dataset (hillstrom.csv)
├── notebooks/
│   └── 01_eda.ipynb     # Phase 1 deliverable: load, validate, explore the data
├── src/
│   └── data_loader.py   # load_hillstrom(): sklift download, with local-file fallback
├── requirements.txt
└── README.md
\`\`\`

## Setup

1. **Create and activate a virtual environment** (from inside `uplift-causal-ml/`):

   \`\`\`bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\\Scripts\\activate
   \`\`\`

2. **Install dependencies:**

   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Register the environment as a Jupyter kernel** (optional, but recommended so the
   notebook uses this venv rather than a global Python install):

   \`\`\`bash
   python -m ipykernel install --user --name uplift-causal-ml
   \`\`\`

## Running the notebook

\`\`\`bash
jupyter notebook notebooks/01_eda.ipynb
\`\`\`

Select the `uplift-causal-ml` kernel if prompted, then run all cells. The notebook loads
the dataset via `src/data_loader.py`, which:

1. First tries `sklift.datasets.fetch_hillstrom()`, the official loader from the
   `scikit-uplift` package (requires outbound internet access to its data host).
2. Falls back automatically to the cached copy at `data/hillstrom.csv` if that download
   is unavailable (e.g. on a restricted network) — so the notebook runs the same either
   way, and you don't need to do anything manually.

## What Phase 1 produces

`notebooks/01_eda.ipynb` loads the data, checks its shape/dtypes/missing values,
identifies treatment/outcome/feature columns, reports the treatment vs. control split, a
naive baseline uplift (raw outcome-rate difference), and a covariate balance check to
confirm treatment assignment looks properly randomized — closing with a markdown summary
of all of the above. This clean, validated dataset is the foundation for the causal/uplift
modeling work in later phases.