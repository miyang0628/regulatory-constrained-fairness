# Regulatory-Constrained Fairness Framework for Credit Scoring

Replication code for the study *Regulatory-Constrained Fairness Framework for Credit
Scoring: A Compliance-Aware Optimization Approach.*

This repository accompanies an anonymous submission and is provided for peer review.
Author and affiliation information has been removed for double-blind review.

## Overview

The study introduces the **Regulatory-Constrained Fairness Score (RCFS)**, a compliance
score (higher = better) that fuses model performance with the satisfaction of regulatory
requirements into a single, tunable measure.

Component satisfaction scores (each in [0,1], higher = better):

- `c_perf` — discriminative adequacy: `max(0, (AUC - 0.5) / 0.5)`
- `c_fair` — fairness: `0.5*min(1, DI) + 0.5*(1 - |SPD|)`
- `c_expl` — explainability: `min(1, k_star / n_eff)`, where `n_eff = exp(entropy of |SHAP|)`
  is the effective number of contributing features (invariant to total feature count)
- `c_rec` — recourse feasibility (evaluation only; disabled by default)

The requirement scores are combined with a weighted **harmonic mean**, so a near-zero on
any single requirement collapses the compliance term ("no requirement may be sacrificed"):

```
Compliance = (sum w_k) / (sum w_k / c_k)
RCFS(gamma) = (1 - gamma) * c_perf + gamma * Compliance
```

`gamma` traces the performance-versus-compliance trade-off. The framework compares
fairness-mitigation techniques (pre / in / post-processing) by the RCFS they achieve and
connects to a regulatory-constrained training loss.

A single-regulation, threshold-free continuous score is used (cross-regulation comparison
is out of scope here).

## Datasets

The datasets are **not redistributed** here. Obtain them from the original sources and
place the files in `data/` as described below.

| Dataset                          | Source               | Expected file(s) in `data/`                   |
|----------------------------------|----------------------|-----------------------------------------------|
| German Credit (Statlog)          | UCI ML Repository    | `german.data` (downloaded automatically)      |
| Default of Credit Card Clients   | UCI ML Repository    | `default of credit card clients.xls` (auto)   |
| Credit Card Approval Prediction  | Kaggle (dataset)     | `application_record.csv`, `credit_record.csv` |

Notes:
- The two UCI files are fetched automatically by the first notebook.
- The Kaggle files require a Kaggle account and manual download.
- For the Credit Card Approval data, the binary target is derived from `credit_record.csv`
  using a 30-days-past-due default definition (a 60-days variant is also produced).

## Repository structure

```
.
├── data/                 # place raw datasets here (not tracked)
│   └── processed/        # intermediate parquet files (generated)
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_rcfs.ipynb
│   ├── 05_sensitivity.ipynb
│   ├── 06_visualization.ipynb
│   └── 07_adversarial.ipynb
└── results/
    ├── figures/          # generated figures (PNG + PDF, 600 dpi, grayscale)
    └── tables/           # generated result tables (CSV)
```

## Requirements

Python 3.10+ with:

```
numpy
pandas
scikit-learn
xgboost
lightgbm
catboost
shap
torch
seaborn
matplotlib
pyarrow
xlrd>=2.0.1
```

Install with:

```
pip install numpy pandas scikit-learn xgboost lightgbm catboost shap \
            seaborn matplotlib pyarrow "xlrd>=2.0.1"
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Reproduction

Run the notebooks in order from the `notebooks/` directory:

1. `01_data_loading.ipynb` — load datasets, derive targets, write processed parquet files
2. `02_preprocessing.ipynb` — impute, encode, scale, split
3. `03_baseline_models.ipynb` — baseline classifiers and performance metrics
4. `04_rcfs.ipynb` — RCFS by fairness-mitigation technique and the gamma sweep
5. `05_sensitivity.ipynb` — robustness checks (k*, harmonic weights, gamma, bootstrap)
6. `06_visualization.ipynb` — publication figures
7. `07_adversarial.ipynb` — in-processing (adversarial debiasing) extension and lambda sweep

A fixed random seed (42) is used throughout. Outputs are written to `results/`.

## Notes on interpretation

- Fairness-mitigation techniques improve RCFS in a consistent direction across design
  choices, but the improvement is small on datasets that are already fairly balanced;
  bootstrap confidence intervals reflect this.
- The in-processing (adversarial debiasing) models are neural networks, a different model
  family from the XGBoost-based pre/post techniques, so they are compared within the
  neural family (against a neural baseline) and reported separately.
- The optimal adversarial strength differs by dataset; on low-signal data an RCFS gain can
  coincide with a drop in the fairness component, so components should be read alongside
  the aggregate score.

## License

Provided for peer review. A license will be added upon publication.
