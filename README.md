# Regulatory-Constrained Fairness Score (RCFS) for Credit Scoring

Replication code for the paper *"A Regulatory-Constrained Fairness Score for
Credit Scoring: Optimizing Compliance and Performance as a Single Objective."*

This repository accompanies an anonymized submission and contains all code
required to reproduce the experiments, tables, and figures. **Datasets are
not redistributed**; the notebooks download or expect the original sources
as described below.

---

## Overview

AI-based credit scoring is increasingly governed not by a single fairness
target but by a *conjunction* of regulatory obligations — non-discrimination,
explainability, and contestability — that must hold together (e.g., the EU AI
Act's high-risk regime and Korea's AI Basic Act, effective January 2026).

The **Regulatory-Constrained Fairness Score (RCFS)** is a continuous,
threshold-free score that:

- expresses **performance**, **fairness**, and **explainability** as
  satisfaction components on a common `[0, 1]` scale (higher is better);
- combines the *regulatory* axes (fairness, explainability) through a
  **harmonic mean**, so that a near-failure on any axis collapses the
  aggregate (non-compensatory conjunction);
- balances the resulting compliance score against performance through a
  single parameter **γ**, making the performance–compliance trade-off an
  explicit policy choice;
- is designed to align with a **compliance-constrained loss**, so that
  maximizing the score corresponds to minimizing a penalized training
  objective.

Across three credit datasets and pre-/in-/post-processing mitigation
techniques, the study finds that **no single mitigation technique is
universally best** — the optimal choice depends on the predictive signal
strength of the data, and the ranking transitions with γ. Adversarial
debiasing helps only where signal is sufficient; on weak-signal data it can
degrade the predictor enough to undermine the fairness it targets. RCFS is
therefore best read as a **multi-axis diagnostic**, together with its
components, rather than as a single opaque number.

---

## The RCFS score

Component satisfaction scores (each in `[0, 1]`, higher is better):

- `c_perf  = max(0, (AUC − 0.5) / 0.5)`
- `c_fair  = 0.5 · min(1, DI) + 0.5 · (1 − |SPD|)`
- `c_expl  = min(1, k* / n_eff)`, where `n_eff = exp(H)` and `H` is the
  entropy of the normalized SHAP-importance distribution
- `c_rec`  (contestability / feasible recourse) — defined but **disabled by
  default**

Aggregation:

- `Compliance = (Σ w_k) / (Σ w_k / c_k)`   (weighted harmonic mean of the
  regulatory axes)
- `RCFS(γ)   = (1 − γ) · c_perf + γ · Compliance`

`γ = 0` recovers pure performance; `γ = 1` pure compliance. Unless otherwise
noted, results are reported at `γ = 0.5`.

---

## Datasets

Datasets are **not** included in this repository. UCI files are downloaded
automatically by `01_data_loading`; the Kaggle files must be placed in
`data/` manually.

| Dataset | Source | Sensitive attributes | Target |
|---|---|---|---|
| German Credit (Statlog) | UCI (auto-download) | sex, age group | bad credit = 1 |
| Taiwan Credit Card Default | UCI (auto-download) | sex, education | default next month |
| Credit Card Approval | Kaggle (manual) | sex | 30dpd (primary), 60dpd (alt) |

**Manual step (Credit Card Approval):** download `application_record.csv` and
`credit_record.csv` from the Kaggle "Credit Card Approval Prediction" dataset
and place them in `data/`. The 30+ days-past-due label is primary
(positive rate ≈ 0.118); the 60+ days variant (≈ 0.017) is used for threshold
sensitivity checks.

---

## Repository structure

```
.
├── notebooks/
│   ├── 01_data_loading.ipynb      # load 3 datasets, derive targets & sensitive attrs
│   ├── 02_preprocessing.ipynb     # impute / encode / scale, stratified 75/25 split
│   ├── 03_baseline_models.ipynb   # LogReg, XGBoost, LightGBM, CatBoost baselines
│   ├── 04_rcfs.ipynb              # RCFS: technique comparison + γ sweep
│   ├── 05_sensitivity.ipynb       # sensitivity studies A–D (incl. bootstrap CIs)
│   ├── 06_visualization.ipynb     # publication figures (grayscale, 600 dpi)
│   └── 07_adversarial.ipynb       # in-processing: adversarial debiasing, λ sweep
├── data/
│   ├── processed/                 # generated splits & predictions (created by 01–03)
│   └── (raw dataset files live here; not redistributed)
└── results/
    ├── tables/                    # CSV outputs consumed by the manuscript
    └── figures/                   # PNG + PDF figures
```

---

## Reproducing the results

Run the notebooks in order; each persists intermediate artifacts the next one
consumes.

```
01_data_loading      →  data/processed/*.parquet
02_preprocessing     →  *_train.parquet / *_test.parquet, preprocess_meta.json
03_baseline_models   →  results/tables/03_baseline_metrics.csv, predictions/
04_rcfs              →  results/tables/04_rcfs_by_technique.csv,
                        results/tables/04_rcfs_gamma_sweep.csv
05_sensitivity       →  results/tables/05_sensitivity_bootstrap.csv
07_adversarial       →  results/tables/07_adversarial_lambda_sweep.csv
06_visualization     →  results/figures/fig1..fig6 (PNG + PDF)
```

Note that `06_visualization` reads the table outputs of `04`, `05`, and `07`,
so run it last.

### Result tables → manuscript

| CSV | Used by |
|---|---|
| `04_rcfs_by_technique.csv`    | Table (RCFS by technique), Fig. 1–2 |
| `04_rcfs_gamma_sweep.csv`     | γ-sweep table (appendix), Fig. 3 |
| `05_sensitivity_bootstrap.csv`| Bootstrap CI table (appendix), Fig. 4 |
| `07_adversarial_lambda_sweep.csv` | Adversarial table (appendix), Fig. 5–6 |

### Figures

| File | Content |
|---|---|
| `fig1_rcfs_by_technique` | RCFS per dataset × technique |
| `fig2_component_scores`  | Baseline component scores (c_perf / c_fair / c_expl) |
| `fig3_gamma_curves`      | RCFS vs γ, one panel per dataset |
| `fig4_bootstrap_ci`      | RCFS with bootstrap 95% CIs |
| `fig5_adversarial_lambda`| RCFS vs adversarial strength λ |
| `fig6_adversarial_tradeoff` | Fairness vs performance along the λ sweep |

The schematic figures in the manuscript (framework overview, component
construction, harmonic-vs-arithmetic, mitigation pipeline, loss connection,
graphical abstract) are authored separately and are not produced by these
notebooks.

---

## Methods summary

- **Base models:** Logistic Regression, XGBoost, LightGBM, CatBoost.
- **Imbalance:** class weights / `scale_pos_weight` (no resampling);
  prevalence-based decision thresholds.
- **Mitigation techniques:**
  - *Pre-processing* — Reweighing (portable Kamiran–Calders reimplementation,
    not AIF360-dependent)
  - *Post-processing* — Reject-option (group-specific thresholds)
  - *In-processing* — Adversarial Debiasing (PyTorch gradient reversal,
    neural-network family; compared **within family** against a no-adversary
    neural baseline)
- **Explainability:** SHAP (TreeExplainer / KernelExplainer).
- **Robustness:** sensitivity to `k*` (Study A), harmonic weights (Study B),
  γ (Study C), and bootstrap 95% CIs (Study D).

---

## Environment

- Python 3.10
- Core: `numpy`, `pandas`, `scikit-learn`
- Models: `xgboost`, `lightgbm`, `catboost`
- In-processing: `torch` (PyTorch)
- Explainability: `shap`
- Figures: `matplotlib`, `seaborn`
- Data I/O: `pyarrow` (parquet), `xlrd >= 2.0.1` (Taiwan `.xls`)

All experiments use a fixed random seed (`42`) and a stratified 75/25
train/test split.

```bash
pip install numpy pandas scikit-learn xgboost lightgbm catboost \
            torch shap matplotlib seaborn pyarrow "xlrd>=2.0.1"
```

---

## Notes on reproducibility

- The stratified split and seed are fixed, so table values are deterministic
  up to backend/library-version differences in the gradient-boosting and
  PyTorch implementations.
- Bootstrap confidence intervals use 200 resamples.
- Because datasets are not redistributed, exact reproduction requires
  obtaining the original files as described above.

---

## License

Code is released for review under the terms indicated in this repository.
Datasets remain under their original licenses and are not redistributed here.
