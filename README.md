# A Regulatory-Constrained Fairness Score for Credit Scoring

Reproducibility package for the manuscript *"A Regulatory-Constrained Fairness
Score for Credit Scoring: Optimizing Compliance and Performance as a Single
Objective"* (under review, double-blind).

> **Anonymity note.** This repository is prepared for double-blind review. It
> contains no author names, affiliations, or institutional identifiers, and the
> notebooks never print local filesystem paths. Please keep it anonymous when
> sharing.

The Regulatory-Constrained Fairness Score (RCFS) fuses predictive adequacy with a
harmonic combination of regulatory-requirement satisfaction scores (fairness and
explainability) into a single, threshold-free quantity, and exposes the
performance–compliance posture as an explicit policy choice through a balance
parameter `gamma`. The code reproduces every table and figure in the manuscript.

## Repository layout

```
.
├── data/                     # raw inputs (not redistributed) + generated splits
│   └── processed/            # train/val/test parquet + predictions (regenerated)
├── notebooks/                # one self-contained code cell per notebook
│   ├── 01_data_loading.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_rcfs.ipynb
│   ├── 05_sensitivity.ipynb
│   ├── 06_visualization.ipynb
│   └── 07_adversarial.ipynb
├── results/
│   ├── figures/              # grayscale .png and .pdf at 600 dpi (no captions)
│   └── tables/               # .csv results consumed by the manuscript
├── requirements.txt
└── README.md
```

## Data

Three public credit datasets. The raw files are **not redistributed**; place them
under `data/` before running notebook 01 (which otherwise fetches the two UCI files).

| Dataset | Source | Files expected in `data/` |
|---|---|---|
| German Credit (Statlog) | UCI ML Repository (id 144) | `german.data` |
| Taiwan Credit Card Default | UCI ML Repository (id 350) | `default of credit card clients.xls` |
| Credit Card Approval | Kaggle (application + credit records) | `application_record.csv`, `credit_record.csv` |

For every dataset the adverse event (default) is coded `target = 1`; the
**favourable** decision for an applicant is therefore `y = 0` (non-default).
Sensitive attributes and the privileged reference groups are fixed in notebook 01
and recorded in `data/processed/preprocess_meta.json`:

| Dataset | Sensitive attributes | Privileged reference |
|---|---|---|
| German Credit | sex, age group | sex = male, age ≥ 25 |
| Taiwan CC Default | sex, education | sex = male, education ∈ {graduate, university} |
| Credit Card Approval | sex | sex = male |

## How to run

```bash
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace \
    01_data_loading.ipynb 02_preprocessing.ipynb 03_baseline_models.ipynb \
    04_rcfs.ipynb 05_sensitivity.ipynb 07_adversarial.ipynb 06_visualization.ipynb
```

Run the notebooks in the numeric order above (06 last, as it reads the result
tables). Each notebook is a single code cell and writes its outputs to
`results/`. A global seed (`random_state = 42`) is fixed throughout.

## What each notebook does

1. **01_data_loading** — loads the three datasets, constructs targets, fixes
   sensitive attributes / privileged groups / favourable outcome, writes Table 1.
2. **02_preprocessing** — stratified **train / validation / test** split (60/20/20);
   imputation, scaling and one-hot vocabularies are fit on train only. All decision
   thresholds downstream are tuned on the validation split and frozen for test.
3. **03_baseline_models** — Logistic Regression, XGBoost, LightGBM, CatBoost
   (class-weighted). The classifier carried into the RCFS experiments is fixed here
   (`XGBoost`, `XGB_PARAMS`), so the baseline AUC in the RCFS tables matches Table 2.
4. **04_rcfs** — RCFS and its components. Fairness is measured on the favourable
   outcome, worst-case across all sensitive attributes; explainability uses a stated
   reason-code budget `k* = 10`; techniques are baseline, reweighing (pre),
   reject-option (post, validation-tuned), and exponentiated-gradient (in-processing,
   demographic parity). Reports disparate impact, |SPD|, effective feature count,
   the gamma sweep, and a harmonic-vs-geometric-vs-arithmetic aggregator comparison.
5. **05_sensitivity** — robustness and **statistical significance**: `k*` sweep with
   the compliance-dominant boundary, harmonic-weight sweep, gamma rank stability,
   a **paired bootstrap** of RCFS(technique) − RCFS(baseline) on the frozen split,
   and the same paired difference across **10 independent random splits**.
6. **06_visualization** — grayscale figures (png + pdf, 600 dpi, no captions).
7. **07_adversarial** — adversarial debiasing (gradient reversal, implemented in
   NumPy — no deep-learning framework needed), its lambda sweep, an explainability-
   artifact diagnosis applied to every dataset, and a **controlled signal-degradation**
   experiment that isolates the effect of predictive signal on technique ordering
   on a single dataset at a fixed sample size.

## Changes in this revision (reviewer response)

- **Three-way split**; reject-option and all thresholds tuned on validation, not test.
- **Baseline model named and reconciled**: RCFS baseline AUC equals the XGBoost row
  of the baseline-metrics table by construction.
- **Fairness corrected** to the favourable (non-default) outcome and combined across
  sensitive attributes by a worst-case rule; privileged groups and `k*` stated.
- **Significance added**: paired bootstrap differences and multi-split robustness;
  claims are credited only when the paired interval excludes zero.
- **Constrained-optimisation baseline** (exponentiated gradient) added; the
  compliance aggregator is compared against geometric and arithmetic means.
- **Adversarial section** reports component-level movements and the explainability-
  artifact diagnosis for all datasets; a controlled signal-degradation study isolates
  signal strength from dataset size/features.

## License

Code released for review under the MIT License. The datasets are governed by their
original providers' terms and are not redistributed here.
