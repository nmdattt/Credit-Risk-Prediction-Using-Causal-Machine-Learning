# Causal Machine Learning and Its Application in Credit Risk Prediction

Bachelor's thesis, Faculty of Mathematics and Computer Science (Major: Data Science), University of Science, VNU-HCM, 2026.

This repository contains the thesis report, its LaTeX source, the experiment notebooks and the datasets.

## Overview

The thesis proposes a credit risk prediction framework that combines machine learning with **Bayesian networks** so that the model is both accurate and causally interpretable. The framework runs in five stages:

| Stage | Step | Method |
|-------|------|--------|
| S_Pro | Data balancing | SMOTE-NC |
| L_Pro | Feature selection | Lasso regression |
| K_Pro | Discretisation of continuous variables | K-means |
| BN_Pro | Structure and parameter learning | Tabu Search (BIC score) + Maximum Likelihood Estimation (pgmpy) |
| In_Pro | Inference and interpretation | Sensitivity analysis, forward/backward inference, what-if scenarios, comparison with LIME/SHAP |

The thesis also covers decision threshold optimisation and Decision Curve Analysis (DCA) to assess practical value in credit-granting decisions.

## Datasets

| Dataset | Source | Samples used | Features | Default rate |
|---------|--------|--------------|----------|--------------|
| Australian Credit Approval | Australian Credit (included in this repo via Git LFS) | 690 | 14 | 55.5% |
| German Credit | German Credit (included in this repo via Git LFS) | 1,000 | 20 | 30% |
| Lending Club 2007-2014 | Lending Club public loan data (included in this repo via Git LFS) | 30,000 (subsampled from 466,345) | 46 | 30% |

Each dataset is split 60/40 into training and test sets. Lending Club was subsampled (21,000 non-default and 9,000 default loans) to keep Bayesian network structure learning tractable.

## Main results

Test-set ROC-AUC (full results, PR-AUC and thresholds are in the report):

| Dataset | Bayesian Network | Logistic Reg. | Random Forest | XGBoost | KNN | Stacking |
|---------|------------------|---------------|---------------|---------|-----|----------|
| Australian Credit | **0.9220** | 0.9052 | 0.9046 | 0.8919 | 0.8415 | 0.9055 |
| German Credit | 0.7667 | 0.7311 | 0.7583 | **0.7763** | 0.7325 | 0.7757 |
| Lending Club | 0.6910 | 0.7125 | **0.7133** | 0.7131 | 0.7031 | 0.7055 |

Takeaways:
- On small, low-dimensional data (Australian Credit) the Bayesian network matches or beats all benchmark models, with the smallest train-test gap.
- On large, high-dimensional data (Lending Club) tree ensembles are better in pure accuracy, and the Bayesian network shows signs of structural overfitting.
- Across all three datasets the Bayesian network gives explicit causal structure and posterior-probability analysis that black-box models do not provide natively.
- DCA shows a positive net benefit for the Bayesian network over the Treat All / Treat None strategies within a reasonable threshold range.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── Thesis_22280004_22280009/        # LaTeX source of the report
├── report/
│   └── report.pdf                   # compiled thesis
├── Code/
│   ├── thesis_Autralian_data.ipynb
│   ├── thesis_German_Credit.ipynb
│   └── thesis_lending_club_2007_2014.ipynb
└── Data/
    ├── australian.dat
    ├── german_credit_data.csv
    └── raw_lending_club_2007_2014.csv   # stored with Git LFS
```

Each notebook runs the full pipeline (S_Pro → L_Pro → K_Pro → BN_Pro → In_Pro) for one dataset, and the three notebooks are independent of each other.

## Getting started

**1. Clone the repository.** The Lending Club file is stored with Git LFS, so install it first:

```bash
git lfs install
git clone https://github.com/nmdattt/thesiss.git
cd thesiss
```

If you cloned without Git LFS, run `git lfs pull` to download the large file.

**2. Create an environment and install dependencies** (Python 3.10+ recommended; `pgmpy` 1.0 or newer is required for `DiscreteBayesianNetwork`):

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

**3. Run a notebook.** Start Jupyter from the repository root and open a notebook in `Code/`:

```bash
jupyter notebook
```

Notebooks read data with relative paths such as `../Data/german_credit_data.csv`, so keep the folder layout above.

## Report

The compiled thesis is in [`report/report.pdf`](report/report.pdf). The LaTeX source is in `Thesis_22280004_22280009/` and can be opened directly in Overleaf.

## My contribution

**Nguyen Minh Dat** (Student ID: 22280009)

- **Data processing (code and report):** the data preparation pipeline applied to all three datasets, covering categorical encoding, train/test splitting, class balancing with SMOTE-NC (S_Pro), Lasso feature selection with the regularisation parameter C tuned over a grid of criteria (L_Pro), and K-means discretisation of continuous variables (K_Pro).
- **Causal inference (code and report):** the In_Pro stage on the learned Bayesian networks, covering sensitivity analysis on the Markov blanket of the target, forward inference, backward inference, what-if scenario analysis, and the comparison with LIME and SHAP.
