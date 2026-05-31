# Mineralogical properties and reactivity of natural pozzolans: A machine learning perspective

Bayesian Ridge and Random Forest regression models with **Leave-One-Group-Out Cross-Validation** (LOGO-CV) to predict the reactivity of natural pozzolans.

LOGO-CV addresses data leakage from samples PZ30–PZ33 (derived from PZ20 by grinding), which share identical reactive composition and differ only in particle size (D50).

## Project Structure

```
PaperNPz/
├── config.py                         # Paths, feature selection, parameters
├── utils.py                          # Softplus/sigmoid link functions
├── models/
│   ├── bayesian_loocv.py             # BayesianRidge + LOOCV + softplus
│   ├── rf_loocv.py                   # RandomForest + LOOCV
│   └── logo_cv.py                    # LOGO-CV: BayesianRidge & RF + bootstrap CI
├── notebooks/
│   └── Bayesian_LOOCV.ipynb          # Main analysis notebook
├── data/                             # Place the Excel dataset here
├── output/                           # Generated results (git-ignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Place the dataset file in the `data/` folder:
```
data/Natural pozzolans 34 (data for the model)-3.xlsx
```

## Usage

### From notebook (recommended)
Open `notebooks/Bayesian_LOOCV.ipynb` and run all cells.

### From Google Colab
Upload the project to Google Drive and open the notebook in Colab. The environment is detected automatically.

## Feature Selection

Features are configured in `config.py`. Comment or uncomment features depending on the target variable being predicted:

```python
FEATURE_COLUMNS = [
    # 'SSA',
    'D50',
    'Reactive Al2O3 (%)',
    'Reactive SiO2 (%)',
    # ...
]
```

## Grouped Cross-Validation

Samples PZ20, PZ30, PZ31, PZ32, PZ33 share the same parent material and are grouped into a single CV fold. This is configured via `GROUPED_IDS` in `config.py`.

## Models

- **BayesianRidge + Softplus link**: Ensures positive predictions by fitting in transformed space via `inverse_softplus(y)` and mapping back with `softplus(pred)`. Provides posterior predictive uncertainty via the delta method.
- **RandomForest**: Baseline comparison with configurable `n_estimators` and `max_depth`.

Both models are evaluated using LOGO-CV with:
- R² Training (avg)
- R² LOGO-CV with 95% bootstrap confidence intervals
- RMSE with 95% bootstrap confidence intervals
- MAE with 95% bootstrap confidence intervals

## Sensitivity Analysis

Three scenarios are compared to demonstrate robustness:
- **(A)** 29 samples + LOGO-CV (recommended)
- **(B)** 29 samples + naive LOOCV (for comparison)
- **(C)** 25 samples + LOOCV (excluding PZ30–PZ33)
