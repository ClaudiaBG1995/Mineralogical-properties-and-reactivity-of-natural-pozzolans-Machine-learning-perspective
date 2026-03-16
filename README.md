# Pozzolan Reactivity Prediction

Bayesian Ridge and Random Forest regression models with Leave-One-Out Cross-Validation (LOOCV) for predicting natural pozzolan reactivity indicators.

## Project Structure

```
PaperNPz/
├── config.py                         # Paths, feature selection, parameters
├── utils.py                          # Softplus link function and its inverse
├── models/
│   ├── bayesian_loocv.py             # BayesianRidge + LOOCV + softplus
│   └── rf_loocv.py                   # RandomForest + LOOCV
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

## Models

- **BayesianRidge + Softplus link**: Ensures strictly positive predictions by fitting in transformed space via `inverse_softplus(y)` and mapping back with `softplus(pred)`.
- **RandomForest**: Baseline comparison with configurable `n_estimators` and `max_depth`.

Both models are evaluated using LOOCV with R², RMSE, and MAE metrics.
