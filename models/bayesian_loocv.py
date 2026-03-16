"""Bayesian Ridge regression with LOOCV and softplus link function."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from utils import softplus, inverse_softplus


def loocv_bayesian_softplus(df, feature_cols, target_col):
    """Run Leave-One-Out Cross-Validation with BayesianRidge + softplus link.

    Workflow per fold:
      1. Scale features with StandardScaler (fit on train only).
      2. Transform target to unconstrained space via inverse_softplus.
      3. Fit BayesianRidge in transformed space.
      4. Predict and map back to original space via softplus (always > 0).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe containing features and target.
    feature_cols : list[str]
        Column names used as predictors.
    target_col : str
        Column name of the response variable.

    Returns
    -------
    df_folds : pd.DataFrame
        Per-fold results with columns: fold, y_true, y_pred, r2_train.
        Global metrics are appended as constant columns.
    """
    X = df[feature_cols].values
    y = df[target_col].values

    loo = LeaveOneOut()
    records = []
    r2_train_list = []

    for fold, (train_idx, test_idx) in enumerate(loo.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        # Transform target to unconstrained space
        y_train_t = inverse_softplus(y_train)

        # Fit model in transformed space
        model = BayesianRidge()
        model.fit(X_train_sc, y_train_t)

        # R² training (in original scale)
        y_pred_train = softplus(model.predict(X_train_sc))
        r2_train_fold = r2_score(y_train, y_pred_train)
        r2_train_list.append(r2_train_fold)

        # Predict and map back via softplus
        y_pred = softplus(model.predict(X_test_sc))

        records.append({
            'fold': fold,
            'y_true': float(y_test[0]),
            'y_pred': float(y_pred[0]),
            'r2_train': float(r2_train_fold),
        })

    df_folds = pd.DataFrame(records)

    # Compute LOOCV metrics in original scale
    avg_r2_train = np.mean(r2_train_list)
    r2   = r2_score(df_folds['y_true'], df_folds['y_pred'])
    rmse = np.sqrt(mean_squared_error(df_folds['y_true'], df_folds['y_pred']))
    mae  = mean_absolute_error(df_folds['y_true'], df_folds['y_pred'])

    print("BayesianRidge LOOCV (softplus link):")
    print(f"  R2 Training (avg) = {avg_r2_train:.4f}")
    print(f"  R2 LOOCV          = {r2:.4f}")
    print(f"  RMSE              = {rmse:.4f}")
    print(f"  MAE               = {mae:.4f}")

    df_folds['avg_r2_train'] = avg_r2_train
    df_folds['loocv_r2']     = r2
    df_folds['loocv_rmse']   = rmse
    df_folds['loocv_mae']    = mae

    return df_folds
