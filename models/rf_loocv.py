"""Random Forest regression with LOOCV."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def loocv_random_forest(df, feature_cols, target_col,
                        n_estimators=100, max_depth=3, random_state=42):
    """Run Leave-One-Out Cross-Validation with RandomForest.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe containing features and target.
    feature_cols : list[str]
        Column names used as predictors.
    target_col : str
        Column name of the response variable.
    n_estimators : int
        Number of trees in the forest.
    max_depth : int
        Maximum depth of each tree.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    df_folds : pd.DataFrame
        Per-fold results with columns: fold, y_true, y_pred.
        Global metrics are appended as constant columns.
    """
    X = df[feature_cols].values
    y = df[target_col].values

    loo = LeaveOneOut()
    records = []

    for fold, (train_idx, test_idx) in enumerate(loo.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)

        # Fit Random Forest
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        rf.fit(X_train_sc, y_train)
        y_pred = rf.predict(X_test_sc)[0]

        records.append({
            'fold': fold,
            'y_true': float(y_test[0]),
            'y_pred': float(y_pred),
        })

    df_folds = pd.DataFrame(records)

    # Compute LOOCV metrics
    r2   = r2_score(df_folds['y_true'], df_folds['y_pred'])
    rmse = np.sqrt(mean_squared_error(df_folds['y_true'], df_folds['y_pred']))
    mae  = mean_absolute_error(df_folds['y_true'], df_folds['y_pred'])

    print("RandomForest LOOCV:")
    print(f"  R2   = {r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE  = {mae:.4f}")

    df_folds['loocv_r2']   = r2
    df_folds['loocv_rmse'] = rmse
    df_folds['loocv_mae']  = mae

    return df_folds
