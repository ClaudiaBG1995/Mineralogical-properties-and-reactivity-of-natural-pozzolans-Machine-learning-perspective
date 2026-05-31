"""Leave-One-Group-Out Cross-Validation with bootstrap confidence intervals.

Addresses data leakage from grouped samples (PZ20-family) sharing identical
reactive composition by keeping all derived samples in the same CV fold.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from utils import softplus, inverse_softplus, sigmoid

CI_Z = 1.96  # z-score for 95% confidence interval


def build_group_vector(df, id_col, grouped_ids):
    """Assign integer group labels. Samples in grouped_ids share label 0;
    all others get unique labels."""
    missing = [sid for sid in grouped_ids if sid not in df[id_col].values]
    if missing:
        raise ValueError(
            f"GROUPED_IDS not found in dataset: {missing}. "
            f"Check that ID column matches expected sample names."
        )
    groups = np.empty(len(df), dtype=int)
    next_label = 1
    for i, sid in enumerate(df[id_col]):
        if sid in grouped_ids:
            groups[i] = 0
        else:
            groups[i] = next_label
            next_label += 1
    return groups


def _bootstrap_ci(y_true, y_pred, n_bootstrap=1000, random_state=42):
    """Compute 95% bootstrap confidence intervals for R2, RMSE, MAE."""
    rng = np.random.default_rng(random_state)
    boot_r2, boot_rmse, boot_mae = [], [], []
    n = len(y_true)

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        if np.var(yt) > 1e-12:
            boot_r2.append(r2_score(yt, yp))
        boot_rmse.append(np.sqrt(mean_squared_error(yt, yp)))
        boot_mae.append(mean_absolute_error(yt, yp))

    return {
        'r2_ci': np.percentile(boot_r2, [2.5, 97.5]),
        'rmse_ci': np.percentile(boot_rmse, [2.5, 97.5]),
        'mae_ci': np.percentile(boot_mae, [2.5, 97.5]),
    }


def logo_cv_bayesian_softplus(df, feature_cols, target_col, groups,
                              n_bootstrap=1000, random_state=42):
    """LOGO-CV for BayesianRidge with softplus link and bootstrap CI.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe containing features and target.
    feature_cols : list[str]
        Column names used as predictors.
    target_col : str
        Column name of the response variable.
    groups : array-like
        Group labels for each sample (same label = same CV fold).
    n_bootstrap : int
        Number of bootstrap iterations for confidence intervals.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    df_folds : pd.DataFrame
        Per-sample results with predictions, CI, and global metrics.
    """
    X = df[feature_cols].values
    y = df[target_col].values
    logo = LeaveOneGroupOut()
    records, r2_train_list = [], []

    n_samples = len(df)
    n_groups = len(set(groups))

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        model = BayesianRidge()
        model.fit(X_tr_sc, inverse_softplus(y_tr))

        # R2 training
        r2_tr = r2_score(y_tr, softplus(model.predict(X_tr_sc)))
        r2_train_list.append(r2_tr)

        # Posterior predictive uncertainty (delta method)
        mu_t, sigma_t = model.predict(X_te_sc, return_std=True)
        y_pred = softplus(mu_t)
        sigma_orig = sigmoid(mu_t) * sigma_t
        ci_lower = y_pred - CI_Z * sigma_orig
        ci_upper = y_pred + CI_Z * sigma_orig

        for j, tidx in enumerate(test_idx):
            records.append({
                'fold': fold,
                'sample_index': int(tidx),
                'y_true': float(y_te[j]),
                'y_pred': float(y_pred[j]),
                'ci_lower': float(ci_lower[j]),
                'ci_upper': float(ci_upper[j]),
                'std': float(sigma_orig[j]),
                'r2_train': float(r2_tr),
            })

    df_f = pd.DataFrame(records)

    # Global metrics
    y_true_all = df_f['y_true'].values
    y_pred_all = df_f['y_pred'].values

    avg_r2_tr = np.mean(r2_train_list)
    r2 = r2_score(y_true_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    mae = mean_absolute_error(y_true_all, y_pred_all)

    # Bootstrap CI
    ci = _bootstrap_ci(y_true_all, y_pred_all, n_bootstrap, random_state)

    # Bayesian CI coverage
    coverage = np.mean(
        (df_f['y_true'] >= df_f['ci_lower']) &
        (df_f['y_true'] <= df_f['ci_upper'])
    )
    avg_ci_width = (df_f['ci_upper'] - df_f['ci_lower']).mean()

    print(f'BayesianRidge LOGO-CV (N={n_samples}, groups={n_groups}):')
    print(f'  R2 Training (avg) = {avg_r2_tr:.4f}')
    print(f'  R2 LOGO-CV        = {r2:.4f} [{ci["r2_ci"][0]:.4f}, {ci["r2_ci"][1]:.4f}]')
    print(f'  RMSE              = {rmse:.2f} [{ci["rmse_ci"][0]:.2f}, {ci["rmse_ci"][1]:.2f}]')
    print(f'  MAE               = {mae:.2f} [{ci["mae_ci"][0]:.2f}, {ci["mae_ci"][1]:.2f}]')
    print(f'  95% CI coverage   = {coverage:.2%} | Avg CI width = {avg_ci_width:.2f}')

    # Store metrics in dataframe
    df_f['avg_r2_train'] = avg_r2_tr
    df_f['logo_cv_r2'] = r2
    df_f['logo_cv_rmse'] = rmse
    df_f['logo_cv_mae'] = mae
    df_f['bootstrap_r2_ci_lower'] = ci['r2_ci'][0]
    df_f['bootstrap_r2_ci_upper'] = ci['r2_ci'][1]
    df_f['bootstrap_rmse_ci_lower'] = ci['rmse_ci'][0]
    df_f['bootstrap_rmse_ci_upper'] = ci['rmse_ci'][1]
    df_f['bootstrap_mae_ci_lower'] = ci['mae_ci'][0]
    df_f['bootstrap_mae_ci_upper'] = ci['mae_ci'][1]
    df_f['bayesian_ci_coverage'] = coverage
    df_f['avg_bayesian_ci_width'] = avg_ci_width
    df_f['n_bootstrap'] = n_bootstrap

    return df_f


def logo_cv_random_forest(df, feature_cols, target_col, groups,
                          n_estimators=100, max_depth=3,
                          random_state=42, n_bootstrap=1000):
    """LOGO-CV for RandomForest with bootstrap CI.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe containing features and target.
    feature_cols : list[str]
        Column names used as predictors.
    target_col : str
        Column name of the response variable.
    groups : array-like
        Group labels for each sample (same label = same CV fold).
    n_estimators : int
        Number of trees in the forest.
    max_depth : int
        Maximum depth of each tree.
    random_state : int
        Seed for reproducibility.
    n_bootstrap : int
        Number of bootstrap iterations for confidence intervals.

    Returns
    -------
    df_folds : pd.DataFrame
        Per-sample results with predictions and global metrics.
    """
    X = df[feature_cols].values
    y = df[target_col].values
    logo = LeaveOneGroupOut()
    records, r2_train_list = [], []

    n_samples = len(df)
    n_groups = len(set(groups))

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_te_sc = scaler.transform(X_te)

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        rf.fit(X_tr_sc, y_tr)

        # R2 training
        r2_tr = r2_score(y_tr, rf.predict(X_tr_sc))
        r2_train_list.append(r2_tr)

        y_pred = rf.predict(X_te_sc)

        for j, tidx in enumerate(test_idx):
            records.append({
                'fold': fold,
                'sample_index': int(tidx),
                'y_true': float(y_te[j]),
                'y_pred': float(y_pred[j]),
                'r2_train': float(r2_tr),
            })

    df_f = pd.DataFrame(records)

    # Global metrics
    y_true_all = df_f['y_true'].values
    y_pred_all = df_f['y_pred'].values

    avg_r2_tr = np.mean(r2_train_list)
    r2 = r2_score(y_true_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    mae = mean_absolute_error(y_true_all, y_pred_all)

    # Bootstrap CI
    ci = _bootstrap_ci(y_true_all, y_pred_all, n_bootstrap, random_state)

    print(f'RandomForest LOGO-CV (N={n_samples}, groups={n_groups}):')
    print(f'  R2 Training (avg) = {avg_r2_tr:.4f}')
    print(f'  R2 LOGO-CV        = {r2:.4f} [{ci["r2_ci"][0]:.4f}, {ci["r2_ci"][1]:.4f}]')
    print(f'  RMSE              = {rmse:.2f} [{ci["rmse_ci"][0]:.2f}, {ci["rmse_ci"][1]:.2f}]')
    print(f'  MAE               = {mae:.2f} [{ci["mae_ci"][0]:.2f}, {ci["mae_ci"][1]:.2f}]')

    # Store metrics in dataframe
    df_f['avg_r2_train'] = avg_r2_tr
    df_f['logo_cv_r2'] = r2
    df_f['logo_cv_rmse'] = rmse
    df_f['logo_cv_mae'] = mae
    df_f['bootstrap_r2_ci_lower'] = ci['r2_ci'][0]
    df_f['bootstrap_r2_ci_upper'] = ci['r2_ci'][1]
    df_f['bootstrap_rmse_ci_lower'] = ci['rmse_ci'][0]
    df_f['bootstrap_rmse_ci_upper'] = ci['rmse_ci'][1]
    df_f['bootstrap_mae_ci_lower'] = ci['mae_ci'][0]
    df_f['bootstrap_mae_ci_upper'] = ci['mae_ci'][1]
    df_f['n_bootstrap'] = n_bootstrap

    return df_f
