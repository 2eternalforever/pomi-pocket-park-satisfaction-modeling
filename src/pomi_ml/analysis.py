# -*- coding: utf-8 -*-
"""Numerical utilities for the pocket-park satisfaction workflow."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.svm import SVR

from .config import (
    FEATURE_LABELS,
    FEATURES,
    MODEL_LABELS,
    N_BOOTSTRAP,
    RANDOM_STATE,
    RF_MIN_SAMPLES_LEAF,
    RF_TREES,
    TARGET,
    TRAIN_FRACTION,
)


@dataclass(frozen=True)
class SplitData:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    x_train_std: np.ndarray
    x_test_std: np.ndarray


def corr_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return float(corr * corr)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def read_survey_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    missing = [col for col in [TARGET, *FEATURES] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


def split_like_matlab(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Approximate MATLAB rng(42); randsample(n_total, floor(0.7*n_total))."""
    n_total = len(y)
    n_train = int(np.floor(TRAIN_FRACTION * n_total))
    rng = np.random.RandomState(RANDOM_STATE)
    train_idx = rng.choice(np.arange(n_total), size=n_train, replace=False)
    mask = np.zeros(n_total, dtype=bool)
    mask[train_idx] = True
    return x[mask], x[~mask], y[mask], y[~mask]


def standardize(train_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0, ddof=1)
    std[std == 0] = 1
    return (train_x - mean) / std, (test_x - mean) / std


def prepare_split(df: pd.DataFrame) -> SplitData:
    x = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    x_train, x_test, y_train, y_test = split_like_matlab(x, y)
    x_train_std, x_test_std = standardize(x_train, x_test)
    return SplitData(x_train, x_test, y_train, y_test, x_train_std, x_test_std)


def make_models(random_state: int = RANDOM_STATE, rf_trees: int = RF_TREES) -> dict[str, object]:
    gpr_kernel = (
        ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
        * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    )
    return {
        "Linear Reg": LinearRegression(),
        "SVR": SVR(kernel="rbf", gamma="scale"),
        "Random Forest": RandomForestRegressor(
            n_estimators=rf_trees,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            random_state=random_state,
            n_jobs=1,
            bootstrap=True,
            oob_score=True,
        ),
        "GPR": GaussianProcessRegressor(
            kernel=gpr_kernel,
            normalize_y=False,
            random_state=random_state,
            n_restarts_optimizer=2,
        ),
    }


def ols_fit(train_x: np.ndarray, train_y: np.ndarray) -> dict[str, np.ndarray | float]:
    design = np.column_stack([np.ones(len(train_y)), train_x])
    beta = np.linalg.lstsq(design, train_y, rcond=None)[0]
    fitted = design @ beta
    residuals = train_y - fitted
    n_samples, n_params = design.shape
    sse = np.sum(residuals**2)
    sst = np.sum((train_y - train_y.mean()) ** 2)
    sigma2 = sse / (n_samples - n_params)
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(covariance))
    t_values = beta / se
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_values), df=n_samples - n_params))
    r2 = 1 - sse / sst
    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_params)
    return {
        "beta": beta,
        "se": se,
        "t_values": t_values,
        "p_values": p_values,
        "r2": r2,
        "adjusted_r2": adjusted_r2,
    }


def linear_regression_outputs(split: SplitData) -> dict[str, object]:
    fit = ols_fit(split.x_train_std, split.y_train)
    beta = np.asarray(fit["beta"], dtype=float)
    p_values = np.asarray(fit["p_values"], dtype=float)
    y_pred = beta[0] + split.x_test_std @ beta[1:]
    residuals = split.y_test - y_pred
    coefficient_table = pd.DataFrame(
        {
            "variable": FEATURE_LABELS,
            "coefficient": beta[1:],
            "abs_coefficient": np.abs(beta[1:]),
            "std_error": np.asarray(fit["se"], dtype=float)[1:],
            "t_value": np.asarray(fit["t_values"], dtype=float)[1:],
            "p_value": p_values[1:],
        }
    )
    metrics = pd.DataFrame(
        [
            {
                "adjusted_r2_train": fit["adjusted_r2"],
                "test_mae": mean_absolute_error(split.y_test, y_pred),
                "test_rmse": rmse(split.y_test, y_pred),
                "test_r2_correlation": corr_r2(split.y_test, y_pred),
                "test_mape_percent": np.mean(np.abs((split.y_test - y_pred) / split.y_test)) * 100,
                "residual_mean": residuals.mean(),
                "residual_sd": residuals.std(ddof=1),
            }
        ]
    )
    return {
        "fit": fit,
        "prediction": y_pred,
        "residuals": residuals,
        "coefficient_table": coefficient_table,
        "metrics": metrics,
    }


def metric_row(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | str]:
    return {
        "model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2_correlation": corr_r2(y_true, y_pred),
        "MAPE_percent": np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    }


def fit_predict_all(split: SplitData) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    predictions: dict[str, np.ndarray] = {}
    fitted: dict[str, object] = {}
    for name, model in make_models().items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(split.x_train_std, split.y_train)
        predictions[name] = np.asarray(model.predict(split.x_test_std), dtype=float)
        fitted[name] = model
    metrics = pd.DataFrame([metric_row(name, split.y_test, predictions[name]) for name in MODEL_LABELS])
    return metrics, predictions, fitted


def baseline_metrics(split: SplitData) -> pd.DataFrame:
    baselines = {
        "Training mean baseline": np.full_like(split.y_test, split.y_train.mean(), dtype=float),
        "Training median baseline": np.full_like(split.y_test, np.median(split.y_train), dtype=float),
    }
    rows = []
    for name, pred in baselines.items():
        rows.append(
            {
                "model": name,
                "MAE": mean_absolute_error(split.y_test, pred),
                "RMSE": rmse(split.y_test, pred),
                "R2_score": 1
                - np.sum((split.y_test - pred) ** 2)
                / np.sum((split.y_test - split.y_test.mean()) ** 2),
            }
        )
    return pd.DataFrame(rows)


def compute_learning_curves(split: SplitData) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    train_sizes = np.round(np.linspace(50, len(split.y_train), 7)).astype(int)
    rows = []
    for size in train_sizes:
        subset = rng.choice(np.arange(len(split.y_train)), size=size, replace=False)
        x_subset = split.x_train[subset]
        y_subset = split.y_train[subset]
        x_subset_scaled, x_test_scaled = standardize(x_subset, split.x_test)
        row = {"training_set_size": size}
        for name, model in make_models(rf_trees=50).items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_subset_scaled, y_subset)
            row[name] = rmse(split.y_test, np.asarray(model.predict(x_test_scaled), dtype=float))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_cv_mae(split: SplitData) -> pd.DataFrame:
    rows = []
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(split.x_train), start=1):
        x_fold_train = split.x_train[train_idx]
        x_fold_valid = split.x_train[valid_idx]
        y_fold_train = split.y_train[train_idx]
        y_fold_valid = split.y_train[valid_idx]
        x_fold_train_scaled, x_fold_valid_scaled = standardize(x_fold_train, x_fold_valid)
        row = {"fold": fold}
        for name, model in make_models(rf_trees=50).items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(x_fold_train_scaled, y_fold_train)
            prediction = np.asarray(model.predict(x_fold_valid_scaled), dtype=float)
            row[name] = mean_absolute_error(y_fold_valid, prediction)
        rows.append(row)
    return pd.DataFrame(rows)


def oob_permutation_importance(
    model: RandomForestRegressor,
    train_x_scaled: np.ndarray,
    train_y: np.ndarray,
    n_repeats: int = 50,
) -> pd.DataFrame:
    n_samples, n_features = train_x_scaled.shape
    samples = model.estimators_samples_

    def oob_prediction(feature_idx: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
        pred_sum = np.zeros(n_samples)
        pred_count = np.zeros(n_samples)
        for tree, sample_idx in zip(model.estimators_, samples):
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[sample_idx] = False
            if not np.any(oob_mask):
                continue
            x_oob = train_x_scaled[oob_mask].copy()
            if feature_idx is not None and rng is not None:
                x_oob[:, feature_idx] = rng.permutation(x_oob[:, feature_idx])
            pred_sum[oob_mask] += tree.predict(x_oob)
            pred_count[oob_mask] += 1
        prediction = np.full(n_samples, np.nan)
        valid = pred_count > 0
        prediction[valid] = pred_sum[valid] / pred_count[valid]
        return prediction

    baseline = oob_prediction()
    valid = ~np.isnan(baseline)
    baseline_mae = mean_absolute_error(train_y[valid], baseline[valid])
    rng = np.random.default_rng(RANDOM_STATE)
    importances = np.zeros(n_features)
    for feature_idx in range(n_features):
        deltas = []
        for _ in range(n_repeats):
            prediction = oob_prediction(feature_idx, rng)
            valid = ~np.isnan(prediction)
            deltas.append(mean_absolute_error(train_y[valid], prediction[valid]) - baseline_mae)
        importances[feature_idx] = np.mean(deltas)
    importances = np.maximum(importances, 0)
    if importances.sum() == 0:
        importances = np.full(n_features, 1 / n_features)
    else:
        importances = importances / importances.sum()
    return pd.DataFrame({"feature": FEATURE_LABELS, "relative_importance": importances})


def acf_values(errors: np.ndarray, max_lags: int) -> np.ndarray:
    centered = errors - errors.mean()
    denom = np.sum(centered**2)
    acf = np.empty(max_lags + 1)
    acf[0] = 1.0
    for lag in range(1, max_lags + 1):
        acf[lag] = np.sum(centered[:-lag] * centered[lag:]) / denom
    return acf


def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    n_samples = len(y_true)
    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(np.arange(n_samples), size=n_samples, replace=True)
        y_boot = y_true[idx]
        pred_boot = y_pred[idx]
        rows.append(
            {
                "MAE": mean_absolute_error(y_boot, pred_boot),
                "RMSE": rmse(y_boot, pred_boot),
                "R2_correlation": corr_r2(y_boot, pred_boot),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_summary(boot: pd.DataFrame) -> pd.DataFrame:
    values = np.percentile(boot[["MAE", "RMSE", "R2_correlation"]].to_numpy(), [2.5, 25, 50, 75, 97.5], axis=0)
    return pd.DataFrame(
        values,
        index=["CI_2.5", "IQR_25", "Median", "IQR_75", "CI_97.5"],
        columns=["MAE", "RMSE", "R2_correlation"],
    )


def error_summary(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    errors = y_true - y_pred
    std_error = errors.std(ddof=1)
    return pd.DataFrame(
        [
            {
                "selected_model": model_name,
                "mean_error": errors.mean(),
                "std_error": std_error,
                "median_error": np.median(errors),
                "iqr_error": stats.iqr(errors),
                "error_ci_2.5": np.percentile(errors, 2.5),
                "error_ci_97.5": np.percentile(errors, 97.5),
                "pi_half_width": 1.96 * std_error,
                "empirical_coverage_percent": np.mean(np.abs(errors) <= 1.96 * std_error) * 100,
            }
        ]
    )
