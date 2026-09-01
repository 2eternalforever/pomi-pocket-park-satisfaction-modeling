# -*- coding: utf-8 -*-
"""Plotting utilities for the pocket-park satisfaction workflow."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .analysis import acf_values
from .config import FEATURE_LABELS, MODEL_LABELS, PLOT_COLORS


def style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#8a8a8a", alpha=0.28, linewidth=0.55)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.75)
    ax.tick_params(direction="out", top=False, right=False, width=0.75, length=3.5)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.4,
            "ytick.labelsize": 8.4,
            "axes.linewidth": 0.75,
            "savefig.facecolor": "white",
        }
    )


def coefficient_colors(p_values: np.ndarray) -> list[tuple[float, float, float] | str]:
    colors: list[tuple[float, float, float] | str] = []
    for p_value in p_values:
        if p_value < 0.001:
            colors.append(tuple(PLOT_COLORS[0]))
        elif p_value < 0.01:
            colors.append(tuple(PLOT_COLORS[1]))
        elif p_value < 0.05:
            colors.append(tuple(PLOT_COLORS[2]))
        else:
            colors.append("#cfcfcf")
    return colors


def gradient_bar_colors(values: np.ndarray, offset: int = 0) -> np.ndarray:
    if np.max(values) == np.min(values):
        normalized = np.full_like(values, 0.5, dtype=float)
    else:
        normalized = (values - np.min(values)) / (np.max(values) - np.min(values))
    colors = []
    for i, value in enumerate(normalized):
        base = PLOT_COLORS[(i + offset) % len(PLOT_COLORS)]
        pale = base * 0.3 + np.array([0.7, 0.7, 0.7])
        colors.append((1 - value) * pale + value * base)
    return np.array(colors)


def add_value_labels(ax: plt.Axes, bars, fmt: str = "{:.2f}") -> None:
    ymin, ymax = ax.get_ylim()
    offset = (ymax - ymin) * 0.02
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
            color="#222222",
        )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=dpi)


def plot_regression_diagnostics(
    output_dir: Path,
    coefficients,
    linear_fit: dict[str, np.ndarray | float],
    y_test: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    dpi: int = 600,
) -> None:
    configure_matplotlib()
    beta = np.asarray(linear_fit["beta"], dtype=float)
    p_values = np.asarray(linear_fit["p_values"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(1184 / 150, 981 / 150), dpi=150, facecolor="white")

    ax = axes[0, 0]
    xpos = np.arange(len(FEATURE_LABELS))
    ax.bar(
        xpos,
        coefficients["abs_coefficient"].to_numpy(),
        color=coefficient_colors(p_values[1:]),
        edgecolor="#1f1f1f",
        linewidth=0.6,
        width=0.72,
        alpha=0.95,
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(FEATURE_LABELS, rotation=45, ha="right")
    ax.set_ylabel("Coefficient", fontweight="bold")
    ax.set_title("Regression Coefficients", fontweight="bold", pad=4)
    ax.set_ylim(0, 0.16)
    ax.set_yticks(np.arange(0, 0.161, 0.02))
    style_axes(ax)

    ax = axes[0, 1]
    ax.scatter(
        y_test,
        y_pred,
        s=26,
        facecolor=PLOT_COLORS[1],
        edgecolor=PLOT_COLORS[1] * 0.7,
        linewidth=0.5,
        alpha=0.6,
    )
    lo, hi = 0.75, 5.55
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.8, dashes=(5, 5))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks(np.arange(1, 6, 1))
    ax.set_yticks(np.arange(1, 5.6, 0.5))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Actual Satisfaction", fontweight="bold")
    ax.set_ylabel("Predicted Satisfaction", fontweight="bold")
    ax.set_title("Test Set: Predicted vs Actual", fontweight="bold", pad=4)
    style_axes(ax)

    ax = axes[1, 0]
    ax.scatter(
        y_pred,
        residuals,
        s=26,
        facecolor=PLOT_COLORS[2],
        edgecolor=PLOT_COLORS[2] * 0.7,
        linewidth=0.5,
        alpha=0.6,
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1.8, dashes=(5, 5))
    ax.set_xlim(1, 6)
    ax.set_ylim(-0.8, 1.2)
    ax.set_xticks(np.arange(1, 6.1, 1))
    ax.set_yticks(np.arange(-0.8, 1.21, 0.2))
    ax.set_xlabel("Predicted Value", fontweight="bold")
    ax.set_ylabel("Residual", fontweight="bold")
    ax.set_title("Residual Plot", fontweight="bold", pad=4)
    style_axes(ax)

    ax = axes[1, 1]
    osm, osr = stats.probplot(residuals, dist="norm", fit=False)
    slope, intercept, _ = stats.probplot(residuals, dist="norm", fit=True)[1]
    ax.scatter(osm, osr, s=18, marker="+", color="#4c4c4c", linewidth=1.3)
    x_line = np.linspace(min(osm), max(osm), 100)
    ax.plot(x_line, slope * x_line + intercept, "--", color=PLOT_COLORS[1], linewidth=1.6)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.8, 1.2)
    ax.set_xticks(np.arange(-3, 3.1, 1))
    ax.set_yticks(np.arange(-0.8, 1.21, 0.2))
    ax.set_xlabel("Theoretical Quantiles", fontweight="bold")
    ax.set_ylabel("Sample Quantiles", fontweight="bold")
    ax.set_title("Residual Q-Q Plot", fontweight="bold", pad=4)
    style_axes(ax)

    for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axes.flat):
        ax.text(-0.24, 1.12, label, transform=ax.transAxes, fontsize=15, fontweight="bold", fontfamily="serif")

    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.095, top=0.945, wspace=0.42, hspace=0.40)
    save_figure(fig, output_dir, "regression_diagnostics", dpi)
    plt.close(fig)


def _bar_panel(ax: plt.Axes, values: np.ndarray, title: str, ylabel: str, ypad: float = 0.18) -> None:
    x = np.arange(len(MODEL_LABELS))
    bars = ax.bar(
        x,
        values,
        color=gradient_bar_colors(values),
        edgecolor="#1f1f1f",
        linewidth=0.7,
        alpha=0.92,
        width=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, rotation=35, ha="right")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontsize=10.2, fontweight="bold", pad=4)
    upper = max(values) * (1 + ypad)
    lower = max(0, min(values) - (max(values) - min(values)) * 0.35)
    ax.set_ylim(lower, upper)
    add_value_labels(ax, bars)
    style_axes(ax)


def plot_model_comparison(output_dir: Path, metrics, learning_curves, cv_mae, importance, dpi: int = 600) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(12.38, 6.78), dpi=150, facecolor="white")

    _bar_panel(axes[0, 0], metrics["MAE"].to_numpy(), "Mean Absolute Error (Test Set)", "MAE")
    _bar_panel(axes[0, 1], metrics["RMSE"].to_numpy(), "Root Mean Square Error (Test Set)", "RMSE")
    _bar_panel(axes[0, 2], metrics["R2_correlation"].to_numpy(), "R\u00b2 Score (Test Set)", "R\u00b2", ypad=0.08)

    ax = axes[1, 0]
    for idx, name in enumerate(MODEL_LABELS):
        ax.plot(
            learning_curves["training_set_size"],
            learning_curves[name],
            "o-",
            color=PLOT_COLORS[idx],
            linewidth=2.2,
            markersize=6,
            markerfacecolor=PLOT_COLORS[idx],
            label=name,
        )
    ax.set_xlabel("Training Set Size", fontweight="bold")
    ax.set_ylabel("Test RMSE", fontweight="bold")
    ax.set_title("Learning Curves", fontsize=10.2, fontweight="bold", pad=4)
    ax.legend(loc="best", frameon=False, fontsize=7.4)
    style_axes(ax)

    ax = axes[1, 1]
    box = ax.boxplot(
        [cv_mae[name].to_numpy() for name in MODEL_LABELS],
        tick_labels=MODEL_LABELS,
        patch_artist=True,
        widths=0.56,
        showfliers=False,
        medianprops={"color": "#222222", "linewidth": 1.8},
        whiskerprops={"color": "#555555", "linewidth": 0.8},
        capprops={"color": "#555555", "linewidth": 0.8},
    )
    for idx, patch in enumerate(box["boxes"]):
        patch.set_facecolor(PLOT_COLORS[idx])
        patch.set_alpha(0.62)
        patch.set_edgecolor(PLOT_COLORS[idx] * 0.7)
        patch.set_linewidth(1.5)
    ax.set_xticklabels(MODEL_LABELS, rotation=35, ha="right")
    ax.set_ylabel("CV MAE (Training Set)", fontweight="bold")
    ax.set_title("5-Fold Cross-Validation MAE", fontsize=10.2, fontweight="bold", pad=4)
    style_axes(ax)

    ax = axes[1, 2]
    x_pos = np.arange(len(FEATURE_LABELS))
    bars = ax.bar(
        x_pos,
        importance["relative_importance"],
        color=gradient_bar_colors(importance["relative_importance"].to_numpy()),
        edgecolor="#1f1f1f",
        linewidth=0.65,
        alpha=0.92,
        width=0.72,
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(FEATURE_LABELS, rotation=45, ha="right")
    ax.set_ylabel("Relative Importance", fontweight="bold")
    ax.set_title("Feature Importance (Random Forest)", fontsize=10.2, fontweight="bold", pad=4)
    ax.set_ylim(0, max(importance["relative_importance"]) * 1.22)
    add_value_labels(ax, bars)
    style_axes(ax)

    for label, ax in zip(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"], axes.flat):
        ax.text(-0.23, 1.12, label, transform=ax.transAxes, fontsize=15, fontweight="bold", fontfamily="serif")

    fig.subplots_adjust(left=0.065, right=0.99, bottom=0.115, top=0.94, wspace=0.34, hspace=0.42)
    save_figure(fig, output_dir, "model_comparison", dpi)
    plt.close(fig)


def plot_model_robustness(
    output_dir: Path,
    y_test: np.ndarray,
    y_pred_best: np.ndarray,
    boot_summary,
    dpi: int = 600,
) -> None:
    configure_matplotlib()
    errors = y_test - y_pred_best
    ci_values = boot_summary.loc[["CI_2.5", "IQR_25", "Median", "IQR_75", "CI_97.5"], ["MAE", "RMSE", "R2_correlation"]].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(1335 / 150, 997 / 150), dpi=150, facecolor="white")

    ax = axes[0, 0]
    ax.hist(errors, bins=25, density=True, color=PLOT_COLORS[0], edgecolor=PLOT_COLORS[0] * 0.7, linewidth=0.55, alpha=0.82)
    x_range = np.linspace(errors.min(), errors.max(), 200)
    ax.plot(x_range, stats.norm.pdf(x_range, loc=errors.mean(), scale=errors.std(ddof=1)), color=PLOT_COLORS[1], linewidth=2.2)
    ax.set_xlabel("Prediction Error", fontweight="bold")
    ax.set_ylabel("Probability Density", fontweight="bold")
    ax.set_title("Error Distribution", fontweight="bold", pad=4)
    style_axes(ax)

    ax = axes[0, 1]
    x_pos = np.arange(1, 4)
    medians = ci_values[2, :]
    lower = medians - ci_values[0, :]
    upper = ci_values[4, :] - medians
    ax.errorbar(
        x_pos,
        medians,
        yerr=np.vstack([lower, upper]),
        fmt="o",
        linewidth=2.0,
        markersize=8.0,
        capsize=7,
        color=PLOT_COLORS[2],
        markerfacecolor=PLOT_COLORS[2],
        label="95% CI",
    )
    for idx, xpos in enumerate(x_pos):
        ax.plot([xpos, xpos], [ci_values[1, idx], ci_values[3, idx]], color=PLOT_COLORS[4], linewidth=7.0, alpha=0.9, solid_capstyle="butt", label="IQR" if idx == 0 else None)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["MAE", "RMSE", "R\u00b2"])
    ax.set_ylabel("Metric Value", fontweight="bold")
    ax.set_title("Bootstrap 95% CI & IQR", fontweight="bold", pad=4)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(name) for name in ["95% CI", "IQR"]]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="best", frameon=False, fontsize=8)
    style_axes(ax)

    ax = axes[1, 0]
    max_lags = min(20, len(errors) // 4)
    lags = np.arange(max_lags + 1)
    acf = acf_values(errors, max_lags)
    confidence = 1.96 / math.sqrt(len(errors))
    markerline, stemlines, _ = ax.stem(lags, acf, basefmt=" ")
    plt.setp(markerline, marker="o", markersize=5.2, color=PLOT_COLORS[3], markerfacecolor=PLOT_COLORS[3])
    plt.setp(stemlines, color=PLOT_COLORS[3], linewidth=1.6)
    ax.axhline(0, color="#777777", linewidth=0.75)
    ax.axhline(confidence, linestyle="--", color=PLOT_COLORS[0], linewidth=1.6, dashes=(5, 5))
    ax.axhline(-confidence, linestyle="--", color=PLOT_COLORS[0], linewidth=1.6, dashes=(5, 5))
    ax.set_xlim(-0.8, max_lags + 0.8)
    ax.set_ylim(min(-0.25, acf.min() - 0.05), 1.08)
    ax.set_xticks(np.arange(0, max_lags + 1, 5))
    ax.set_xlabel("Lag", fontweight="bold")
    ax.set_ylabel("Autocorrelation", fontweight="bold")
    ax.set_title("Error Autocorrelation", fontweight="bold", pad=4)
    style_axes(ax)

    ax = axes[1, 1]
    sort_idx = np.argsort(y_test)
    y_sorted = y_test[sort_idx]
    pred_sorted = y_pred_best[sort_idx]
    std_error = errors.std(ddof=1)
    lower_bound = pred_sorted - 1.96 * std_error
    upper_bound = pred_sorted + 1.96 * std_error
    ax.fill_between(y_sorted, lower_bound, upper_bound, color=PLOT_COLORS[4], alpha=0.22, linewidth=0, label="95% PI")
    ax.plot(y_sorted, pred_sorted, color=PLOT_COLORS[5], linewidth=1.7, label="Predictions")
    ax.scatter(y_sorted, pred_sorted, s=24, facecolor=PLOT_COLORS[2], edgecolor=PLOT_COLORS[2] * 0.75, linewidth=0.45, alpha=0.55)
    lo = min(y_test.min(), y_pred_best.min()) - 0.1
    hi = max(y_test.max(), y_pred_best.max()) + 0.1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.7, dashes=(5, 5), label="Perfect Fit")
    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(max(0.5, lower_bound.min() - 0.1), min(6.0, upper_bound.max() + 0.1))
    ax.set_xticks(np.arange(1, 6, 1))
    ax.set_xlabel("Actual Value", fontweight="bold")
    ax.set_ylabel("Predicted Value", fontweight="bold")
    ax.set_title("Prediction Intervals (95%)", fontweight="bold", pad=4)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(name) for name in ["Predictions", "95% PI", "Perfect Fit"]]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="best", frameon=False, fontsize=7.8)
    style_axes(ax)

    for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axes.flat):
        ax.text(-0.18, 1.10, label, transform=ax.transAxes, fontsize=15, fontweight="bold", fontfamily="serif")

    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.095, top=0.94, wspace=0.34, hspace=0.40)
    save_figure(fig, output_dir, "model_robustness", dpi)
    plt.close(fig)
