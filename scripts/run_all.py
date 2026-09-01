# -*- coding: utf-8 -*-
"""Run the complete analysis and export reusable outputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pomi_ml.analysis import (
    baseline_metrics,
    bootstrap_metrics,
    bootstrap_summary,
    compute_cv_mae,
    compute_learning_curves,
    error_summary,
    fit_predict_all,
    linear_regression_outputs,
    oob_permutation_importance,
    prepare_split,
    read_survey_data,
)
from pomi_ml.config import MODEL_LABELS
from pomi_ml.plotting import plot_model_comparison, plot_model_robustness, plot_regression_diagnostics
from sklearn.ensemble import RandomForestRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pocket-park satisfaction modeling workflow.")
    parser.add_argument("--data", default="data/survey_data.xlsx", help="Path to the anonymized survey Excel file.")
    parser.add_argument("--out", default="outputs", help="Output directory.")
    parser.add_argument("--dpi", type=int, default=600, help="Figure export DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    output_dir = Path(args.out)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = read_survey_data(str(data_path))
    split = prepare_split(df)

    linear = linear_regression_outputs(split)
    linear["coefficient_table"].to_csv(table_dir / "regression_coefficients.csv", index=False, encoding="utf-8-sig")
    linear["metrics"].to_csv(table_dir / "regression_metrics.csv", index=False, encoding="utf-8-sig")
    plot_regression_diagnostics(
        figure_dir,
        linear["coefficient_table"],
        linear["fit"],
        split.y_test,
        linear["prediction"],
        linear["residuals"],
        dpi=args.dpi,
    )

    holdout_metrics, predictions, fitted_models = fit_predict_all(split)
    holdout_metrics.to_csv(table_dir / "holdout_model_metrics.csv", index=False, encoding="utf-8-sig")
    baseline_metrics(split).to_csv(table_dir / "baseline_model_metrics.csv", index=False, encoding="utf-8-sig")
    learning_curves = compute_learning_curves(split)
    learning_curves.to_csv(table_dir / "learning_curves.csv", index=False, encoding="utf-8-sig")
    cv_mae = compute_cv_mae(split)
    cv_mae.to_csv(table_dir / "cross_validation_mae.csv", index=False, encoding="utf-8-sig")

    rf_model = fitted_models["Random Forest"]
    if not isinstance(rf_model, RandomForestRegressor):
        raise TypeError("Unexpected random forest model type.")
    importance = oob_permutation_importance(rf_model, split.x_train_std, split.y_train)
    importance.to_csv(table_dir / "random_forest_importance.csv", index=False, encoding="utf-8-sig")
    plot_model_comparison(figure_dir, holdout_metrics, learning_curves, cv_mae, importance, dpi=args.dpi)

    best_idx = int(holdout_metrics["MAE"].to_numpy().argmin())
    best_model = str(holdout_metrics.loc[best_idx, "model"])
    y_pred_best = predictions[best_model]
    boot = bootstrap_metrics(split.y_test, y_pred_best)
    boot.to_csv(table_dir / "bootstrap_metrics.csv", index=False, encoding="utf-8-sig")
    boot_summary = bootstrap_summary(boot)
    boot_summary.to_csv(table_dir / "bootstrap_summary.csv", encoding="utf-8-sig")
    error_summary(best_model, split.y_test, y_pred_best).to_csv(
        table_dir / "error_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    holdout_metrics.to_csv(table_dir / "model_selection_metrics.csv", index=False, encoding="utf-8-sig")
    plot_model_robustness(figure_dir, split.y_test, y_pred_best, boot_summary, dpi=args.dpi)

    print("Analysis complete.")
    print(f"Selected model: {best_model}")
    print(f"Tables: {table_dir.resolve()}")
    print(f"Figures: {figure_dir.resolve()}")
    print(holdout_metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
