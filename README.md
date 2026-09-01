# POMI: Pocket-Park Satisfaction Modeling

This repository contains a reproducible Python workflow for analyzing questionnaire-based pocket-park satisfaction data.

The workflow produces:

- multivariable linear-regression estimates and diagnostics;
- holdout comparisons for linear regression, support vector regression, random forest, and Gaussian process regression;
- five-fold cross-validation results and learning curves;
- random-forest permutation importance;
- bootstrap summaries and prediction-error diagnostics;
- reusable CSV tables and high-resolution PNG graphics.

## Analysis configuration

- Target: `Satisfaction_Overall`
- Predictors: six satisfaction dimensions and five facility ratings
- Data split: 70% training and 30% holdout
- Random seed: `42`
- Candidate models: linear regression, SVR, random forest, and GPR
- Validation: five-fold cross-validation and 1,000 bootstrap resamples

## Repository structure

```text
pomi_ml_github/
  README.md
  LICENSE
  requirements.txt
  .gitignore
  PUBLIC_RELEASE_CHECKLIST.md
  data/
    README.md
  scripts/
    run_all.py
  src/
    pomi_ml/
      __init__.py
      analysis.py
      config.py
      plotting.py
  outputs/
    README.md
    figures/
    tables/
```

## Data requirements

The participant-level dataset is not included in this repository. Use only an anonymized dataset that has been approved for sharing.

By default, the workflow looks for:

```text
data/survey_data.xlsx
```

The file must contain one row per response and the following columns:

```text
Satisfaction_Overall
Satisfaction_Access
Satisfaction_Comfort
Satisfaction_Cleanliness
Satisfaction_Safety
Satisfaction_Greenery
Satisfaction_Maintenance
Facility_Seating
Facility_Fitness
Facility_Children
Facility_Shade
Facility_WiFi
```

The workflow also accepts a different input path:

```bash
python scripts/run_all.py --data "path/to/survey_data.xlsx"
```

See `data/README.md` for variable definitions and privacy checks.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the environment before running the analysis. On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

## Run the workflow

From the repository root:

```bash
python scripts/run_all.py
```

To use another output directory or resolution:

```bash
python scripts/run_all.py --out "path/to/outputs" --dpi 300
```

## Generated outputs

Graphics are written to `outputs/figures/`:

```text
regression_diagnostics.png
model_comparison.png
model_robustness.png
```

Tables are written to `outputs/tables/`:

```text
baseline_model_metrics.csv
regression_coefficients.csv
regression_metrics.csv
holdout_model_metrics.csv
cross_validation_mae.csv
learning_curves.csv
random_forest_importance.csv
bootstrap_metrics.csv
bootstrap_summary.csv
error_summary.csv
model_selection_metrics.csv
```

## Reproducibility notes

- Random seeds are fixed at `42`.
- Standardization parameters are estimated from the training data only.
- The same holdout observations are used for all candidate models.
- `R2_correlation` is the squared Pearson correlation between observed and predicted values; it is not the conventional coefficient of determination defined as `1 - SSE/SST`.
- Random-forest importance represents relative predictive contribution and does not establish causality.
- Existing outputs can be regenerated from an approved dataset with the required columns.

## Expected results for the reference dataset

| Model | MAE | RMSE | Squared correlation |
|---|---:|---:|---:|
| Linear regression | 0.1902 | 0.2419 | 0.9288 |
| SVR | 0.2325 | 0.4627 | 0.7627 |
| Random forest | 0.1772 | 0.3092 | 0.9117 |
| GPR | 0.1420 | 0.3090 | 0.8806 |

These values are provided as a reproducibility check for the included workflow.
