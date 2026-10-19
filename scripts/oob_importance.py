import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
RANDOM_STATE=42
FEATURE_LABELS=["Accessibility","Comfort","Cleanliness","Safety","Green Quality","Maintenance","Seating Facilities","Fitness Equipment","Children's Facilities","Shading Facilities","WiFi"]

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
