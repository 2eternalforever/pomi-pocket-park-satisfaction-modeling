"""Read-only statistical audit for Reviewer 3 Comment 6. No source edits."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.decomposition import FactorAnalysis, PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
source = ROOT/'data'/'survey_data.xlsx'

df = pd.read_excel(source)
groups = {
    'Satisfaction_6': [c for c in df if c.startswith('Satisfaction_') and c != 'Satisfaction_Overall'],
    'Facility_8': [c for c in df if c.startswith('Facility_')],
    'Impact_3': [c for c in df if c.startswith('Impact_')],
}
columns = sum(groups.values(), [])
X = df[columns].to_numpy(dtype=float)
assert X.shape == (361, 17) and np.isfinite(X).all()
n, p = X.shape
R = np.corrcoef(X, rowvar=False)
inv = np.linalg.inv(R)
partial = -inv / np.sqrt(np.outer(np.diag(inv), np.diag(inv)))
np.fill_diagonal(partial, 0)
R0 = R.copy()
np.fill_diagonal(R0, 0)
kmo = (R0**2).sum() / ((R0**2).sum() + (partial**2).sum())
bartlett = -(n - 1 - (2*p + 5)/6) * np.linalg.slogdet(R)[1]

def alpha(values):
    k = values.shape[1]
    return k/(k-1) * (1 - values.var(axis=0, ddof=1).sum()/values.sum(axis=1).var(ddof=1))

def reduced_eig(correlation):
    reduced = correlation.copy()
    np.fill_diagonal(reduced, 1 - 1/np.diag(np.linalg.inv(correlation)))
    return np.linalg.eigvalsh(reduced)[::-1]

rng = np.random.default_rng(42)
pa_pca, pa_fa = [], []
for _ in range(2000):
    shuffled = np.column_stack([rng.permutation(X[:, j]) for j in range(p)])
    null_R = np.corrcoef(shuffled, rowvar=False)
    pa_pca.append(np.linalg.eigvalsh(null_R)[::-1])
    pa_fa.append(reduced_eig(null_R))
observed = np.linalg.eigvalsh(R)[::-1]
observed_reduced = reduced_eig(R)
q95 = np.quantile(pa_pca, .95, axis=0)
q95_fa = np.quantile(pa_fa, .95, axis=0)

def leading_retained(observed_values, thresholds):
    # Retain only consecutive leading roots. Later crossings among tiny or
    # negative reduced-matrix roots do not define additional leading factors.
    failures = np.flatnonzero(observed_values <= thresholds)
    return int(failures[0]) if failures.size else len(observed_values)

Z = StandardScaler().fit_transform(X)
fa = FactorAnalysis(n_components=1, rotation=None, svd_method='lapack', tol=1e-9, max_iter=10000).fit(Z)
loading = fa.components_[0].copy()
if loading.sum() < 0:
    loading *= -1
table = pd.DataFrame({'item': columns, 'ML_loading': loading, 'communality': loading**2, 'uniqueness': fa.noise_variance_})
out = ROOT/'outputs'/'tables'
out.mkdir(parents=True, exist_ok=True)
table.to_csv(out/'R3_C6_single_factor_loadings.csv', index=False)
pd.DataFrame({'component': np.arange(1,p+1), 'observed': observed, 'permutation_p95': q95,
              'observed_SMC_reduced': observed_reduced, 'permutation_SMC_p95': q95_fa}).to_csv(out/'R3_C6_parallel_analysis.csv', index=False)
model_R = np.outer(loading, loading) + np.diag(fa.noise_variance_)
off_diag = ~np.eye(p, dtype=bool)
pca = PCA().fit(Z)
summary = {
    'source': 'data/survey_data.xlsx', 'n': n, 'p': p, 'KMO': kmo,
    'Bartlett_chi2': bartlett, 'Bartlett_df': int(p*(p-1)/2), 'Bartlett_p_less_than': 0.001,
    'alphas': {name: alpha(df[cols].to_numpy()) for name, cols in groups.items()},
    'alpha_all17': alpha(X), 'eigenvalues': observed.tolist(),
    'permutation_PA_replicates': 2000, 'random_seed': 42,
    'PA_pca_retained': leading_retained(observed, q95),
    'PA_SMC_reduced_retained': leading_retained(observed_reduced, q95_fa),
    'PA_SMC_reduced_all_crossings_not_retention': int((observed_reduced > q95_fa).sum()),
    'PC1_variance_percent': 100*pca.explained_variance_ratio_[0],
    'PC1_eigenvector_range_abs': [float(abs(pca.components_[0]).min()), float(abs(pca.components_[0]).max())],
    'PC1_correlation_loadings_range_abs': [float((abs(pca.components_[0])*np.sqrt(observed[0])).min()), float((abs(pca.components_[0])*np.sqrt(observed[0])).max())],
    'FA1_loading_minmax': [float(loading.min()), float(loading.max())],
    'FA1_total_variance_percent': 100*sum(loading**2)/p,
    'FA1_RMS_off_diagonal_residual': float(np.sqrt(np.mean((R-model_R)[off_diag]**2))),
    'FA1_iterations': fa.n_iter_,
}
(out/'R3_C6_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
print(table.to_string(index=False))
