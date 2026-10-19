"""Check published numerical claims against independently regenerated outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
T = ROOT/'outputs'/'tables'
checks = []

def check(label, actual, expected, atol=5e-5):
    ok = bool(np.isclose(actual, expected, atol=atol, rtol=0))
    checks.append(dict(claim=label, actual=float(actual), expected=float(expected), passed=ok))
    if not ok:
        raise AssertionError(f'{label}: {actual} != {expected}')

h = pd.read_csv(T/'holdout_metrics.csv').set_index('model')
for name, values in {
    'Linear regression': [.1902, .2419, .9267],
    'SVR': [.2325, .4627, .7319],
    'Random forest': [.1772, .3092, .8802],
    'GPR': [.1420, .3090, .8804],
    'Six-attribute mean': [.2171, .3675, .8308],
}.items():
    for key, expected in zip(['MAE', 'RMSE', 'R2'], values):
        check(f'{name} holdout {key}', h.loc[name, key], expected)
cv = pd.read_csv(T/'training_cv.csv').groupby('model').MAE.mean()
assert cv.idxmin() == 'GPR'
check('GPR training CV MAE', cv['GPR'], .1830)
outer = pd.read_csv(T/'nested_cv_folds.csv')
assert outer.loc[outer.model.eq('Nested selected procedure'), 'selected'].eq('GPR').all()
for name, expected in [('Nested selected procedure', .1536), ('Linear regression', .1907), ('Six-attribute mean', .2227)]:
    check(f'{name} outer-fold mean MAE', outer.loc[outer.model.eq(name), 'MAE'].mean(), expected)
ordinal = pd.read_csv(T/'ordinal_sensitivity.csv').set_index('model')
check('Ordered-logit accuracy', ordinal.loc['Ordered logit (median)', 'exact_accuracy'], .9541)
check('Ordered-logit quadratic kappa', ordinal.loc['Ordered logit (median)', 'quadratic_weighted_kappa'], .9701)
r = json.loads((T/'verified_results.json').read_text())
for key, bounds in {'MAE': [.0954, .2004], 'RMSE': [.1907, .4154], 'R2': [.7684, .9514]}.items():
    for i, expected in enumerate(bounds):
        check(f'Bootstrap {key} bound {i+1}', r['bootstrap_CI'][key][i], expected)
f = json.loads((T/'R3_C6_audit.json').read_text())
check('KMO', f['KMO'], .971, atol=.0005)
check('Bartlett chi-square', f['Bartlett_chi2'], 3511.898, atol=.0005)
check('ML factor percent', f['FA1_total_variance_percent'], 51.3, atol=.05)
assert f['PA_pca_retained'] == f['PA_SMC_reduced_retained'] == 1
c = json.loads((T/'cluster_and_record_audit.json').read_text())
assert [g['n'] for g in c['clusters']] == [109, 252]
check('Lower cluster satisfaction SD', c['clusters'][0]['SD'], .65, atol=.005)
check('Higher cluster satisfaction SD', c['clusters'][1]['SD'], .46, atol=.005)
for row, expected in zip(c['tests'], [.842, .727, .113, .739, .815, .557, .673]):
    check('Cluster test '+row['variable']+' p', row['p'], expected, atol=.0005)
importance = pd.read_csv(T/'rf_importance_verified.csv').sort_values('relative_importance', ascending=False)
assert importance.feature.head(5).tolist() == ['Fitness Equipment', 'WiFi', 'Accessibility', 'Safety', 'Comfort']
pca = pd.read_csv(T/'pca_variance.csv').set_index('component')
check('Eight-component cumulative percent', pca.loc[8, 'cumulative_percent'], 79.9, atol=.05)
check('Nine-component cumulative percent', pca.loc[9, 'cumulative_percent'], 82.7, atol=.05)
result = dict(status='passed', checks=checks,
              additional_assertions=['training-CV GPR selection', 'all five nested selections',
                                     'one-factor retention', 'cluster counts', 'top-five RF ranking'])
(ROOT/'validation/numerical_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(f'Passed {len(checks)} numerical checks and 5 structural assertions.')
