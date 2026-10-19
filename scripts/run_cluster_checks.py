"""Reproduce descriptive clustering tests and the recommendation-item audit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'/'tables'
OUT.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_excel(ROOT/'data'/'survey_data.xlsx')
    cols = [c for c in df if c.startswith(('Satisfaction_', 'Facility_', 'Impact_')) and c != 'Satisfaction_Overall']
    x = df[cols].to_numpy(float)
    z = (x-x.mean(0))/x.std(0, ddof=1)
    raw = KMeans(n_clusters=2, n_init=20, max_iter=500, random_state=42).fit_predict(z)
    low = int(df.assign(cluster=raw).groupby('cluster').Satisfaction_Overall.mean().idxmin())
    labels = np.where(raw == low, 0, 1)
    sil = silhouette_samples(z, labels, metric='sqeuclidean')
    summaries = []
    for k in [0, 1]:
        y = df.loc[labels == k, 'Satisfaction_Overall']
        summaries.append(dict(cluster=k+1, n=int(len(y)), share=float(len(y)/len(df)),
                              mean=float(y.mean()), SD=float(y.std(ddof=1)),
                              silhouette=float(sil[labels == k].mean())))
    pd.DataFrame(summaries).to_csv(OUT/'cluster_summary.csv', index=False)
    tests = []
    for c in ['Age', 'Distance', 'Frequency', 'Duration']:
        result = stats.mannwhitneyu(df.loc[labels == 0, c], df.loc[labels == 1, c],
                                   alternative='two-sided', method='asymptotic', use_continuity=False)
        tests.append(dict(variable=c, test='Mann-Whitney U (asymptotic; tie correction; no continuity correction)',
                          statistic=float(result.statistic), p=float(result.pvalue)))
    for c in ['Gender', 'District', 'ResidenceType']:
        result = stats.chi2_contingency(pd.crosstab(labels, df[c]))
        tests.append(dict(variable=c, test='Pearson chi-square', statistic=float(result.statistic), p=float(result.pvalue)))
    pd.DataFrame(tests).to_csv(OUT/'cluster_background_tests.csv', index=False)
    y = df.Satisfaction_Overall.to_numpy(float)
    r = df.NPS_Score.to_numpy(float)
    pearson = stats.pearsonr(y, r)
    spearman = stats.spearmanr(y, r)
    assert np.isclose(stats.pearsonr(y, r-1).statistic, pearson.statistic)
    assert np.isclose(stats.spearmanr(y, r-1).statistic, spearman.statistic)
    report = dict(n=int(len(df)), cells_missing=int(df.isna().sum().sum()),
                  exact_duplicate_rows=int(df.duplicated().sum()),
                  silhouette_overall=float(sil.mean()), clusters=summaries,
                  tests=tests, recommendation_audit_only=dict(stored_min=float(r.min()), stored_max=float(r.max()),
                      pearson_r=float(pearson.statistic), pearson_p=float(pearson.pvalue),
                      spearman_rho=float(spearman.statistic), spearman_p=float(spearman.pvalue),
                      subtracting_one_changes_correlations=False, underlying_cause='unresolved'))
    (OUT/'cluster_and_record_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
