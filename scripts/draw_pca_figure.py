"""Redraw Figure 7 from the retained data with signed eigenvector coefficients."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'/'figures'
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_excel(ROOT/'data'/'survey_data.xlsx')
cols = [c for c in df if c.startswith(('Satisfaction_', 'Facility_', 'Impact_')) and c != 'Satisfaction_Overall']
x = df[cols].to_numpy(float)
z = (x-x.mean(0))/x.std(0, ddof=1)
pca = PCA().fit(z)
score = pca.transform(z)
coeff = pca.components_.copy()
if coeff[0].sum() < 0:
    coeff[0] *= -1
    score[:, 0] *= -1
if coeff[1, cols.index('Satisfaction_Maintenance')] < 0:
    coeff[1] *= -1
    score[:, 1] *= -1
ratio = pca.explained_variance_ratio_*100
cumulative = np.cumsum(ratio)
pd.DataFrame({'component': np.arange(1, 18), 'variance_percent': ratio,
              'cumulative_percent': cumulative}).to_csv(ROOT/'outputs/tables/pca_variance.csv', index=False)
pd.DataFrame({'item': cols, 'PC1_coefficient': coeff[0], 'PC2_coefficient': coeff[1]}).to_csv(
    ROOT/'outputs/tables/pca_coefficients.csv', index=False)
plt.rcParams.update({'font.family': 'Arial', 'font.size': 9, 'axes.titlesize': 10.5,
                     'axes.titleweight': 'bold', 'axes.labelsize': 9.5, 'axes.labelweight': 'bold',
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'savefig.facecolor': 'white'})
fig, axes = plt.subplots(2, 2, figsize=(8.22, 6.61))
a = axes[0, 0]
a.bar(np.arange(1, 11), ratio[:10], color='#d83b50', edgecolor='#8f1f31', alpha=.78)
a.plot(np.arange(1, 11), ratio[:10], 'o-', color='#f2ab62', linewidth=2, markersize=6)
a.set(xlabel='Principal Component', ylabel='Variance Explained (%)', title='Scree Plot',
      xticks=np.arange(1, 11), ylim=(0, 60))
a = axes[0, 1]
a.plot(np.arange(1, 11), cumulative[:10], 'o-', color='#76a9c6', markeredgecolor='#456d85',
       linewidth=2, markersize=6)
a.axhline(80, color='#777777', ls='--', linewidth=1.5)
a.annotate('PC8: 79.9%', xy=(8, cumulative[7]), xytext=(5.6, 86),
           arrowprops={'arrowstyle': '-', 'color': '#555555'}, fontsize=8)
a.set(xlabel='Number of Components', ylabel='Cumulative Variance Explained (%)',
      title='Cumulative Variance Explained', xticks=np.arange(1, 11), ylim=(50, 90))
a = axes[1, 0]
cmap = LinearSegmentedColormap.from_list('rating', ['#d83b50', '#cbc4df', '#76a9c6'])
points = a.scatter(score[:, 0], score[:, 1], c=df.Satisfaction_Overall, cmap=cmap,
                   vmin=1, vmax=5, s=15, alpha=.66, edgecolors='none')
fig.colorbar(points, ax=a, fraction=.046, pad=.035, label='Overall satisfaction')
a.set(xlabel=f'PC1 ({ratio[0]:.1f}% variance)', ylabel=f'PC2 ({ratio[1]:.1f}% variance)',
      title='PC1–PC2 Score Plot')
a = axes[1, 1]
labels = ['Access', 'Comfort', 'Clean', 'Safety', 'Green', 'Maintain', 'Seating', 'Fitness',
          'Children', 'Shade', 'WiFi', 'Art', 'Plant', 'Privacy', 'Stress', 'Interaction', 'Community quality']
i = np.arange(17)
a.bar(i-.18, coeff[0], width=.36, color='#34458b', edgecolor='#333333', lw=.4, label='PC1')
a.bar(i+.18, coeff[1], width=.36, color='#df9696', edgecolor='#333333', lw=.4, label='PC2')
a.axhline(0, color='#333333', linewidth=.7)
a.set_xticks(i, labels, rotation=52, ha='right')
a.set(ylabel='Eigenvector Coefficient', title='PC1–PC2 Coefficients', ylim=(-.45, .6))
a.legend(frameon=False, fontsize=8, loc='upper right')
for a, letter in zip(axes.flat, 'abcd'):
    a.grid(True, color='#888888', alpha=.25, linewidth=.55)
    a.set_axisbelow(True)
    a.tick_params(direction='out', width=.75, length=3)
    for spine in a.spines.values():
        spine.set_linewidth(.75)
    a.text(-.22, 1.08, '('+letter+')', transform=a.transAxes,
           fontsize=14, fontweight='bold', fontfamily='serif')
fig.subplots_adjust(left=.115, right=.985, bottom=.145, top=.945, wspace=.43, hspace=.42)
fig.savefig(OUT/'Figure_7_corrected_600dpi.png', dpi=600)
fig.savefig(OUT/'Figure_7_preview.png', dpi=150)
plt.close(fig)
print('PC8 cumulative:', cumulative[7], 'PC9 cumulative:', cumulative[8])
