from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'/'survey_data.xlsx'; OUT=ROOT/'outputs'/'figures'; OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_excel(DATA)
cols=[c for c in df.columns if c.startswith(('Satisfaction_','Facility_','Impact_')) and c!='Satisfaction_Overall']
X=df[cols].to_numpy(float); Z=(X-X.mean(0))/X.std(0,ddof=1)
km=KMeans(n_clusters=2,n_init=20,max_iter=500,random_state=42); raw=km.fit_predict(Z)
means=df.assign(_c=raw).groupby('_c').Satisfaction_Overall.mean(); low_raw=int(means.idxmin()); labels=np.where(raw==low_raw,0,1)
sil=silhouette_samples(Z,labels,metric='sqeuclidean')
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':10.5,'axes.titleweight':'bold','axes.labelsize':9.5,'axes.labelweight':'bold','xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.75,'savefig.facecolor':'white'})
C=['#d83b50','#f2ab62','#76a9c6','#34458b','#df9696','#f5d3af']
fig,ax=plt.subplots(2,3,figsize=(12.55,6.55))
wcss=[KMeans(n_clusters=k,n_init=20,max_iter=500,random_state=42).fit(Z).inertia_ for k in range(1,11)]
a=ax[0,0]; a.plot(range(1,11),wcss,'o-',color=C[0],markerfacecolor=C[0],markeredgecolor='#8f1f31',linewidth=2.2,markersize=7); a.set(xlabel='Number of Clusters (k)',ylabel='Within-Cluster Sum of Squares',title='Elbow Method',xticks=range(0,11,2))
a=ax[0,1]
for cl,col in [(0,C[0]),(1,C[1])]:
 v=np.sort(sil[labels==cl]); a.bar(np.arange(len(v)),v,color=col,edgecolor='none',width=1,alpha=.82)
a.axhline(float(sil.mean()),color='red',linestyle='--',linewidth=1.8); a.set(xlabel='Sample Index',ylabel='Silhouette Value',title='Silhouette Plot',xlim=(0,len(sil)),ylim=(-.1,.8))
pca=PCA(n_components=2,random_state=42); score=pca.fit_transform(Z); a=ax[0,2]
for cl,col,name in [(0,C[0],'Cluster 1 (Low)'),(1,C[1],'Cluster 2 (High)')]:
 m=labels==cl; a.scatter(score[m,0],score[m,1],s=30,c=col,alpha=.62,edgecolors='#7a3a40' if cl==0 else '#9a6d3f',linewidth=.35,label=name)
cent=pca.transform(km.cluster_centers_[[low_raw,1-low_raw],:]); a.scatter(cent[:,0],cent[:,1],s=115,c='black',marker='X',linewidth=.6,label='Centroids'); a.set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',ylabel=f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',title='Clusters in PCA Space'); a.legend(frameon=False,fontsize=7.5,loc='upper right')
a=ax[1,0]; age_labels=['18–25','26–35','36–45','46–55','56–65','66+']; width=.72/6
for i,(lab,code,col) in enumerate(zip(age_labels,range(1,7),C)):
 vals=[int(((labels==cl)&(df.Age.to_numpy()==code)).sum()) for cl in (0,1)]; a.bar(np.array([0,1])+(i-2.5)*width,vals,width=width,color=col,edgecolor='#333333',linewidth=.35,label=lab)
a.set(xticks=[0,1],xticklabels=['Cluster 1','Cluster 2'],ylabel='Count',title='Cluster Age Distribution'); a.legend(frameon=False,fontsize=7,loc='upper left',ncol=2)
a=ax[1,1]; sat=['Satisfaction_Access','Satisfaction_Comfort','Satisfaction_Cleanliness','Satisfaction_Safety','Satisfaction_Greenery','Satisfaction_Maintenance']; sl=['Access','Comfort','Clean','Safety','Green','Maintain']; width=.72/6
for i,(col,lab,color) in enumerate(zip(sat,sl,C)):
 vals=[df.loc[labels==cl,col].mean() for cl in (0,1)]; a.bar(np.array([0,1])+(i-2.5)*width,vals,width=width,color=color,edgecolor='#333333',linewidth=.35,label=lab)
a.set(xticks=[0,1],xticklabels=['Cluster 1','Cluster 2'],ylabel='Mean Satisfaction',title='Cluster Satisfaction Profiles',ylim=(.5,4.75)); a.legend(frameon=False,fontsize=7,loc='upper left',ncol=2)
a=ax[1,2]; freq=['Daily','3–4/week','1–2/week','2–3/month','Rarely']; width=.72/5
for i,(lab,code,color) in enumerate(zip(freq,range(1,6),C[:5])):
 vals=[int(((labels==cl)&(df.Frequency.to_numpy()==code)).sum()) for cl in (0,1)]; a.bar(np.array([0,1])+(i-2)*width,vals,width=width,color=color,edgecolor='#333333',linewidth=.35,label=lab)
a.set(xticks=[0,1],xticklabels=['Cluster 1','Cluster 2'],ylabel='Count',title='Cluster Usage Frequency'); a.legend(frameon=False,fontsize=7,loc='upper left')
for aa in ax.flat:
 aa.grid(True,color='#8a8a8a',alpha=.28,linewidth=.55); aa.set_axisbelow(True)
 for s in aa.spines.values(): s.set_color('#222222'); s.set_linewidth(.75)
 aa.tick_params(direction='out',width=.75,length=3.5)
for aa,letter in zip(ax.flat,'abcdef'): aa.text(-.2,1.1,f'({letter})',transform=aa.transAxes,fontsize=14,fontweight='bold',fontfamily='serif',va='top')
fig.subplots_adjust(left=.065,right=.985,bottom=.085,top=.935,wspace=.43,hspace=.42)
out=OUT/'Figure_8_clustering_600dpi.png'; fig.savefig(out,dpi=600); plt.close(fig)
print(out, 'mean silhouette',float(sil.mean()),'sizes',int((labels==0).sum()),int((labels==1).sum()))
