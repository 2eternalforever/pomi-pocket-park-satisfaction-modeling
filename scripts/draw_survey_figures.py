"""Read-only round-four data checks and original-style 600-dpi figures.
Raw data and manuscript are never overwritten.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"/"tables"; PNG=ROOT/"outputs"/"figures"; PNG.mkdir(parents=True,exist_ok=True)
d=pd.read_excel(ROOT/"data"/"survey_data.xlsx")
n=len(d)
sat=["Satisfaction_"+s for s in ["Access","Comfort","Cleanliness","Safety","Greenery","Maintenance"]]
fac=["Facility_"+s for s in ["Seating","Fitness","Children","Shade","WiFi","Art","PlantDiversity","Privacy"]]
imp=["Impact_Stress","Impact_Community","Impact_Quality"]
sl=["Access","Comfort","Clean","Safety","Green","Maintain"]
fl=["Seating","Fitness","Children","Shade","WiFi","Art","Plant","Privacy"]
ages=["18–25","26–35","36–45","46–55","56–65","66+"]
freq=["Daily","3–4/week","1–2/week","2–3/month","≤1/month"]
duration=["≤10 min","11–30 min","31–60 min","1–2 h",">2 h"]
distance=["≤5 min","6–10 min","11–15 min","16–20 min",">20 min"]
C=["#d83b50","#f2ab62","#76a9c6","#34458b","#df9696","#f5d3af","#bfd3de","#a0c49e","#cbc4df"]
plt.rcParams.update({"font.family":"Arial","font.size":9,"axes.titlesize":10.5,"axes.titleweight":"bold","axes.labelsize":9.5,"axes.labelweight":"bold","xtick.labelsize":8,"ytick.labelsize":8,"axes.linewidth":.75,"savefig.facecolor":"white","axes.unicode_minus":False})
corrmap=LinearSegmentedColormap.from_list("blue_red",["#053061","#4393c3","#d1e5f0","#ffffff","#f4a582","#b2182b"])

def style(a):
    a.grid(True,color="#888888",alpha=.25,linewidth=.55);a.set_axisbelow(True)
    for s in a.spines.values():s.set_color("#333333");s.set_linewidth(.75)

def bar(a,vals,labs,title,ylabel="Count",rotate=0,decimal=0):
    b=a.bar(range(len(vals)),vals,color=C[:len(vals)],edgecolor="#444444",lw=.5,alpha=.85)
    a.set_xticks(range(len(vals)),labs,rotation=rotate,ha="right" if rotate else "center")
    a.set(title=title,ylabel=ylabel)
    a.bar_label(b,fmt="%."+str(decimal)+"f",padding=3,fontsize=7.5)
    a.set_ylim(0,float(np.max(vals))*1.19);style(a)

def count(col,codes):return d[col].value_counts().reindex(codes,fill_value=0).to_numpy()

def heat(a,m,xlabs,ylabs,title,cmap=corrmap,vmin=0,vmax=1,annot=False):
    im=a.imshow(m,aspect="auto",cmap=cmap,vmin=vmin,vmax=vmax)
    a.set_xticks(range(len(xlabs)),xlabs,rotation=45,ha="right")
    a.set_yticks(range(len(ylabs)),ylabs);a.set_title(title)
    if annot:
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                a.text(j,i,f"{m[i,j]:.2f}",ha="center",va="center",fontsize=6.5,
                       color="white" if m[i,j]>.82 else "#222222")
    fig=a.figure;fig.colorbar(im,ax=a,fraction=.045,pad=.025)

def save(fig,axes,num):
    for a,letter in zip(axes,"abcdef"):
        a.text(-.20,1.09,f"({letter})",transform=a.transAxes,fontsize=14,fontfamily="serif",fontweight="bold",va="top")
    p=PNG/f"Figure_{num}_corrected_600dpi.png";fig.savefig(p,dpi=600);plt.close(fig)
    with Image.open(p) as im:
        assert abs(im.info["dpi"][0]-600)<.1
        im.thumbnail((1880,1200));im.save(PNG/f"Figure_{num}_preview.png")
    print(p,flush=True)

fig,aa=plt.subplots(2,2,figsize=(10,7));a=aa.flat
bar(a[0],count("Gender",[1,2,3]),["Male","Female","Other / prefer not to say"],"Gender Distribution")
bar(a[1],count("Age",range(1,7)),ages,"Age Distribution")
bar(a[2],count("Frequency",range(1,6)),freq,"Visit Frequency",rotate=30)
bar(a[3],count("Duration",range(1,6)),duration,"Stay Duration",rotate=30)
fig.subplots_adjust(left=.10,right=.98,bottom=.12,top=.94,wspace=.36,hspace=.43)
save(fig,list(aa.flat),3)

fig,aa=plt.subplots(2,3,figsize=(12.8,7.0));a=list(aa.flat)
time=["Time_"+s for s in ["Morning","AM","Noon","PM","Evening","Night","Late"]]
bar(a[0],d[time].sum().to_numpy(),["06–09","09–12","12–14","14–18","18–20","20–22","After 22"],"Usage Time Periods",ylabel="Respondents",rotate=45)
purpose=["Purpose_"+s for s in ["Relax","Exercise","Walk","Social","Commute","Study","View","Work","Other"]]
pl=["Relax","Exercise","Dog/child walk","Social","Commute","Study","View","Phone/work","Other"]
bar(a[1],d[purpose].sum().to_numpy(),pl,"Purpose Distribution",ylabel="Respondents",rotate=50)
comp=["Companion_"+s for s in ["Alone","Partner","Children","Parents","Friends","Colleagues","Pets"]]
bar(a[2],d[comp].sum().to_numpy(),["Alone","Partner","Children","Parents","Friends","Colleagues","Pets"],"Companion Distribution",ylabel="Respondents",rotate=45)
m=pd.crosstab(d.Distance,d.Frequency).reindex(index=range(1,6),columns=range(1,6),fill_value=0).to_numpy()
heat(a[3],m,freq,distance,"Distance–Frequency Counts",cmap=LinearSegmentedColormap.from_list("bf",["white",C[2]]),vmin=0,vmax=m.max())
a[3].set(xlabel="Visit Frequency",ylabel="Walking Time from Home")
m=pd.crosstab(d.Distance,d.Duration).reindex(index=range(1,6),columns=range(1,6),fill_value=0).to_numpy()
heat(a[4],m,duration,distance,"Distance–Duration Counts",cmap=LinearSegmentedColormap.from_list("pf",["white",C[4]]),vmin=0,vmax=m.max())
a[4].set(xlabel="Stay Duration",ylabel="Walking Time from Home")
m=d.groupby("Age")[purpose[:5]].sum().reindex(range(1,7),fill_value=0).to_numpy()
heat(a[5],m,pl[:5],ages,"Age–Purpose Counts",cmap=LinearSegmentedColormap.from_list("nf",["white",C[3]]),vmin=0,vmax=m.max())
a[5].set(xlabel="Purpose",ylabel="Age Group")
fig.subplots_adjust(left=.075,right=.985,bottom=.16,top=.94,wspace=.49,hspace=.64)
save(fig,a,4)

fig,aa=plt.subplots(2,3,figsize=(12.8,7.0));a=list(aa.flat)
bar(a[0],count("Satisfaction_Overall",range(1,6)),["1","2","3","4","5"],"Overall Satisfaction")
a[0].set_xlabel("Rating")
bar(a[1],d[sat].mean().to_numpy(),sl,"Satisfaction Attributes",ylabel="Mean Rating",rotate=40,decimal=2);a[1].set_ylim(0,5)
m=d[["Satisfaction_Overall"]+sat].corr().to_numpy()
heat(a[2],m,["Overall"]+sl,["Overall"]+sl,"Satisfaction Correlations",vmin=0,vmax=1,annot=True)
bar(a[3],d[fac].mean().to_numpy(),fl,"Facility Ratings",ylabel="Mean Rating",rotate=45,decimal=2);a[3].set_ylim(0,5)
bar(a[4],d[imp].mean().to_numpy(),["Stress relief","Neighbour contact","Community quality"],"Perceived Impacts",ylabel="Mean Rating",rotate=25,decimal=2);a[4].set_ylim(0,5)
a[5].axis("off")
fig.subplots_adjust(left=.075,right=.955,bottom=.14,top=.94,wspace=.47,hspace=.58)
save(fig,a[:5],5)

fig,aa=plt.subplots(1,3,figsize=(12.8,4.35));a=list(aa.flat)
m=d[sat+fac].corr().loc[sat,fac].to_numpy()
heat(a[0],m,fl,sl,"Attribute–Facility Correlations",vmin=0,vmax=1)
oc=["Age","Distance","Frequency","Duration"]
rho=np.array([stats.spearmanr(d[c],d.Satisfaction_Overall).statistic for c in oc])
pvals=np.array([stats.spearmanr(d[c],d.Satisfaction_Overall).pvalue for c in oc])
bars=a[1].bar(range(4),rho,color=C[:4],edgecolor="#444444",lw=.5)
a[1].set_xticks(range(4),oc,rotation=40,ha="right")
a[1].axhline(0,color="#444444",lw=.7)
for b,r,p in zip(bars,rho,pvals):
    a[1].text(b.get_x()+b.get_width()/2,r+(.007 if r>=0 else -.007),f"{r:.3f}",ha="center",va="bottom" if r>=0 else "top",fontsize=7.5)
a[1].set(ylabel="Spearman correlation with satisfaction",title="Ordinal Background Variables",ylim=(-.10,.09));style(a[1])
m=d[fac].corr().to_numpy()
heat(a[2],m,fl,fl,"Facility Correlations",vmin=0,vmax=1)
fig.subplots_adjust(left=.075,right=.955,bottom=.27,top=.87,wspace=.53)
save(fig,a,6)

r,p=stats.pearsonr(d.Satisfaction_Overall,d.NPS_Score)
rho,sp=stats.spearmanr(d.Satisfaction_Overall,d.NPS_Score)
audit={
 "n":n,"missing_cells":int(d.isna().sum().sum()),"exact_duplicate_rows":int(d.duplicated().sum()),
 "recommendation_stored_range":[int(d.NPS_Score.min()),int(d.NPS_Score.max())],
 "pearson":{"r":float(r),"p":float(p)},"spearman":{"rho":float(rho),"p":float(sp)},
 "age_counts":{str(k):int(v) for k,v in d.Age.value_counts().sort_index().items()},
 "validity_item_counts":{str(k):int(v) for k,v in d.Validity_Check.value_counts().sort_index().items()},
 "ordinal_background":{c:{"rho":float(stats.spearmanr(d[c],d.Satisfaction_Overall).statistic),"p":float(pp)} for c,pp in zip(oc,pvals)},
 "gender_kruskal":{k:float(v) for k,v in zip(["H","p"],stats.kruskal(*[d.loc[d.Gender.eq(g),"Satisfaction_Overall"] for g in [1,2,3]]))}
}
(OUT/"data_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
print(json.dumps(audit,indent=2))
