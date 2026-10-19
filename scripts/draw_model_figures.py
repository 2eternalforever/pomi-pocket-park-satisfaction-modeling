"""Create 600-dpi replacements styled after the user's embedded figures."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch,Rectangle
from matplotlib.colors import LinearSegmentedColormap,Normalize
from scipy import stats
from PIL import Image

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'outputs'/'tables'
OUT=ROOT/'outputs'/'figures';OUT.mkdir(parents=True,exist_ok=True)
C=np.array([[.8353,.2431,.3098],[.9725,.6706,.3804],[.4784,.6784,.8039],[.2000,.2627,.5490],[.91,.55,.58],[.40,.63,.67]])
BLUE='#72B0D8';GREEN='#64956B';ORANGE='#F3AB52'
plt.rcParams.update({'font.family':'Arial','font.size':9,'axes.titlesize':10.5,'axes.titleweight':'bold','axes.labelsize':9.5,'axes.labelweight':'bold','xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.75,'savefig.facecolor':'white'})
pred=pd.read_csv(DATA/'holdout_predictions.csv');metrics=pd.read_csv(DATA/'holdout_metrics.csv').set_index('model')
coef=pd.read_csv(DATA/'table2_corrected.csv');cv=pd.read_csv(DATA/'training_cv.csv');outer=pd.read_csv(DATA/'nested_cv_folds.csv');boot=pd.read_csv(DATA/'paired_bootstrap.csv');report=json.loads((DATA/'verified_results.json').read_text())
names=['Linear regression','SVR','Random forest','GPR'];short=['Linear Reg','SVR','Random Forest','GPR']
labs=['Access','Comfort','Clean','Safety','Green','Maintain','Seating','Fitness','Children','Shade','Wi-Fi']
full=['Accessibility','Comfort','Cleanliness','Safety','Green Quality','Maintenance','Seating Facilities','Fitness Equipment',"Children's Facilities",'Shading Facilities','WiFi']

def style(ax):
    ax.grid(True,color='#8a8a8a',alpha=.28,linewidth=.55);ax.set_axisbelow(True)
    for s in ax.spines.values():s.set_color('#222222');s.set_linewidth(.75)
    ax.tick_params(direction='out',width=.75,length=3.5)
def letters(axes):
    for ax,s in zip(axes.flat,'abcdef'):
        ax.text(-.21,1.1,f'({s})',transform=ax.transAxes,fontsize=14,fontweight='bold',fontfamily='serif',va='top')
def save(fig,n):
    filename=OUT/f'Figure_{n}_original_style_600dpi.png';fig.savefig(filename,dpi=600)
    fig.savefig(OUT/f'Figure_{n}_preview.png',dpi=130);plt.close(fig)
    with Image.open(filename) as im:
        assert abs(im.info['dpi'][0]-600)<.1
        print(filename.name,im.size,im.info['dpi'],flush=True)

def fig9():
    fig,ax=plt.subplots(2,2,figsize=(1184/150,981/150));cc=coef.set_index('variable').loc[full];p=pred['Linear regression'].to_numpy();y=pred.observed.to_numpy();res=y-p
    colors=[C[0] if v<.001 else C[1] if v<.01 else C[2] if v<.05 else '#cfcfcf' for v in cc.p]
    a=ax[0,0];a.bar(range(11),cc.coefficient,color=colors,edgecolor='#222222',lw=.6,alpha=.95);a.set_xticks(range(11),labs,rotation=45,ha='right');a.set(ylabel='Coefficient (points per 1 SD)',title='Regression Coefficients',ylim=(0,.17))
    a=ax[0,1];a.scatter(y,p,s=25,c=[C[1]],alpha=.6,edgecolors=[C[1]*.7],linewidth=.5);a.plot([.75,5.55],[.75,5.55],'k--',lw=1.8);a.set(xlim=(.75,5.55),ylim=(.75,5.55),xlabel='Actual Satisfaction',ylabel='Predicted Satisfaction',title='Holdout: Predicted vs Actual');a.set_aspect('equal',adjustable='box')
    a=ax[1,0];a.scatter(p,res,s=25,c=[C[2]],alpha=.6,edgecolors=[C[2]*.7],linewidth=.5);a.axhline(0,color='black',ls='--',lw=1.8);a.set(xlabel='Predicted Value',ylabel='Residual',title='Residual Plot',xlim=(1,6),ylim=(-.8,1.2))
    a=ax[1,1];(osm,osr),(slope,intercept,_)=stats.probplot(res);a.scatter(osm,osr,s=18,marker='+',color='#4c4c4c');a.plot(osm,intercept+slope*osm,'--',color=C[1],lw=1.6);a.set(xlabel='Theoretical Quantiles',ylabel='Sample Quantiles',title='Residual Q-Q Plot',xlim=(-3,3),ylim=(-.8,1.2))
    for a in ax.flat:style(a)
    letters(ax);fig.subplots_adjust(left=.115,right=.985,bottom=.10,top=.94,wspace=.45,hspace=.44);save(fig,9)

def fig10():
    fig,ax=plt.subplots(2,3,figsize=(12.38,6.78))
    for a,key,title in zip(ax[0],['MAE','RMSE','R2'],['Mean Absolute Error (Holdout)','Root Mean Square Error (Holdout)','Predictive R² (Holdout)']):
        vals=metrics.loc[names,key].to_numpy();bars=a.bar(range(4),vals,color=C[:4],edgecolor='#222222',lw=.65,alpha=.92)
        for b,h in zip(bars,['','//','..','xx']):b.set_hatch(h)
        a.bar_label(bars,fmt='%.4f',padding=3,fontsize=7.5)
        a.set_xticks(range(4),short,rotation=30,ha='right');a.set(ylabel=key if key!='R2' else 'R²',title=title,ylim=(0,vals.max()*1.2))
        a.axhline(metrics.loc['Six-attribute mean',key],color='#555555',ls='--',lw=1,label='Six-attribute mean');a.legend(fontsize=7,frameon=False,loc='upper right' if key=='R2' else 'upper right')
    a=ax[1,0];order=['Six-attribute mean','Linear regression','Nested selected procedure'];g=outer.groupby('model').MAE
    for i,(name,color,marker,ls) in enumerate(zip(order,[C[1],C[0],C[3]],['s','o','D'],['--','-',':'])):
        vals=outer[outer.model.eq(name)].sort_values('fold');a.plot(vals.fold,vals.MAE,marker=marker,color=color,ls=ls,lw=1.8,ms=5,label=['Six-attribute mean','Linear regression','Nested selection'][i])
    a.set(xlabel='Outer Validation Fold',ylabel='Outer-fold MAE',title='Nested Cross-Validation',xticks=range(1,6));a.legend(fontsize=7.1,frameon=False,loc='best')
    a=ax[1,1];box=a.boxplot([cv[cv.model.eq(n)].MAE for n in names],tick_labels=short,patch_artist=True,showfliers=True,widths=.56,medianprops={'color':'#222222','lw':1.8})
    for i,b in enumerate(box['boxes']):b.set_facecolor(C[i]);b.set_alpha(.62);b.set_edgecolor(C[i]*.7)
    a.tick_params(axis='x',rotation=35);a.set(ylabel='CV MAE (Training Set)',title='5-Fold Cross-Validation MAE')
    a=ax[1,2];imp=pd.read_csv(DATA/'rf_importance_verified.csv').set_index('feature').loc[full].relative_importance.to_numpy();cmap=LinearSegmentedColormap.from_list('orig',[C[1],C[2],C[3]]);bars=a.bar(range(11),imp,color=cmap(Normalize(imp.min(),imp.max())(imp)),edgecolor='#222222',lw=.6)
    a.bar_label(bars,fmt='%.2f',padding=2,fontsize=6.5);a.set_xticks(range(11),labs,rotation=45,ha='right');a.set(ylabel='Relative Importance',title='Feature Importance (Random Forest)',ylim=(0,imp.max()*1.18))
    for a in ax.flat:style(a)
    letters(ax);fig.subplots_adjust(left=.065,right=.985,bottom=.135,top=.935,wspace=.36,hspace=.53);save(fig,10)

def fig11():
    # Keep the original four-panel geometry; replace unsupported ACF and PI panels.
    fig,ax=plt.subplots(2,2,figsize=(1335/150,997/150));err=pred.observed-pred.GPR
    a=ax[0,0];a.hist(err,bins=25,density=True,color=C[0],edgecolor=C[0]*.7,lw=.55,alpha=.82);xx=np.linspace(err.min(),err.max(),200);a.plot(xx,stats.norm.pdf(xx,loc=err.mean(),scale=err.std(ddof=1)),color=C[1],lw=2,label='Descriptive normal fit');a.set(xlabel='Observed − Predicted',ylabel='Probability Density',title='Error Distribution');a.legend(fontsize=7,frameon=False)
    a=ax[0,1];keys=['MAE','RMSE','R2'];point=metrics.loc['GPR',keys].to_numpy(float);lo=np.array([report['bootstrap_CI'][k][0] for k in keys]);hi=np.array([report['bootstrap_CI'][k][1] for k in keys]);a.errorbar([1,2,3],point,yerr=[point-lo,hi-point],fmt='o',lw=2,ms=7,capsize=6,color=C[2],label='Point estimate & 95% CI')
    for i,k in enumerate(keys,1):
        q1,q3=boot[k].quantile([.25,.75]);a.plot([i,i],[q1,q3],color=C[4],lw=7,alpha=.8,label='Bootstrap IQR' if i==1 else None)
    a.set(xticks=[1,2,3],xticklabels=['MAE','RMSE','R²'],ylabel='Metric Value',title='Conditional Bootstrap 95% CI & IQR');a.legend(fontsize=7,frameon=False)
    a=ax[1,0]
    for i,(name,color,marker,ls) in enumerate(zip(['Nested selected procedure','Linear regression','Six-attribute mean'],[C[3],C[0],C[1]],['D','o','s'],[':', '-', '--'])):
        rows=outer[outer.model.eq(name)].sort_values('fold');a.plot(rows.fold,rows.MAE,color=color,marker=marker,ls=ls,lw=1.6,ms=5,label=['Nested selection','Linear regression','Six-attribute mean'][i])
    a.set(xlabel='Outer Validation Fold',ylabel='MAE',title='Nested-CV Error by Fold',xticks=range(1,6));a.legend(fontsize=7,frameon=False)
    a=ax[1,1];a.scatter(pred.observed,pred.GPR,s=25,c=[C[2]],edgecolors=[C[2]*.75],lw=.45,alpha=.6);a.plot([.8,5.2],[.8,5.2],'k--',lw=1.7,label='Identity line');a.set(xlabel='Actual Satisfaction',ylabel='Predicted Satisfaction',title='GPR Holdout Predictions',xlim=(.8,5.2),ylim=(.5,5.5),xticks=range(1,6));a.legend(fontsize=7,frameon=False)
    for a in ax.flat:style(a)
    letters(ax);fig.subplots_adjust(left=.095,right=.985,bottom=.1,top=.94,wspace=.34,hspace=.40);save(fig,11)

def box(ax,x,y,w,h,text,face='white',fontsize=9,weight='normal'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.008,rounding_size=.035',facecolor=face,edgecolor='#c8d0d6',lw=.7));ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize,weight=weight,linespacing=1.22)
def arrow(ax,x1,y1,x2,y2):ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,lw=1,color='#52616b',shrinkA=4,shrinkB=4))
def panels(ax,headers):
    for i,((title,color,light),letter) in enumerate(zip(headers,'ABC')):
        x=.12+i*3.96;box(ax,x,.2,3.75,6.45,'',light)
        ax.add_patch(FancyBboxPatch((x,5.8),3.75,.85,boxstyle='round,pad=.008,rounding_size=.035',facecolor=color,edgecolor='none'))
        ax.text(x+.18,6.23,letter,color='white',weight='bold',fontsize=23,va='center');ax.text(x+2.05,6.23,title,ha='center',va='center',color='white',fontsize=10.2,weight='bold')

def fig2():
    fig,ax=plt.subplots(figsize=(12,6.7));ax.set(xlim=(0,12),ylim=(0,6.85));ax.axis('off');panels(ax,[('DATA PREPARATION',BLUE,'#dcecf6'),('MODEL TRAINING\n& COMPARISON',GREEN,'#e0eee2'),('VALIDATION\n& INTERPRETATION',ORANGE,'#fcebd6')])
    blocks=[
        ['Separate overall-satisfaction item\n11 rating predictors; n=361', 'Original split retained (seed 42)\nTraining n=252; holdout n=109\nHoldout previously inspected', 'Standardize within training only\nTraining mean / median baselines\nUnweighted six-attribute mean'],
        ['Linear regression, SVR, RF, GPR\nSelect by training-only 5-fold MAE\nRefit using training respondents', 'Retrospective nested 5 × 5 CV\nInner folds select; outer folds evaluate\nFold-specific standardization', 'Ordered-logit sensitivity check\nSame 11 predictors and original split\nCategory accuracy and weighted κ'],
        ['Holdout MAE, RMSE, predictive R²\nReport separately from nested-CV results\nR² = 1 − SSE / SST', 'Conditional paired bootstrap\n1,000 resamples; predictions fixed\nNo selection-uncertainty claim', 'Training OOB permutation importance\nInternal, same-survey associations\nNo causal or external-validity claim']]
    for i in range(3):
        x=.12+i*3.96
        for j in range(3):
            y=4.15-j*1.65;box(ax,x+.2,y,3.35,1.18,blocks[i][j],fontsize=8.6)
            if j<2:arrow(ax,x+1.875,y,x+1.875,y-.44)
        if i<2:arrow(ax,x+3.6,4.74,x+4.1,4.74)
    fig.subplots_adjust(left=.005,right=.995,bottom=.015,top=.99);save(fig,2)

def fig1():
    fig,ax=plt.subplots(figsize=(12,6.7));ax.set(xlim=(0,12),ylim=(0,6.85));ax.axis('off');panels(ax,[('STUDY PROFILE &\nBASELINE METRICS',BLUE,'#dcecf6'),('UNSUPERVISED\nSTRUCTURAL ANALYSIS',GREEN,'#e0eee2'),('PREDICTIVE MODELING\n& KEY ASSOCIATIONS',ORANGE,'#fcebd6')])
    content=[
        [('Data scope and demographics','361 valid online questionnaires\nShanghai pocket-park users\nDescriptive, convenience sample'),('Overall satisfaction','Mean 3.84; SD 0.90\n70.1% rated 4 or 5'),('Correlations with satisfaction','Accessibility r=0.748\nSafety r=0.731; cleanliness r=0.710')],
        [('PCA: shared questionnaire variation','First component: 54.1% variance\nQuestionnaire structure exploration'),('Rating-based clustering','Two groups: 30.2% and 69.8%\nLower and higher rating patterns'),('Exploratory factor analysis','One retained factor; 51.3% variance\nParallel analysis: 2,000 permutations\nNo four-construct interpretation')],
        [('Internal validation','Four continuous-response models\nTraining-only selection; nested CV\nSimple-mean and ordinal comparisons'),('Original holdout performance','GPR: MAE 0.1420\nLinear: RMSE 0.2419; R² 0.9267\nPreviously inspected holdout'),('Model-dependent associations','Linear: seating, access, cleanliness\nRF: fitness, WiFi, access, safety, comfort\nPredictive contribution; no causal ranking')]]
    for i,items in enumerate(content):
        x=.12+i*3.96
        for j,(title,text) in enumerate(items):
            y=4.1-j*1.7;box(ax,x+.16,y,3.43,1.35,'',face='white');ax.text(x+1.875,y+1.08,title,ha='center',va='center',weight='bold',fontsize=9.8);ax.text(x+1.875,y+.50,text,ha='center',va='center',fontsize=8.7,linespacing=1.3)
    fig.subplots_adjust(left=.005,right=.995,bottom=.015,top=.99);save(fig,1)

def fig12():
    # Three panels required by the already-revised caption, with original colors/grid.
    pa=pd.read_csv(DATA/'R3_C6_parallel_analysis.csv');load=pd.read_csv(DATA/'R3_C6_single_factor_loadings.csv')
    label17=labs+['Art','Plant','Privacy','Stress','Community','Quality']
    fig,ax=plt.subplots(1,3,figsize=(12.4,4.55));a=ax[0];a.plot(pa.component,pa.observed,'o-',color=C[0],lw=1.8,ms=4,label='Observed');a.plot(pa.component,pa.permutation_p95,'s--',color=C[1],lw=1.6,ms=3,label='Permutation 95th percentile');a.axhline(1,ls=':',color='#555555',lw=1);a.set(xlabel='Ordered Component',ylabel='Eigenvalue',title='Parallel Analysis (2,000 Permutations)',xticks=[1,4,8,12,17]);a.legend(fontsize=7,frameon=False)
    a=ax[1];a.barh(np.arange(17),load.ML_loading,color=C[2],edgecolor='#333333',lw=.4);a.set_yticks(range(17),label17);a.invert_yaxis();a.set(xlabel='Standardized ML Loading',title='One-Factor Loadings',xlim=(0,.9))
    a=ax[2];a.bar(range(17),load.communality,color=[C[i%4] for i in range(17)],edgecolor='#333333',lw=.4);a.set_xticks(range(17),label17,rotation=55,ha='right');a.set(ylabel='Communality',title='Item Communalities',ylim=(0,.75))
    for a in ax:style(a)
    letters(ax);fig.subplots_adjust(left=.06,right=.985,bottom=.24,top=.88,wspace=.44);save(fig,12)

if __name__=='__main__':
    for fn in [fig1,fig2,fig9,fig10,fig11,fig12]:fn()
