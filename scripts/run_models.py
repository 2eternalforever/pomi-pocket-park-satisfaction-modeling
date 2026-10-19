"""Reproducible retrospective audit; never edits the manuscript or response letter."""
from pathlib import Path
import json, warnings, sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, cohen_kappa_score
from statsmodels.miscmodels.ordinal_model import OrderedModel
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
FEATURES = ['Satisfaction_Access','Satisfaction_Comfort','Satisfaction_Cleanliness','Satisfaction_Safety','Satisfaction_Greenery','Satisfaction_Maintenance','Facility_Seating','Facility_Fitness','Facility_Children','Facility_Shade','Facility_WiFi']
LABELS = ['Accessibility','Comfort','Cleanliness','Safety','Green Quality','Maintenance','Seating Facilities','Fitness Equipment',"Children's Facilities",'Shading Facilities','WiFi']
MODELS = {
    'Linear regression': LinearRegression(),
    'SVR': SVR(kernel='rbf', C=1., epsilon=.1, gamma='scale'),
    'Random forest': RandomForestRegressor(n_estimators=100,min_samples_leaf=5,random_state=42,n_jobs=1),
    'GPR': GaussianProcessRegressor(kernel=ConstantKernel(1.,(1e-3,1e3))*RBF(1.,(1e-2,1e2))+WhiteKernel(.1,(1e-5,1e1)),normalize_y=False,random_state=42,n_restarts_optimizer=2),
}

def standardize(a,b):
    mu=a.mean(0); sd=a.std(0,ddof=1); sd[sd==0]=1
    return (a-mu)/sd,(b-mu)/sd

def metrics(y,p):
    return dict(MAE=float(mean_absolute_error(y,p)),RMSE=float(np.sqrt(mean_squared_error(y,p))),R2=float(r2_score(y,p)))

def fitpred(name,a,y,b):
    aa,bb=standardize(a,b)
    model=clone(MODELS[name])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore'); model.fit(aa,y)
    return model.predict(bb),model

def crossvalidate(x,y,seed=42):
    records=[]
    for fold,(ti,vi) in enumerate(KFold(5,shuffle=True,random_state=seed).split(x),1):
        for name in MODELS:
            pred,_=fitpred(name,x[ti],y[ti],x[vi])
            records.append(dict(fold=fold,model=name,**metrics(y[vi],pred)))
    return pd.DataFrame(records)

def main():
    df=pd.read_excel(ROOT/'data'/'survey_data.xlsx')
    x=df[FEATURES].to_numpy(float); y=df.Satisfaction_Overall.to_numpy(float)
    mask=np.zeros(len(y),bool)
    mask[np.random.RandomState(42).choice(np.arange(len(y)),size=int(.7*len(y)),replace=False)]=True
    xt,xe,yt,ye=x[mask],x[~mask],y[mask],y[~mask]
    pd.DataFrame({'excel_row':np.arange(len(y))+2,'partition':np.where(mask,'training','previously_inspected_holdout')}).to_csv(OUT/'split_rows.csv',index=False)
    records=[]; predictions={}; fitted={}
    for name in MODELS:
        p,m=fitpred(name,xt,yt,xe); predictions[name]=p;fitted[name]=m
        records.append(dict(model=name,**metrics(ye,p)))
    for name,p in [('Training mean',np.full(len(ye),yt.mean())),('Training median',np.full(len(ye),np.median(yt))),('Six-attribute mean',xe[:,:6].mean(1))]:
        predictions[name]=p; records.append(dict(model=name,**metrics(ye,p)))
    cv=crossvalidate(xt,yt);cv.to_csv(OUT/'training_cv.csv',index=False)
    cv_summary=cv.groupby('model')['MAE'].agg(['mean','std'])
    selected=cv_summary['mean'].idxmin()
    print('CV selection:',selected,cv_summary.to_dict(),flush=True)

    aa,bb=standardize(xt,xe)
    ordinal_model=OrderedModel(yt.astype(int),aa,distr='logit')
    ordinal_fit=ordinal_model.fit(method='bfgs',maxiter=3000,disp=False,gtol=1e-7)
    probs=np.asarray(ordinal_fit.model.predict(ordinal_fit.params,exog=bb))
    expected=probs@np.arange(1,6)
    med=1+(np.cumsum(probs,axis=1)<.5).sum(1)
    mode=1+probs.argmax(1)
    ordinal=dict(converged=bool(ordinal_fit.mle_retvals.get('converged')),max_abs_gradient=float(np.max(np.abs(ordinal_fit.mle_retvals['gopt']))),expected_score=metrics(ye,expected),log_likelihood=float(ordinal_fit.llf),max_abs_coefficient=float(np.max(np.abs(ordinal_fit.params[:11]))))
    ordinal['thresholds']=ordinal_fit.model.transform_threshold_params(ordinal_fit.params)[1:-1].tolist()
    category_records=[]
    cat_preds={'Ordered logit (median)':med,'Ordered logit (mode)':mode,'Linear regression (rounded)':np.clip(np.rint(predictions['Linear regression']),1,5),'GPR (rounded)':np.clip(np.rint(predictions['GPR']),1,5),'Six-attribute mean (rounded)':np.clip(np.rint(predictions['Six-attribute mean']),1,5)}
    for name,p in cat_preds.items():
        category_records.append(dict(model=name,category_MAE=float(mean_absolute_error(ye,p)),exact_accuracy=float(np.mean(ye==p)),quadratic_weighted_kappa=float(cohen_kappa_score(ye,p,weights='quadratic')),errors=int(np.sum(ye!=p))))
    pd.DataFrame(category_records).to_csv(OUT/'ordinal_sensitivity.csv',index=False)
    predictions['Ordered logit expected score']=expected
    records.append(dict(model='Ordered logit expected score',**metrics(ye,expected)))
    pd.DataFrame(records).to_csv(OUT/'holdout_metrics.csv',index=False)
    pd.DataFrame({'excel_row':np.where(~mask)[0]+2,'observed':ye,**predictions,**cat_preds}).to_csv(OUT/'holdout_predictions.csv',index=False)
    print('Ordinal:',ordinal,category_records,flush=True)

    # A paired bootstrap conditional on one fitted model; no model refitting.
    rng=np.random.default_rng(42);boot=[]
    for b in range(1000):
        ix=rng.integers(0,len(ye),size=len(ye));boot.append(metrics(ye[ix],predictions[selected][ix]))
    boot=pd.DataFrame(boot);boot.to_csv(OUT/'paired_bootstrap.csv',index=False)
    cis={k:[float(v) for v in np.quantile(boot[k],[.025,.975])] for k in ['MAE','RMSE','R2']}

    design=np.column_stack([np.ones(len(yt)),aa]);beta=np.linalg.lstsq(design,yt,rcond=None)[0]
    resid=yt-design@beta;dof=len(yt)-design.shape[1];var=resid@resid/dof
    se=np.sqrt(np.diag(var*np.linalg.inv(design.T@design)));t=beta/se;ps=2*stats.t.sf(abs(t),dof)
    coef=pd.DataFrame({'variable':LABELS,'coefficient':beta[1:],'SE':se[1:],'t':t[1:],'p':ps[1:]}).sort_values('coefficient',ascending=False)
    coef.to_csv(OUT/'table2_corrected.csv',index=False)

    # Nested CV evaluates a model-selection procedure without consulting outer fold outcomes.
    oof=np.zeros(len(y));outer_records=[];selection=[];inner_records=[]
    for fold,(ti,vi) in enumerate(KFold(5,shuffle=True,random_state=42).split(x),1):
        inner=crossvalidate(x[ti],y[ti],seed=42);inner['outer_fold']=fold;inner_records.append(inner)
        winner=inner.groupby('model').MAE.mean().idxmin();selection.append(winner)
        p,_=fitpred(winner,x[ti],y[ti],x[vi]);oof[vi]=p
        outer_records.append(dict(fold=fold,model='Nested selected procedure',selected=winner,**metrics(y[vi],p)))
        for name in ['Linear regression','Six-attribute mean']:
            q=fitpred(name,x[ti],y[ti],x[vi])[0] if name in MODELS else x[vi,:6].mean(1)
            outer_records.append(dict(fold=fold,model=name,selected='',**metrics(y[vi],q)))
        print('Nested fold',fold,'selected',winner,metrics(y[vi],p),flush=True)
    outer=pd.DataFrame(outer_records);outer.to_csv(OUT/'nested_cv_folds.csv',index=False)
    pd.concat(inner_records).to_csv(OUT/'nested_cv_inner.csv',index=False)
    pd.DataFrame({'excel_row':np.arange(len(y))+2,'observed':y,'outer_fold_prediction':oof}).to_csv(OUT/'nested_cv_predictions.csv',index=False)
    nested=outer.groupby('model')[['MAE','RMSE','R2']].agg(['mean','std'])
    nested.to_csv(OUT/'nested_cv_summary.csv')

    from oob_importance import oob_permutation_importance
    importance = oob_permutation_importance(fitted['Random forest'], aa, yt)
    importance.to_csv(OUT/'rf_importance_verified.csv', index=False)
    report=dict(n_train=len(yt),n_holdout=len(ye),selected=selected,holdout=records,cv=cv_summary.to_dict(),ordinal=ordinal,category_metrics=category_records,bootstrap_CI=cis,mean_error=float((ye-predictions[selected]).mean()),error_SD=float((ye-predictions[selected]).std(ddof=1)),nested_selected=selection,nested_pooled=metrics(y,oof),adjusted_train_r2=float(1-(1-r2_score(yt,design@beta))*(len(yt)-1)/dof))
    (OUT/'verified_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
