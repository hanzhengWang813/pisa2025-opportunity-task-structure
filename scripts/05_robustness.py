# -*- coding: utf-8 -*-
"""Robustness checks: collinearity, threshold-group profiles, alternative measures.
"""
import pandas as pd, numpy as np, os, warnings
from scipy import stats
warnings.filterwarnings('ignore')
from paths import BASE  # repo root (override with PISA_BASE)
q = pd.read_parquet(os.path.join(BASE,'data','qci_analysis.parquet'))
REP=[f'W_FSTURWT{i}' for i in range(1,81)]; FAY,M_PV,R=0.5,10,80

q['female']=(q.ST004D01T==1).astype(float); q['age']=q.AGE; q['grade']=q.GRADE
q['repeat']=(q.REPEAT>0).astype(float); q['immig1']=(q.IMMIG==1).astype(float)
q['immig2']=(q.IMMIG==2).astype(float); q['escs']=q.ESCS
for v in ['AI_total','AI_deep','AI_sub','ICTINFO','ICTQUAL','ICTHOME','ICTWKDY','JOYSCIE',
          'EFFSCIE','SELFREG','PERSEV','CURIO','ENGSCIPR','TEACHSUP','COGACSC','DISCLISCI',
          'SCIECOLLR','DIGPREP','SCTRAIN','EDUSHORT','STAFFSHORT','NEGSCLIM']:
    if v in q.columns: q[v+'_z']=(q[v]-q[v].mean())/q[v].std()
g=q.groupby('CNTSCHID'); q['SCH_ESCS']=g['ESCS'].transform('mean')
q['SCH_ESCS_z']=(q.SCH_ESCS-q.SCH_ESCS.mean())/q.SCH_ESCS.std()
q['AI_grp']=q.AI_total.round()
for k in [2,3,4,5]: q[f'AI_g{k}']=(q.AI_grp==k).astype(float)

BASE_COV=['female','age','grade','repeat','immig1','immig2','escs']
DISP=['JOYSCIE_z','EFFSCIE_z','SELFREG_z','PERSEV_z','CURIO_z','ENGSCIPR_z']
CLAS=['TEACHSUP_z','COGACSC_z','DISCLISCI_z','SCIECOLLR_z']
ICT=['ICTHOME_z','ICTQUAL_z','ICTWKDY_z','ICTINFO_z']
SCH=['DIGPREP_z','SCTRAIN_z','EDUSHORT_z','STAFFSHORT_z','NEGSCLIM_z','SCH_ESCS_z']
FULL=BASE_COV+DISP+CLAS+ICT+SCH

def build(df,cols):
    cols=list(dict.fromkeys(cols))
    return np.column_stack([np.ones(len(df))]+[df[c].values.astype(float) for c in cols]),['Intercept']+cols
def wls(X,y,w):
    Xt=X.T*w; A=Xt@X; b=Xt@y
    try: return np.linalg.solve(A,b)
    except np.linalg.LinAlgError: return np.linalg.lstsq(A,b,rcond=None)[0]
def run(df,yname,cols,label):
    X,names=build(df,cols); Y=df[[f'PV{k}{yname}' for k in range(1,M_PV+1)]].values
    W=df[REP].values; w0=df['W_FSTUWT'].values; cs,vs=[],[]
    for j in range(Y.shape[1]):
        y=Y[:,j]; b0=wls(X,y,w0)
        if not np.all(np.isfinite(b0)): continue
        D=np.array([wls(X,y,W[:,r])-b0 for r in range(R)])
        cs.append(b0); vs.append((D**2).sum(0)/(R*(1-FAY)**2))
    cs,vs=np.array(cs),np.array(vs); m=len(cs)
    Qb,Ub=cs.mean(0),vs.mean(0); B=cs.var(0,ddof=1) if m>1 else np.zeros_like(Ub)
    T=Ub+(1+1/m)*B; se=np.sqrt(np.where(T>0,T,np.nan))
    with np.errstate(divide='ignore',invalid='ignore'):
        dfr=np.where(B>0,(m-1)*(1+Ub/((1+1/m)*B))**2,np.inf)
        t=Qb/se; p=2*stats.t.sf(np.abs(t),np.where(np.isfinite(dfr),dfr,1e6))
    return pd.DataFrame({'model':label,'outcome':yname,'term':names,'b':Qb,'se':se,'p':p,'n':len(df)})

need=list(dict.fromkeys(BASE_COV+DISP+CLAS+ICT+SCH+['AI_total','AI_deep','AI_sub','AI_grp',
      'ST438Q01DA','ST438Q02DA','ST438Q03DA','ST438Q04DA']))
need=[c for c in need if c in q.columns]
s=q.dropna(subset=need+['PV1SCIE']).copy()

# ---------- 1. Collinearity ----------
print('='*66); print('1. Collinearity diagnostics (C1 model)'); print('='*66)
X,names=build(s,FULL+['AI_deep_z','AI_sub_z'])
Xz=X[:,1:]; nz=names[1:]
vif=[]
for i,c in enumerate(nz):
    yv=Xz[:,i]; others=np.delete(Xz,i,axis=1)
    r2=np.corrcoef(yv, others@np.linalg.lstsq(others,yv,rcond=None)[0])[0,1]**2
    vif.append((c, 1/(1-r2)))
v=pd.DataFrame(vif,columns=['变量','VIF']).sort_values('VIF',ascending=False)
print(v[v.变量.isin(['AI_deep_z','AI_sub_z','escs','ICTINFO_z'])].to_string(index=False))
print('最大 VIF = %.2f (%s)'%(v.VIF.max(), v.iloc[0].变量))

# ---------- 2. Profile of the threshold group ----------
print('\n'+'='*66); print('2. Profile of the "daily use" group (is it an anomalous subgroup?)'); print('='*66)
def cmp(col):
    a=s.loc[s.AI_grp==5,col].dropna(); b=s.loc[s.AI_grp<5,col].dropna()
    if len(a)==0: return np.nan, np.nan
    return (np.average(a.values, weights=s.loc[a.index,'W_FSTUWT'].values),
            np.average(b.values, weights=s.loc[b.index,'W_FSTUWT'].values))
for c,lab in [('ESCS','family ESCS'),('female','share female'),('grade','grade'),('ICTWKDY','weekday ICT use'),
              ('SELFREG','self-regulation'),('JOYSCIE','interest in science'),('REPEAT','share repeating a grade'),
              ('SCH_ESCS','school mean ESCS')]:
    if c in s.columns:
        a,b=cmp(c); print(f'  {lab:12s} daily={a:6.2f}  others={b:6.2f}  diff={a-b:+6.2f}')

# ---------- 3. Robustness: alternative measures and specifications ----------
print('\n'+'='*66); print('3. Robustness checks'); print('='*66)
s['AI_q04_grp']=s.ST438Q04DA.round()
s['AI_high']=(s.AI_total>4.5).astype(float)
s['AI_sq']=s.AI_total_z**2
rows=[]
rows.append(run(s,'SCIE',FULL+['AI_g2','AI_g3','AI_g4','AI_g5'],'R1 基准(四题均值分组)'))
rows.append(run(s,'SCIE',FULL+['AI_high'],'R2 仅「每天」二分类'))
rows.append(run(s,'SCIE',FULL+['AI_total_z','AI_sq'],'R3 线性+二次'))
# R4: single items only, Q03 (draft writing) and Q02 (research)
rows.append(run(s,'SCIE',FULL+['ST438Q02DA','ST438Q03DA'],'R4 单题(查资料/写初稿)'))
# R5: does the linear term survive once the daily group is excluded?
s_no5=s[s.AI_grp<5]
rows.append(run(s_no5,'SCIE',FULL+['AI_total_z'],'R5 剔除每天组后的线性项'))
# R6: population weights (SENWT)
s2=s.copy(); s2['W_FSTUWT_save']=s2.W_FSTUWT
rows.append(run(s2,'SCIE',FULL+['AI_total_z'],'R6 线性项(正文模型)'))
res=pd.concat(rows,ignore_index=True)
fc=['AI_g2','AI_g3','AI_g4','AI_g5','AI_high','AI_total_z','AI_sq','ST438Q02DA','ST438Q03DA']
for lab in res.model.unique():
    print(f'\n--- {lab} (n={res[res.model==lab].n.iloc[0]}) ---')
    for _,w in res[(res.model==lab)&(res.term.isin(fc))].iterrows():
        st='***' if w.p<.001 else '**' if w.p<.01 else '*' if w.p<.05 else ''
        print(f'  {w.term:12s} b={w.b:8.2f} se={w.se:6.2f} p={w.p:.4f}{st}')
res.to_csv(os.path.join(BASE,'outputs','robustness.csv'),index=False,encoding='utf-8-sig')

# ---------- 4. Influence diagnostics: drop the extremes ----------
print('\n'+'='*66); print('4. Influence diagnostics: score distribution inside the daily group'); print('='*66)
d5=s[s.AI_grp==5]
print('  daily group n=%d, SCIE mean=%.1f, median=%.1f, P10=%.1f, P90=%.1f'%(
    len(d5), d5.PV1SCIE.mean(), d5.PV1SCIE.median(), d5.PV1SCIE.quantile(.1), d5.PV1SCIE.quantile(.9)))
print('  other groups n=%d, SCIE mean=%.1f, median=%.1f'%(len(s)-len(d5), s[s.AI_grp<5].PV1SCIE.mean(), s[s.AI_grp<5].PV1SCIE.median()))
