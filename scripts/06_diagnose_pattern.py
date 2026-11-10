# -*- coding: utf-8 -*-
"""Diagnostics: specification sensitivity of the AI use-purpose effects.

The two-dimension specification (C1) and the single-item specification (R4)
point in opposite directions; this script decomposes the conflict to establish
whether it is a collinearity artefact.
"""
import pandas as pd, numpy as np, os, warnings
from scipy import stats
warnings.filterwarnings('ignore')
from paths import BASE  # repo root (override with PISA_BASE)
q=pd.read_parquet(os.path.join(BASE,'data','qci_analysis.parquet'))
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
ITEMS=['ST438Q01DA','ST438Q02DA','ST438Q03DA','ST438Q04DA']
for c in ITEMS: q[c+'_z']=(q[c]-q[c].mean())/q[c].std()
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
    Xt=X.T*w;A=Xt@X;b=Xt@y
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
    return pd.DataFrame({'model':label,'term':names,'b':Qb,'se':se,'p':p,'n':len(df)})

need=list(dict.fromkeys(BASE_COV+DISP+CLAS+ICT+SCH+['AI_total','AI_deep','AI_sub']+ITEMS))
need=[c for c in need if c in q.columns]
s=q.dropna(subset=need+['PV1SCIE']).copy()
print('分析样本 n=%d'%len(s))

print('\n'+'='*70)
print('A. 单题零阶相关（未控制）')
print('='*70)
print(s[ITEMS+['PV1SCIE']].corr().loc[ITEMS,['PV1SCIE']].round(3).to_string())

print('\n'+'='*70)
print('B. 四题同时进入模型（全部标准化）')
print('='*70)
r=run(s,'SCIE',FULL+[c+'_z' for c in ITEMS],'B 四题全入')
for _,w in r[r.term.isin([c+'_z' for c in ITEMS])].iterrows():
    st='***' if w.p<.001 else '**' if w.p<.01 else '*' if w.p<.05 else ''
    print(f'  {w.term}  b={w.b:8.2f} se={w.se:5.2f} p={w.p:.4f}{st}')

print('\n'+'='*70)
print('C. 两两配对：哪个组合给出"一正一负"')
print('='*70)
pairs={'深化(Q02+Q04) vs 替代(Q01+Q03)':(('ST438Q02DA','ST438Q04DA'),('ST438Q01DA','ST438Q03DA')),
       '理解(Q01+Q04) vs 产出(Q02+Q03)':(('ST438Q01DA','ST438Q04DA'),('ST438Q02DA','ST438Q03DA')),
       '帮学习(Q04) vs 其余三题':(('ST438Q04DA',),('ST438Q01DA','ST438Q02DA','ST438Q03DA'))}
for lab,(a,b) in pairs.items():
    s['_A']=s[list(a)].mean(axis=1); s['_B']=s[list(b)].mean(axis=1)
    s['_Az']=(s._A-s._A.mean())/s._A.std(); s['_Bz']=(s._B-s._B.mean())/s._B.std()
    s['_Az_b']=(s._A-s._A.mean())/s._A.std(); s['_Bz_b']=(s._B-s._B.mean())/s._B.std()
    r2=run(s,'SCIE',FULL+['_Az_b','_Bz_b'],lab)
    v=r2[r2.term.isin(['_Az_b','_Bz_b'])]
    print(f'  {lab}:  A={v.iloc[0].b:7.2f}(p={v.iloc[0].p:.3f})   B={v.iloc[1].b:7.2f}(p={v.iloc[1].p:.3f})')

print('\n'+'='*70)
print('D. 单题逐题进入（控制其余三题均值）')
print('='*70)
s['other_mean']=s[ITEMS].mean(axis=1)
s['other_mean_z']=(s.other_mean-s.other_mean.mean())/s.other_mean.std()
for c,tag in zip(ITEMS,['Q01概括文本','Q02查资料','Q03写初稿','Q04帮学习']):
    s['_one']=(s[c]-s[c].mean())/s[c].std()
    r3=run(s,'SCIE',FULL+['_one','other_mean_z'],f'D {tag}')
    v=r3[r3.term=='_one'].iloc[0]
    st='***' if v.p<.001 else '**' if v.p<.01 else '*' if v.p<.05 else ''
    print(f'  {tag:10s} 净效应 b={v.b:7.2f} se={v.se:5.2f} p={v.p:.4f}{st}')

print('\n'+'='*70)
print('E. 组内 R² 增量：AI 双维度是否真的解释额外变异')
print('='*70)
def r2_of(cols):
    X,names=build(s,cols); y=s.PV1SCIE.values; w=s.W_FSTUWT.values
    b=wls(X,y,w); e=y-X@b
    ybar=np.average(y,weights=w)
    return 1-np.average(e**2,weights=w)/np.average((y-ybar)**2,weights=w)
r_base=r2_of(FULL); r_tot=r2_of(FULL+['AI_total_z'])
r_two=r2_of(FULL+['AI_deep_z','AI_sub_z'])
print(f'  基础模型 R²={r_base:.4f}')
print(f'  +AI总强度        ΔR²={r_tot-r_base:.5f}')
print(f'  +AI双维度        ΔR²={r_two-r_base:.5f}')
