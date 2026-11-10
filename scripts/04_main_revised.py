# -*- coding: utf-8 -*-
"""Revised main analysis: non-linearity, moderation, and two-level decomposition.

Fixes collinearity by de-duplicating predictors, switches digital literacy to
ICTINFO, and adds school-level AI governance.
"""
import pandas as pd, numpy as np, os, warnings
from scipy import stats
warnings.filterwarnings('ignore')

from paths import BASE  # repo root (override with PISA_BASE)
OUT  = os.path.join(BASE,'outputs'); os.makedirs(OUT, exist_ok=True)
q = pd.read_parquet(os.path.join(BASE,'data','qci_analysis.parquet'))
REP = [f'W_FSTURWT{i}' for i in range(1,81)]
FAY, M_PV, R = 0.5, 10, 80

# ---------- Variables ----------
q['female']=(q.ST004D01T==1).astype(float); q['age']=q.AGE; q['grade']=q.GRADE
q['repeat']=(q.REPEAT>0).astype(float)
q['immig1']=(q.IMMIG==1).astype(float); q['immig2']=(q.IMMIG==2).astype(float)
q['escs']=q.ESCS

# School AI governance and training
for c in ['SC190Q01JA','SC190Q05JA','SC190Q06JA','SC190Q09JA']:
    q[c+'_b'] = (q[c]==1).astype(float).where(q[c].notna())
q['SCH_AIPOL']  = q[[c+'_b' for c in ['SC190Q01JA','SC190Q05JA','SC190Q06JA','SC190Q09JA']]].mean(axis=1)
q['SCH_AITRAIN']= q['SC265Q13DA']

Z = ['AI_total','AI_deep','AI_sub','ICTINFO','ICTQUAL','ICTHOME','ICTWKDY','DL_crit',
     'JOYSCIE','EFFSCIE','SELFREG','PERSEV','CURIO','ENGSCIPR','TEACHSUP','COGACSC',
     'DISCLISCI','SCIECOLLR','DIGPREP','SCTRAIN','EDUSHORT','STAFFSHORT','NEGSCLIM',
     'SCH_AIPOL','SCH_AITRAIN','HOMEPOS']
for v in Z:
    if v in q.columns: q[v+'_z'] = (q[v]-q[v].mean())/q[v].std()
q['escs_z'] = (q.escs-q.escs.mean())/q.escs.std()

g = q.groupby('CNTSCHID')
q['SCH_ESCS'] = g['ESCS'].transform('mean')
q['SCH_AI']   = g['AI_total'].transform('mean')
q['SCH_ESCS_z'] = (q.SCH_ESCS-q.SCH_ESCS.mean())/q.SCH_ESCS.std()
q['SCH_AI_z']   = (q.SCH_AI-q.SCH_AI.mean())/q.SCH_AI.std()
q['AI_within']  = q.AI_total_z - q.SCH_AI_z

# AI non-linearity: groupings + squared term
q['AI_grp'] = q.AI_total.round()
for k in [2,3,4,5]: q[f'AI_g{k}'] = (q.AI_grp==k).astype(float)
q['AI_sq'] = q.AI_total_z**2

# Interaction terms
for a,b,nm in [('AI_total_z','ICTINFO_z','AIxDL'),('AI_total_z','escs_z','AIxSES'),
               ('AI_total_z','DIGPREP_z','AIxPREP'),('AI_total_z','SCTRAIN_z','AIxTRN'),
               ('AI_total_z','SCH_AIPOL_z','AIxPOL'),('AI_total_z','SCH_AITRAIN_z','AIxATRN'),
               ('AI_total_z','SCH_AI_z','AIxSCHAI')]:
    q[nm] = q[a]*q[b]

BASE_COV = ['female','age','grade','repeat','immig1','immig2','escs']
DISP = ['JOYSCIE_z','EFFSCIE_z','SELFREG_z','PERSEV_z','CURIO_z','ENGSCIPR_z']
CLAS = ['TEACHSUP_z','COGACSC_z','DISCLISCI_z','SCIECOLLR_z']
ICT  = ['ICTHOME_z','ICTQUAL_z','ICTWKDY_z','ICTINFO_z']
SCH  = ['DIGPREP_z','SCTRAIN_z','EDUSHORT_z','STAFFSHORT_z','NEGSCLIM_z','SCH_ESCS_z']
FULL = BASE_COV + DISP + CLAS + ICT + SCH

def build(df, cols):
    cols = list(dict.fromkeys(cols))          # key step: de-duplicate to avoid collinearity
    X = np.column_stack([np.ones(len(df))]+[df[c].values.astype(float) for c in cols])
    return X, ['Intercept']+cols

def wls(X,y,w):
    Xt = X.T*w; A = Xt@X; b = Xt@y
    try: return np.linalg.solve(A,b)
    except np.linalg.LinAlgError: return np.linalg.lstsq(A,b,rcond=None)[0]

def run(df, yname, cols, label):
    X, names = build(df, cols)
    Y = df[[f'PV{k}{yname}' for k in range(1,M_PV+1)]].values
    W = df[REP].values; w0 = df['W_FSTUWT'].values
    cs, vs = [], []
    for j in range(Y.shape[1]):
        y = Y[:,j]; b0 = wls(X,y,w0)
        if not np.all(np.isfinite(b0)): continue
        D = np.array([wls(X,y,W[:,r])-b0 for r in range(R)])
        cs.append(b0); vs.append((D**2).sum(axis=0)/(R*(1-FAY)**2))
    cs, vs = np.array(cs), np.array(vs); m = len(cs)
    Qb, Ub = cs.mean(0), vs.mean(0)
    B = cs.var(0,ddof=1) if m>1 else np.zeros_like(Ub)
    T = Ub + (1+1/m)*B
    se = np.sqrt(np.where(T>0,T,np.nan))
    with np.errstate(divide='ignore',invalid='ignore'):
        dfr = np.where(B>0,(m-1)*(1+Ub/((1+1/m)*B))**2,np.inf)
        t = Qb/se; p = 2*stats.t.sf(np.abs(t), np.where(np.isfinite(dfr),dfr,1e6))
    return pd.DataFrame({'model':label,'outcome':yname,'term':names,'b':Qb,'se':se,
                         't':t,'p':p,'n':len(df),'M_PV':m})

print('='*72); print('Analysis sample'); print('='*72)
need = list(dict.fromkeys(BASE_COV+DISP+CLAS+ICT+SCH+
        ['AI_total','AI_deep','AI_sub','AI_grp','DIGPREP','SCH_AIPOL','SCH_AITRAIN','SCH_AI']))
need = [c for c in need if c in q.columns]
s = q.dropna(subset=need+['PV1SCIE','PV1CMPS']).copy()
print(f'Complete cases n={len(s)} ({len(s)/len(q)*100:.1f}%), schools={s.CNTSCHID.nunique()}')

# ---------- Model set ----------
MODELS = {
 'A1 AI分组(基础)':  BASE_COV+['AI_g2','AI_g3','AI_g4','AI_g5'],
 'A2 AI分组(全模型)': FULL+['AI_g2','AI_g3','AI_g4','AI_g5'],
 'A3 AI线性+平方':    FULL+['AI_total_z','AI_sq'],
 'B1 AI×数字素养':    FULL+['AI_total_z','AIxDL'],
 'B2 AI×家庭SES':     FULL+['AI_total_z','AIxSES'],
 'B3 AI×学校数字准备度': FULL+['AI_total_z','AIxPREP'],
 'B4 AI×科学教师培训':  FULL+['AI_total_z','AIxTRN'],
 'B5 AI×学校AI治理':   FULL+['AI_total_z','AIxPOL'],
 'B6 AI×教师AI应对':   FULL+['AI_total_z','AIxATRN'],
 'C1 AI双维度(全模型)': FULL+['AI_deep_z','AI_sub_z'],
 'D1 组内组间分解':    FULL+['AI_within','SCH_AI_z'],
}
FOCUS = lambda c: ('AI' in c) or c in ('ICTINFO_z','escs','DIGPREP_z','SCTRAIN_z','SCH_AIPOL_z','SCH_AITRAIN_z')

rows=[]
for lab, cols in MODELS.items():
    for yv in ['SCIE','CMPS']:
        r = run(s, yv, cols, lab); rows.append(r)
        print(f'\n--- {lab} | {yv} ---')
        for _,w in r[r.term.map(FOCUS)].iterrows():
            st = '***' if w.p<.001 else '**' if w.p<.01 else '*' if w.p<.05 else ''
            print(f'  {w.term:16s} b={w.b:8.2f} se={w.se:6.2f} p={w.p:.4f} {st}')
res = pd.concat(rows, ignore_index=True)
res.to_csv(os.path.join(OUT,'regression_main.csv'), index=False, encoding='utf-8-sig')

# ---------- ICC ----------
print('\n'+'='*72); print('Multilevel ICC (school share of variance in the outcome)'); print('='*72)
import statsmodels.formula.api as smf
for yv in ['PV1SCIE','PV1CMPS']:
    d = s[[yv,'CNTSCHID']].rename(columns={yv:'y'}).dropna()
    m0 = smf.mixedlm('y ~ 1', d, groups=d['CNTSCHID']).fit()
    vb = float(m0.cov_re.iloc[0,0]); vw = float(m0.scale)
    print(f'  {yv}: between-school variance={vb:.1f}, within-school variance={vw:.1f}, ICC={vb/(vb+vw):.3f}')

# ---------- Weighted group means (descriptive) ----------
print('\n'+'='*72); print('AI use group x weighted score'); print('='*72)
def wm(x,w):
    m=~np.isnan(x); return np.average(x[m],weights=w[m]) if m.any() else np.nan
lab5={1:'never/almost never',2:'1-2 times a year',3:'1-2 times a month',4:'1-2 times a week',5:'every day/almost every day'}
for yv in ['PV1SCIE','PV1CMPS']:
    print(f'\n{yv}:')
    for k in range(1,6):
        sub=s[s.AI_grp==k]
        print(f'  {lab5[k]:12s} n={len(sub):5d}  weighted mean={wm(sub[yv].values,sub.W_FSTUWT.values):6.1f}')
print('\nSaved outputs/regression_main.csv')
