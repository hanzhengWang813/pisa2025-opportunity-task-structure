# -*- coding: utf-8 -*-
"""03 主分析：PV(10) x BRR-Fay(80) 引擎 + 调节效应模型
规范：PISA 2025 用 10 组能力值、80 个 BRR-Fay 重复权重(ρ=0.5)、Rubin 规则合并
"""
import pandas as pd, numpy as np, os, json, warnings
from scipy import stats
warnings.filterwarnings('ignore')

from paths import BASE  # repo root (override with PISA_BASE)
OUT  = os.path.join(BASE,'outputs'); os.makedirs(OUT, exist_ok=True)
q = pd.read_parquet(os.path.join(BASE,'data','qci_analysis.parquet'))
REP = [f'W_FSTURWT{i}' for i in range(1,81)]
FAY = 0.5; M_PV = 10; R = 80

# ---------------- 变量准备 ----------------
q['female'] = (q.ST004D01T==1).astype(float)
q['age']    = q.AGE
q['grade']  = q.GRADE
q['repeat'] = (q.REPEAT>0).astype(float)
q['immig1'] = (q.IMMIG==1).astype(float)   # 第二代
q['immig2'] = (q.IMMIG==2).astype(float)   # 第一代
q['escs']   = q.ESCS

def zscore(s):
    return (s - s.mean())/s.std()

for v in ['AI_total','AI_deep','AI_sub','DL_crit','DL_ai','JOYSCIE','EFFSCIE','SELFREG',
          'PERSEV','CURIO','TEACHSUP','COGACSC','DISCLISCI','EXPECEDU','ICTHOME','ICTQUAL',
          'ICTWKDY','HOMEPOS','DIGPREP','AVLRSOFT','SCTRAIN','EDUSHORT','STAFFSHORT',
          'NEGSCLIM','SCIECOLLR','ENGSCIPR']:
    if v in q.columns: q[v+'_z'] = zscore(q[v])

# 学校层：学校平均 SES 与平均 AI 使用（用于分离组内/组间效应）
g = q.groupby('CNTSCHID')
q['SCH_ESCS']  = g['ESCS'].transform('mean')
q['SCH_AI']    = g['AI_total'].transform('mean')

BASE_COV = ['female','age','grade','repeat','immig1','immig2','escs']
DISP_COV = ['JOYSCIE_z','EFFSCIE_z','SELFREG_z','PERSEV_z','CURIO_z','ENGSCIPR_z']
CLAS_COV = ['TEACHSUP_z','COGACSC_z','DISCLISCI_z','SCIECOLLR_z']
ICT_COV  = ['ICTHOME_z','ICTQUAL_z','ICTWKDY_z']
SCH_COV  = ['DIGPREP_z','SCTRAIN_z','EDUSHORT_z','STAFFSHORT_z','NEGSCLIM_z','SCH_ESCS']

def build(df, cols):
    X = np.column_stack([np.ones(len(df))] + [df[c].values.astype(float) for c in cols])
    return X, ['Intercept']+list(cols)

def wls(X, y, w):
    XtW = X.T * w
    A = XtW @ X
    b = XtW @ y
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]

def run_model(df, yname, cols, label):
    """PV x BRR-Fay + Rubin 合并"""
    X, names = build(df, cols)
    Y = df[[f'PV{k}{yname}' for k in range(1,M_PV+1)]].values
    W = df[REP].values                       # n x 80
    w0 = df['W_FSTUWT'].values
    M = Y.shape[1]

    coefs, vars_ = [], []
    for j in range(M):
        y = Y[:, j]
        b0 = wls(X, y, w0)
        if not np.all(np.isfinite(b0)): continue
        diffs = np.empty((R, len(b0)))
        for r in range(R):
            br = wls(X, y, W[:, r])
            diffs[r] = br - b0
        V = (diffs**2).sum(axis=0) / (R * (1-FAY)**2)
        coefs.append(b0); vars_.append(V)
    coefs = np.array(coefs); vars_ = np.array(vars_)
    M = len(coefs)

    Qbar = coefs.mean(axis=0)
    Ubar = vars_.mean(axis=0)
    B    = coefs.var(axis=0, ddof=1) if M > 1 else np.zeros_like(Ubar)
    T    = Ubar + (1 + 1/M)*B
    se   = np.sqrt(np.where(T > 0, T, np.nan))
    with np.errstate(divide='ignore', invalid='ignore'):
        dfr = np.where(B > 0, (M-1)*(1 + Ubar/((1+1/M)*B))**2, np.inf)
        t   = Qbar/se
        p   = 2*stats.t.sf(np.abs(t), np.where(np.isfinite(dfr), dfr, 1e6))
    return pd.DataFrame({'model':label,'outcome':yname,'term':names,'b':Qbar,'se':se,
                         't':t,'p':p,'n':len(df),'M_PV':M})

print('='*70); print('样本与结果变量'); print('='*70)
print('总学生 n =', len(q))

# ---------------- 模型定义（先定义，再据此确定分析样本）----------------
sample = q.copy()
models = {
 'M0 基础':            BASE_COV,
 'M1 +AI强度':         BASE_COV + ['AI_total_z'],
 'M2 +AI双维度':        BASE_COV + ['AI_deep_z','AI_sub_z'],
 'M3 +倾向与课堂':      BASE_COV + ['AI_total_z'] + DISP_COV + CLAS_COV + ICT_COV,
 'M4 +学校层':          BASE_COV + ['AI_total_z'] + DISP_COV + CLAS_COV + ICT_COV + SCH_COV,
 'M5 数字素养调节':      BASE_COV + ['AI_total_z','DL_crit_z','AI_total_z:DL_crit_z'] + DISP_COV + CLAS_COV + ICT_COV + SCH_COV,
 'M6 SES调节':         BASE_COV + ['AI_total_z','escs','AI_total_z:escs'] + DISP_COV + CLAS_COV + ICT_COV + SCH_COV,
 'M7 学校准备度跨层调节': BASE_COV + ['AI_total_z','DIGPREP_z','AI_total_z:DIGPREP_z'] + DISP_COV + CLAS_COV + ICT_COV + SCH_COV,
}

sample['AI_total_z:DL_crit_z'] = sample.AI_total_z * sample.DL_crit_z
sample['AI_total_z:escs']      = sample.AI_total_z * sample.escs
sample['AI_total_z:DIGPREP_z'] = sample.AI_total_z * sample.DIGPREP_z

# 分析样本：所有模型用到的变量取交集后 listwise 删除
need_all = sorted({c for cols in models.values() for c in cols})
need_all += ['AI_deep','AI_sub','DIGPREP','SCH_ESCS']
need_all = [c for c in dict.fromkeys(need_all) if c in sample.columns]
sample = sample.dropna(subset=need_all + ['PV1SCIE','PV1CMPS']).copy()
print('完整案例分析样本 n =', len(sample), f'({len(sample)/len(q)*100:.1f}%)')
print('学校数 =', sample.CNTSCHID.nunique())
print('模型中变量数 =', len(need_all))

allres = []
for label, cols in models.items():
    for yv in ['SCIE','CMPS']:
        r = run_model(sample, yv, cols, label)
        allres.append(r)
        key = [c for c in cols if 'AI' in c or 'DL' in c or 'escs' in c or 'DIGPREP' in c]
        sub = r[r.term.isin(key)]
        print(f'\n--- {label} | 结果变量 {yv} ---')
        for _,row in sub.iterrows():
            star = '***' if row.p<0.001 else '**' if row.p<0.01 else '*' if row.p<0.05 else ''
            print(f'  {row.term:26s} b={row.b:8.3f} se={row.se:6.3f} p={row.p:.4f} {star}')

res = pd.concat(allres, ignore_index=True)
res.to_csv(os.path.join(OUT,'regression_results.csv'), index=False, encoding='utf-8-sig')
print('\n已保存 outputs/regression_results.csv  (行数 %d)' % len(res))

# ---------------- 组内/组间分解 (M7 补充) ----------------
print('\n' + '='*70); print('学校层效应分解'); print('='*70)
sample['SCH_AI_z'] = zscore(sample.SCH_AI)
sample['AI_within'] = sample.AI_total_z - sample.SCH_AI_z
r = run_model(sample, 'SCIE', BASE_COV + ['AI_within','SCH_AI_z'] + DISP_COV + CLAS_COV + ICT_COV + SCH_COV, 'M8 组内组间分解')
print(r[r.term.isin(['AI_within','SCH_AI_z'])].round(3).to_string(index=False))
r.to_csv(os.path.join(OUT,'regression_within_between.csv'), index=False, encoding='utf-8-sig')
