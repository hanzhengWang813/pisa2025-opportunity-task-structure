# -*- coding: utf-8 -*-
"""02 变量构造 + AI 使用模式的维度结构检验"""
import pandas as pd, numpy as np, pickle, os, json, warnings
warnings.filterwarnings('ignore')
from scipy import stats

from paths import BASE  # repo root (override with PISA_BASE)
q = pd.read_parquet(os.path.join(BASE,'data','qci_merged.parquet'))
M_STU = pickle.load(open(os.path.join(BASE,'meta.pkl'),'rb'))
M_SCH = pickle.load(open(os.path.join(BASE,'meta_sch.pkl'),'rb'))

MISS_PAT = ['valid skip','system missing','not administered','invalid','no response','n/a']
def missing_codes(vl):
    return {int(c) for c,l in vl.items() if l and any(p in str(l).lower() for p in MISS_PAT)}

VLAB = {**M_STU['vlabels'], **M_SCH['vlabels']}
n_clean = 0
for c in q.columns:
    if c in VLAB and q[c].dtype.kind in 'if':
        mc = missing_codes(VLAB[c])
        if mc:
            mask = q[c].isin(mc)
            if mask.any():
                q.loc[mask, c] = np.nan
                n_clean += int(mask.sum())
print('清理缺失码单元格数:', n_clean)

# 额外：ST436 系列有效值为 1-4
for c in [x for x in q.columns if x.startswith('ST436Q')]:
    q.loc[q[c] > 4, c] = np.nan

AI = ['ST438Q01DA','ST438Q02DA','ST438Q03DA','ST438Q04DA']

# ============ 维度结构检验 ============
print('\n' + '='*60)
print('AI 使用模式维度结构检验')
print('='*60)
d = q[AI].dropna()
print('完整案例 n =', len(d))

# 1. KMO & Bartlett
from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
kmo_all, kmo_model = calculate_kmo(d)
chi2, p = calculate_bartlett_sphericity(d)
print(f'KMO = {kmo_model:.3f} | Bartlett chi2={chi2:.1f}, p={p:.3g}')

# 2. 特征值 + 平行分析
R = d.corr().values
ev = np.sort(np.linalg.eigvalsh(R))[::-1]
rng = np.random.default_rng(42)
sim = []
for _ in range(500):
    X = rng.normal(size=d.shape)
    Rs = np.corrcoef(X, rowvar=False)
    sim.append(np.sort(np.linalg.eigvalsh(Rs))[::-1])
sim = np.array(sim)
print(f'特征值: {np.round(ev,3)}')
print(f'平行分析 95 分位随机特征值: {np.round(np.percentile(sim,95,axis=0),3)}')
print(f'建议保留因子数(平行分析): {(ev > np.percentile(sim,95,axis=0)).sum()}')
print(f'方差解释率(第1因子): {ev[0]/4*100:.1f}%')

# 3. CFA 模型比较
import semopy
def cfa(desc, data):
    m = semopy.Model(desc)
    m.fit(data)
    st = semopy.calc_stats(m).T
    vals = st[0].to_dict()
    return {k: float(vals.get(k, np.nan)) for k in ['CFI','TLI','RMSEA','SRMR','AIC','BIC','chi2','DoF']}, m
res = {}
for name, desc in [
    ('单因子(使用强度)', 'F =~ ST438Q01DA + ST438Q02DA + ST438Q03DA + ST438Q04DA'),
    ('双因子A 代写vs辅助', 'F1 =~ ST438Q03DA\nF2 =~ ST438Q02DA + ST438Q04DA + ST438Q01DA'),
    ('双因子B 文本vs探究', 'F1 =~ ST438Q01DA + ST438Q03DA\nF2 =~ ST438Q02DA + ST438Q04DA'),
]:
    try:
        r, mm = cfa(desc, d)
        res[name] = r
    except Exception as e:
        res[name] = {'err': str(e)[:60]}
print('\nCFA 模型比较:')
print(pd.DataFrame(res).T.round(3).to_string())

# 4. 信度
def alpha(X):
    k = X.shape[1]; return k/(k-1)*(1 - X.var(ddof=1).sum()/X.sum(axis=1).var(ddof=1))
print(f'\nCronbach α (四题单维) = {alpha(d):.3f}')
def sb2(r): return 2*r/(1+r)
print(f'Spearman-Brown 深化型(Q02,Q04) = {sb2(d.ST438Q02DA.corr(d.ST438Q04DA)):.3f}')
print(f'Spearman-Brown 替代型(Q01,Q03) = {sb2(d.ST438Q01DA.corr(d.ST438Q03DA)):.3f}')

# ============ 构造变量 ============
q['AI_deep'] = q[['ST438Q02DA','ST438Q04DA']].mean(axis=1)      # 深化型：查资料+帮学习
q['AI_sub']  = q[['ST438Q01DA','ST438Q03DA']].mean(axis=1)      # 替代型：概括文本+写初稿
q['AI_total']= q[AI].mean(axis=1)                                # 使用强度
z = lambda s: (s - q.AI_total.mean())/q.AI_total.std()           # 占位, 下面重算
mu, sd = q.AI_deep.mean(), q.AI_deep.std()
q['AI_bal'] = (q.AI_deep - q.AI_sub) / q[['ST438Q02DA','ST438Q04DA','ST438Q01DA','ST438Q03DA']].stack().std()

# 数字素养：批判性评估（线上信息 + AI输出）
q['DL_info'] = q['ST436Q02DA']
q['DL_ai']   = q['ST436Q16DA']
dl = q[['DL_info','DL_ai']].dropna()
print(f'\nDL_info 与 DL_ai 相关 r = {dl.DL_info.corr(dl.DL_ai):.3f}, n={len(dl)}')
qz = (q[['DL_info','DL_ai']] - q[['DL_info','DL_ai']].mean()) / q[['DL_info','DL_ai']].std()
q['DL_crit'] = qz.mean(axis=1)
q['DL_high'] = (q.DL_ai >= 3).astype(float).where(q.DL_ai.notna())   # 经常/非常频繁评估AI输出

# 结果与协变量
q['female'] = (q['ST004D01T']==1).astype(float)
q['SES']    = q['ESCS']
q['REPEAT_bin'] = (q['REPEAT']>0).astype(float).where(q['REPEAT'].notna())
print('\n核心变量描述:')
cols=['AI_total','AI_deep','AI_sub','AI_bal','DL_crit','DL_ai','DL_info','DL_high','SES','female','PV1SCIE']
print(q[cols].describe().T[['count','mean','std','min','max']].round(3).to_string())

q.to_parquet(os.path.join(BASE,'data','qci_analysis.parquet'), index=False)
print('\n已保存 data/qci_analysis.parquet', q.shape)
