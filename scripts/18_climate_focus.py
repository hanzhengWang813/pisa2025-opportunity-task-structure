# -*- coding: utf-8 -*-
"""
18. 聚焦检验：学校学术氛围（校均值合成）× 学生 AI 使用

合成 SCH_CLIM = 校均(认知激活, 纪律氛围, 自我调节, 科学兴趣) 的 z 分数均值
（四个成分同向、高度相关，合并以避免多重比较）

稳健性：
  R1 加入校均 ESCS / 校均 AI 使用 / 学校规模
  R2 学校固定效应（校内比较，消除全部校间构成）
  R3 用「每天使用」哑变量替代连续强度
  R4 剔除每天使用组
  R5 四个成分分别入模
输出 outputs/climate_focus.csv、outputs/climate_robust.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from pisa_engine import PISAEngine, _wls, FAY

from paths import BASE  # repo root (override with PISA_BASE)
OUT = os.path.join(BASE, 'outputs')
q = pd.read_parquet(os.path.join(BASE, 'data/qci_school.parquet'))

q['female'] = (q['ST004D01T'] == 1).astype(float).where(q['ST004D01T'].notna())
q['repeat'] = (q['REPEAT'] > 0).astype(float).where(q['REPEAT'].notna())
q['escs'] = q['ESCS']
for c in ['ESCS', 'GRADE', 'JOYSCIE', 'EFFSCIE', 'SELFREG', 'COGACSC', 'TEACHSUP',
          'DISCLISCI', 'ICTWKDY', 'ICTHOME', 'SCIECOLLR']:
    q[c + '_z'] = (q[c] - q[c].mean()) / q[c].std()
q['grade_z'] = q['GRADE_z']
q['AIz'] = (q['AI_total'] - q['AI_total'].mean()) / q['AI_total'].std()
q['AI_daily'] = (q['AI_total'].round() >= 5).astype(float).where(q['AI_total'].notna())
q['SCH_SIZE'] = q['CNTSCHID'].map(q.groupby('CNTSCHID')['W_FSTUWT'].size())
q['SCH_AI_MEAN'] = q['CNTSCHID'].map(q.groupby('CNTSCHID')['AI_total'].mean())

# ---------- 学校学术氛围合成 ----------
COMP = {'COGACSC': '认知激活', 'DISCLISCI': '纪律氛围', 'SELFREG': '自我调节', 'JOYSCIE': '科学兴趣'}
for c in COMP:
    q['SM_' + c] = q['CNTSCHID'].map(q.groupby('CNTSCHID')[c].mean())
    q['SM_' + c + '_z'] = (q['SM_' + c] - q['SM_' + c].mean()) / q['SM_' + c].std()
cols = ['SM_' + c + '_z' for c in COMP]
# 校层信度（四个校均值的 alpha）
sch = q.groupby('CNTSCHID')[cols].first().dropna()
k = len(cols)
a = k / (k - 1) * (1 - sch.var(ddof=1).sum() / sch.sum(axis=1).var(ddof=1))
print(f'学校学术氛围合成：4 个校均值的 α = {a:.3f}（n={len(sch)} 校）')
print(sch.corr().round(2).to_string())

q['SCH_CLIM'] = q[cols].mean(axis=1)
q['SCH_CLIM_z'] = (q['SCH_CLIM'] - q['SCH_CLIM'].mean()) / q['SCH_CLIM'].std()
q['AIxCLIM'] = q['AIz'] * q['SCH_CLIM_z']
q['DAILYxCLIM'] = q['AI_daily'] * q['SCH_CLIM_z']
q['SCH_MEAN_ESCS_z'] = (q['SCH_MEAN_ESCS'] - q['SCH_MEAN_ESCS'].mean()) / q['SCH_MEAN_ESCS'].std()
q['SCH_AI_MEAN_z'] = (q['SCH_AI_MEAN'] - q['SCH_AI_MEAN'].mean()) / q['SCH_AI_MEAN'].std()
q['SCH_SIZE_z'] = (q['SCH_SIZE'] - q['SCH_SIZE'].mean()) / q['SCH_SIZE'].std()

STU = ['female', 'escs', 'repeat', 'grade_z', 'JOYSCIE_z', 'EFFSCIE_z', 'SELFREG_z',
       'COGACSC_z', 'TEACHSUP_z', 'DISCLISCI_z', 'ICTWKDY_z', 'ICTHOME_z']

need = STU + ['AIz', 'W_FSTUWT', 'CNTSCHID', 'SCH_CLIM_z', 'AIxCLIM',
              'SCH_MEAN_ESCS_z', 'SCH_AI_MEAN_z', 'SCH_SIZE_z'] + cols
s = q.dropna(subset=[c for c in need if c in q.columns]).copy()
print('样本 n=%d 学校=%d' % (len(s), s.CNTSCHID.nunique()))

rows = []
def rec(model, outcome, df):
    for _, r in df.iterrows():
        rows.append(dict(model=model, outcome=outcome, term=r.term, b=r.b, se=r.se, p=r.p))

for outcome in ['SCIE', 'CMPS']:
    eng = PISAEngine(s, 'PV', outcome)
    rec('F1 基础', outcome, eng.fit(STU + ['AIz'] + cols, 'F1 基础'))
    rec('F2 +氛围主效应与交互', outcome,
        eng.fit(STU + ['AIz', 'SCH_CLIM_z', 'AIxCLIM'], 'F2'))
    rec('F3 +校均控制', outcome,
        eng.fit(STU + ['AIz', 'SCH_CLIM_z', 'AIxCLIM', 'SCH_MEAN_ESCS_z',
                       'SCH_AI_MEAN_z', 'SCH_SIZE_z'], 'F3'))
    rec('R3 每天哑变量', outcome,
        eng.fit(STU + ['AI_daily', 'SCH_CLIM_z', 'DAILYxCLIM', 'SCH_MEAN_ESCS_z',
                       'SCH_AI_MEAN_z', 'SCH_SIZE_z'], 'R3'))
    # R4 剔除每天使用组
    s2 = s[s.AI_daily < 1]
    eng2 = PISAEngine(s2, 'PV', outcome)
    rec('R4 剔除每天组', outcome,
        eng2.fit(STU + ['AIz', 'SCH_CLIM_z', 'AIxCLIM', 'SCH_MEAN_ESCS_z'], 'R4'))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'climate_focus.csv'), index=False, encoding='utf-8-sig')

def show(model, outcome, terms=None):
    sub = res[(res.model == model) & (res.outcome == outcome)]
    if terms:
        sub = sub[sub.term.isin(terms)]
    for _, r in sub.iterrows():
        st = '***' if r.p < .001 else '**' if r.p < .01 else '*' if r.p < .05 else ''
        print(f'   {r.term:20s} {r.b:8.2f} ({r.se:5.2f}) p={r.p:.4f}{st}')

print('\n=== SCIE ===')
for m in ['F2 +氛围主效应与交互', 'F3 +校均控制', 'R3 每天哑变量', 'R4 剔除每天组']:
    print(' --', m); show(m, 'SCIE', ['AIz', 'AI_daily', 'SCH_CLIM_z', 'AIxCLIM', 'DAILYxCLIM'])
print('\n=== CMPS ===')
for m in ['F2 +氛围主效应与交互', 'F3 +校均控制']:
    print(' --', m); show(m, 'CMPS', ['AIz', 'SCH_CLIM_z', 'AIxCLIM'])

# ---------- 简单斜率 ----------
print('\n=== 简单斜率（SCIE，F3 模型）===')
f3 = res[(res.model == 'F3 +校均控制') & (res.outcome == 'SCIE')].set_index('term')
b_ai, b_int = f3.loc['AIz', 'b'], f3.loc['AIxCLIM', 'b']
for lab, z in [('低氛围 (−1SD)', -1), ('中', 0), ('高氛围 (+1SD)', 1)]:
    print(f'   {lab:14s} AI 斜率 = {b_ai + b_int * z:+.2f} 分/SD')

# ---------- R2 学校固定效应（校内比较） ----------
print('\n=== R2 学校固定效应（校内识别，SCIE）===')
sfe = s.dropna(subset=['PV1SCIE']).copy()
dummies = pd.get_dummies(sfe.CNTSCHID, drop_first=True, dtype=float)
base_cols = ['AIz', 'SCH_CLIM_z', 'AIxCLIM']
X = np.column_stack([np.ones(len(sfe))] +
                    [sfe[c].values for c in base_cols] +
                    [dummies[c].values for c in dummies.columns])
names = ['(截距)'] + base_cols
pts, samp = [], []
for pv in [f'PV{i}SCIE' for i in range(1, 11)]:
    y = sfe[pv].values
    b0 = _wls(X, y, sfe.W_FSTUWT.values)
    reps = np.array([_wls(X, y, w) for w in
                     [sfe[f'W_FSTURWT{i}'].values for i in range(1, 81)]])
    v = ((reps - b0) ** 2).sum(axis=0) / (80 * (1 - FAY) ** 2)
    pts.append(b0); samp.append(v)
P = np.array(pts); qbar = P.mean(axis=0)
Ubar = np.mean(samp, axis=0); Bv = P.var(axis=0, ddof=1)
T = Ubar + (1 + 1 / 10) * Bv
se = np.sqrt(T)
dfree = np.where(Bv > 1e-12, 9 * (1 + Ubar / ((1 + 1 / 10) * np.maximum(Bv, 1e-12))) ** 2, 1e6)
p = 2 * stats.t.sf(np.abs(qbar / se), df=dfree)
for nm in ['AIz', 'SCH_CLIM_z', 'AIxCLIM']:
    i = names.index(nm)
    st = '***' if p[i] < .001 else '**' if p[i] < .01 else '*' if p[i] < .05 else ''
    print(f'   {nm:14s} {qbar[i]:8.2f} ({se[i]:5.2f}) p={p[i]:.4f}{st}')
pd.DataFrame(dict(term=names[:4], b=qbar[:4], se=se[:4], p=p[:4])).to_csv(
    os.path.join(OUT, 'climate_schoolFE.csv'), index=False, encoding='utf-8-sig')

# ---------- R5 四成分分别 ----------
print('\n=== R5 四成分分别（含校均控制，SCIE）===')
eng = PISAEngine(s, 'PV', 'SCIE')
r5 = []
for c, lab in COMP.items():
    cc = 'SM_' + c + '_z'
    s['AIx_' + c] = s.AIz * s[cc]
    df = eng.fit(STU + ['AIz', cc, 'AIx_' + c, 'SCH_MEAN_ESCS_z', 'SCH_AI_MEAN_z'],
                 f'R5 {lab}')
    gi = df[df.term == 'AIx_' + c].iloc[0]
    gm = df[df.term == cc].iloc[0]
    print(f'   {lab:8s} 交互 {gi.b:7.2f} ({gi.se:5.2f}) p={gi.p:.4f} | 主效应 {gm.b:7.2f} p={gm.p:.4f}')
    r5.append(dict(component=lab, b_int=gi.b, se_int=gi.se, p_int=gi.p,
                   b_main=gm.b, p_main=gm.p))
pd.DataFrame(r5).to_csv(os.path.join(OUT, 'climate_components.csv'), index=False, encoding='utf-8-sig')
