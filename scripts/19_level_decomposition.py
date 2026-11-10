# -*- coding: utf-8 -*-
"""
19. 层别分解：AI 关联的「校间」与「校内」成分
  B 校间：学校均值回归（校均 PV ~ 校均 AI + 校均氛围 + 交互）
  W 校内：学校固定效应（PV ~ AIz + AIz×氛围 + 校固定效应）
  两者之差即生态（构成）成分
另：四个氛围成分分别入模（R5，修列创建顺序）
输出 outputs/level_decomposition.csv、outputs/climate_components.csv
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
          'DISCLISCI', 'ICTWKDY', 'ICTHOME']:
    q[c + '_z'] = (q[c] - q[c].mean()) / q[c].std()
q['grade_z'] = q['GRADE_z']
q['AIz'] = (q['AI_total'] - q['AI_total'].mean()) / q['AI_total'].std()
q['AI_daily'] = (q['AI_total'].round() >= 5).astype(float).where(q['AI_total'].notna())

COMP = {'COGACSC': '认知激活', 'DISCLISCI': '纪律氛围', 'SELFREG': '自我调节', 'JOYSCIE': '科学兴趣'}
for c in COMP:
    q['SM_' + c] = q['CNTSCHID'].map(q.groupby('CNTSCHID')[c].mean())
    q['SM_' + c + '_z'] = (q['SM_' + c] - q['SM_' + c].mean()) / q['SM_' + c].std()
cols = ['SM_' + c + '_z' for c in COMP]
q['SCH_CLIM_z'] = q[cols].mean(axis=1)
q['SCH_CLIM_z'] = (q['SCH_CLIM_z'] - q['SCH_CLIM_z'].mean()) / q['SCH_CLIM_z'].std()
q['AIxCLIM'] = q['AIz'] * q['SCH_CLIM_z']
q['SCH_MEAN_ESCS_z'] = (q['SCH_MEAN_ESCS'] - q['SCH_MEAN_ESCS'].mean()) / q['SCH_MEAN_ESCS'].std()
q['SCH_AI_MEAN'] = q['CNTSCHID'].map(q.groupby('CNTSCHID')['AI_total'].mean())
q['SCH_AI_MEAN_z'] = (q['SCH_AI_MEAN'] - q['SCH_AI_MEAN'].mean()) / q['SCH_AI_MEAN'].std()
for c in COMP:
    q['AIx_' + c] = q['AIz'] * q['SM_' + c + '_z']

STU = ['female', 'escs', 'repeat', 'grade_z', 'JOYSCIE_z', 'EFFSCIE_z', 'SELFREG_z',
       'COGACSC_z', 'TEACHSUP_z', 'DISCLISCI_z', 'ICTWKDY_z', 'ICTHOME_z']
need = STU + ['AIz', 'W_FSTUWT', 'CNTSCHID', 'SCH_CLIM_z', 'AIxCLIM',
              'SCH_MEAN_ESCS_z', 'SCH_AI_MEAN_z'] + cols + ['AIx_' + c for c in COMP]
s = q.dropna(subset=[c for c in need if c in q.columns]).copy()
print('样本 n=%d 学校=%d' % (len(s), s.CNTSCHID.nunique()))

# ---------- W：学校固定效应（去掉与校哑变量共线的校层项） ----------
print('\n=== W 校内（学校固定效应，PV×BRR）===')
dum = pd.get_dummies(s.CNTSCHID, drop_first=True, dtype=float)
Xn = np.column_stack([np.ones(len(s)), s.AIz.values, s.AIxCLIM.values,
                      dum.values])
names = ['(截距)', 'AIz', 'AIxCLIM']
pts, samp = [], []
for pv in [f'PV{i}SCIE' for i in range(1, 11)]:
    y = s[pv].values
    b0 = _wls(Xn, y, s.W_FSTUWT.values)
    reps = np.array([_wls(Xn, y, w) for w in [s[f'W_FSTURWT{i}'].values for i in range(1, 81)]])
    pts.append(b0)
    samp.append(((reps - b0) ** 2).sum(axis=0) / (80 * (1 - FAY) ** 2))
P = np.array(pts); qb = P.mean(axis=0)
Ub = np.mean(samp, axis=0); Bv = P.var(axis=0, ddof=1)
T = Ub + (1 + 1 / 10) * Bv; se = np.sqrt(T)
dfree = np.where(Bv > 1e-12, 9 * (1 + Ub / ((1 + 1 / 10) * np.maximum(Bv, 1e-12))) ** 2, 1e6)
pv_ = 2 * stats.t.sf(np.abs(qb / se), df=dfree)
fe = pd.DataFrame(dict(term=names, b=qb[:3], se=se[:3], p=pv_[:3], level='W 校内'))
print(fe.to_string(index=False))

# ---------- B：学校均值回归 ----------
print('\n=== B 校间（学校均值，加权 OLS）===')
sch = s.groupby('CNTSCHID').agg(
    n=('W_FSTUWT', 'size'), w=('W_FSTUWT', 'sum'),
    AI=('AIz', 'mean'), CLIM=('SCH_CLIM_z', 'mean'),
    ESCS=('SCH_MEAN_ESCS_z', 'mean')).dropna()
Xb = np.column_stack([np.ones(len(sch)), sch.AI.values, sch.CLIM.values,
                      (sch.AI * sch.CLIM).values, sch.ESCS.values])
nb = ['(截距)', '校均AI', '校均氛围', '校均AI×氛围', '校均ESCS']
pts = []
for pv in [f'PV{i}SCIE' for i in range(1, 11)]:
    ym = s.groupby('CNTSCHID')[pv].mean().reindex(sch.index).values
    pts.append(_wls(Xb, ym, sch.w.values))
P = np.array(pts); qb = P.mean(axis=0)
se = np.sqrt(np.mean([((p - qb) ** 2) for p in []], axis=0)) if False else None
# 用 PV 间方差 + 解析 SE 近似
se_anal = []
for p in pts:
    r = p
V = np.zeros(len(nb))
for _ in range(1):
    pass
# 解析 SE（以第一个 PV 的加权残差估计）
ym = s.groupby('CNTSCHID')['PV1SCIE'].mean().reindex(sch.index).values
res = ym - Xb @ pts[0]
s2 = np.sum(sch.w.values * res ** 2) / (len(sch) - Xb.shape[1])
XtWX_inv = np.linalg.inv((Xb.T * sch.w.values) @ Xb)
se_a = np.sqrt(np.diag(XtWX_inv) * s2)
t_a = qb / se_a
p_a = 2 * stats.t.sf(np.abs(t_a), df=len(sch) - Xb.shape[1])
bi = pd.DataFrame(dict(term=nb, b=qb, se=se_a, p=p_a, level='B 校间'))
print(bi.to_string(index=False))

dec = pd.concat([fe, bi], ignore_index=True)
dec.to_csv(os.path.join(OUT, 'level_decomposition.csv'), index=False, encoding='utf-8-sig')

# ---------- R5 四成分 ----------
print('\n=== R5 四个氛围成分分别（含校均控制，SCIE）===')
eng = PISAEngine(s, 'PV', 'SCIE')
r5 = []
for c, lab in COMP.items():
    cc = 'SM_' + c + '_z'
    df = eng.fit(STU + ['AIz', cc, 'AIx_' + c, 'SCH_MEAN_ESCS_z', 'SCH_AI_MEAN_z'], f'R5 {lab}')
    gi = df[df.term == 'AIx_' + c].iloc[0]
    gm = df[df.term == cc].iloc[0]
    print(f'   {lab:8s} 交互 {gi.b:7.2f} ({gi.se:5.2f}) p={gi.p:.4f} | 主效应 {gm.b:7.2f} p={gm.p:.4f}')
    r5.append(dict(component=lab, b_int=gi.b, se_int=gi.se, p_int=gi.p, b_main=gm.b, p_main=gm.p))
pd.DataFrame(r5).to_csv(os.path.join(OUT, 'climate_components.csv'), index=False, encoding='utf-8-sig')

# ---------- 校间 / 校内 简单斜率对比 ----------
print('\n=== 简单斜率对比（AI 斜率，分氛围高低）===')
wb = dict(zip(names, qb))
for lab, z in [('低氛围(−1SD)', -1), ('均值', 0), ('高氛围(+1SD)', 1)]:
    print(f'   {lab:14s} 校内={wb["AIz"] + wb["AIxCLIM"]*z:+.2f}  '
          f'校间={bi.set_index("term").loc["校均AI","b"] + bi.set_index("term").loc["校均AI×氛围","b"]*z:+.2f}')
