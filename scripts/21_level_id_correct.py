# -*- coding: utf-8 -*-
"""
21. 统一协变量下的校间/校内识别（修正 19、20 的设定不一致）

所有模型使用同一套学生层协变量 STU：
  A 单层：            y ~ STU + AIz
  B Mundlak：         y ~ STU + AI_within + AI_between
  C 校内 FE：         y ~ STU + AI_within + 校固定效应   （应与 B 的 within 系数一致）
  D 氛围交互三版：     单层 / Mundlak / FE
  E 四用途：          单层 vs 校内（组均值中心化）

输出 outputs/level_id.csv、outputs/uses_within.csv
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

IT = ['ST438Q01DA', 'ST438Q02DA', 'ST438Q03DA', 'ST438Q04DA']
LAB = dict(zip(IT, ['总结阅读材料', '初步研究新主题', '起草写作', '帮助学习']))
for c in IT:
    q[c] = q[c].where(q[c] < 95)

q['AIz'] = (q['AI_total'] - q['AI_total'].mean()) / q['AI_total'].std()
q['AI_daily'] = (q['AI_total'].round() >= 5).astype(float).where(q['AI_total'].notna())
q['AI_between'] = q['CNTSCHID'].map(q.groupby('CNTSCHID')['AIz'].mean())
q['AI_within'] = q['AIz'] - q['AI_between']
q['AI_between_z'] = (q['AI_between'] - q['AI_between'].mean()) / q['AI_between'].std()
q['D_between'] = q['CNTSCHID'].map(q.groupby('CNTSCHID')['AI_daily'].mean())
q['D_within'] = q['AI_daily'] - q['D_between']

COMP = {'COGACSC': '认知激活', 'DISCLISCI': '纪律氛围', 'SELFREG': '自我调节', 'JOYSCIE': '科学兴趣'}
for c in COMP:
    q['SM_' + c] = q['CNTSCHID'].map(q.groupby('CNTSCHID')[c].mean())
    q['SM_' + c + '_z'] = (q['SM_' + c] - q['SM_' + c].mean()) / q['SM_' + c].std()
cols = ['SM_' + c + '_z' for c in COMP]
q['SCH_CLIM_z'] = q[cols].mean(axis=1)
q['SCH_CLIM_z'] = (q['SCH_CLIM_z'] - q['SCH_CLIM_z'].mean()) / q['SCH_CLIM_z'].std()
q['AIxCLIM'] = q['AI_within'] * q['SCH_CLIM_z']
q['AIzxCLIM'] = q['AIz'] * q['SCH_CLIM_z']
q['SCH_MEAN_ESCS_z'] = (q['SCH_MEAN_ESCS'] - q['SCH_MEAN_ESCS'].mean()) / q['SCH_MEAN_ESCS'].std()

for c in IT:
    q['bw_' + c] = q['CNTSCHID'].map(q.groupby('CNTSCHID')[c].mean())
    q['wi_' + c] = q[c] - q['bw_' + c]
    q['ciz_' + c] = (q['wi_' + c] - q['wi_' + c].mean()) / q['wi_' + c].std()
    q['bzz_' + c] = (q['bw_' + c] - q['bw_' + c].mean()) / q['bw_' + c].std()

STU = ['female', 'escs', 'repeat', 'grade_z', 'JOYSCIE_z', 'EFFSCIE_z', 'SELFREG_z',
       'COGACSC_z', 'TEACHSUP_z', 'DISCLISCI_z', 'ICTWKDY_z', 'ICTHOME_z']


def fe_fit(df, cols, outcomes, cluster='CNTSCHID'):
    """加权固定效应（校哑变量剥离），PV×BRR"""
    dum = pd.get_dummies(df[cluster], drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols] + [dum.values])
    keep = ['(截距)'] + cols
    k = len(keep)
    out = []
    for oc in outcomes:
        pts, samp = [], []
        for pv in [f'PV{i}{oc}' for i in range(1, 11)]:
            y = df[pv].values
            b0 = _wls(X, y, df.W_FSTUWT.values)[:k]
            reps = np.array([_wls(X, y, w)[:k] for w in
                             [df[f'W_FSTURWT{i}'].values for i in range(1, 81)]])
            pts.append(b0)
            samp.append(((reps - b0) ** 2).sum(axis=0) / (80 * (1 - FAY) ** 2))
        P = np.array(pts); qb = P.mean(axis=0)
        Ub = np.mean(samp, axis=0); Bv = P.var(axis=0, ddof=1)
        T = Ub + (1 + 1 / 10) * Bv
        se = np.sqrt(np.maximum(T, 1e-12))
        dfree = np.where(Bv > 1e-12, 9 * (1 + Ub / ((1 + 1 / 10) * np.maximum(Bv, 1e-12))) ** 2, 1e6)
        pv_ = 2 * stats.t.sf(np.abs(qb / se), df=dfree)
        out.append(pd.DataFrame(dict(outcome=oc, term=keep, b=qb, se=se, p=pv_)))
    return pd.concat(out, ignore_index=True)


rows = []
# ---------- A/B/C ----------
need = STU + ['AIz', 'AI_within', 'AI_between_z', 'AI_daily', 'D_within', 'D_between',
              'W_FSTUWT', 'CNTSCHID']
s = q.dropna(subset=need).copy()
print('样本 n=%d 学校=%d' % (len(s), s.CNTSCHID.nunique()))

for oc in ['SCIE', 'CMPS']:
    eng = PISAEngine(s, 'PV', oc)
    rows.append(eng.fit(STU + ['AIz'], f'{oc}|A 单层(强度)').assign(outcome=oc))
    rows.append(eng.fit(STU + ['AI_within', 'AI_between_z'], f'{oc}|B Mundlak(强度)').assign(outcome=oc))
    rows.append(eng.fit(STU + ['AI_daily'], f'{oc}|A 单层(每天)').assign(outcome=oc))
    rows.append(eng.fit(STU + ['D_within', 'D_between'], f'{oc}|B Mundlak(每天)').assign(outcome=oc))

fe = fe_fit(s, STU + ['AI_within'], ['SCIE', 'CMPS'])
fe['model'] = fe.outcome + '|C 校内FE(强度)'
rows.append(fe)
fe2 = fe_fit(s, STU + ['D_within'], ['SCIE', 'CMPS'])
fe2['model'] = fe2.outcome + '|C 校内FE(每天)'
rows.append(fe2)

# ---------- D 氛围交互 ----------
for oc in ['SCIE', 'CMPS']:
    eng = PISAEngine(s, 'PV', oc)
    rows.append(eng.fit(STU + ['AIz', 'SCH_CLIM_z', 'AIzxCLIM', 'SCH_MEAN_ESCS_z'],
                        f'{oc}|D1 交互 单层').assign(outcome=oc))
    rows.append(eng.fit(STU + ['AI_within', 'AI_between_z', 'SCH_CLIM_z', 'AIxCLIM',
                               'SCH_MEAN_ESCS_z'], f'{oc}|D2 交互 Mundlak').assign(outcome=oc))
fe3 = fe_fit(s, STU + ['AI_within', 'AIxCLIM'], ['SCIE', 'CMPS'])
fe3['model'] = fe3.outcome + '|D3 交互 校内FE'
rows.append(fe3)

res = pd.concat(rows, ignore_index=True)
res.to_csv(os.path.join(OUT, 'level_id.csv'), index=False, encoding='utf-8-sig')

def star(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''

print('\n=== 强度：三种识别 ===')
for m in [x for x in res.model.unique() if '强度' in x]:
    sub = res[(res.model == m) & res.term.isin(['AIz', 'AI_within', 'AI_between_z'])]
    print(' ', m)
    for _, r in sub.iterrows():
        print(f'    {r.term:14s} {r.b:8.2f} ({r.se:5.2f}) p={r.p:.4f}{star(r.p)}')

print('\n=== 每天使用：三种识别 ===')
for m in [x for x in res.model.unique() if '每天' in x]:
    sub = res[(res.model == m) & res.term.isin(['AI_daily', 'D_within', 'D_between'])]
    print(' ', m)
    for _, r in sub.iterrows():
        print(f'    {r.term:12s} {r.b:8.2f} ({r.se:5.2f}) p={r.p:.4f}{star(r.p)}')

print('\n=== 氛围交互：三种识别 ===')
for m in [x for x in res.model.unique() if '交互' in x]:
    sub = res[(res.model == m) & res.term.str.contains('CLIM|xCLIM|AIxCLIM')]
    print(' ', m)
    for _, r in sub.iterrows():
        print(f'    {r.term:14s} {r.b:8.2f} ({r.se:5.2f}) p={r.p:.4f}{star(r.p)}')

# ---------- E 四用途：单层 vs 校内 ----------
need2 = STU + ['ciz_' + c for c in IT] + ['bzz_' + c for c in IT] + ['W_FSTUWT', 'CNTSCHID']
s2 = q.dropna(subset=need2).copy()
print('\n四用途样本 n=%d' % len(s2))
rows = []
for oc in ['SCIE', 'CMPS']:
    eng = PISAEngine(s2, 'PV', oc)
    rows.append(eng.fit(STU + ['ciz_' + c for c in IT], f'{oc}|E1 四用途 单层').assign(outcome=oc))
    rows.append(eng.fit(STU + ['ciz_' + c for c in IT] + ['bzz_' + c for c in IT],
                        f'{oc}|E2 四用途 含校间').assign(outcome=oc))
fe4 = fe_fit(s2, STU + ['ciz_' + c for c in IT], ['SCIE', 'CMPS'])
fe4['model'] = fe4.outcome + '|E3 四用途 校内FE'
rows.append(fe4)
u = pd.concat(rows, ignore_index=True)
u.to_csv(os.path.join(OUT, 'uses_within.csv'), index=False, encoding='utf-8-sig')

print('\n=== 四用途：单层 vs 校内FE（标准化）===')
for oc in ['SCIE', 'CMPS']:
    a = u[(u.model == f'{oc}|E1 四用途 单层')].set_index('term')
    b = u[(u.model == f'{oc}|E3 四用途 校内FE')].set_index('term')
    print(f'-- {oc}')
    for i, c in enumerate(IT):
        t = 'ciz_' + c
        print(f'    {LAB[c]:8s} 单层 {a.loc[t,"b"]:7.2f} ({a.loc[t,"se"]:5.2f})  '
              f'校内 {b.loc[t,"b"]:7.2f} ({b.loc[t,"se"]:5.2f}) p={b.loc[t,"p"]:.4f}')
