# -*- coding: utf-8 -*-
"""Between/within identification under a unified covariate set (fixes the inconsistency between 19 and 20).

All models use the same student covariates STU:
  A Single level:        y ~ STU + AIz
  B Mundlak:             y ~ STU + AI_within + AI_between
  C Within (school FE):  y ~ STU + AIz + school fixed effects
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
LAB = dict(zip(IT, ['summarise source text', 'initial research on a new topic', 'draft writing', 'help learning']))
for c in IT:
    q[c] = q[c].where(q[c] < 95)

q['AIz'] = (q['AI_total'] - q['AI_total'].mean()) / q['AI_total'].std()
q['AI_daily'] = (q['AI_total'].round() >= 5).astype(float).where(q['AI_total'].notna())
q['AI_between'] = q['CNTSCHID'].map(q.groupby('CNTSCHID')['AIz'].mean())
q['AI_within'] = q['AIz'] - q['AI_between']
q['AI_between_z'] = (q['AI_between'] - q['AI_between'].mean()) / q['AI_between'].std()
q['D_between'] = q['CNTSCHID'].map(q.groupby('CNTSCHID')['AI_daily'].mean())
q['D_within'] = q['AI_daily'] - q['D_between']

COMP = {'COGACSC': 'cognitive activation', 'DISCLISCI': 'disciplinary climate', 'SELFREG': 'self-regulation', 'JOYSCIE': 'interest in science'}
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
    """Weighted fixed effects (school dummies partialled out), PV x BRR"""
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
print('Sample n=%d schools=%d' % (len(s), s.CNTSCHID.nunique()))

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

# ---------- D climate interaction ----------
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

print('\n=== Intensity: three identification strategies ===')
for m in [x for x in res.model.unique() if '强度' in x]:
    sub = res[(res.model == m) & res.term.isin(['AIz', 'AI_within', 'AI_between_z'])]
    print(' ', m)
    for _, r in sub.iterrows():
        print(f'    {r.term:14s} {r.b:8.2f} ({r.se:5.2f}) p={r.p:.4f}{star(r.p)}')

print('\n=== Daily use: three identification strategies ===')
for m in [x for x in res.model.unique() if '每天' in x]:
    sub = res[(res.model == m) & res.term.isin(['AI_daily', 'D_within', 'D_between'])]
    print(' ', m)
    for _, r in sub.iterrows():
        print(f'    {r.term:12s} {r.b:8.2f} ({r.se:5.2f}) p={r.p:.4f}{star(r.p)}')

print('\n=== Climate interaction: three identification strategies ===')
for m in [x for x in res.model.unique() if '交互' in x]:
    sub = res[(res.model == m) & res.term.str.contains('CLIM|xCLIM|AIxCLIM')]
    print(' ', m)
    for _, r in sub.iterrows():
        print(f'    {r.term:14s} {r.b:8.2f} ({r.se:5.2f}) p={r.p:.4f}{star(r.p)}')

# ---------- E Four purposes: single level vs within-school ----------
need2 = STU + ['ciz_' + c for c in IT] + ['bzz_' + c for c in IT] + ['W_FSTUWT', 'CNTSCHID']
s2 = q.dropna(subset=need2).copy()
print('\nFour-purpose sample n=%d' % len(s2))
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

print('\n=== Four purposes: single level vs within-school FE (standardised) ===')
for oc in ['SCIE', 'CMPS']:
    a = u[(u.model == f'{oc}|E1 四用途 单层')].set_index('term')
    b = u[(u.model == f'{oc}|E3 四用途 校内FE')].set_index('term')
    print(f'-- {oc}')
    for i, c in enumerate(IT):
        t = 'ciz_' + c
        print(f'    {LAB[c]:8s} single level {a.loc[t,"b"]:7.2f} ({a.loc[t,"se"]:5.2f})  '
              f'within-school {b.loc[t,"b"]:7.2f} ({b.loc[t,"se"]:5.2f}) p={b.loc[t,"p"]:.4f}')
