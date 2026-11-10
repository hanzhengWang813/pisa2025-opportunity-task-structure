# -*- coding: utf-8 -*-
"""Between- versus within-school: Mundlak decomposition plus the within-school four-purpose test.

  Mundlak: y ~ AI_within (group-mean centred) + AI_between (school mean) + covariates
  Both coefficients sit in one model, so they can be compared directly
  (Hausman-type test of equality).
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
# Mundlak decomposition
q['AI_between'] = q['CNTSCHID'].map(q.groupby('CNTSCHID')['AIz'].mean())
q['AI_within'] = q['AIz'] - q['AI_between']
q['AI_between_z'] = (q['AI_between'] - q['AI_between'].mean()) / q['AI_between'].std()
# centring of the four purposes (within-school / between-school)
for c in IT:
    q['bw_' + c] = q['CNTSCHID'].map(q.groupby('CNTSCHID')[c].mean())
    q['wi_' + c] = q[c] - q['bw_' + c]
    q['bz_' + c] = (q['bw_' + c] - q['bw_' + c].mean()) / q['bw_' + c].std()
    q['ci_' + c] = (q['wi_' + c] - q['wi_' + c].mean()) / q['wi_' + c].std()

STU = ['female', 'escs', 'repeat', 'grade_z', 'JOYSCIE_z', 'EFFSCIE_z', 'SELFREG_z',
       'COGACSC_z', 'TEACHSUP_z', 'DISCLISCI_z', 'ICTWKDY_z', 'ICTHOME_z']

# ---------- Mundlak ----------
need = STU + ['AI_within', 'AI_between_z', 'W_FSTUWT', 'CNTSCHID', 'AI_daily']
s = q.dropna(subset=need).copy()
print('Mundlak sample n=%d schools=%d' % (len(s), s.CNTSCHID.nunique()))
rows = []
for outcome in ['SCIE', 'CMPS']:
    eng = PISAEngine(s, 'PV', outcome)
    df = eng.fit(STU + ['AI_within', 'AI_between_z'], f'{outcome}|Mundlak 强度')
    rows.append(df.assign(outcome=outcome))
    d2 = eng.fit(STU + ['AI_daily', 'AI_between_z'], f'{outcome}|Mundlak 每天哑变量')
    rows.append(d2.assign(outcome=outcome))
    print(f'-- {outcome}')
    for _, r in df[df.term.isin(['AI_within', 'AI_between_z'])].iterrows():
        st = '***' if r.p < .001 else '**' if r.p < .01 else '*' if r.p < .05 else ''
        print(f'   {r.term:14s} {r.b:8.2f} ({r.se:5.2f}) p={r.p:.4f}{st}')

mun = pd.concat(rows, ignore_index=True)
mun.to_csv(os.path.join(OUT, 'mundlak.csv'), index=False, encoding='utf-8-sig')

# ---------- Within-school four purposes (FE) ----------
need2 = STU + ['ci_' + c for c in IT] + ['bz_' + c for c in IT] + ['W_FSTUWT', 'CNTSCHID']
s2 = q.dropna(subset=need2).copy()
print('\nWithin-school four-purpose sample n=%d schools=%d' % (len(s2), s2.CNTSCHID.nunique()))

rows = []
for outcome in ['SCIE', 'CMPS']:
    eng = PISAEngine(s2, 'PV', outcome)
    base = STU
    # single level (benchmark: the existing approach)
    d1 = eng.fit(base + ['ci_' + c for c in IT], f'{outcome}|四用途 单层')
    # add the between-school component
    d2 = eng.fit(base + ['ci_' + c for c in IT] + ['bz_' + c for c in IT], f'{outcome}|四用途 单层+校间')
    rows += [d1.assign(outcome=outcome), d2.assign(outcome=outcome)]
    # FE
    dum = pd.get_dummies(s2.CNTSCHID, drop_first=True, dtype=float)
    cols = base + ['ci_' + c for c in IT]
    Xn = np.column_stack([np.ones(len(s2))] + [s2[c].values for c in cols] + [dum.values])
    names = ['(截距)'] + cols
    pts, samp = [], []
    for pv in [f'PV{i}{outcome}' for i in range(1, 11)]:
        y = s2[pv].values
        b0 = _wls(Xn, y, s2.W_FSTUWT.values)
        reps = np.array([_wls(Xn, y, w) for w in
                         [s2[f'W_FSTURWT{i}'].values for i in range(1, 81)]])
        pts.append(b0)
        samp.append(((reps - b0) ** 2).sum(axis=0) / (80 * (1 - FAY) ** 2))
    P = np.array(pts); qb = P.mean(axis=0)
    Ub = np.mean(samp, axis=0); Bv = P.var(axis=0, ddof=1)
    T = Ub + (1 + 1 / 10) * Bv; se = np.sqrt(T)
    dfree = np.where(Bv > 1e-12, 9 * (1 + Ub / ((1 + 1 / 10) * np.maximum(Bv, 1e-12))) ** 2, 1e6)
    pv_ = 2 * stats.t.sf(np.abs(qb / se), df=dfree)
    dfe = pd.DataFrame(dict(model=f'{outcome}|四用途 校内FE', term=names,
                            b=qb, se=se, p=pv_, df=dfree)).iloc[1:]
    rows.append(dfe.assign(outcome=outcome))
    print(f'\n-- {outcome} four purposes (standardised, within-school vs single level)')
    for i, c in enumerate(IT):
        a = d1[d1.term == 'ci_' + c].iloc[0]
        f = dfe[dfe.term == 'ci_' + c].iloc[0]
        print(f'   {LAB[c]:8s} single level {a.b:7.2f} ({a.se:5.2f}) | within-school FE {f.b:7.2f} ({f.se:5.2f}) p={f.p:.4f}')

w = pd.concat(rows, ignore_index=True)
w.to_csv(os.path.join(OUT, 'within_uses.csv'), index=False, encoding='utf-8-sig')
print('\nDone')
