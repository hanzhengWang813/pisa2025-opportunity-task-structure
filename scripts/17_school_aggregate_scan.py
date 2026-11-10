# -*- coding: utf-8 -*-
"""Second family of school-level moderators: school means of in-school teaching practice.

Rationale: principal-reported digital conditions converge across schools
(ceiling effect) and carry little discriminating power, whereas in-school
teaching practice (cognitive activation, teacher support, disciplinary climate,
collaborative reasoning, self-regulation) shows genuine between-school variance.
Also diagnoses whether the opposite signs of A_DIGCOND and B_TCHCAP reflect
suppression.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pisa_engine import PISAEngine

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

# ---------- School-mean aggregation (in-school teaching practice) ----------
AGG = {'SCH_COG': 'COGACSC', 'SCH_TS': 'TEACHSUP', 'SCH_DC': 'DISCLISCI',
       'SCH_SCOLL': 'SCIECOLLR', 'SCH_SREG': 'SELFREG', 'SCH_JOY': 'JOYSCIE'}
g = q.groupby('CNTSCHID')
for k, c in AGG.items():
    q[k] = q['CNTSCHID'].map(g[c].mean())
    q[k + '_z'] = (q[k] - q[k].mean()) / q[k].std()
    q['AIx' + k] = q['AIz'] * q[k + '_z']
q['SCH_MEAN_ESCS_z'] = (q['SCH_MEAN_ESCS'] - q['SCH_MEAN_ESCS'].mean()) / q['SCH_MEAN_ESCS'].std()
q['AIxSCH_MEAN_ESCS'] = q['AIz'] * q['SCH_MEAN_ESCS_z']

STU = ['female', 'escs', 'repeat', 'grade_z', 'JOYSCIE_z', 'EFFSCIE_z', 'SELFREG_z',
       'COGACSC_z', 'TEACHSUP_z', 'DISCLISCI_z', 'ICTWKDY_z', 'ICTHOME_z']
MODS = list(AGG.keys()) + ['SCH_MEAN_ESCS']

need = STU + ['AIz', 'W_FSTUWT', 'CNTSCHID'] + MODS + [k + '_z' for k in MODS] + ['AIx' + k for k in MODS]
s = q.dropna(subset=[c for c in need if c in q.columns]).copy()
print('Sample n=%d schools=%d' % (len(s), s.CNTSCHID.nunique()))

rows = []
for outcome in ['SCIE', 'CMPS']:
    eng = PISAEngine(s, 'PV', outcome)
    for k in MODS:
        df = eng.fit(STU + ['AIz', k + '_z', 'AIx' + k], f'{outcome}|{k}')
        gi = df[df.term == 'AIx' + k].iloc[0]
        gm = df[df.term == k + '_z'].iloc[0]
        ga = df[df.term == 'AIz'].iloc[0]
        rows.append(dict(outcome=outcome, moderator=k, b_AI=ga.b,
                         b_int=gi.b, se_int=gi.se, p_int=gi.p,
                         b_main=gm.b, se_main=gm.se, p_main=gm.p))
        print(f'{outcome} {k:16s} interaction {gi.b:7.2f} ({gi.se:5.2f}) p={gi.p:.4f} | main effect {gm.b:7.2f} p={gm.p:.4f}')

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'school_aggregate_scan.csv'), index=False, encoding='utf-8-sig')

# ---------- 抑制效应诊断：A 与 B 分开入模 ----------
print('\n=== 抑制效应诊断（SCIE）===')
eng = PISAEngine(s, 'PV', 'SCIE')
diag = []
specs = {
    'A 单独': ['A_DIGCOND_z'],
    'B 单独': ['B_TCHCAP_z'],
    'A+B 同时': ['A_DIGCOND_z', 'B_TCHCAP_z'],
    'A+B+校均ESCS': ['A_DIGCOND_z', 'B_TCHCAP_z', 'SCH_MEAN_ESCS_z'],
    'A+B+数字使用取向': ['A_DIGCOND_z', 'B_TCHCAP_z', 'C_DIGPURP_z'],
}
for lab, cols in specs.items():
    df = eng.fit(STU + ['AIz'] + cols, lab)
    for c in cols:
        r = df[df.term == c].iloc[0]
        diag.append(dict(spec=lab, var=c, b=r.b, se=r.se, p=r.p))
dd = pd.DataFrame(diag)
dd.to_csv(os.path.join(OUT, 'suppression_check.csv'), index=False, encoding='utf-8-sig')
for lab in specs:
    sub = dd[dd.spec == lab]
    print(' ', lab)
    for _, r in sub.iterrows():
        print(f'    {r["var"]:16s} {r.b:8.2f} ({r.se:5.2f}) p={r.p:.4f}')

# 相关
print('\nA/B/C 相关：')
print(s[['A_DIGCOND_z', 'B_TCHCAP_z', 'C_DIGPURP_z', 'SCH_MEAN_ESCS_z']].corr().round(2).to_string())
