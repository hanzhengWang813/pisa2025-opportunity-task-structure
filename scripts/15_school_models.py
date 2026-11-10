# -*- coding: utf-8 -*-
"""Design-based inference: school-level main effects and cross-level interactions (PV x 10 + BRR-Fay x 80).

Writes outputs/school_models.csv (all coefficients) and outputs/school_r2.csv
(incremental explanatory power).
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

# ---------- derived variables ----------
q['female'] = (q['ST004D01T'] == 1).astype(float).where(q['ST004D01T'].notna())
q['repeat'] = (q['REPEAT'] > 0).astype(float).where(q['REPEAT'].notna())
q['escs'] = q['ESCS']
for c in ['ESCS', 'GRADE', 'JOYSCIE', 'EFFSCIE', 'SELFREG', 'COGACSC', 'TEACHSUP',
          'DISCLISCI', 'ICTWKDY', 'ICTHOME']:
    q[c + '_z'] = (q[c] - q[c].mean()) / q[c].std()
q['grade_z'] = q['GRADE_z']
q['AIz'] = (q['AI_total'] - q['AI_total'].mean()) / q['AI_total'].std()
# Daily-use group (used for the threshold-form robustness check)
q['AI_daily'] = (q['AI_total'].round() >= 5).astype(float).where(q['AI_total'].notna())

SCH = {'A_DIGCOND': 'A 数字条件', 'B_TCHCAP': 'B 教师能力时间', 'C_DIGPURP': 'C 数字使用取向',
       'D_RULES': 'D 规范共定', 'F_SRL': 'F 自主学习优先', 'G_DIGORG': 'G 数字教学组织',
       'H_MONITOR': 'H 过程追踪', 'I_ENABLE': 'I 赋能取向(单题)'}
for c in SCH:
    q[c + '_z'] = (q[c] - q[c].mean()) / q[c].std()
    q['AIx' + c] = q['AIz'] * q[c + '_z']
q['SCH_MEAN_ESCS_z'] = (q['SCH_MEAN_ESCS'] - q['SCH_MEAN_ESCS'].mean()) / q['SCH_MEAN_ESCS'].std()

STU = ['female', 'escs', 'repeat', 'grade_z']
DISP = ['JOYSCIE_z', 'EFFSCIE_z', 'SELFREG_z', 'COGACSC_z', 'TEACHSUP_z', 'DISCLISCI_z']
ICT = ['ICTWKDY_z', 'ICTHOME_z']
SCHZ = [c + '_z' for c in SCH]
BASE_COV = STU + DISP + ICT

need = ['W_FSTUWT', 'CNTSCHID', 'AIz'] + BASE_COV + SCHZ + ['SCH_MEAN_ESCS_z']
s = q.dropna(subset=[c for c in need if c in q.columns]).copy()
print('Sample n=%d  schools=%d' % (len(s), s.CNTSCHID.nunique()))

allres = []
r2rows = []
for outcome in ['SCIE', 'CMPS']:
    eng = PISAEngine(s, 'PV', outcome)
    # S1 student level
    allres.append(eng.fit(BASE_COV + ['AIz'], f'{outcome}|S1 学生层'))
    # S2 school-level main effects
    allres.append(eng.fit(BASE_COV + ['AIz'] + SCHZ, f'{outcome}|S2 +学校层主效应'))
    # S3 one cross-level interaction at a time
    for c, lab in SCH.items():
        allres.append(eng.fit(BASE_COV + ['AIz'] + SCHZ + ['AIx' + c],
                              f'{outcome}|S3 交互×{lab}'))
    # S4 all interactions jointly
    allres.append(eng.fit(BASE_COV + ['AIz'] + SCHZ + ['AIx' + c for c in SCH],
                          f'{outcome}|S4 全部交互'))
    # incremental R-squared
    a, b, d = eng.r2(BASE_COV + ['AIz'], BASE_COV + ['AIz'] + SCHZ)
    a2, b2, d2 = eng.r2(BASE_COV + ['AIz'] + SCHZ, BASE_COV + ['AIz'] + SCHZ + ['AIx' + c for c in SCH])
    r2rows.append(dict(outcome=outcome, R2_base=round(a, 4), R2_school=round(b, 4), dR2_school=round(d, 4),
                       R2_inter=round(b2, 4), dR2_inter=round(d2, 4)))
    print(f'{outcome}: school-level dR2={d:.4f}  interaction dR2={d2:.4f}')

res = pd.concat(allres, ignore_index=True)
res.to_csv(os.path.join(OUT, 'school_models.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame(r2rows).to_csv(os.path.join(OUT, 'school_r2.csv'), index=False, encoding='utf-8-sig')

def star(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''

print('\n=== School-level main effects (S2) ===')
sub = res[res.model.str.contains('S2')]
for m in sub.model:
    for _, r in sub[sub.model == m].iterrows():
        if r.term in SCHZ or r.term in ['AIz', 'escs']:
            print(f'  {m:16s} {r.term:18s} {r.b:8.2f} ({r.se:5.2f}) p={r.p:.4f}{star(r.p)}')

print('\n=== Cross-level interactions (S3, one at a time) ===')
sub = res[res.model.str.contains('S3')]
for m in sub.model:
    row = sub[(sub.model == m) & (sub.term.str.startswith('AIx'))]
    for _, r in row.iterrows():
        print(f'  {m:24s} {r.term:18s} {r.b:8.2f} ({r.se:5.2f}) p={r.p:.4f}{star(r.p)}')

print('\n=== All interactions (S4) ===')
sub = res[res.model.str.contains('S4')]
for m in sub.model:
    for _, r in sub[sub.model == m].iterrows():
        if r.term.startswith('AIx') or r.term == 'AIz':
            print(f'  {m:16s} {r.term:18s} {r.b:8.2f} ({r.se:5.2f}) p={r.p:.4f}{star(r.p)}')
