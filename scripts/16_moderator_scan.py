# -*- coding: utf-8 -*-
"""Systematic scan of school-level moderators (PV x BRR design-based inference).

Candidates: WLE indices, self-built constructs, school aggregates (AI culture,
screen culture, composition) and selected single items. Each candidate enters
its own model: y ~ student covariates + AIz + W + AIz:W.
Writes outputs/moderator_scan.csv including Benjamini-Hochberg FDR correction.
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

# ---------- Derived variables ----------
q['female'] = (q['ST004D01T'] == 1).astype(float).where(q['ST004D01T'].notna())
q['repeat'] = (q['REPEAT'] > 0).astype(float).where(q['REPEAT'].notna())
q['escs'] = q['ESCS']
for c in ['ESCS', 'GRADE', 'JOYSCIE', 'EFFSCIE', 'SELFREG', 'COGACSC', 'TEACHSUP',
          'DISCLISCI', 'ICTWKDY', 'ICTHOME']:
    q[c + '_z'] = (q[c] - q[c].mean()) / q[c].std()
q['grade_z'] = q['GRADE_z']
q['AIz'] = (q['AI_total'] - q['AI_total'].mean()) / q['AI_total'].std()

# ---------- School aggregates (AI culture / screen culture / dispersion) ----------
g = q.groupby('CNTSCHID')
q['SCH_AI_MEAN'] = q['CNTSCHID'].map(g['AI_total'].mean())
q['SCH_AI_SD'] = q['CNTSCHID'].map(g['AI_total'].std())
q['SCH_DAILY_SHARE'] = q['CNTSCHID'].map((q['AI_total'].round() >= 5).groupby(q['CNTSCHID']).mean())
q['SCH_ICT_MEAN'] = q['CNTSCHID'].map(g['ICTWKDY'].mean())
q['SCH_SIZE'] = q['CNTSCHID'].map(g['W_FSTUWT'].size())

# ---------- Candidate moderators ----------
CANDS = {
    # self-built constructs
    'A 数字条件': 'A_DIGCOND', 'B 教师能力时间': 'B_TCHCAP', 'C 数字使用取向': 'C_DIGPURP',
    'D 规范共定': 'D_RULES', 'F 自主学习优先': 'F_SRL', 'G 数字教学组织': 'G_DIGORG',
    'H 过程追踪(单题)': 'H_MONITOR', 'I 赋能取向(单题)': 'I_ENABLE', 'J 手机禁令': 'D_BAN',
    # PISA derived indices
    'K 数字教学准备度': 'DIGPREP', 'L 科学教师培训': 'SCTRAIN', 'M 教育短缺': 'EDUSHORT',
    'N 师资短缺': 'STAFFSHORT', 'O 软件可用性': 'AVLRSOFT', 'P 教学领导': 'EDULEAD',
    'Q 负面校风': 'NEGSCLIM', 'R 科学合作文化': 'SCIECOLLR',
    # key single items (concrete school governance practices)
    'S 书面数字设备规定': 'SC190Q01JA', 'T 教师自定使用规则': 'SC190Q06JA',
    'U 师生共定规则': 'SC190Q07JA', 'V 网络素养课程': 'SC190Q08JA',
    'W 社交网络政策': 'SC190Q09JA', 'X 教师应对学生用AI': 'SC265Q13DA',
    'Y 数字教学PD需求': 'SC265Q03DA', 'Z 自主学习教学支持': 'SC266Q02DA',
    # school aggregates
    'AA 校均AI使用': 'SCH_AI_MEAN', 'AB 校内AI离散度': 'SCH_AI_SD',
    'AC 校均屏幕时间': 'SCH_ICT_MEAN', 'AD 每天使用占比': 'SCH_DAILY_SHARE',
    'AE 学校规模': 'SCH_SIZE',
}
STU = ['female', 'escs', 'repeat', 'grade_z', 'JOYSCIE_z', 'EFFSCIE_z', 'SELFREG_z',
       'COGACSC_z', 'TEACHSUP_z', 'DISCLISCI_z', 'ICTWKDY_z', 'ICTHOME_z']

# Standardise + build interaction terms
for lab, c in CANDS.items():
    if c not in q.columns:
        print('missing', c)
        continue
    q[c + '_M'] = (q[c] - q[c].mean()) / q[c].std()
    q['AIx_' + c] = q['AIz'] * q[c + '_M']

need = STU + ['AIz', 'W_FSTUWT', 'CNTSCHID'] + [c + '_M' for c in CANDS.values() if c in q.columns]
s = q.dropna(subset=[c for c in need if c in q.columns]).copy()
print('Sample n=%d schools=%d' % (len(s), s.CNTSCHID.nunique()))

rows = []
for outcome in ['SCIE', 'CMPS']:
    eng = PISAEngine(s, 'PV', outcome)
    for lab, c in CANDS.items():
        if c not in s.columns:
            continue
        mod = c + '_M'
        inter = 'AIx_' + c
        df = eng.fit(STU + ['AIz', mod, inter], f'{outcome}|{lab}')
        gi = df[df.term == inter].iloc[0]
        gm = df[df.term == mod].iloc[0]
        ga = df[df.term == 'AIz'].iloc[0]
        rows.append(dict(outcome=outcome, moderator=lab, var=c,
                         b_int=gi.b, se_int=gi.se, p_int=gi.p,
                         b_main=gm.b, se_main=gm.se, p_main=gm.p,
                         b_AI=ga.b, p_AI=ga.p))
        print(f'{outcome} {lab:18s} interaction {gi.b:7.2f} ({gi.se:5.2f}) p={gi.p:.4f} | main effect {gm.b:7.2f} p={gm.p:.4f}')

res = pd.DataFrame(rows)
# Benjamini-Hochberg FDR (corrected separately within each outcome domain)
res['p_fdr'] = np.nan
for oc in res.outcome.unique():
    m = res.outcome == oc
    p = res.loc[m, 'p_int'].values
    order = np.argsort(p)
    n = len(p)
    qv = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        val = min(prev, p[idx] * n / (rank + 1))
        qv[idx] = val
        prev = val
    res.loc[m, 'p_fdr'] = qv
res.to_csv(os.path.join(OUT, 'moderator_scan.csv'), index=False, encoding='utf-8-sig')
print('\n=== Still significant after FDR (q<.05) ===')
sig = res[res.p_fdr < .05]
print(sig[['outcome', 'moderator', 'b_int', 'se_int', 'p_int', 'p_fdr']].round(4).to_string(index=False)
      if len(sig) else '  none')
print('\n=== Top 8 by |interaction b| ===')
print(res.reindex(res.b_int.abs().sort_values(ascending=False).index)
      .head(8)[['outcome', 'moderator', 'b_int', 'p_int', 'p_fdr']].round(4).to_string(index=False))
