# -*- coding: utf-8 -*-
"""
16. 学校层调节变量系统扫描（PV×BRR 设计法）
候选：WLE 指数 + 自建构念 + 学校聚合（AI 文化/屏幕文化/构成）+ 关键单题
每个候选单独入模：y ~ 学生层 + AIz + W + AIz×W
输出 outputs/moderator_scan.csv（含 FDR 校正）
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

# ---------- 派生 ----------
q['female'] = (q['ST004D01T'] == 1).astype(float).where(q['ST004D01T'].notna())
q['repeat'] = (q['REPEAT'] > 0).astype(float).where(q['REPEAT'].notna())
q['escs'] = q['ESCS']
for c in ['ESCS', 'GRADE', 'JOYSCIE', 'EFFSCIE', 'SELFREG', 'COGACSC', 'TEACHSUP',
          'DISCLISCI', 'ICTWKDY', 'ICTHOME']:
    q[c + '_z'] = (q[c] - q[c].mean()) / q[c].std()
q['grade_z'] = q['GRADE_z']
q['AIz'] = (q['AI_total'] - q['AI_total'].mean()) / q['AI_total'].std()

# ---------- 学校聚合（AI 文化 / 屏幕文化 / 离散度） ----------
g = q.groupby('CNTSCHID')
q['SCH_AI_MEAN'] = q['CNTSCHID'].map(g['AI_total'].mean())
q['SCH_AI_SD'] = q['CNTSCHID'].map(g['AI_total'].std())
q['SCH_DAILY_SHARE'] = q['CNTSCHID'].map((q['AI_total'].round() >= 5).groupby(q['CNTSCHID']).mean())
q['SCH_ICT_MEAN'] = q['CNTSCHID'].map(g['ICTWKDY'].mean())
q['SCH_SIZE'] = q['CNTSCHID'].map(g['W_FSTUWT'].size())

# ---------- 候选调节变量 ----------
CANDS = {
    # 自建构念
    'A 数字条件': 'A_DIGCOND', 'B 教师能力时间': 'B_TCHCAP', 'C 数字使用取向': 'C_DIGPURP',
    'D 规范共定': 'D_RULES', 'F 自主学习优先': 'F_SRL', 'G 数字教学组织': 'G_DIGORG',
    'H 过程追踪(单题)': 'H_MONITOR', 'I 赋能取向(单题)': 'I_ENABLE', 'J 手机禁令': 'D_BAN',
    # PISA 派生指数
    'K 数字教学准备度': 'DIGPREP', 'L 科学教师培训': 'SCTRAIN', 'M 教育短缺': 'EDUSHORT',
    'N 师资短缺': 'STAFFSHORT', 'O 软件可用性': 'AVLRSOFT', 'P 教学领导': 'EDULEAD',
    'Q 负面校风': 'NEGSCLIM', 'R 科学合作文化': 'SCIECOLLR',
    # 关键单题（学校治理具体做法）
    'S 书面数字设备规定': 'SC190Q01JA', 'T 教师自定使用规则': 'SC190Q06JA',
    'U 师生共定规则': 'SC190Q07JA', 'V 网络素养课程': 'SC190Q08JA',
    'W 社交网络政策': 'SC190Q09JA', 'X 教师应对学生用AI': 'SC265Q13DA',
    'Y 数字教学PD需求': 'SC265Q03DA', 'Z 自主学习教学支持': 'SC266Q02DA',
    # 学校聚合
    'AA 校均AI使用': 'SCH_AI_MEAN', 'AB 校内AI离散度': 'SCH_AI_SD',
    'AC 校均屏幕时间': 'SCH_ICT_MEAN', 'AD 每天使用占比': 'SCH_DAILY_SHARE',
    'AE 学校规模': 'SCH_SIZE',
}
STU = ['female', 'escs', 'repeat', 'grade_z', 'JOYSCIE_z', 'EFFSCIE_z', 'SELFREG_z',
       'COGACSC_z', 'TEACHSUP_z', 'DISCLISCI_z', 'ICTWKDY_z', 'ICTHOME_z']

# 标准化 + 交互
for lab, c in CANDS.items():
    if c not in q.columns:
        print('缺少', c)
        continue
    q[c + '_M'] = (q[c] - q[c].mean()) / q[c].std()
    q['AIx_' + c] = q['AIz'] * q[c + '_M']

need = STU + ['AIz', 'W_FSTUWT', 'CNTSCHID'] + [c + '_M' for c in CANDS.values() if c in q.columns]
s = q.dropna(subset=[c for c in need if c in q.columns]).copy()
print('样本 n=%d 学校=%d' % (len(s), s.CNTSCHID.nunique()))

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
        print(f'{outcome} {lab:18s} 交互 {gi.b:7.2f} ({gi.se:5.2f}) p={gi.p:.4f} | 主效应 {gm.b:7.2f} p={gm.p:.4f}')

res = pd.DataFrame(rows)
# Benjamini-Hochberg FDR（按结果域分别校正）
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
print('\n=== FDR 后仍显著（q<.05）===')
sig = res[res.p_fdr < .05]
print(sig[['outcome', 'moderator', 'b_int', 'se_int', 'p_int', 'p_fdr']].round(4).to_string(index=False)
      if len(sig) else '  无')
print('\n=== 按交互 |b| 排序前 8 ===')
print(res.reindex(res.b_int.abs().sort_values(ascending=False).index)
      .head(8)[['outcome', 'moderator', 'b_int', 'p_int', 'p_fdr']].round(4).to_string(index=False))
