# -*- coding: utf-8 -*-
"""Construct and validate the school-level constructs (the paper's main thread).

Four construct families:
  A Digital conditions (availability)      SC155Q01-Q05
  B Teacher capacity                       SC155Q06-Q07
  C Purpose of digital use                 ...
  D Rules and regulation                   ...
Writes outputs/school_constructs.csv and the reliability diagnostics.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

from paths import BASE  # repo root (override with PISA_BASE)
OUT = os.path.join(BASE, 'outputs')
q = pd.read_parquet(os.path.join(BASE, 'data/qci_analysis.parquet'))

# ---------- Missing-code cleaning ----------
def clean(df, cols, hi):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].where(df[c] < hi)
    return df

# 4-point rating (1-4); values of 5 and above are missing / not applicable
L4 = [f'SC155Q0{i}HA' for i in range(1, 11)] + [f'SC268Q0{i}DA' for i in range(1, 9)] + \
     [f'SC266Q0{i}DA' for i in range(1, 8)] + [f'SC267Q0{i}DA' for i in range(1, 8)]
q = clean(q, L4, 5)
# Yes/No：1=Yes 2=No → 1/0
YN = [f'SC190Q0{i}JA' for i in [1, 2, 5, 6, 7, 8, 9, 10, 11]]
q = clean(q, YN, 3)
for c in YN:
    if c in q.columns:
        q[c] = (q[c] == 1).astype(float).where(q[c].notna())

# ---------- School-level subsample (one row per school) ----------
sch = q.groupby('CNTSCHID').first()
N_SCH = len(sch)
print(f'Number of schools = {N_SCH}')

def alpha(df, cols):
    cols = [c for c in cols if c in df.columns]
    d = df[cols].dropna()
    if len(d) < 30 or len(cols) < 2:
        return np.nan, len(d)
    k = len(cols)
    v = d.var(ddof=1)
    tot = d.sum(axis=1).var(ddof=1)
    return k / (k - 1) * (1 - v.sum() / tot), len(d)

def mk(df, cols, name, min_frac=0.7):
    cols = [c for c in cols if c in df.columns]
    d = df[cols]
    v = d.mean(axis=1, skipna=True)
    v[d.notna().mean(axis=1) < min_frac] = np.nan
    a, n = alpha(df, cols)
    print(f'  {name:32s} items={len(cols):2d}  alpha={a:.3f}  n={n:3d}  usable={v.notna().sum():3d}  '
          f'mean={v.mean():.2f} sd={v.std():.2f}')
    return v

print('\n=== Construct reliability ===')
sch['A_DIGCOND'] = mk(sch, [f'SC155Q0{i}HA' for i in [1, 2, 3, 4, 5]], 'A Digital conditions')
sch['B_TCHCAP'] = mk(sch, ['SC155Q06HA', 'SC155Q07HA'], 'B Teacher capacity and time')
# Six SC268 items correlate at 0.93 and cannot be dichotomised; use one overall "purpose of digital use" index
sch['C_DIGPURP'] = mk(sch, [f'SC268Q0{i}DA' for i in range(1, 7)], 'C Purpose of digital use')
sch['D_RULES'] = mk(sch, ['SC190Q01JA', 'SC190Q05JA', 'SC190Q06JA', 'SC190Q07JA', 'SC190Q08JA'], 'D Co-designed rules')
sch['F_SRL'] = mk(sch, [f'SC266Q0{i}DA' for i in range(1, 7)], 'F Priority on self-regulation')
sch['G_DIGORG'] = mk(sch, [f'SC267Q0{i}DA' for i in range(1, 7)], 'G Organisation of digital teaching')

# Single-item constructs (no composite)
sch['D_BAN'] = sch['SC190Q02JA']            # phone ban, 1 = ban in place
sch['H_MONITOR'] = sch['SC267Q04DA']        # LMS logs used for tracking
sch['I_ENABLE'] = sch['SC268Q02DA']         # single item: digital resources used to strengthen self-regulated learning

print('\n=== Phone-ban distribution ===')
print(sch['D_BAN'].value_counts(dropna=False).to_string())

# ---------- Correlations among school-level constructs ----------
CC = ['A_DIGCOND', 'B_TCHCAP', 'C_DIGPURP', 'D_RULES', 'F_SRL', 'G_DIGORG']
print('\n=== School-level construct correlation matrix ===')
print(sch[CC].corr().round(2).to_string())
print('\n=== School-level construct VIF (mutual) ===')
from numpy.linalg import lstsq
X = sch[CC].dropna()
for c in CC:
    others = [x for x in CC if x != c]
    Xo = X[others].values
    Xo = np.column_stack([np.ones(len(Xo)), Xo])
    y = X[c].values
    b = lstsq(Xo, y, rcond=None)[0]
    r2 = 1 - ((y - Xo @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f'  {c:10s} R2={r2:.3f} VIF={1/(1-r2):.2f}')

# ---------- School-level aggregation (student composition) ----------
agg = q.groupby('CNTSCHID').agg(
    n_stu=('W_FSTUWT', 'size'),
    SCH_MEAN_ESCS=('ESCS', 'mean'),
    SCH_MEAN_AI=('AI_total', 'mean'),
)
sch = sch.join(agg)
print(f'\nSize: median students per school={sch.n_stu.median():.0f}  school-mean ESCS={sch.SCH_MEAN_ESCS.mean():.2f}')

# ---------- Write back to the student level ----------
KEEP = ['A_DIGCOND', 'B_TCHCAP', 'C_DIGPURP', 'D_RULES', 'D_BAN',
        'F_SRL', 'G_DIGORG', 'H_MONITOR', 'I_ENABLE',
        'SCH_MEAN_ESCS', 'SCH_MEAN_AI', 'n_stu']
q = q.merge(sch[KEEP], left_on='CNTSCHID', right_index=True, how='left', suffixes=('', '_new'))
for c in KEEP:
    if c + '_new' in q.columns:
        q[c] = q[c + '_new']
        q.drop(columns=[c + '_new'], inplace=True)
q.to_parquet(os.path.join(BASE, 'data/qci_school.parquet'), index=False)
print(f'\nWritten to data/qci_school.parquet  {q.shape}')

# ---------- School-level construct summary table ----------
rows = []
for c in CC + ['D_BAN', 'H_MONITOR', 'I_ENABLE', 'SCH_MEAN_ESCS', 'SCH_MEAN_AI']:
    s = sch[c].dropna()
    rows.append(dict(construct=c, n=len(s), mean=round(s.mean(), 3), sd=round(s.std(), 3),
                     p10=round(s.quantile(.1), 2), p90=round(s.quantile(.9), 2)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'school_constructs.csv'), index=False,
                          encoding='utf-8-sig')
print('\nWrote outputs/school_constructs.csv')
print(pd.DataFrame(rows).to_string(index=False))
