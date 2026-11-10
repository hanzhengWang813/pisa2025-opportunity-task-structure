# -*- coding: utf-8 -*-
"""
13. 构造并验证学校层构念（论文主线：学校能做什么）

四类构念：
  A 数字条件（有没有）      SC155Q01-Q05
  B 教师能力与时间（会不会） SC155Q06-Q07
  C 数字资源使用取向（怎么用）SC268（效率/监控 vs 赋能学生）
  D 治理规范（管不管）      SC190（书面规定/指南/师生共定规则/网络素养课程）、SC190Q02 手机禁令
  附：SC266 自主学习教学优先度、SC267 数字教学组织形态
输出：data/qci_school.parquet（学生层 + 校层构念）、outputs/school_constructs.csv
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

from paths import BASE  # repo root (override with PISA_BASE)
OUT = os.path.join(BASE, 'outputs')
q = pd.read_parquet(os.path.join(BASE, 'data/qci_analysis.parquet'))

# ---------- 缺失码清洗 ----------
def clean(df, cols, hi):
    for c in cols:
        if c in df.columns:
            df[c] = df[c].where(df[c] < hi)
    return df

# 4 点评分（1-4），5 起为缺失/不适用
L4 = [f'SC155Q0{i}HA' for i in range(1, 11)] + [f'SC268Q0{i}DA' for i in range(1, 9)] + \
     [f'SC266Q0{i}DA' for i in range(1, 8)] + [f'SC267Q0{i}DA' for i in range(1, 8)]
q = clean(q, L4, 5)
# Yes/No：1=Yes 2=No → 1/0
YN = [f'SC190Q0{i}JA' for i in [1, 2, 5, 6, 7, 8, 9, 10, 11]]
q = clean(q, YN, 3)
for c in YN:
    if c in q.columns:
        q[c] = (q[c] == 1).astype(float).where(q[c].notna())

# ---------- 学校层子样本（每校一条） ----------
sch = q.groupby('CNTSCHID').first()
N_SCH = len(sch)
print(f'学校数 = {N_SCH}')

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
    print(f'  {name:14s} 题数={len(cols):2d}  α={a:.3f}  n={n:3d}  可用={v.notna().sum():3d}  '
          f'均值={v.mean():.2f} sd={v.std():.2f}')
    return v

print('\n=== 构念信度 ===')
sch['A_DIGCOND'] = mk(sch, [f'SC155Q0{i}HA' for i in [1, 2, 3, 4, 5]], 'A 数字条件')
sch['B_TCHCAP'] = mk(sch, ['SC155Q06HA', 'SC155Q07HA'], 'B 教师能力时间')
# SC268 六题相关 0.93，无法二分，改为单一「数字资源使用取向」总指数
sch['C_DIGPURP'] = mk(sch, [f'SC268Q0{i}DA' for i in range(1, 7)], 'C 数字使用取向')
sch['D_RULES'] = mk(sch, ['SC190Q01JA', 'SC190Q05JA', 'SC190Q06JA', 'SC190Q07JA', 'SC190Q08JA'], 'D 规范共定')
sch['F_SRL'] = mk(sch, [f'SC266Q0{i}DA' for i in range(1, 7)], 'F 自主学习优先')
sch['G_DIGORG'] = mk(sch, [f'SC267Q0{i}DA' for i in range(1, 7)], 'G 数字教学组织')

# 单题构念（不做合成）
sch['D_BAN'] = sch['SC190Q02JA']            # 手机禁令 1=有禁令
sch['H_MONITOR'] = sch['SC267Q04DA']        # LMS 日志用于追踪
sch['I_ENABLE'] = sch['SC268Q02DA']         # 单题：用数字资源强化自主学习能力

print('\n=== 手机禁令分布 ===')
print(sch['D_BAN'].value_counts(dropna=False).to_string())

# ---------- 校层构念之间相关 ----------
CC = ['A_DIGCOND', 'B_TCHCAP', 'C_DIGPURP', 'D_RULES', 'F_SRL', 'G_DIGORG']
print('\n=== 校层构念相关矩阵 ===')
print(sch[CC].corr().round(2).to_string())
print('\n=== 校层构念 VIF（相互）===')
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

# ---------- 学校层聚合（学生构成） ----------
agg = q.groupby('CNTSCHID').agg(
    n_stu=('W_FSTUWT', 'size'),
    SCH_MEAN_ESCS=('ESCS', 'mean'),
    SCH_MEAN_AI=('AI_total', 'mean'),
)
sch = sch.join(agg)
print(f'\n规模：学生/校 中位数={sch.n_stu.median():.0f}  校均ESCS={sch.SCH_MEAN_ESCS.mean():.2f}')

# ---------- 写回学生层 ----------
KEEP = ['A_DIGCOND', 'B_TCHCAP', 'C_DIGPURP', 'D_RULES', 'D_BAN',
        'F_SRL', 'G_DIGORG', 'H_MONITOR', 'I_ENABLE',
        'SCH_MEAN_ESCS', 'SCH_MEAN_AI', 'n_stu']
q = q.merge(sch[KEEP], left_on='CNTSCHID', right_index=True, how='left', suffixes=('', '_new'))
for c in KEEP:
    if c + '_new' in q.columns:
        q[c] = q[c + '_new']
        q.drop(columns=[c + '_new'], inplace=True)
q.to_parquet(os.path.join(BASE, 'data/qci_school.parquet'), index=False)
print(f'\n已写入 data/qci_school.parquet  {q.shape}')

# ---------- 校层构念汇总表 ----------
rows = []
for c in CC + ['D_BAN', 'H_MONITOR', 'I_ENABLE', 'SCH_MEAN_ESCS', 'SCH_MEAN_AI']:
    s = sch[c].dropna()
    rows.append(dict(construct=c, n=len(s), mean=round(s.mean(), 3), sd=round(s.std(), 3),
                     p10=round(s.quantile(.1), 2), p90=round(s.quantile(.9), 2)))
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'school_constructs.csv'), index=False,
                          encoding='utf-8-sig')
print('\n已写 outputs/school_constructs.csv')
print(pd.DataFrame(rows).to_string(index=False))
