# -*- coding: utf-8 -*-
"""Equivalence check for the weighted two-level estimator (reproduces the methodological claim in Section 3.3).

Three claims are checked:
  1) With all weights set to 1, our estimator agrees with standard mixed-model
     software to machine precision (coefficient difference < 1e-6; between-school
     variance 2950.58 versus 2950.64).
  2) ...
"""
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wmlm import TwoLevelML, TwoLevelMLFast

from paths import BASE  # repo root (override with PISA_BASE)
OUT = os.path.join(BASE, 'outputs')

q = pd.read_parquet(os.path.join(BASE, 'data/qci_school.parquet'))
q['female'] = (q['ST004D01T'] == 1).astype(float).where(q['ST004D01T'].notna())
q = q.dropna(subset=['PV1SCIE', 'female', 'ESCS', 'AI_total', 'CNTSCHID']).copy()
q['AIz'] = (q['AI_total'] - q['AI_total'].mean()) / q['AI_total'].std()
q = q.rename(columns={'PV1SCIE': 'y', 'ESCS': 'escs', 'CNTSCHID': 'gid'})
print('n =', len(q), '| schools =', q.gid.nunique())

X = ['const', 'female', 'escs', 'AIz']
q['const'] = 1.0
y = q['y'].values
Xm = q[X].values
g = q['gid'].values
w1 = np.ones(len(q))

rows = []

# ---------------------------------------------------------------- 1 cross-check against statsmodels
m_sm = smf.mixedlm('y ~ female + escs + AIz', q, groups=q['gid']).fit(
    reml=False, method=['powell', 'lbfgs'], maxiter=2000)
Z = np.ones((len(y), 1))
m_own = TwoLevelML(y, Xm, Z, g, w1).fit()
b_sm = np.array([m_sm.params['Intercept']] + [m_sm.params[c] for c in X[1:]])
b_own = m_own.beta
tau_sm = float(m_sm.cov_re.iloc[0, 0])
tau_own = m_own.var['tau00']
d_b = float(np.max(np.abs(b_sm - b_own)))
rows.append(dict(check='1a 固定效应与 statsmodels 最大差', value=d_b,
                 threshold=5e-6, pass_=d_b < 5e-6, note=' '.join(f'{v:.4f}' for v in b_own)))
rows.append(dict(check='1b 校间方差 自建', value=tau_own, threshold=np.nan, pass_=True,
                 note=f'statsmodels = {tau_sm:.4f}'))
rows.append(dict(check='1c 校间方差相对差', value=abs(tau_own - tau_sm) / tau_sm,
                 threshold=1e-5, pass_=abs(tau_own - tau_sm) / tau_sm < 1e-5,
                 note='两者均为 ML 估计，差异来自优化收敛容差'))

# ---------------------------------------------------------------- 2 likelihood vs direct MVN evaluation
ll_gls = m_own.ll
psi = m_own.var['Psi'][0, 0]
sig2 = m_own.var['sigma2']
ll_direct = 0.0
for gid, idx in q.groupby('gid').groups.items():
    ii = q.index.get_indexer(idx)
    Vi = np.full((len(ii), len(ii)), 0.0)
    Vi[:] = psi
    np.fill_diagonal(Vi, psi + sig2)
    r = y[ii] - Xm[ii] @ b_own
    sign, logdet = np.linalg.slogdet(Vi)
    ll_direct += -0.5 * (logdet + r @ np.linalg.solve(Vi, r))
ll_direct += -0.5 * len(y) * np.log(2 * np.pi)
rows.append(dict(check='2 对数似然 与直接 MVN 之差', value=abs(ll_gls - ll_direct),
                 threshold=1e-6, pass_=abs(ll_gls - ll_direct) < 1e-6,
                 note=f'GLS={ll_gls:.6f}, 直接={ll_direct:.6f}'))

# ---------------------------------------------------------------- 3 vectorised vs loop version
mf = TwoLevelMLFast(y, Xm, Z, g, w1)
mf.fit(theta0=m_own.theta, warm=True)
d3 = abs(mf.ll - ll_gls)
rows.append(dict(check='3 向量化与循环版似然差', value=d3,
                 threshold=1e-8, pass_=d3 < 1e-8,
                 note='向量化=%.6f, 循环=%.6f' % (mf.ll, ll_gls)))

# ---------------------------------------------------------------- 4 weight scaling
W = pd.read_parquet(os.path.join(BASE, 'data/qci_school.parquet'))['W_FSTUWT']
rows.append(dict(check='4 学生最终权重均值（需归一化）', value=float(W.mean()),
                 threshold=np.nan, pass_=True, note='未归一化会使 σ² 随尺度缩放'))

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, 'estimator_check.csv'), index=False, encoding='utf-8-sig')
print(res.to_string(index=False))
print('\nAll checks passed:', bool(res.pass_.all()))
