# -*- coding: utf-8 -*-
"""
14. 两层模型阶梯（学校为主角）
  M0 空模型                          → ICC
  M1 + 学生层协变量                    → 学校间方差还剩多少
  M2 + AI 使用强度（固定斜率）          → 平均关联
  M3 + AI 随机斜率                    → 斜率在学校间是否不同（核心）
  M4 + 学校层构念（主效应）             → 学校条件解释多少校间方差 / 斜率方差
  M5 + 跨层交互（AI × 学校构念）        → 谁能缓冲
估计：加权 ML（w=W_FSTUWT），PV1–PV10 逐一估计后 Rubin 合并。
推断：模型法 SE（设计法 SE 见 15 脚本的 BRR 版本）
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))
from wmlm import TwoLevelMLFast

from paths import BASE  # repo root (override with PISA_BASE)
OUT = os.path.join(BASE, 'outputs')
q = pd.read_parquet(os.path.join(BASE, 'data/qci_school.parquet'))

PV = [f'PV{i}SCIE' for i in range(1, 11)]
STU = ['female', 'escs', 'repeat', 'grade_z']
DISP = ['JOYSCIE_z', 'EFFSCIE_z', 'SELFREG_z', 'COGACSC_z', 'TEACHSUP_z', 'DISCLISCI_z']
ICT = ['ICTWKDY_z', 'ICTHOME_z']
SCH = ['A_DIGCOND_z', 'B_TCHCAP_z', 'C_DIGPURP_z', 'D_RULES_z', 'F_SRL_z',
       'G_DIGORG_z', 'H_MONITOR_z', 'I_ENABLE_z']

# ---------- 派生变量 ----------
q['female'] = (q['ST004D01T'] == 1).astype(float).where(q['ST004D01T'].notna())
q['repeat'] = (q['REPEAT'] > 0).astype(float).where(q['REPEAT'].notna())
q['escs'] = q['ESCS']
for c in ['ESCS', 'GRADE', 'JOYSCIE', 'EFFSCIE', 'SELFREG', 'COGACSC', 'TEACHSUP',
          'DISCLISCI', 'ICTWKDY', 'ICTHOME']:
    q[c + '_z'] = (q[c] - q[c].mean()) / q[c].std()
q['grade_z'] = q['GRADE_z']
q['AIz'] = (q['AI_total'] - q['AI_total'].mean()) / q['AI_total'].std()
for c in ['A_DIGCOND', 'B_TCHCAP', 'C_DIGPURP', 'D_RULES', 'F_SRL', 'G_DIGORG',
          'H_MONITOR', 'I_ENABLE']:
    q[c + '_z'] = (q[c] - q[c].mean()) / q[c].std()

need = PV + ['W_FSTUWT', 'CNTSCHID', 'AIz'] + STU + DISP + ICT + SCH
s = q.dropna(subset=[c for c in need if c in q.columns]).copy()
s['g'] = s['CNTSCHID'].astype('category').cat.codes
print('样本 n=%d  学校=%d' % (len(s), s.g.nunique()))
print('学校构念相关（与 AI 斜率有关的候选）:')
print(s[['AIz'] + SCH].corr()['AIz'].round(3).to_string())

IG = np.ones(len(s))
Z1 = IG.reshape(-1, 1)
Z2 = np.column_stack([IG, s.AIz.values])
W = s.W_FSTUWT.values
W = W / W.mean()   # 归一化：否则 σ² 随权重尺度缩放，ICC 无法直接计算

def design(cols):
    return np.column_stack([IG] + [s[c].values for c in cols])

MODELS = {
    'M0 空模型':                ([], Z1),
    'M1 +学生层':               (STU, Z1),
    'M2 +AI(固定斜率)':         (STU + DISP + ICT + ['AIz'], Z1),
    'M3 +AI随机斜率':           (STU + DISP + ICT + ['AIz'], Z2),
    'M4 +学校层主效应':          (STU + DISP + ICT + ['AIz'] + SCH, Z2),
    'M5 +跨层交互':             (STU + DISP + ICT + ['AIz'] + SCH + [f'AIx{k}' for k in ['A', 'B', 'C', 'D', 'F', 'G', 'H', 'I']], Z2),
}
# M5 交互项
for k, c in zip(['A', 'B', 'C', 'D', 'F', 'G', 'H', 'I'],
                ['A_DIGCOND_z', 'B_TCHCAP_z', 'C_DIGPURP_z', 'D_RULES_z',
                 'F_SRL_z', 'G_DIGORG_z', 'H_MONITOR_z', 'I_ENABLE_z']):
    s[f'AIx{k}'] = s.AIz * s[c]

t0 = time.time()
rows = []
for name, (cols, Z) in MODELS.items():
    cols = [c for c in cols if c != 'dummy']
    X = design(cols)
    betas, lls, ses, vars_ = [], [], [], []
    prev_theta = None
    for pv in PV:
        y = s[pv].values
        warm = prev_theta if (prev_theta is not None and len(prev_theta) == Z.shape[1] + 1) else None
        m = TwoLevelMLFast(y, X, Z, s.g.values, w=W).fit(theta0=warm, warm=warm is not None)
        prev_theta = m.theta
        betas.append(m.beta)
        ses.append(m.se())
        lls.append(m.ll)
        vars_.append(m.var)
    B = np.array(betas)
    # Rubin 合并
    names = ['(截距)'] + cols
    k = B.shape[1]
    qbar = B.mean(axis=0)
    Ubar = np.mean([se ** 2 for se in ses], axis=0)
    Bvar = B.var(axis=0, ddof=1)
    T = Ubar + (1 + 1 / len(PV)) * Bvar
    se = np.sqrt(T)
    dfree = (len(PV) - 1) * (1 + Ubar / ((1 + 1 / len(PV)) * np.maximum(Bvar, 1e-12))) ** 2
    from scipy import stats as st
    tval = qbar / se
    pval = 2 * st.t.sf(np.abs(tval), df=dfree)
    for i, nm in enumerate(names):
        rows.append(dict(model=name, term=nm, b=qbar[i], se=se[i], p=pval[i], df=dfree[i]))
    tau00 = np.mean([v['tau00'] for v in vars_])
    tau11 = np.mean([v['tau11'] for v in vars_])
    s2 = np.mean([v['sigma2'] for v in vars_])
    ll = np.mean(lls)
    rows.append(dict(model=name, term='__var__', tau00=tau00, tau11=tau11, sigma2=s2,
                     icc=tau00 / (tau00 + s2), ll=ll))
    print(f'{name:22s} tau00={tau00:8.1f} tau11={tau11:7.2f} sigma2={s2:8.1f} '
          f'ICC={tau00/(tau00+s2):.3f}  斜率SD={np.sqrt(max(tau11,0)):5.2f}  ({time.time()-t0:.0f}s)')

res = pd.DataFrame(rows)
# 固定效应行补齐方差栏位
for c in ['tau00', 'tau11', 'sigma2', 'icc', 'll']:
    if c not in res.columns:
        res[c] = np.nan
res.to_csv(os.path.join(OUT, 'mlm_ladder.csv'), index=False, encoding='utf-8-sig')

# ---------- LRT：斜率方差是否显著（M2 vs M3，同固定效应，PV 逐一） ----------
print('\n=== 斜率方差 LRT（M2 无随机斜率 vs M3 有）===')
lr_all, p_all = [], []
for pv in PV:
    y = s[pv].values
    X = design(STU + DISP + ICT + ['AIz'])
    m2 = TwoLevelMLFast(y, X, Z1, s.g.values, w=W).fit()
    m3 = TwoLevelMLFast(y, X, Z2, s.g.values, w=W).fit()
    lr, p = m3.lrt_slope(m2)
    lr_all.append(lr); p_all.append(p)
    print(f'  {pv}: LR={lr:7.2f}  p={p:.4f}  tau11={m3.var["tau11"]:6.2f}')
print(f'  平均：LR={np.mean(lr_all):.2f}  p(合并近似)={np.mean(p_all):.4f}')
pd.DataFrame({'PV': PV, 'LR': lr_all, 'p': p_all}).to_csv(
    os.path.join(OUT, 'mlm_lrt_slope.csv'), index=False, encoding='utf-8-sig')

# ---------- ICC 的抽样分布（空模型） ----------
print('\n注：ICC 由 M0 的 tau00/(tau00+sigma2) 得出；设计法 SE 见 15 脚本。')
print('总耗时 %.0fs' % (time.time() - t0))
