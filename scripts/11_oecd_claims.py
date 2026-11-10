# -*- coding: utf-8 -*-
"""Direct tests of the core OECD claims on the B-S-J-Z subsample.

OECD PISA 2025 Vol. I, Box I.4.2 states that
  (i) about six in ten students are asked to evaluate AI-generated information
      (OECD average 62.6%, B-S-J-Z 44.6%), and
  (ii) students who use AI frequently for learning AND were taught at school to
      evaluate AI information score slightly higher in science than non-users
      or low-frequency users.
Both claims are re-estimated here on the same subsample used in the paper.
"""
import pandas as pd, numpy as np, os, warnings
from scipy import stats
warnings.filterwarnings('ignore')

from paths import BASE  # repo root (override with PISA_BASE)
OUT = os.path.join(BASE, 'outputs')
q = pd.read_parquet(os.path.join(BASE, 'data', 'qci_analysis.parquet'))
REP = [f'W_FSTURWT{i}' for i in range(1, 81)]
FAY, M_PV, R = 0.5, 10, 80

q['female'] = (q.ST004D01T == 1).astype(float)
q['age'] = q.AGE; q['grade'] = q.GRADE
q['repeat'] = (q.REPEAT > 0).astype(float)
q['immig1'] = (q.IMMIG == 1).astype(float)
q['immig2'] = (q.IMMIG == 2).astype(float)
q['escs'] = q.ESCS
for c in ['SC190Q01JA', 'SC190Q05JA', 'SC190Q06JA', 'SC190Q09JA']:
    q[c + '_b'] = (q[c] == 1).astype(float).where(q[c].notna())
q['SCH_AIPOL'] = q[[c + '_b' for c in ['SC190Q01JA', 'SC190Q05JA', 'SC190Q06JA', 'SC190Q09JA']]].mean(axis=1)
q['SCH_AITRAIN'] = q['SC265Q13DA']

Z = ['AI_total', 'ICTINFO', 'ICTQUAL', 'ICTHOME', 'ICTWKDY', 'JOYSCIE', 'EFFSCIE',
     'SELFREG', 'PERSEV', 'CURIO', 'ENGSCIPR', 'TEACHSUP', 'COGACSC', 'DISCLISCI',
     'SCIECOLLR', 'DIGPREP', 'SCTRAIN', 'EDUSHORT', 'STAFFSHORT', 'NEGSCLIM',
     'SCH_AIPOL', 'SCH_AITRAIN']
for v in Z:
    if v in q.columns:
        q[v + '_z'] = (q[v] - q[v].mean()) / q[v].std()
q['escs_z'] = (q.escs - q.escs.mean()) / q.escs.std()
g = q.groupby('CNTSCHID')
q['SCH_ESCS'] = g['ESCS'].transform('mean')
q['SCH_ESCS_z'] = (q.SCH_ESCS - q.SCH_ESCS.mean()) / q.SCH_ESCS.std()
q['AI_grp'] = q.AI_total.round()
for k in [2, 3, 4, 5]:
    q[f'AI_g{k}'] = (q.AI_grp == k).astype(float)

# ---- Key OECD constructs ----
# ST436Q16: at school, asked to evaluate the quality of AI-generated information
q['LO_AI'] = q['ST436Q16DA'].where(q['ST436Q16DA'] < 5)
q['LO_AI_any']   = (q.LO_AI >= 2).astype(float).where(q.LO_AI.notna())   # at least sometimes
q['LO_AI_freq']  = (q.LO_AI >= 3).astype(float).where(q.LO_AI.notna())   # often / very often
# ST436Q02: evaluate the quality of online information (general digital literacy teaching)
q['LO_NET'] = q['ST436Q02DA'].where(q['ST436Q02DA'] < 5)
q['LO_NET_any'] = (q.LO_NET >= 2).astype(float).where(q.LO_NET.notna())
q['AI_weekly'] = (q.AI_total >= 3.5).astype(float)   # once a week or more
q['AIxLOAI'] = q.AI_total_z * q.LO_AI_freq
q['AIxLONET'] = q.AI_total_z * q.LO_NET_any

BASE_COV = ['female', 'age', 'grade', 'repeat', 'immig1', 'immig2', 'escs']
DISP = ['JOYSCIE_z', 'EFFSCIE_z', 'SELFREG_z', 'PERSEV_z', 'CURIO_z', 'ENGSCIPR_z']
CLAS = ['TEACHSUP_z', 'COGACSC_z', 'DISCLISCI_z', 'SCIECOLLR_z']
ICT = ['ICTHOME_z', 'ICTQUAL_z', 'ICTWKDY_z', 'ICTINFO_z']
SCH = ['DIGPREP_z', 'SCTRAIN_z', 'EDUSHORT_z', 'STAFFSHORT_z', 'NEGSCLIM_z', 'SCH_ESCS_z']
FULL = BASE_COV + DISP + CLAS + ICT + SCH


def build(df, cols):
    cols = list(dict.fromkeys(cols))
    X = np.column_stack([np.ones(len(df))] + [df[c].values.astype(float) for c in cols])
    return X, ['Intercept'] + cols


def wls(X, y, w):
    Xt = X.T * w; A = Xt @ X; b = Xt @ y
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def run(df, yname, cols, label):
    X, names = build(df, cols)
    Y = df[[f'PV{k}{yname}' for k in range(1, M_PV + 1)]].values
    W = df[REP].values; w0 = df['W_FSTUWT'].values
    cs, vs = [], []
    for j in range(Y.shape[1]):
        y = Y[:, j]; b0 = wls(X, y, w0)
        if not np.all(np.isfinite(b0)):
            continue
        D = np.array([wls(X, y, W[:, r]) - b0 for r in range(R)])
        cs.append(b0); vs.append((D ** 2).sum(axis=0) / (R * (1 - FAY) ** 2))
    cs, vs = np.array(cs), np.array(vs); m = len(cs)
    Qb, Ub = cs.mean(0), vs.mean(0)
    B = cs.var(0, ddof=1) if m > 1 else np.zeros_like(Ub)
    T = Ub + (1 + 1 / m) * B
    se = np.sqrt(np.where(T > 0, T, np.nan))
    with np.errstate(divide='ignore', invalid='ignore'):
        dfr = np.where(B > 0, (m - 1) * (1 + Ub / ((1 + 1 / m) * B)) ** 2, np.inf)
        t = Qb / se
        p = 2 * stats.t.sf(np.abs(t), np.where(np.isfinite(dfr), dfr, 1e6))
    return pd.DataFrame({'model': label, 'outcome': yname, 'term': names,
                         'b': Qb, 'se': se, 't': t, 'p': p, 'n': len(df)})


need = list(dict.fromkeys(FULL + ['AI_total', 'AI_grp', 'LO_AI', 'LO_AI_freq',
                                  'LO_NET', 'LO_NET_any', 'AIxLOAI', 'AIxLONET']))
need = [c for c in need if c in q.columns]
s = q.dropna(subset=need + ['PV1SCIE']).copy()
print('=' * 76)
print(f'Sample n={len(s)}, schools {s.CNTSCHID.nunique()}')
print('=' * 76)

# ---------- Claim 1: opportunity coverage ----------
print('\n[OECD claim 1] Share asked at school to evaluate AI-generated information')
def wshare(mask, w):
    m = ~np.isnan(mask.astype(float)); return np.average(mask.astype(float)[m], weights=w[m]) * 100
for nm, v in [('at least sometimes', 'LO_AI_any'), ('often / very often', 'LO_AI_freq')]:
    print(f'  AI-generated information evaluation · {nm}: {wshare(s[v].values, s.W_FSTUWT.values):.1f}%')
print(f'  Online information evaluation · at least sometimes: {wshare(s.LO_NET_any.values, s.W_FSTUWT.values):.1f}%')
print('  (OECD report: B-S-J-Z 44.6%, OECD average 62.6%)')

# Do opportunities skew towards advantaged families (claim 3)?
print('\n[OECD claim 3] Social distribution of the opportunity')
for v in ['LO_AI_freq', 'LO_NET_any']:
    sub = s.dropna(subset=[v])
    lo = sub[sub.escs <= sub.escs.quantile(.25)]; hi = sub[sub.escs >= sub.escs.quantile(.75)]
    print(f'  {v}: lowest SES quartile {wshare(lo[v].values, lo.W_FSTUWT.values):.1f}%  '
          f'| highest SES quartile {wshare(hi[v].values, hi.W_FSTUWT.values):.1f}%  '
          f'| diff {wshare(hi[v].values, hi.W_FSTUWT.values) - wshare(lo[v].values, lo.W_FSTUWT.values):+.1f} pp')
print(f'  corr(LO_AI_freq, ESCS) = {s[["LO_AI_freq", "escs"]].corr().iloc[0,1]:.3f}')

# ---------- Claim 2: protective effect ----------
print('\n[OECD claim 2] AI-use effect with vs without access to AI evaluation (PV+BRR, full covariate set)')
rows = []
for cols, lab in [
    (FULL + ['AI_total_z'], 'P0 基准'),
    (FULL + ['AI_total_z', 'LO_AI_freq', 'AIxLOAI'], 'P1 ×AI评估机会(频繁)'),
    (FULL + ['AI_total_z', 'LO_NET_any', 'AIxLONET'], 'P2 ×线上信息评估'),
    (FULL + ['AI_g2', 'AI_g3', 'AI_g4', 'AI_g5', 'LO_AI_freq', 'AIxLOAI'], 'P3 分组+交互'),
]:
    r = run(s, 'SCIE', cols, lab); rows.append(r)
    for _, w in r[r.term.str.contains('AI') | r.term.str.startswith('LO_')].iterrows():
        st = '***' if w.p < .001 else '**' if w.p < .01 else '*' if w.p < .05 else ''
        print(f'  {lab:16s} {w.term:12s} b={w.b:7.2f} se={w.se:5.2f} p={w.p:.4f}{st}')

# ---------- OECD-style stratified description ----------
print('\n[OECD-style stratification] Weighted scores by AI intensity group, split by whether students are often asked to evaluate AI information')
d = s.dropna(subset=['LO_AI_freq'])
def wm(sub):
    x = sub.PV1SCIE.values; w = sub.W_FSTUWT.values; m = ~np.isnan(x)
    return np.average(x[m], weights=w[m]) if m.sum() else np.nan
for lv, lb in [(0, 'no AI-evaluation teaching'), (1, 'frequent AI-evaluation teaching')]:
    sub = d[d.LO_AI_freq == lv]
    line = f'  {lb:16s} n={len(sub):5d} |'
    for k in [1, 2, 3, 4, 5]:
        gk = sub[sub.AI_grp == k]
        line += f' {["","never","yearly","monthly","weekly","daily"][k]}:{wm(gk):6.1f}'
    print(line)
print('  (OECD argues: frequent AI use for learning plus access to evaluation teaching scores slightly higher than non-/low-frequency use)')

pd.concat(rows).to_csv(os.path.join(OUT, 'oecd_claim_tests.csv'), index=False, encoding='utf-8-sig')
print('\nSaved to outputs/oecd_claim_tests.csv')
