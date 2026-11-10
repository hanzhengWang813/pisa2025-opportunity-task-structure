# -*- coding: utf-8 -*-
"""Intervention-lever analysis (quantitative basis for the policy discussion).

Three additions beyond the original report:
1. Put every malleable predictor on one common scale, the score difference from
   -1 SD to +1 SD, so the levers can be ranked by priority.
2. Build a single output-orientation versus input-orientation composite and
   enter it alongside total use.
3. Report the levers separately for the within-school and the between-school
   channel, since the two channels carry different causal content.
"""
import pandas as pd, numpy as np, os, warnings
from scipy import stats
warnings.filterwarnings('ignore')

from paths import BASE  # repo root (override with PISA_BASE)
OUT = os.path.join(BASE, 'outputs')
q = pd.read_parquet(os.path.join(BASE, 'data', 'qci_analysis.parquet'))
REP = [f'W_FSTURWT{i}' for i in range(1, 81)]
FAY, M_PV, R = 0.5, 10, 80

# ---------- Base variables (identical to script 04) ----------
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

Z = ['AI_total', 'AI_deep', 'AI_sub', 'ICTINFO', 'ICTQUAL', 'ICTHOME', 'ICTWKDY',
     'JOYSCIE', 'EFFSCIE', 'SELFREG', 'PERSEV', 'CURIO', 'ENGSCIPR', 'TEACHSUP',
     'COGACSC', 'DISCLISCI', 'SCIECOLLR', 'DIGPREP', 'SCTRAIN', 'EDUSHORT',
     'STAFFSHORT', 'NEGSCLIM', 'SCH_AIPOL', 'SCH_AITRAIN', 'HOMEPOS']
for v in Z:
    if v in q.columns:
        q[v + '_z'] = (q[v] - q[v].mean()) / q[v].std()
q['escs_z'] = (q.escs - q.escs.mean()) / q.escs.std()

g = q.groupby('CNTSCHID')
q['SCH_ESCS'] = g['ESCS'].transform('mean')
q['SCH_AI'] = g['AI_total'].transform('mean')
q['SCH_ESCS_z'] = (q.SCH_ESCS - q.SCH_ESCS.mean()) / q.SCH_ESCS.std()
q['SCH_AI_z'] = (q.SCH_AI - q.SCH_AI.mean()) / q.SCH_AI.std()

q['AI_grp'] = q.AI_total.round()
for k in [2, 3, 4, 5]:
    q[f'AI_g{k}'] = (q.AI_grp == k).astype(float)
q['AI_sq'] = q.AI_total_z ** 2

# ===== Added: single composition contrast =====
# Input-oriented  = (Q01 summarise a given text + Q02 look up material)/2
# Output-oriented = (Q03 write a first draft + Q04 help learning)/2
IT = ['ST438Q01DA', 'ST438Q02DA', 'ST438Q03DA', 'ST438Q04DA']
d = q[IT].where(q[IT] < 95)
q['AI_in'] = d[['ST438Q01DA', 'ST438Q02DA']].mean(axis=1)
q['AI_out'] = d[['ST438Q03DA', 'ST438Q04DA']].mean(axis=1)
q['AI_CONTRAST'] = q['AI_out'] - q['AI_in']          # output-oriented − input-oriented
q['AI_CONTRAST_z'] = (q.AI_CONTRAST - q.AI_CONTRAST.mean()) / q.AI_CONTRAST.std()

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
                         'b': Qb, 'se': se, 't': t, 'p': p, 'n': len(df), 'M_PV': m})


need = list(dict.fromkeys(FULL + ['AI_total', 'AI_CONTRAST', 'AI_grp', 'DIGPREP',
                                  'SCH_AIPOL', 'SCH_AITRAIN', 'SCH_AI', 'ICTWKDY',
                                  'ST438Q01DA', 'ST438Q02DA', 'ST438Q03DA', 'ST438Q04DA']))
need = [c for c in need if c in q.columns]
s = q.dropna(subset=need + ['PV1SCIE', 'PV1CMPS']).copy()
print('=' * 78)
print(f'Analysis sample n={len(s)} ({len(s)/len(q)*100:.1f}%), schools={s.CNTSCHID.nunique()}')
print('=' * 78)

rows = []
# ---------- 1. Composition contrast entered separately ----------
print('\n[1] Total use vs single composition contrast (entered together)')
for cols, lab in [
    (FULL + ['AI_total_z'], 'T0 仅总量'),
    (FULL + ['AI_CONTRAST_z'], 'T1 仅构成对比'),
    (FULL + ['AI_total_z', 'AI_CONTRAST_z'], 'T2 总量+构成'),
]:
    r = run(s, 'SCIE', cols, lab); rows.append(r)
    for _, w in r[r.term.isin(['AI_total_z', 'AI_CONTRAST_z'])].iterrows():
        print(f'  {lab:14s} {w.term:14s} b={w.b:7.2f} se={w.se:5.2f} p={w.p:.4f}')

# VIF
from numpy.linalg import lstsq
Xv = s[['AI_total_z', 'AI_CONTRAST_z']].values
def vif(cols):
    out = {}
    for i, c in enumerate(cols):
        y_ = s[c].values; Xo = s[[x for x in cols if x != c]].values
        Xo = np.column_stack([np.ones(len(s)), Xo])
        b = lstsq(Xo, y_, rcond=None)[0]
        r2 = 1 - ((y_ - Xo @ b) ** 2).sum() / ((y_ - y_.mean()) ** 2).sum()
        out[c] = 1 / (1 - r2)
    return out
print('  VIF:', {k: round(v, 2) for k, v in vif(['AI_total_z', 'AI_CONTRAST_z']).items()})
print('  corr(AI_total, AI_CONTRAST) = %.3f' % s[['AI_total_z', 'AI_CONTRAST_z']].corr().iloc[0, 1])

# ---------- 2. Intervention levers on one common scale ----------
print('\n[2] Intervention levers: weighted score difference from −1SD to +1SD (points)')
rfull = run(s, 'SCIE', FULL + ['AI_total_z', 'AI_CONTRAST_z'], 'L0 全模型')
rows.append(rfull)
c = rfull.set_index('term')
lever = []
spec = [
    ('AI_CONTRAST_z', 'AI 用途构成（输入导向→产出导向）', '学生', '高'),
    ('ICTWKDY_z', '上学日数字设备使用总量', '学生/家庭', '高'),
    ('SELFREG_z', '自我调节学习', '学生（可教）', '中高'),
    ('EFFSCIE_z', '科学自我效能', '学生（可教）', '中高'),
    ('COGACSC_z', '课堂认知激活教学', '教师（可培训）', '高'),
    ('JOYSCIE_z', '科学兴趣', '教师/课程', '中'),
    ('DISCLISCI_z', '探究式科学教学', '教师（可培训）', '高'),
    ('AI_total_z', 'AI 使用强度（线性项）', '学生', '高'),
    ('SCTRAIN_z', '学校科学教师培训', '学校（可投入）', '中高'),
    ('ICTQUAL_z', '学校 ICT 资源质量', '学校（可投入）', '中'),
    ('ICTINFO_z', '学生数字信息素养', '学生（可教）', '中'),
    ('DIGPREP_z', '学校数字教学准备度', '学校（可投入）', '中'),
]
for v, nm, lvl, act in spec:
    if v in c.index:
        b, se, p = c.loc[v, 'b'], c.loc[v, 'se'], c.loc[v, 'p']
        lever.append([nm, lvl, act, b, se, p, 2 * b, 2 * se, 2 * 1.96 * se])
lev = pd.DataFrame(lever, columns=['变量', '层级', '可干预性', 'b', 'se', 'p',
                                   '−1SD→+1SD', 'SE_span', 'CI95半宽'])
lev = lev.sort_values('−1SD→+1SD', key=lambda x: x.abs(), ascending=False)
print(lev.round(2).to_string(index=False))
lev.to_csv(os.path.join(OUT, 'intervention_levers.csv'), index=False, encoding='utf-8-sig')

# ---------- 3. Scenario simulation ----------
print('\n[3] Scenario simulation (based on the A2 grouped-model coefficients)')
a2 = pd.read_csv(os.path.join(OUT, 'regression_main.csv'))
a2 = a2[(a2.model == 'A2 AI分组(全模型)') & (a2.outcome == 'SCIE')].set_index('term')
gap = a2.loc['AI_g5', 'b'] - a2.loc['AI_g4', 'b']
print(f'  Scenario A daily use → weekly use: {gap:+.1f} points  (daily {a2.loc["AI_g5","b"]:.1f}, weekly {a2.loc["AI_g4","b"]:.1f})')
bc = c.loc['AI_CONTRAST_z', 'b']
print(f'  Scenario B composition shifts up by 1 SD overall (more output-oriented): {bc:+.1f} points')
print(f'        composition moves from −0.5SD to +0.5SD: {bc:+.1f} points')
bt = c.loc['AI_total_z', 'b']
print(f'  Scenario C AI intensity shifts down by 1 SD overall (incl. suppressing the daily group): {bt:+.1f} points')
print(f'  Scenario D school teacher training from −1SD to +1SD: {2*c.loc["SCTRAIN_z","b"]:+.1f} points')
print(f'  Scenario E school-day digital device use from +1SD down to the mean: {c.loc["ICTWKDY_z","b"]*(0-1):+.1f} points')

# ---------- 4. ICTWKDY groups ----------
print('\n[4] School-day digital device use (ICTWKDY) groups and science literacy')
qq = s.dropna(subset=['ICTWKDY']).copy()
qq['grp'] = pd.qcut(qq.ICTWKDY, 5, labels=['Q1 最少', 'Q2', 'Q3', 'Q4', 'Q5 最多'])
def wm(x, w):
    m = ~np.isnan(x); return np.average(x[m], weights=w[m])
for gname in ['Q1 最少', 'Q2', 'Q3', 'Q4', 'Q5 最多']:
    sub = qq[qq.grp == gname]
    print(f'  {gname:8s} n={len(sub):5d}  SCIE={wm(sub.PV1SCIE.values, sub.W_FSTUWT.values):6.1f}  '
          f'CMPS={wm(sub.PV1CMPS.values, sub.W_FSTUWT.values):6.1f}  '
          f'ICTWKDY mean={sub.ICTWKDY.mean():6.2f}')
print(f'  Q5−Q1 science-literacy gap = {wm(qq[qq.grp=="Q5 最多"].PV1SCIE.values, qq[qq.grp=="Q5 最多"].W_FSTUWT.values) - wm(qq[qq.grp=="Q1 最少"].PV1SCIE.values, qq[qq.grp=="Q1 最少"].W_FSTUWT.values):+.1f} points')

# ---------- 5. Composition distribution: how many students lean input-oriented ----------
print('\n[5] Use-composition distribution (AI_CONTRAST = output-oriented − input-oriented)')
print('  Mean %.2f, SD %.2f' % (s.AI_CONTRAST.mean(), s.AI_CONTRAST.std()))
print('  Share input-oriented (contrast<0): %.1f%%' % ((s.AI_CONTRAST < 0).mean() * 100))
print('  Share output-oriented (contrast>0): %.1f%%' % ((s.AI_CONTRAST > 0).mean() * 100))
sub5 = s[s.AI_grp == 5]
print('  Share input-oriented among daily users: %.1f%%' % ((sub5.AI_CONTRAST < 0).mean() * 100))
print('  Share for whom "summarise text" is the most frequent item, among daily users: %.1f%%' %
      ((sub5[IT].idxmax(axis=1) == 'ST438Q01DA').mean() * 100))
print('  Share for whom "summarise text" is the most frequent item, full sample: %.1f%%' %
      ((s[IT].idxmax(axis=1) == 'ST438Q01DA').mean() * 100))

# ---------- 6. Basis for tiered recommendations: composition differences across AI-intensity groups ----------
print('\n[6] Use composition by usage-intensity group')
for k in [1, 2, 3, 4, 5]:
    sub = s[s.AI_grp == k]
    if len(sub) < 30: continue
    print(f'  {["","Never","Yearly","Monthly","Weekly","Daily"][k]:8s} n={len(sub):5d}  '
          f'input-oriented={sub.AI_in.mean():.2f}  output-oriented={sub.AI_out.mean():.2f}  '
          f'contrast={sub.AI_CONTRAST.mean():+.2f}')

pd.concat(rows).to_csv(os.path.join(OUT, 'levers_regression.csv'), index=False, encoding='utf-8-sig')
print('\nDone: outputs/intervention_levers.csv, outputs/levers_regression.csv')
