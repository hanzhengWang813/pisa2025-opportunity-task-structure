# -*- coding: utf-8 -*-
"""Replicate Xu & Schleicher (2026) to test whether our conclusions are specification-sensitive.

That paper uses exactly the same PISA 2025 B-S-J-Z data as this project
(374 schools, 12,385 students) and also enters the four use purposes
simultaneously. Its reported science coefficients are +22.06 for help with
learning, -19.07 for summarising and reading, significance for reading only
under the initial-research purpose, and no significance for drafting/writing.
"""
import pandas as pd, numpy as np, os, warnings
from scipy import stats
warnings.filterwarnings('ignore')

from paths import BASE  # repo root (override with PISA_BASE)
OUT = os.path.join(BASE, 'outputs')
q = pd.read_parquet(os.path.join(BASE, 'data', 'qci_analysis.parquet'))
REP = [f'W_FSTURWT{i}' for i in range(1, 81)]
FAY, M_PV, R = 0.5, 10, 80
IT = ['ST438Q01DA', 'ST438Q02DA', 'ST438Q03DA', 'ST438Q04DA']
LAB = dict(zip(IT, ['Summarising reading material', 'Initial research on a new topic', 'Drafting written assignments', 'Help with learning']))

# ---------- Variables ----------
d = q[IT].where(q[IT] < 95)
for c in IT:
    q[c] = d[c]
q['AIUSESCH'] = d.mean(axis=1)                     # mean of the four items (overall intensity)
q['female'] = (q.ST004D01T == 1).astype(float)
q['senior'] = (q.GRADE >= 10).astype(float)        # upper secondary = 1 (PISA age 15: lower secondary grades 7-9, upper 10-12)
q['escs'] = q.ESCS

# Xu et al. specification: binary "weekly or more" indicator for each of the four purposes
for c in IT:
    q['W_' + c] = (q[c] >= 4).astype(float).where(q[c].notna())

# Four intensity groups (Xu et al. definition)
mx = d.max(axis=1)
q['AI4'] = pd.Series(np.select(
    [d.isna().any(axis=1), (d == 1).all(axis=1), mx == 2, mx.isin([3, 4]), mx == 5],
    ['NA', '1未使用', '2低频', '3适度', '4高频'], default='NA'), index=q.index)
q.loc[q.AI4 == 'NA', 'AI4'] = np.nan

need = IT + ['W_' + c for c in IT] + ['AIUSESCH', 'female', 'senior', 'escs', 'CNTSCHID', 'W_FSTUWT'] + REP
s = q.dropna(subset=need + [f'PV{k}SCIE' for k in range(1, 11)]).copy()
print('=' * 80)
print(f'Analysis sample n={len(s)}, schools={s.CNTSCHID.nunique()}  (Xu et al. report 12,241-12,360)')
print('=' * 80)


# ---------- PV + BRR-Fay engine ----------
def wls(X, y, w):
    Xt = X.T * w
    return np.linalg.solve(Xt @ X + 1e-10 * np.eye(X.shape[1]), Xt @ y)


def run(df, yname, cols, label):
    X = np.column_stack([np.ones(len(df))] + [df[c].values for c in cols])
    names = ['const'] + cols
    Y = df[[f'PV{k}{yname}' for k in range(1, M_PV + 1)]].values
    W = df[REP].values; w0 = df['W_FSTUWT'].values
    cs, vs = [], []
    for j in range(Y.shape[1]):
        y = Y[:, j]; b0 = wls(X, y, w0)
        D = np.array([wls(X, y, W[:, r]) - b0 for r in range(R)])
        cs.append(b0); vs.append((D ** 2).sum(axis=0) / (R * (1 - FAY) ** 2))
    cs, vs = np.array(cs), np.array(vs); m = len(cs)
    Qb, Ub = cs.mean(0), vs.mean(0)
    B = cs.var(0, ddof=1)
    T = Ub + (1 + 1 / m) * B
    se = np.sqrt(np.where(T > 0, T, np.nan))
    with np.errstate(divide='ignore', invalid='ignore'):
        dfr = np.where(B > 0, (m - 1) * (1 + Ub / ((1 + 1 / m) * B)) ** 2, np.inf)
        t = Qb / se
        p = 2 * stats.t.sf(np.abs(t), np.where(np.isfinite(dfr), dfr, 1e6))
    return pd.DataFrame({'model': label, 'outcome': yname, 'term': names,
                         'b': Qb, 'se': se, 'p': p, 'n': len(df)})


CTRL = ['escs', 'female', 'senior']
rows = []

# ---------- 1. Strict Xu et al. specification: binary + three controls ----------
print('\n[1] Replicate the Xu et al. specification (binary "weekly or more", all four purposes entered together, controlling ESCS/gender/grade band)')
for y in ['SCIE', 'MATH', 'READ']:
    r = run(s, y, ['W_' + c for c in IT] + CTRL, f'A 徐文二分/{y}')
    rows.append(r)
    w = r.set_index('term')
    line = f'  {y:5s}'
    for c in IT:
        line += f' | {LAB[c]} {w.loc["W_"+c,"b"]:+7.2f}{"***" if w.loc["W_"+c,"p"]<.001 else "**" if w.loc["W_"+c,"p"]<.01 else "*" if w.loc["W_"+c,"p"]<.05 else "  "}'
    print(line)
print('  (Xu et al. report: science help with learning +22.06 / summarising -19.07; maths +22.47 / -20.30; reading +18.85 / -15.92)')

# ---------- 2. Continuous variables under the same control set ----------
print('\n[2] Same control set, four items entered as continuous values (our operationalisation, unstandardised)')
for y in ['SCIE', 'MATH', 'READ']:
    r = run(s, y, IT + CTRL, f'B 连续/{y}')
    rows.append(r)
    w = r.set_index('term')
    line = f'  {y:5s}'
    for c in IT:
        line += f' | {LAB[c]} {w.loc[c,"b"]:+7.2f}{"***" if w.loc[c,"p"]<.001 else "**" if w.loc[c,"p"]<.01 else "*" if w.loc[c,"p"]<.05 else "  "}'
    print(line)

# ---------- 3. Binary + our full control set (test whether the control set drives the difference) ----------
FULLCTRL = None
try:
    import importlib.util
    FULLCTRL = ['escs', 'female', 'senior', 'repeat', 'immig1', 'immig2',
                'JOYSCIE_z', 'EFFSCIE_z', 'SELFREG_z', 'PERSEV_z', 'CURIO_z', 'ENGSCIPR_z',
                'TEACHSUP_z', 'COGACSC_z', 'DISCLISCI_z', 'ICTWKDY_z', 'ICTINFO_z', 'ICTHOME_z',
                'DIGPREP_z', 'SCTRAIN_z', 'SCH_ESCS_z']
    FULLCTRL = [c for c in FULLCTRL if c in q.columns]
except Exception:
    pass
if FULLCTRL:
    s2 = s.dropna(subset=[c for c in FULLCTRL if c not in s.columns or s[c].isna().any()]).copy()
    print(f'\n[3] Binary + extended control set (n={len(s2)})')
    for y in ['SCIE']:
        r = run(s2, y, ['W_' + c for c in IT] + FULLCTRL, f'C 二分+全控制/{y}')
        rows.append(r)
        w = r.set_index('term')
        for c in IT:
            print(f'  {LAB[c]:8s} b={w.loc["W_"+c,"b"]:+7.2f} (se={w.loc["W_"+c,"se"]:5.2f}) p={w.loc["W_"+c,"p"]:.4f}')

# ---------- 4. Four intensity groups: adjusted means ----------
print('\n[4] Replicate the Xu et al. four-group intensity classification (non-use / low / moderate / high)')
sub = s.dropna(subset=['AI4']).copy()
for g in ['1未使用', '2低频', '3适度', '4高频']:
    sub['g_' + g] = (sub.AI4 == g).astype(float)
r = run(sub, 'SCIE', ['g_' + g for g in ['2低频', '3适度', '4高频']] + CTRL, 'D 四分组')
rows.append(r)
w = r.set_index('term')
descr = sub.groupby('AI4').apply(lambda x: pd.Series({
    'n': len(x), 'share%': len(x) / len(sub) * 100,
    'weighted_mean': np.average(x.PV1SCIE, weights=x.W_FSTUWT)}), include_groups=False)
print(descr.round(1).to_string())
print('\n  Adjusted differences relative to "non-use":')
for g in ['2低频', '3适度', '4高频']:
    st = '***' if w.loc['g_' + g, 'p'] < .001 else '**' if w.loc['g_' + g, 'p'] < .01 else '*' if w.loc['g_' + g, 'p'] < .05 else ''
    print(f'    {g}: b={w.loc["g_"+g,"b"]:+7.2f} (se={w.loc["g_"+g,"se"]:5.2f}) p={w.loc["g_"+g,"p"]:.4f} {st}')
print('  (Xu et al.: high-frequency users account for 10.8%; non-users and moderate users do not differ significantly)')

def star(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''

pd.concat(rows).to_csv(os.path.join(OUT, 'replicate_xu2026.csv'), index=False, encoding='utf-8-sig')
print('\nResults saved to outputs/replicate_xu2026.csv')

# ---------- 5. Comparison with our original specification (continuous + standardised + full controls, three-way) ----------
print('\n[5] Coefficients for "drafting/writing" and "initial research" under three specifications')
cur = pd.read_csv(os.path.join(OUT, 'regression_main.csv'))
b1 = cur[(cur.model.str.contains('双维度', na=False)) & (cur.outcome == 'SCIE')]
if len(b1):
    for t, nm in [('AI_sub_z', 'drafting (component)'), ('AI_deep_z', 'researching sources (component)')]:
        row = b1[b1.term == t]
        if len(row):
            print(f'    Our original specification {nm}: b={row.b.iloc[0]:+.2f} p={row.p.iloc[0]:.4f}')
print('    Xu et al. specification: see panels [1] [2] [3] above')
