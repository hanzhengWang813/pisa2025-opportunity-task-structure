# -*- coding: utf-8 -*-
"""12 复现徐瑾劼 & Schleicher (2026, 开放教育研究) 的规范，检验我们的结论是否对设定敏感

背景：该文与本项目使用完全相同的 PISA 2025 B-S-J-Z 数据（374 校、12,385 人），
同样对四类用途"同时入模"。其结论：帮助学习 +22.06、总结阅读 −19.07、
初步研究仅阅读显著、起草写作三科均不显著（科学）。
我们（连续变量、四题同入）得到：帮学习 +9.7、写初稿 +3.4、查资料 −3.5、概括 −12.3（标准化）。

本脚本做三件事：
1. 严格按其规范复现（"每周及以上"二分 + ESCS/性别/学段 三个控制变量 + PV/BRR）
2. 同一控制变量下换成连续变量，隔离"操作化"与"控制变量集"两个来源
3. 复现其四组强度分组（未使用/低频/适度/高频）的调整后均值，对照我们的阈值结论
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
LAB = dict(zip(IT, ['概括阅读材料', '初步研究新主题', '起草写作作业', '帮助学习']))

# ---------- 变量 ----------
d = q[IT].where(q[IT] < 95)
for c in IT:
    q[c] = d[c]
q['AIUSESCH'] = d.mean(axis=1)                     # 四题均值（总体强度）
q['female'] = (q.ST004D01T == 1).astype(float)
q['senior'] = (q.GRADE >= 10).astype(float)        # 高中=1（PISA 15 岁，初中 7-9 / 高中 10-12）
q['escs'] = q.ESCS

# 徐文规范：四类用途的"每周及以上"二分
for c in IT:
    q['W_' + c] = (q[c] >= 4).astype(float).where(q[c].notna())

# 强度四分组（徐文定义）
mx = d.max(axis=1)
q['AI4'] = pd.Series(np.select(
    [d.isna().any(axis=1), (d == 1).all(axis=1), mx == 2, mx.isin([3, 4]), mx == 5],
    ['NA', '1未使用', '2低频', '3适度', '4高频'], default='NA'), index=q.index)
q.loc[q.AI4 == 'NA', 'AI4'] = np.nan

need = IT + ['W_' + c for c in IT] + ['AIUSESCH', 'female', 'senior', 'escs', 'CNTSCHID', 'W_FSTUWT'] + REP
s = q.dropna(subset=need + [f'PV{k}SCIE' for k in range(1, 11)]).copy()
print('=' * 80)
print(f'分析样本 n={len(s)}, 学校={s.CNTSCHID.nunique()}  （徐文报告 12,241—12,360）')
print('=' * 80)


# ---------- PV + BRR-Fay 引擎 ----------
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

# ---------- 1. 严格按徐文规范：二分 + 三控制变量 ----------
print('\n【1】复现徐文规范（"每周及以上"二分，四类同时入模，控制 ESCS/性别/学段）')
for y in ['SCIE', 'MATH', 'READ']:
    r = run(s, y, ['W_' + c for c in IT] + CTRL, f'A 徐文二分/{y}')
    rows.append(r)
    w = r.set_index('term')
    line = f'  {y:5s}'
    for c in IT:
        line += f' | {LAB[c]} {w.loc["W_"+c,"b"]:+7.2f}{"***" if w.loc["W_"+c,"p"]<.001 else "**" if w.loc["W_"+c,"p"]<.01 else "*" if w.loc["W_"+c,"p"]<.05 else "  "}'
    print(line)
print('  （徐文报告：科学 帮助学习 +22.06 / 总结 −19.07；数学 +22.47 / −20.30；阅读 +18.85 / −15.92）')

# ---------- 2. 同一控制变量下改连续变量 ----------
print('\n【2】同一控制变量集，改用四题连续值（我们的操作化，未标准化）')
for y in ['SCIE', 'MATH', 'READ']:
    r = run(s, y, IT + CTRL, f'B 连续/{y}')
    rows.append(r)
    w = r.set_index('term')
    line = f'  {y:5s}'
    for c in IT:
        line += f' | {LAB[c]} {w.loc[c,"b"]:+7.2f}{"***" if w.loc[c,"p"]<.001 else "**" if w.loc[c,"p"]<.01 else "*" if w.loc[c,"p"]<.05 else "  "}'
    print(line)

# ---------- 3. 二分 + 我们的完整控制变量（判断是否控制变量集造成差异） ----------
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
    print(f'\n【3】二分 + 扩展控制变量集（n={len(s2)}）')
    for y in ['SCIE']:
        r = run(s2, y, ['W_' + c for c in IT] + FULLCTRL, f'C 二分+全控制/{y}')
        rows.append(r)
        w = r.set_index('term')
        for c in IT:
            print(f'  {LAB[c]:8s} b={w.loc["W_"+c,"b"]:+7.2f} (se={w.loc["W_"+c,"se"]:5.2f}) p={w.loc["W_"+c,"p"]:.4f}')

# ---------- 4. 强度四分组：调整后均值 ----------
print('\n【4】复现徐文四组强度分组（未使用/低频/适度/高频）')
sub = s.dropna(subset=['AI4']).copy()
for g in ['1未使用', '2低频', '3适度', '4高频']:
    sub['g_' + g] = (sub.AI4 == g).astype(float)
r = run(sub, 'SCIE', ['g_' + g for g in ['2低频', '3适度', '4高频']] + CTRL, 'D 四分组')
rows.append(r)
w = r.set_index('term')
descr = sub.groupby('AI4').apply(lambda x: pd.Series({
    'n': len(x), '占比%': len(x) / len(sub) * 100,
    '加权均值': np.average(x.PV1SCIE, weights=x.W_FSTUWT)}), include_groups=False)
print(descr.round(1).to_string())
print('\n  以"未使用"为参照的调整后差异：')
for g in ['2低频', '3适度', '4高频']:
    st = '***' if w.loc['g_' + g, 'p'] < .001 else '**' if w.loc['g_' + g, 'p'] < .01 else '*' if w.loc['g_' + g, 'p'] < .05 else ''
    print(f'    {g}: b={w.loc["g_"+g,"b"]:+7.2f} (se={w.loc["g_"+g,"se"]:5.2f}) p={w.loc["g_"+g,"p"]:.4f} {st}')
print('  （徐文：高频使用者占 10.8%，未使用与适度使用者成绩无显著差异）')

def star(p):
    return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''

pd.concat(rows).to_csv(os.path.join(OUT, 'replicate_xu2026.csv'), index=False, encoding='utf-8-sig')
print('\n结果已保存 outputs/replicate_xu2026.csv')

# ---------- 5. 我们原规范的对照（连续+标准化+完整控制，用于三方比较） ----------
print('\n【5】三种设定下"起草写作"与"初步研究"的系数对照')
cur = pd.read_csv(os.path.join(OUT, 'regression_main.csv'))
b1 = cur[(cur.model.str.contains('双维度', na=False)) & (cur.outcome == 'SCIE')]
if len(b1):
    for t, nm in [('AI_sub_z', '写初稿(构成)'), ('AI_deep_z', '查资料(构成)')]:
        row = b1[b1.term == t]
        if len(row):
            print(f'    我们原设定 {nm}: b={row.b.iloc[0]:+.2f} p={row.p.iloc[0]:.4f}')
print('    徐文设定见上表【1】【2】【3】')
