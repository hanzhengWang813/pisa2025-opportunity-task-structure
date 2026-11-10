# -*- coding: utf-8 -*-
"""Main paper figures with the school as the organising thread.

Figure 1 Variance decomposition and slope heterogeneity
Figure 2 School-level moderator scan (forest plot; none significant)
and the remaining school-side exhibits.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from wmlm import TwoLevelMLFast

from paths import BASE  # repo root (override with PISA_BASE)
OUT = os.path.join(BASE, 'outputs')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DengXian']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9.5
INK, MUTED, RED, TEAL, GREY = '#2C2C2A', '#5F5E5A', '#A32D2D', '#0F6E56', '#888780'

q = pd.read_parquet(os.path.join(BASE, 'data/qci_school.parquet'))
q['female'] = (q['ST004D01T'] == 1).astype(float).where(q['ST004D01T'].notna())
q['repeat'] = (q['REPEAT'] > 0).astype(float).where(q['REPEAT'].notna())
q['escs'] = q['ESCS']
for c in ['ESCS', 'GRADE', 'JOYSCIE', 'EFFSCIE', 'SELFREG', 'COGACSC', 'TEACHSUP',
          'DISCLISCI', 'ICTWKDY', 'ICTHOME']:
    q[c + '_z'] = (q[c] - q[c].mean()) / q[c].std()
q['grade_z'] = q['GRADE_z']
q['AIz'] = (q['AI_total'] - q['AI_total'].mean()) / q['AI_total'].std()
COMP = {'COGACSC': '认知激活', 'DISCLISCI': '纪律氛围', 'SELFREG': '自我调节', 'JOYSCIE': '科学兴趣'}
for c in COMP:
    q['SM_' + c] = q['CNTSCHID'].map(q.groupby('CNTSCHID')[c].mean())
    q['SM_' + c + '_z'] = (q['SM_' + c] - q['SM_' + c].mean()) / q['SM_' + c].std()
cols = ['SM_' + c + '_z' for c in COMP]
q['SCH_CLIM_z'] = q[cols].mean(axis=1)
q['SCH_CLIM_z'] = (q['SCH_CLIM_z'] - q['SCH_CLIM_z'].mean()) / q['SCH_CLIM_z'].std()

STU = ['female', 'escs', 'repeat', 'grade_z', 'JOYSCIE_z', 'EFFSCIE_z', 'SELFREG_z',
       'COGACSC_z', 'TEACHSUP_z', 'DISCLISCI_z', 'ICTWKDY_z', 'ICTHOME_z']
need = STU + ['AIz', 'W_FSTUWT', 'CNTSCHID', 'SCH_CLIM_z']
s = q.dropna(subset=need).copy()
s['g'] = s.CNTSCHID.astype('category').cat.codes
W = s.W_FSTUWT.values
W = W / W.mean()
IG = np.ones(len(s))
Z1, Z2 = IG.reshape(-1, 1), np.column_stack([IG, s.AIz.values])

def X_of(cols):
    return np.column_stack([IG] + [s[c].values for c in cols])

specs = [('空模型', [], Z1), ('+学生协变量', STU, Z1), ('+AI 强度', STU + ['AIz'], Z1),
         ('+AI 随机斜率', STU + ['AIz'], Z2), ('+学校氛围', STU + ['AIz', 'SCH_CLIM_z'], Z2)]
rows = []
for nm, cols, Z in specs:
    X = X_of(cols)
    t00, t11, s2, lls = [], [], [], []
    for i in range(1, 11):
        m = TwoLevelMLFast(s[f'PV{i}SCIE'].values, X, Z, s.g.values, w=W).fit()
        t00.append(m.var['tau00']); t11.append(m.var['tau11'])
        s2.append(m.var['sigma2']); lls.append(m.ll)
    rows.append(dict(model=nm, tau00=np.mean(t00), tau11=np.nanmean(t11),
                     sigma2=np.mean(s2), icc=np.mean(t00) / (np.mean(t00) + np.mean(s2))))
    print(f'{nm:14s} tau00={rows[-1]["tau00"]:7.1f} tau11={rows[-1]["tau11"]:6.2f} '
          f'ICC={rows[-1]["icc"]:.3f}')
ml = pd.DataFrame(rows)
ml.to_csv(os.path.join(OUT, 'mlm_variance.csv'), index=False, encoding='utf-8-sig')

# ---------- Figure 1 ----------
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axes[0]
labels = ml.model.tolist()
bt = ml.tau00.values; be = ml.sigma2.values
x = np.arange(len(labels))
ax.bar(x, bt, 0.58, color=TEAL, label='学校间方差 $\\tau_{00}$')
ax.bar(x, be, 0.58, bottom=bt, color='#D8E4E0', label='校内方差 $\\sigma^2$')
for i, (a, b) in enumerate(zip(bt, be)):
    ax.text(i, a + b + 220, f'ICC {a/(a+b):.2f}', ha='center', fontsize=8.6, color=INK)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.6, rotation=14, ha='right')
ax.set_ylabel('方差（分$^2$）'); ax.set_title('科学素养的方差分解', fontsize=10.5, pad=8)
ax.legend(frameon=False, fontsize=8.2, loc='upper right')
ax.set_ylim(0, max(bt + be) * 1.16)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)

ax = axes[1]
# Slope heterogeneity: illustrative distribution generated from the M4 random-slope variance tau11
tau11 = ml.loc[ml.model == '+AI 随机斜率', 'tau11'].iloc[0]
tau00 = ml.loc[ml.model == '+AI 随机斜率', 'tau00'].iloc[0]
rng = np.random.default_rng(7)
mu = -2.9
slopes = rng.normal(mu, np.sqrt(tau11), 4000)
ax.hist(slopes, bins=44, color='#CFE0DB', edgecolor='white', linewidth=.5)
ax.axvline(mu, color=INK, lw=1.6, label=f'平均斜率 {mu:+.1f}')
ax.axvline(mu - 2 * np.sqrt(tau11), color=RED, lw=1.1, ls='--')
ax.axvline(mu + 2 * np.sqrt(tau11), color=RED, lw=1.1, ls='--',
           label=f'±2SD（斜率的 95% 范围，{np.sqrt(tau11):.1f}）')
ax.axvline(0, color=MUTED, lw=.8)
ax.set_xlabel('学校层面的 AI 斜率（分 / 1 SD AI 使用）')
ax.set_ylabel('学校数（示意）')
ax.set_title(f'AI 关联强度在学校间显著不同（$\\tau_{{11}}$={tau11:.1f}，LRT p<.001）', fontsize=9.6, pad=8)
ax.legend(frameon=False, fontsize=8)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig6_variance.png'), dpi=210, bbox_inches='tight')
plt.close()
print('Figure 1 done')

# ---------- Figure 2 Moderator-scan forest plot ----------
scan = pd.read_csv(os.path.join(OUT, 'moderator_scan.csv'))
scan = scan[scan.outcome == 'SCIE'].copy()
agg = pd.read_csv(os.path.join(OUT, 'school_aggregate_scan.csv'))
agg = agg[agg.outcome == 'SCIE'].copy()
scan['源'] = '校长报告'
agg['源'] = '校内实践聚合'
comb = pd.concat([scan[['moderator', 'b_int', 'se_int', 'p_int', '源']],
                  agg.rename(columns={'moderator': 'moderator'})[['moderator', 'b_int', 'se_int', 'p_int', '源']]],
                 ignore_index=True)
comb = comb.sort_values('b_int')
comb.to_csv(os.path.join(OUT, 'moderator_all.csv'), index=False, encoding='utf-8-sig')

fig, ax = plt.subplots(figsize=(7.6, 6.6))
y = np.arange(len(comb))
for i, (_, r) in enumerate(comb.iterrows()):
    col = RED if r.源 == '校长报告' else TEAL
    lo, hi = r.b_int - 1.96 * r.se_int, r.b_int + 1.96 * r.se_int
    ax.plot([lo, hi], [i, i], color=col, lw=1.5, solid_capstyle='round', alpha=.85)
    ax.plot(r.b_int, i, 'o', color=col, ms=4.4)
ax.axvline(0, color=INK, lw=1.0)
ax.axvspan(-3, 3, color='#F1EFE8', zorder=0)
ax.text(0, len(comb) - 0.2, '灰色带：|效应| < 3 分/SD\n可被本研究排除', ha='center', va='top',
        fontsize=7.6, color=MUTED)
ax.set_yticks(y); ax.set_yticklabels(comb.moderator, fontsize=8.3)
ax.set_xlabel('跨层交互系数（分 / 学校特征 1 SD × AI 使用 1 SD）')
ax.set_title('学校层特征与 AI 使用强度的跨层交互：26 个候选无一稳健显著', fontsize=10.2, pad=9)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=RED, marker='o', ls='-', label='校长报告的条件'),
                   Line2D([], [], color=TEAL, marker='o', ls='-', label='学生报告的校内实践（校均值）')],
          frameon=False, fontsize=8.2, loc='lower right')
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig7_moderator_forest.png'), dpi=210, bbox_inches='tight')
plt.close()
print('Figure 2 done')

# ---------- Figure 3 Single-level vs within-school ----------
lvl = pd.read_csv(os.path.join(OUT, 'level_id.csv'))
use = pd.read_csv(os.path.join(OUT, 'uses_within.csv'))
IT = ['ST438Q01DA', 'ST438Q02DA', 'ST438Q03DA', 'ST438Q04DA']
LAB = dict(zip(IT, ['总结阅读材料', '初步研究新主题', '起草写作', '帮助学习']))

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9), gridspec_kw=dict(width_ratios=[.85, 1.15]))
ax = axes[0]
items = [('AI 强度（每 SD）', 'SCIE|A 单层(强度)', 'AIz', 'SCIE|C 校内FE(强度)', 'AI_within'),
         ('每天使用（相对其他）', 'SCIE|A 单层(每天)', 'AI_daily', 'SCIE|C 校内FE(每天)', 'D_within')]
yy = np.arange(len(items))[::-1]
for k, (lab, m1, t1, m2, t2) in enumerate(items):
    a = lvl[(lvl.model == m1) & (lvl.term == t1)].iloc[0]
    b = lvl[(lvl.model == m2) & (lvl.term == t2)].iloc[0]
    ax.plot([a.b - 1.96 * a.se, a.b + 1.96 * a.se], [yy[k] + .12] * 2, color=GREY, lw=2.1)
    ax.plot(a.b, yy[k] + .12, 'o', color=GREY, ms=6)
    ax.plot([b.b - 1.96 * b.se, b.b + 1.96 * b.se], [yy[k] - .12] * 2, color=TEAL, lw=2.1)
    ax.plot(b.b, yy[k] - .12, 'o', color=TEAL, ms=6)
    ax.text(a.b, yy[k] + .26, f'{a.b:+.1f}', ha='center', fontsize=8.2, color=MUTED)
    ax.text(b.b, yy[k] - .40, f'{b.b:+.1f}', ha='center', fontsize=8.2, color=TEAL)
ax.axvline(0, color=INK, lw=1.0)
ax.set_yticks(yy); ax.set_yticklabels([i[0] for i in items], fontsize=9)
ax.set_xlabel('科学素养差（分）')
ax.set_title('关联在校内比较中依然成立', fontsize=10.2, pad=8)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=GREY, marker='o', ls='-', label='单层（含校间差异）'),
                   Line2D([], [], color=TEAL, marker='o', ls='-', label='校内固定效应（仅个体比较）')],
          frameon=False, fontsize=8, loc='lower left')
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)

ax = axes[1]
m1 = use[use.model == 'SCIE|E1 四用途 单层'].set_index('term')
m2 = use[use.model == 'SCIE|E3 四用途 校内FE'].set_index('term')
yy = np.arange(len(IT))[::-1]
for k, c in enumerate(IT):
    a = m1.loc['ciz_' + c]; b = m2.loc['ciz_' + c]
    ax.plot([a.b - 1.96 * a.se, a.b + 1.96 * a.se], [yy[k] + .12] * 2, color=GREY, lw=2.1)
    ax.plot(a.b, yy[k] + .12, 'o', color=GREY, ms=6)
    ax.plot([b.b - 1.96 * b.se, b.b + 1.96 * b.se], [yy[k] - .12] * 2, color=TEAL, lw=2.1)
    ax.plot(b.b, yy[k] - .12, 'o', color=TEAL, ms=6)
    ax.text(b.b, yy[k] - .38, f'{b.b:+.1f}', ha='center', fontsize=8.2, color=TEAL)
ax.axvline(0, color=INK, lw=1.0)
ax.set_yticks(yy); ax.set_yticklabels([LAB[c] for c in IT], fontsize=9)
ax.set_xlabel('科学素养差（分 / 各用途 1 SD，四题同时入模）')
ax.set_title('"怎么用"的效应在本校比较中稳定保留', fontsize=10.2, pad=8)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig8_within_between.png'), dpi=210, bbox_inches='tight')
plt.close()
print('Figure 3 done')

# ---------- Figure 4 School main effects (re-estimated: school-mean ESCS and academic climate included) ----------
from pisa_engine import PISAEngine
SCHZ = ['A_DIGCOND_z', 'B_TCHCAP_z', 'C_DIGPURP_z', 'D_RULES_z', 'F_SRL_z',
        'G_DIGORG_z', 'H_MONITOR_z', 'I_ENABLE_z']
for c in SCHZ:
    base = c.replace('_z', '')
    q[c] = (q[base] - q[base].mean()) / q[base].std()
q['SCH_MEAN_ESCS_z'] = (q['SCH_MEAN_ESCS'] - q['SCH_MEAN_ESCS'].mean()) / q['SCH_MEAN_ESCS'].std()
need2 = STU + ['AIz', 'W_FSTUWT'] + SCHZ + ['SCH_MEAN_ESCS_z', 'SCH_CLIM_z']
s3 = q.dropna(subset=need2).copy()
eng = PISAEngine(s3, 'PV', 'SCIE')
main = eng.fit(STU + ['AIz', 'SCH_MEAN_ESCS_z', 'SCH_CLIM_z'] + SCHZ, 'MAIN')
main.to_csv(os.path.join(OUT, 'school_main.csv'), index=False, encoding='utf-8-sig')
print(main[main.term.isin(['SCH_MEAN_ESCS_z', 'SCH_CLIM_z'] + SCHZ)]
      .assign(p=lambda d: d.p.round(4))[['term', 'b', 'se', 'p']].round(2).to_string(index=False))
sub = main.copy()
KEEP = [('SCH_MEAN_ESCS_z', '校均家庭 ESCS'), ('SCH_CLIM_z', '学校学术氛围'),
        ('A_DIGCOND_z', '数字条件（设备/带宽）'), ('B_TCHCAP_z', '教师技术与备课时间'),
        ('C_DIGPURP_z', '数字资源使用取向'), ('D_RULES_z', '数字使用规范'),
        ('F_SRL_z', '自主学习教学优先'), ('G_DIGORG_z', '数字教学组织形态'),
        ('H_MONITOR_z', '学习过程追踪'), ('I_ENABLE_z', '以数字资源促自主学习')]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ys = np.arange(len(KEEP))[::-1]
for k, (v, lab) in enumerate(KEEP):
    r = sub[sub.term == v]
    if len(r) == 0:
        continue
    r = r.iloc[0]
    sig = r.p < .05
    col = TEAL if sig else GREY
    ax.plot([r.b - 1.96 * r.se, r.b + 1.96 * r.se], [ys[k]] * 2, color=col, lw=2.1)
    ax.plot(r.b, ys[k], 'o', color=col, ms=5.6)
    ax.text(r.b + (1.2 if r.b >= 0 else -1.2), ys[k], f'{r.b:+.1f}',
            va='center', ha='left' if r.b >= 0 else 'right', fontsize=8.4,
            color=INK if sig else MUTED)
ax.axvline(0, color=INK, lw=1.0)
ax.set_yticks(ys); ax.set_yticklabels([k[1] for k in KEEP], fontsize=9)
ax.set_xlabel('与科学素养的关联（分 / 1 SD，含学生层 12 个协变量与 AI 使用）')
ax.set_title('学校层特征与平均成绩：教育性条件有作用，装备条件作用有限', fontsize=10.2, pad=9)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig9_school_effects.png'), dpi=210, bbox_inches='tight')
plt.close()
print('Figure 4 done')
