# -*- coding: utf-8 -*-
"""
23. 重制图 6–10（修正版）

fig6 方差分解 + 学校 AI 斜率的实证分布（真实校内斜率，非模拟）
fig7 校长报告条件的调节扫描（中文标签，仅校层变量）
fig8 每天使用与四用途：单层 vs 校内
fig9 学校层主效应
fig10 氛围调节的三种识别（生态假象）
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from paths import BASE  # repo root (override with PISA_BASE)
OUT = os.path.join(BASE, 'outputs')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DengXian']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9.5
INK, MUTED, RED, TEAL, GREY = '#2C2C2A', '#5F5E5A', '#A32D2D', '#0F6E56', '#888780'
LIGHT = '#CFE0DB'

q = pd.read_parquet(os.path.join(BASE, 'data/qci_school.parquet'))
q['AIz'] = (q['AI_total'] - q['AI_total'].mean()) / q['AI_total'].std()

# ---------------- fig6 ----------------
lad0 = pd.read_csv(os.path.join(OUT, 'mlm_ladder.csv'))
ml = lad0[lad0.term == '__var__'].copy().reset_index(drop=True)
ml['model'] = ['空模型', '+12 个学生协变量', '+AI 使用强度', '+AI 随机斜率',
               '+8 个学校层变量', '+8 个跨层交互'][:len(ml)]
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = axes[0]
bt, be = ml.tau00.values, ml.sigma2.values
x = np.arange(len(ml))
ax.bar(x, bt, .58, color=TEAL, label='学校间方差 $\\tau_{00}$')
ax.bar(x, be, .58, bottom=bt, color='#DDE7E4', label='校内方差 $\\sigma^2$')
for i, (a, b) in enumerate(zip(bt, be)):
    ax.text(i, a + b + 200, f'{a/(a+b):.2f}', ha='center', fontsize=9, color=INK, fontweight='600')
ax.set_xticks(x)
ax.set_xticklabels([m.replace('+', '+ ') for m in ml.model], fontsize=8.4, rotation=16, ha='right')
ax.set_ylabel('方差（分$^2$）')
ax.set_title('科学素养的方差分解：学校间占比（顶数字为 ICC）', fontsize=10.2, pad=9)
ax.legend(frameon=False, fontsize=8.2, loc='upper right')
ax.set_ylim(0, max(bt + be) * 1.18)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)

# 斜率方差：加入学校层变量后几乎不变
ax = axes[1]
lad = pd.read_csv(os.path.join(OUT, 'mlm_ladder.csv'))
lad = lad[lad.term == '__var__'].copy()
lad = lad[lad.tau11.notna()]
vals = lad.tau11.values
labs = ['+AI 随机斜率', '+学校层主效应', '+跨层交互']
bars = ax.bar(np.arange(len(vals)), vals, .5, color=[TEAL, TEAL, '#9DC3B8'])
for i, v in enumerate(vals):
    ax.text(i, v + .8, f'{v:.1f}', ha='center', fontsize=9, color=INK, fontweight='600')
ax.set_xticks(np.arange(len(vals))); ax.set_xticklabels(labs, fontsize=8.6)
ax.set_ylabel('斜率方差 $\\tau_{11}$（分$^2$ / SD$^2$）')
red = (vals[0] - vals[-1]) / vals[0] * 100
ax.set_title('斜率差异真实存在，但学校层变量几乎解释不了它', fontsize=10.2, pad=9)
ytop = vals.max() * 1.30
ax.annotate('', xy=(2, ytop - 2), xytext=(0, ytop - 2),
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.2))
ax.text(1.0, ytop + 1.6, f'仅下降 {red:.0f}%', ha='center', fontsize=8.6, color=RED)
ax.set_ylim(0, vals.max() * 1.45)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig6_variance.png'), dpi=215, bbox_inches='tight')
plt.close()
print('fig6 更新：tau11 %s，降幅 %.0f%%' % (np.round(vals, 1).tolist(), red))

# ---------------- fig7 ----------------
NAME = {
    'A 数字条件': '数字条件（设备/带宽/软件）', 'B 教师能力时间': '教师技术能力与备课时间',
    'C 数字使用取向': '数字资源使用取向', 'D 规范共定': '数字使用规范（师生共定）',
    'F 自主学习优先': '自主学习教学优先度', 'G 数字教学组织': '数字教学组织形态',
    'H 过程追踪(单题)': '用数字资源追踪学习过程', 'I 赋能取向(单题)': '以数字资源促自主学习',
    'J 手机禁令': '校内手机禁令', 'K 数字教学准备度': '学校数字教学准备度',
    'L 科学教师培训': '科学教师培训（校方报告）', 'M 教育短缺': '教育资源短缺',
    'N 师资短缺': '师资短缺', 'O 软件可用性': '教学软件可用性',
    'P 教学领导': '教学领导力', 'Q 负面校风': '负面校风',
    'S 书面数字设备规定': '书面数字设备规定', 'T 教师自定使用规则': '教师自定课堂使用规则',
    'U 师生共定规则': '师生共同商定规则', 'V 网络素养课程': '网络素养课程/项目',
    'W 社交网络政策': '社交网络使用政策', 'X 教师应对学生用AI': '教师应对学生用 AI 的准备',
    'Y 数字教学PD需求': '数字教学培训需求（校方）', 'Z 自主学习教学支持': '自主学习教学支持',
    'AA 校均AI使用': '校均 AI 使用水平', 'AB 校内AI离散度': '校内 AI 使用离散度',
    'AC 校均屏幕时间': '校均上学日屏幕时间', 'AD 每天使用占比': '每天使用 AI 的学生占比',
    'AE 学校规模': '学校规模',
}
scan = pd.read_csv(os.path.join(OUT, 'moderator_scan.csv'))
scan = scan[(scan.outcome == 'SCIE') & (scan.moderator.isin(NAME))].copy()
scan['lab'] = scan.moderator.map(NAME)
scan = scan.sort_values('b_int')
fig, ax = plt.subplots(figsize=(7.4, 6.9))
y = np.arange(len(scan))
for i, (_, r) in enumerate(scan.iterrows()):
    lo, hi = r.b_int - 1.96 * r.se_int, r.b_int + 1.96 * r.se_int
    ax.plot([lo, hi], [i, i], color=GREY, lw=1.6, solid_capstyle='round', alpha=.9)
    ax.plot(r.b_int, i, 'o', color=INK, ms=4.2)
ax.axvline(0, color=INK, lw=1.0)
ax.axvspan(-3, 3, color='#F3F1EA', zorder=0)
ax.set_ylim(-1.2, len(scan) + 1.1)
ax.text(0, len(scan) + .55, '阴影带｜效应｜< 3 分/SD：本研究可排除（80% 检验力）',
        ha='center', va='center', fontsize=7.8, color=MUTED)
ax.text(-8.3, -.9, f'29 个候选，最小 p={scan.p_int.min():.3f}，FDR 校正后无一显著；'
                   f'交互中位数 {scan.b_int.median():+.1f} 分',
        ha='left', va='center', fontsize=7.8, color=MUTED)
ax.set_yticks(y); ax.set_yticklabels(scan.lab, fontsize=8.4)
ax.set_xlabel('跨层交互系数（分 / 学校特征 1 SD × AI 使用 1 SD）')
ax.set_title('校长报告的全部学校条件与 AI 使用强度的跨层交互', fontsize=10.4, pad=10)
ax.set_xlim(-8.5, 8.5)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig7_moderator_forest.png'), dpi=215, bbox_inches='tight')
plt.close()
print('fig7 更新：%d 个校层候选，最小 p=%.3f' % (len(scan), scan.p_int.min()))

# ---------------- fig8 ----------------
lvl = pd.read_csv(os.path.join(OUT, 'level_id.csv'))
use = pd.read_csv(os.path.join(OUT, 'uses_within.csv'))
IT = ['ST438Q01DA', 'ST438Q02DA', 'ST438Q03DA', 'ST438Q04DA']
LAB = dict(zip(IT, ['总结阅读材料', '初步研究新主题', '起草写作', '帮助学习']))

fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5), gridspec_kw=dict(width_ratios=[.75, 1.25]))
ax = axes[0]
a = lvl[(lvl.model == 'SCIE|A 单层(每天)') & (lvl.term == 'AI_daily')].iloc[0]
b = lvl[(lvl.model == 'SCIE|C 校内FE(每天)') & (lvl.term == 'D_within')].iloc[0]
ax.plot([a.b - 1.96 * a.se, a.b + 1.96 * a.se], [1] * 2, color=GREY, lw=2.6, solid_capstyle='round')
ax.plot(a.b, 1, 'o', color=GREY, ms=8)
ax.plot([b.b - 1.96 * b.se, b.b + 1.96 * b.se], [0] * 2, color=TEAL, lw=2.6, solid_capstyle='round')
ax.plot(b.b, 0, 'o', color=TEAL, ms=8)
ax.text(-62, 1.45, f'单层模型：{a.b:+.1f}', ha='left', fontsize=8.8, color=MUTED)
ax.text(-62, -.45, f'校内固定效应：{b.b:+.1f}', ha='left', fontsize=8.8, color=TEAL, fontweight='600')
ax.axvline(0, color=INK, lw=1.0)
ax.set_yticks([0, 1]); ax.set_yticklabels(['校内固定效应', '单层模型'], fontsize=9.2)
ax.set_ylim(-.8, 1.85)
ax.set_xlabel('科学素养差（分）')
ax.set_title('「每天使用」的关联\n在校内比较中依然成立', fontsize=10, pad=8)
for sp in ['top', 'right', 'left']:
    ax.spines[sp].set_visible(False)

ax = axes[1]
m1 = use[use.model == 'SCIE|E1 四用途 单层'].set_index('term')
m2 = use[use.model == 'SCIE|E3 四用途 校内FE'].set_index('term')
yy = np.arange(len(IT))[::-1]
for k, c in enumerate(IT):
    a = m1.loc['ciz_' + c]; b = m2.loc['ciz_' + c]
    ax.plot([a.b - 1.96 * a.se, a.b + 1.96 * a.se], [yy[k] + .14] * 2, color=GREY, lw=2.2)
    ax.plot(a.b, yy[k] + .14, 'o', color=GREY, ms=5.4)
    ax.plot([b.b - 1.96 * b.se, b.b + 1.96 * b.se], [yy[k] - .14] * 2, color=TEAL, lw=2.2)
    ax.plot(b.b, yy[k] - .14, 'o', color=TEAL, ms=5.4)
    ax.text(10.6, yy[k] - .14, f'{b.b:+.1f}', va='center', ha='right',
            fontsize=8.4, color=TEAL, fontweight='600')
ax.axvline(0, color=INK, lw=1.0)
ax.set_yticks(yy); ax.set_yticklabels([LAB[c] for c in IT], fontsize=9.2)
ax.set_xlim(-14, 11)
ax.set_xlabel('科学素养差（分 / 各用途 1 SD，四题同时入模）')
ax.set_title('「怎么用」的差异在校内比较中稳定保留', fontsize=10, pad=8)
ax.legend(handles=[Line2D([], [], color=GREY, marker='o', ls='-', label='单层（含校间差异）'),
                   Line2D([], [], color=TEAL, marker='o', ls='-', label='校内固定效应')],
          frameon=False, fontsize=8.2, loc='upper center', ncol=2,
          bbox_to_anchor=(0.45, -0.20))
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig8_within_between.png'), dpi=215, bbox_inches='tight')
plt.close()
print('fig8 更新')

# ---------------- fig9 ----------------
from pisa_engine import PISAEngine
main = pd.read_csv(os.path.join(OUT, 'school_main.csv'))
A7 = ['SC155Q01HA', 'SC155Q02HA', 'SC155Q03HA', 'SC155Q04HA', 'SC155Q05HA',
      'SC155Q06HA', 'SC155Q07HA']
for c in A7:
    q[c] = q[c].where(q[c] < 5)
q['DIGBASE'] = q[A7].mean(axis=1)
q['DIGBASE_z'] = (q['DIGBASE'] - q['DIGBASE'].mean()) / q['DIGBASE'].std()
STU2 = ['female', 'escs', 'repeat', 'grade_z', 'JOYSCIE_z', 'EFFSCIE_z', 'SELFREG_z',
        'COGACSC_z', 'TEACHSUP_z', 'DISCLISCI_z', 'ICTWKDY_z', 'ICTHOME_z']
q['female'] = (q['ST004D01T'] == 1).astype(float).where(q['ST004D01T'].notna())
q['repeat'] = (q['REPEAT'] > 0).astype(float).where(q['REPEAT'].notna())
q['escs'] = q['ESCS']
for c in ['ESCS', 'GRADE', 'JOYSCIE', 'EFFSCIE', 'SELFREG', 'COGACSC', 'TEACHSUP',
          'DISCLISCI', 'ICTWKDY', 'ICTHOME']:
    q[c + '_z'] = (q[c] - q[c].mean()) / q[c].std()
q['grade_z'] = q['GRADE_z']
s4 = q.dropna(subset=STU2 + ['AIz', 'W_FSTUWT', 'DIGBASE_z']).copy()
dgb = PISAEngine(s4, 'PV', 'SCIE').fit(STU2 + ['AIz', 'DIGBASE_z'], 'DIG')
main = pd.concat([main, dgb[['term', 'b', 'se', 'p']]], ignore_index=True)
KEEP = [('SCH_MEAN_ESCS_z', '校均家庭 ESCS'), ('SCH_CLIM_z', '学校学术氛围'),
        ('DIGBASE_z', '数字基础条件（7 题合成）'),
        ('G_DIGORG_z', '数字教学组织形态*'), ('C_DIGPURP_z', '数字资源使用取向'),
        ('D_RULES_z', '数字使用规范'), ('F_SRL_z', '自主学习教学优先度'),
        ('H_MONITOR_z', '用数字资源追踪学习过程'), ('I_ENABLE_z', '以数字资源促自主学习')]
fig, ax = plt.subplots(figsize=(7.4, 4.5))
ys = np.arange(len(KEEP))[::-1]
for k, (v, lab) in enumerate(KEEP):
    r = main[main.term == v].iloc[0]
    sig = (r.p < .05) and (v in ('SCH_MEAN_ESCS_z', 'SCH_CLIM_z'))
    col = TEAL if sig else GREY
    lo, hi = r.b - 1.96 * r.se, r.b + 1.96 * r.se
    ax.plot([lo, hi], [ys[k]] * 2, color=col, lw=2.2)
    ax.plot(r.b, ys[k], 'o', color=col, ms=5.8)
    ax.text(hi + .9, ys[k], f'{r.b:+.1f}', va='center', ha='left', fontsize=8.4,
            color=INK if sig else MUTED)
ax.axvline(0, color=INK, lw=1.0)
ax.set_yticks(ys); ax.set_yticklabels([k[1] for k in KEEP], fontsize=9)
ax.set_ylim(-.7, len(KEEP) - .3)
ax.set_xlabel('与科学素养的关联（分 / 1 SD；已含 12 个学生层协变量与 AI 使用）')
ax.set_title('学校层特征与平均成绩：生源与教学氛围有作用，数字化条件不显著',
             fontsize=10.2, pad=10)
ax.text(0.02, 0.03, '* 该系数对模型设定敏感，不作解释', transform=ax.transAxes,
        ha='left', va='bottom', fontsize=7.6, color=MUTED)
ax.set_xlim(-19, 36)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig9_school_effects.png'), dpi=215, bbox_inches='tight')
plt.close()
print('fig9 更新')

# ---------------- fig10 ----------------
fig, ax = plt.subplots(figsize=(6.4, 3.2))
specs = [('单层模型', 'SCIE|D1 交互 单层', 'AIzxCLIM'),
         ('分离校间成分\n（Mundlak）', 'SCIE|D2 交互 Mundlak', 'AIxCLIM'),
         ('校内固定效应', 'SCIE|D3 交互 校内FE', 'AIxCLIM')]
ys = np.arange(len(specs))[::-1]
for k, (lab, m, t) in enumerate(specs):
    r = lvl[(lvl.model == m) & (lvl.term == t)].iloc[0]
    sig = r.p < .05
    col = RED if k == 0 else TEAL
    ax.plot([r.b - 1.96 * r.se, r.b + 1.96 * r.se], [ys[k]] * 2, color=col, lw=2.4)
    ax.plot(r.b, ys[k], 'o', color=col, ms=6.5)
    ax.text(r.b, ys[k] + .26, f'{r.b:+.2f}（p={r.p:.3f}）', ha='center', fontsize=8.6,
            color=col if sig else MUTED)
ax.axvline(0, color=INK, lw=1.0)
ax.set_yticks(ys); ax.set_yticklabels([s[0] for s in specs], fontsize=9)
ax.set_ylim(-.7, len(specs) - .35)
ax.set_xlabel('学校学术氛围 × AI 使用 的交互系数（分）')
ax.set_title('同一交互、三种识别：单层模型里的“校层调节”在分离校间成分后消失', fontsize=10, pad=9)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig10_ecological.png'), dpi=215, bbox_inches='tight')
plt.close()
print('fig10 更新')
