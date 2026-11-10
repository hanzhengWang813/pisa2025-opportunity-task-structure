# -*- coding: utf-8 -*-
"""Figure 5: rank the intervention levers on a common scale.

Only predictors whose sign is stable across specifications are retained;
suppressor variables (ICTINFO, ICTQUAL) are excluded.
"""
import pandas as pd, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

from paths import BASE  # repo root (override with PISA_BASE)
OUT = os.path.join(BASE, 'outputs')

names = {f.name for f in font_manager.fontManager.ttflist}
for f in ['Microsoft YaHei', 'SimHei', 'DengXian', 'SimSun']:
    if f in names:
        plt.rcParams['font.sans-serif'] = [f]; break
plt.rcParams['axes.unicode_minus'] = False

r = pd.read_csv(os.path.join(OUT, 'levers_regression.csv'))
c = r[r.model == 'L0 全模型'].set_index('term')

# actor responsible -> colour
GROUPS = {
    'AI 使用方式': ('#A32D2D', ['AI_CONTRAST_z', 'AI_total_z']),
    '学生心理资源': ('#2F6F8F', ['EFFSCIE_z', 'SELFREG_z', 'JOYSCIE_z']),
    '教师教学实践': ('#3F7A4F', ['COGACSC_z', 'DISCLISCI_z']),
    '学校可投入':   ('#8A6D2F', ['SCTRAIN_z']),
    '数字实践总量': ('#5B4B8A', ['ICTWKDY_z']),
}
LAB = {
    'AI_CONTRAST_z': 'AI 用途构成（输入导向→产出导向）',
    'AI_total_z': 'AI 使用强度（线性项）',
    'EFFSCIE_z': '科学自我效能',
    'SELFREG_z': '自我调节学习',
    'JOYSCIE_z': '科学兴趣',
    'COGACSC_z': '课堂认知激活教学',
    'DISCLISCI_z': '探究式科学教学',
    'SCTRAIN_z': '学校科学教师培训',
    'ICTWKDY_z': '上学日数字设备使用总量',
}
rows = []
for gname, (col, vlist) in GROUPS.items():
    for v in vlist:
        if v in c.index:
            rows.append((LAB[v], gname, col, c.loc[v, 'b'], c.loc[v, 'se'], c.loc[v, 'p']))
df = pd.DataFrame(rows, columns=['lab', 'grp', 'col', 'b', 'se', 'p'])
df['span'] = 2 * df.b            # −1SD → +1SD
df['span_se'] = 2 * df.se
df = df.reindex(df.span.abs().sort_values().index).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8.6, 4.9))
ypos = np.arange(len(df))
for i, row in df.iterrows():
    ax.barh(i, row.span, height=0.62, color=row.col, alpha=0.86, edgecolor='none', zorder=3)
    lo, hi = row.span - 1.96 * row.span_se, row.span + 1.96 * row.span_se
    ax.plot([lo, hi], [i, i], color='#3A3A38', lw=1.0, zorder=4, solid_capstyle='butt')
    ax.plot([lo, lo], [i - 0.13, i + 0.13], color='#3A3A38', lw=1.0, zorder=4)
    ax.plot([hi, hi], [i - 0.13, i + 0.13], color='#3A3A38', lw=1.0, zorder=4)
    off = 1.8 if row.span > 0 else -1.8
    anchor = hi if row.span > 0 else lo
    ax.text(anchor + off, i, f'{row.span:+.1f}', va='center',
            ha='left' if row.span > 0 else 'right', fontsize=9,
            fontweight='bold', color='#2C2C2A', zorder=5)

ax.axvline(0, color='#8A8A85', lw=0.9, zorder=2)
ax.set_yticks(ypos)
ax.set_yticklabels(df.lab, fontsize=9.2)
ax.set_xlabel('科学素养变化（分）：由 −1 SD 升至 +1 SD', fontsize=9.6)
ax.set_xlim(-45, 28)
ax.grid(axis='x', color='#E4E2DC', lw=0.7, zorder=0)
for s in ['top', 'right', 'left']:
    ax.spines[s].set_visible(False)
ax.spines['bottom'].set_color('#B4B2A9')
ax.tick_params(axis='y', length=0)

handles = [plt.Rectangle((0, 0), 1, 1, color=v[0], alpha=0.86) for v in GROUPS.values()]
ax.legend(handles, list(GROUPS.keys()), loc='upper center', frameon=False,
          fontsize=8.4, handlelength=1.0, ncol=5, columnspacing=1.4,
          bbox_to_anchor=(0.46, -0.13))
ax.set_title('可干预因素的效应量排序（全模型，控制家庭背景与学校特征后）',
             fontsize=10.8, pad=12, loc='left')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig5_levers.png'), dpi=200, bbox_inches='tight',
            facecolor='white')
plt.close()
print('fig5 written')
print(df[['lab', 'span', 'span_se', 'p']].round(2).to_string(index=False))
