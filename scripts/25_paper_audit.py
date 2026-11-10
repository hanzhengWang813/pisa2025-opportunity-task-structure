# -*- coding: utf-8 -*-
"""Number-by-number provenance and reproduction audit of the manuscript.

For every number printed in the paper this script reports the source script,
the source file, the matching key, the recomputed value, the difference from
the printed value, and a verdict.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pisa_engine import PISAEngine

from paths import BASE  # repo root (override with PISA_BASE)
OUT = os.path.join(BASE, 'outputs')

q = pd.read_parquet(os.path.join(BASE, 'data/qci_school.parquet'))
W = q['W_FSTUWT'].values
T = {}          # key -> recomputed value
NOTE = {}       # key -> supplementary note


def wmean(s):
    m = s.notna().values
    return float(np.average(s.values[m], weights=W[m]))


def wsd(s):
    m = s.notna().values
    mu = np.average(s.values[m], weights=W[m])
    return float(np.sqrt(np.average((s.values[m] - mu) ** 2, weights=W[m])))


def wshare(mask, base):
    m = base.notna().values & mask.notna().values
    return float(np.average(mask.values[m].astype(float), weights=W[m]))


# ============================================================ 1 Descriptive statistics
for col in ['PV1SCIE', 'ESCS', 'ICTWKDY', 'ICTHOME', 'AI_total', 'JOYSCIE',
            'EFFSCIE', 'SELFREG', 'COGACSC', 'TEACHSUP', 'DISCLISCI']:
    T[f'd_{col}_mean'] = wmean(q[col])
    T[f'd_{col}_sd'] = wsd(q[col])
    T[f'd_{col}_miss'] = float(q[col].isna().mean() * 100)

T['d_female'] = wshare((q['ST004D01T'] == 1).astype(float).where(q['ST004D01T'].notna()),
                       q['ST004D01T'])
T['d_repeat'] = wshare((q['REPEAT'] > 0).astype(float).where(q['REPEAT'].notna()), q['REPEAT'])
T['n_stu'] = len(q)
T['n_sch'] = int(q['CNTSCHID'].nunique())

# Share answering "every day" on each of the four uses
for c in ['ST438Q01DA', 'ST438Q02DA', 'ST438Q03DA', 'ST438Q04DA']:
    s = q[c].where(q[c] < 95)
    T[f'd_daily_{c}'] = wshare((s == 5).astype(float).where(s.notna()), s)

# Full-use saturation
IT = ['ST438Q01DA', 'ST438Q02DA', 'ST438Q03DA', 'ST438Q04DA']
ai = q[IT].where(q[IT] < 95)
q['_daily'] = (q['AI_total'].round() >= 5).astype(float)   # consistent with script 21
q['_sat'] = (ai.eq(5).sum(axis=1) == 4).astype(float)     # all four items at the top category
daily_n = int((q['_daily'] == 1).sum())
sat_n = int((q['_sat'] == 1).sum())
T['sat_pct'] = sat_n / daily_n * 100
T['sat_n'] = sat_n
T['sat_den'] = daily_n

# School-level construct descriptives (from school_constructs.csv, grade A)
sc = pd.read_csv(os.path.join(OUT, 'school_constructs.csv'))
for _, r in sc.iterrows():
    T[f'sc_{r.construct}_mean'] = r['mean']
    T[f'sc_{r.construct}_sd'] = r['sd']
T['sc_DIGCOND_miss'] = (374 - sc.loc[sc.construct == 'A_DIGCOND', 'n'].iloc[0]) / 374 * 100
T['sc_RULES_miss'] = (374 - sc.loc[sc.construct == 'D_RULES', 'n'].iloc[0]) / 374 * 100
T['sc_BAN_miss'] = (374 - sc.loc[sc.construct == 'D_BAN', 'n'].iloc[0]) / 374 * 100

# School size
g = q.groupby('CNTSCHID').size()
T['sch_min'], T['sch_med'], T['sch_max'] = int(g.min()), int(g.median()), int(g.max())
T['sch_le5'] = int((g <= 5).sum())

# ============================================================ 2 Scale reliability
X = q[IT].where(q[IT] < 95).dropna()
k = X.shape[1]
alpha = k / (k - 1) * (1 - X.var().sum() / X.sum(axis=1).var())
T['ai_alpha'] = float(alpha)
R = X.corr()
T['ai_r_min'] = float(R.values[np.triu_indices(k, 1)].min())
T['ai_r_max'] = float(R.values[np.triu_indices(k, 1)].max())

import semopy
_desc = 'F =~ ' + ' + '.join(IT)
_mod = semopy.Model(_desc)
_mod.fit(X)
_ins = _mod.inspect()
_lam = _ins[(_ins.op == '~') & (_ins.lval.isin(IT))][['lval', 'Estimate']].copy()
_lam['Estimate'] = pd.to_numeric(_lam['Estimate'])
_varF = float(_ins[(_ins.op == '~~') & (_ins.lval == 'F')]['Estimate'].iloc[0])
_sdF = np.sqrt(_varF)
_sd_item = {c: float(X[c].std(ddof=0)) for c in IT}
lam = np.array([abs(r.Estimate) * _sdF / _sd_item[r.lval] for r in _lam.itertuples()])
T['ai_CR'] = float(lam.sum() ** 2 / (lam.sum() ** 2 + np.sum(1 - lam ** 2)))
T['ai_AVE'] = float(np.mean(lam ** 2))
T['ai_lam_min'] = float(lam.min())
T['ai_lam_max'] = float(lam.max())

# School-level construct reliability: intraclass correlation (ICC1) of school means and school-level reliability λ
COMP = {'COGACSC': '认知激活', 'DISCLISCI': '纪律氛围', 'SELFREG': '自我调节', 'JOYSCIE': '科学兴趣'}
rel_rows = []
for c in COMP:
    d = q[['CNTSCHID', c]].dropna()
    grp = [v[c].values for _, v in d.groupby('CNTSCHID') if len(v) >= 2]
    nbar = np.mean([len(x) for x in grp])
    grand = d[c].mean()
    msb = sum(len(x) * (x.mean() - grand) ** 2 for x in grp) / (len(grp) - 1)
    msw = sum(((x - x.mean()) ** 2).sum() for x in grp) / (len(d) - len(grp))
    icc1 = (msb - msw) / (msb + (nbar - 1) * msw)
    rel = (msb - msw) / msb          # reliability of the school mean, λ
    rel_rows.append(dict(component=c, n_school=len(grp), nbar=round(nbar, 1),
                         icc1=round(icc1, 4), reliability=round(rel, 4)))
    T[f'rel_{c}'] = float(rel)
pd.DataFrame(rel_rows).to_csv(os.path.join(OUT, 'school_mean_reliability.csv'),
                              index=False, encoding='utf-8-sig')
T['rel_min'] = min(T[f'rel_{c}'] for c in COMP)
_sm = q.groupby('CNTSCHID')[list(COMP)].mean().dropna()
_ks = _sm.shape[1]
T['clim_alpha_mean'] = float(_ks / (_ks - 1) * (1 - _sm.var().sum() / _sm.sum(axis=1).var()))

# ============================================================ 3 Correlations
s_lv = q.groupby('CNTSCHID').agg(A=('A_DIGCOND', 'mean'), B=('B_TCHCAP', 'mean'),
                                 AI=('AI_total', 'mean'), ESCS=('SCH_MEAN_ESCS', 'first'),
                                 COG=('COGACSC', 'mean')).dropna()
T['r_AB'] = float(s_lv[['A', 'B']].corr().iloc[0, 1])
T['r_AI_ESCS'] = float(s_lv[['AI', 'ESCS']].corr().iloc[0, 1])
T['r_AI_COG'] = float(s_lv[['AI', 'COG']].corr().iloc[0, 1])

# ============================================================ 4 Two-level model ladder
ld = pd.read_csv(os.path.join(OUT, 'mlm_ladder.csv'))
var = ld[ld.term == '__var__'].set_index('model')
for m in var.index:
    tag = m.split(' ')[0]
    T[f'm_{tag}_tau00'] = var.loc[m, 'tau00']
    T[f'm_{tag}_sigma2'] = var.loc[m, 'sigma2']
    T[f'm_{tag}_icc'] = var.loc[m, 'icc']
    if pd.notna(var.loc[m, 'tau11']):
        T[f'm_{tag}_tau11'] = var.loc[m, 'tau11']
T['tau11_drop_pct'] = (T['m_M3_tau11'] - T['m_M5_tau11']) / T['m_M3_tau11'] * 100
T['slope_sd'] = float(np.sqrt(T['m_M3_tau11']))
slope_b = float(ld[(ld.model == 'M3 +AI随机斜率') & (ld.term == 'AIz')]['b'].iloc[0])
T['slope_mean'] = slope_b
T['slope_lo_1sd'] = slope_b - T['slope_sd']
T['slope_hi_1sd'] = slope_b + T['slope_sd']
T['slope_lo_95'] = slope_b - 1.96 * T['slope_sd']
T['slope_hi_95'] = slope_b + 1.96 * T['slope_sd']

lrt = pd.read_csv(os.path.join(OUT, 'mlm_lrt_slope.csv'))
T['lrt_min'] = float(lrt['LR'].min())
T['lrt_max'] = float(lrt['LR'].max())
T['lrt_pmax'] = float(lrt['p'].max())

# Variance change in the climate-only model (mlm_variance.csv, alternative specification)
mv = pd.read_csv(os.path.join(OUT, 'mlm_variance.csv')).set_index('model')
T['mv_tau00_rand'] = float(mv.loc['+AI 随机斜率', 'tau00'])
T['mv_tau00_clim'] = float(mv.loc['+学校氛围', 'tau00'])
T['mv_icc_rand'] = float(mv.loc['+AI 随机斜率', 'icc'])
T['mv_icc_clim'] = float(mv.loc['+学校氛围', 'icc'])
T['mv_drop_pct'] = (T['mv_tau00_rand'] - T['mv_tau00_clim']) / T['mv_tau00_rand'] * 100

# ============================================================ 5 School-level main effects
sm = pd.read_csv(os.path.join(OUT, 'school_main.csv'))
for term, key in [('SCH_MEAN_ESCS_z', 'escs'), ('SCH_CLIM_z', 'clim'),
                  ('C_DIGPURP_z', 'purp'), ('D_RULES_z', 'rules'),
                  ('F_SRL_z', 'srl'), ('H_MONITOR_z', 'mon'),
                  ('I_ENABLE_z', 'enab'), ('G_DIGORG_z', 'org'),
                  ('A_DIGCOND_z', 'digc'), ('B_TCHCAP_z', 'tchcap')]:
    r = sm[sm.term == term]
    if len(r):
        r = r.iloc[0]
        T[f's_{key}_b'], T[f's_{key}_se'], T[f's_{key}_p'] = r.b, r.se, r.p

# Digital infrastructure (7-item composite) — grade B: previously existed only in figure-script memory
r2 = pd.read_csv(os.path.join(OUT, 'school_r2.csv'))
T['dr2_school'] = float(r2.loc[r2.outcome == 'SCIE', 'dR2_school'].iloc[0])

A7 = ['SC155Q01HA', 'SC155Q02HA', 'SC155Q03HA', 'SC155Q04HA', 'SC155Q05HA',
      'SC155Q06HA', 'SC155Q07HA']
qq = q.copy()
qq['AIz'] = (qq['AI_total'] - qq['AI_total'].mean()) / qq['AI_total'].std()
for c in A7:
    qq[c] = qq[c].where(qq[c] < 5)
qq['DIGBASE'] = qq[A7].mean(axis=1)
qq['DIGBASE_z'] = (qq['DIGBASE'] - qq['DIGBASE'].mean()) / qq['DIGBASE'].std()
qq['female'] = (qq['ST004D01T'] == 1).astype(float).where(qq['ST004D01T'].notna())
qq['repeat'] = (qq['REPEAT'] > 0).astype(float).where(qq['REPEAT'].notna())
qq['escs'] = qq['ESCS']
for c in ['ESCS', 'GRADE', 'JOYSCIE', 'EFFSCIE', 'SELFREG', 'COGACSC', 'TEACHSUP',
          'DISCLISCI', 'ICTWKDY', 'ICTHOME']:
    qq[c + '_z'] = (qq[c] - qq[c].mean()) / qq[c].std()
qq['grade_z'] = qq['GRADE_z']
for c in ['A_DIGCOND', 'B_TCHCAP', 'G_DIGORG']:
    qq[c + '_z'] = (qq[c] - qq[c].mean()) / qq[c].std()
qk = qq[A7].ge(0).all(axis=1)
k7 = A7
X7 = qq.loc[qk, k7]
a7 = 7 / 6 * (1 - X7.var().sum() / X7.sum(axis=1).var())
T['digbase_alpha'] = float(a7)

STU2 = ['female', 'escs', 'repeat', 'grade_z', 'JOYSCIE_z', 'EFFSCIE_z', 'SELFREG_z',
        'COGACSC_z', 'TEACHSUP_z', 'DISCLISCI_z', 'ICTWKDY_z', 'ICTHOME_z']
s4 = qq.dropna(subset=STU2 + ['AIz', 'W_FSTUWT', 'DIGBASE_z']).copy()
dgb = PISAEngine(s4, 'PV', 'SCIE').fit(STU2 + ['AIz', 'DIGBASE_z'], 'DIG')
dgb.to_csv(os.path.join(OUT, 'fig9_coefs.csv'), index=False, encoding='utf-8-sig')
row = dgb[dgb.term == 'DIGBASE_z'].iloc[0]
T['s_base_b'], T['s_base_se'], T['s_base_p'] = row.b, row.se, row.p
T['n_digbase'] = len(s4)

# Suppression diagnostics (grade B: previously printed only)
s_sup = qq.dropna(subset=STU2 + ['AIz', 'W_FSTUWT'] + ['A_DIGCOND_z', 'B_TCHCAP_z']).copy()
sep1 = PISAEngine(s_sup, 'PV', 'SCIE').fit(STU2 + ['AIz', 'A_DIGCOND_z'], 'A_only')
sep2 = PISAEngine(s_sup, 'PV', 'SCIE').fit(STU2 + ['AIz', 'B_TCHCAP_z'], 'B_only')
both = PISAEngine(s_sup, 'PV', 'SCIE').fit(STU2 + ['AIz', 'A_DIGCOND_z', 'B_TCHCAP_z'], 'AB')
T['sup_A_sep'] = sep1[sep1.term == 'A_DIGCOND_z'].b.iloc[0]
T['sup_A_sep_p'] = sep1[sep1.term == 'A_DIGCOND_z'].p.iloc[0]
T['sup_B_sep'] = sep2[sep2.term == 'B_TCHCAP_z'].b.iloc[0]
T['sup_B_sep_p'] = sep2[sep2.term == 'B_TCHCAP_z'].p.iloc[0]
T['sup_A_both'] = both[both.term == 'A_DIGCOND_z'].b.iloc[0]
T['sup_A_both_p'] = both[both.term == 'A_DIGCOND_z'].p.iloc[0]
T['sup_B_both'] = both[both.term == 'B_TCHCAP_z'].b.iloc[0]
T['sup_B_both_p'] = both[both.term == 'B_TCHCAP_z'].p.iloc[0]
pd.concat([sep1, sep2, both]).to_csv(os.path.join(OUT, 'suppression_check.csv'),
                                     index=False, encoding='utf-8-sig')

s_org = qq.dropna(subset=STU2 + ['AIz', 'W_FSTUWT', 'G_DIGORG_z']).copy()
org1 = PISAEngine(s_org, 'PV', 'SCIE').fit(STU2 + ['AIz', 'G_DIGORG_z'], 'ORG_only')
T['org_sep'], T['org_sep_p'] = org1[ org1.term == 'G_DIGORG_z'].b.iloc[0], org1[org1.term == 'G_DIGORG_z'].p.iloc[0]

# ============================================================ 6 Moderator scan
NAME = {
    'A 数字条件': '数字条件（设备/带宽/软件）',
    'B 教师能力时间': '教师技术能力与备课时间',
    'C 数字使用取向': '数字资源使用取向',
    'D 规范共定': '数字使用规范（师生共定）',
    'F 自主学习优先': '自主学习教学优先度',
    'G 数字教学组织': '数字教学组织形态',
    'H 过程追踪(单题)': '用数字资源追踪学习过程',
    'I 赋能取向(单题)': '以数字资源促自主学习',
    'J 手机禁令': '校内手机禁令',
    'K 数字教学准备度': '学校数字教学准备度',
    'L 科学教师培训': '科学教师培训（校方报告）',
    'M 教育短缺': '教育资源短缺',
    'N 师资短缺': '师资短缺',
    'O 软件可用性': '教学软件可用性',
    'P 教学领导': '教学领导力',
    'Q 负面校风': '负面校风',
    'S 书面数字设备规定': '书面数字设备规定',
    'T 教师自定使用规则': '教师自定课堂使用规则',
    'U 师生共定规则': '师生共同商定规则',
    'V 网络素养课程': '网络素养课程/项目',
    'W 社交网络政策': '社交网络使用政策',
    'X 教师应对学生用AI': '教师应对学生用 AI 的准备',
    'Y 数字教学PD需求': '数字教学培训需求（校方）',
    'Z 自主学习教学支持': '自主学习教学支持',
    'AA 校均AI使用': '校均 AI 使用水平',
    'AB 校内AI离散度': '校内 AI 使用离散度',
    'AC 校均屏幕时间': '校均上学日屏幕时间',
    'AD 每天使用占比': '每天使用 AI 的学生占比',
    'AE 学校规模': '学校规模',
}
ms = pd.read_csv(os.path.join(OUT, 'moderator_scan.csv'))
ms = ms[(ms.outcome == 'SCIE') & (ms.moderator.isin(NAME))].copy()
ms['lab'] = ms.moderator.map(NAME)
T['mod_n'] = len(ms)
T['mod_bmin'] = float(ms.b_int.min())
T['mod_bmax'] = float(ms.b_int.max())
T['mod_pmin'] = float(ms.p_int.min())
T['mod_qmin'] = float(ms.p_fdr.min())
T['mod_neg'] = int((ms.b_int < 0).sum())
T['mod_se_min'] = float(ms.se_int.min())
T['mod_se_max'] = float(ms.se_int.max())
T['se_med'] = float(ms.se_int.median())
T['mde_med'] = 2.802 * T['se_med']
T['mde_max'] = 2.802 * float(ms.se_int.max())   # 80% power, two-sided .05
for lab, key in [('科学教师培训（校方报告）', 'tr'), ('校内手机禁令', 'ban'),
                 ('教师技术能力与备课时间', 'tch'), ('数字使用规范（师生共定）', 'rules'),
                 ('学校数字教学准备度', 'prep')]:
    r = ms[ms.lab == lab].iloc[0]
    T[f'mod_{key}_b'], T[f'mod_{key}_se'], T[f'mod_{key}_p'] = r.b_int, r.se_int, r.p_int
T['mod_rules_q'] = float(ms[ms.lab == '数字使用规范（师生共定）'].p_fdr.iloc[0])

# ============================================================ 7 Level identification
li = pd.read_csv(os.path.join(OUT, 'level_id.csv'))


def g(model, term):
    r = li[(li.model == model) & (li.term == term)].iloc[0]
    return r.b, r.se, r.p


for model, term, key, tag in [('SCIE|A 单层(强度)', 'AIz', 'a', 'int'),
                              ('SCIE|B Mundlak(强度)', 'AI_within', 'bw', 'int'),
                              ('SCIE|B Mundlak(强度)', 'AI_between_z', 'bb', 'int'),
                              ('SCIE|C 校内FE(强度)', 'AI_within', 'c', 'int'),
                              ('SCIE|A 单层(每天)', 'AI_daily', 'a', 'day'),
                              ('SCIE|B Mundlak(每天)', 'D_within', 'bw', 'day'),
                              ('SCIE|B Mundlak(每天)', 'D_between', 'bb', 'day'),
                              ('SCIE|C 校内FE(每天)', 'D_within', 'c', 'day')]:
    b_, s_, p_ = g(model, term)
    T['lv_%s_%s_b' % (key, tag)] = b_
    T['lv_%s_%s_se' % (key, tag)] = s_
    T['lv_%s_%s_p' % (key, tag)] = p_

for model, term, key in [('SCIE|D1 交互 单层', 'AIzxCLIM', 'd1'),
                         ('SCIE|D2 交互 Mundlak', 'AIxCLIM', 'd2'),
                         ('SCIE|D3 交互 校内FE', 'AIxCLIM', 'd3')]:
    b_, s_, p_ = g(model, term)
    T[f'eco_{key}_b'], T[f'eco_{key}_se'], T[f'eco_{key}_p'] = b_, s_, p_

# Four uses
uw = pd.read_csv(os.path.join(OUT, 'uses_within.csv'))
USE = {'ciz_ST438Q01DA': ('sum', '总结阅读材料'), 'ciz_ST438Q02DA': ('res', '初步研究新主题'),
       'ciz_ST438Q03DA': ('draft', '起草写作'), 'ciz_ST438Q04DA': ('help', '帮助学习')}
for term, (key, _) in USE.items():
    for model, tag in [('SCIE|E1 四用途 单层', 'single'), ('SCIE|E3 四用途 校内FE', 'fe')]:
        r = uw[(uw.model == model) & (uw.term == term)]
        if len(r):
            T[f'use_{key}_{tag}_b'] = r.b.iloc[0]
            T[f'use_{key}_{tag}_se'] = r.se.iloc[0]
            T[f'use_{key}_{tag}_p'] = r.p.iloc[0]
print('uses_within models:', uw.model.unique())

# Sample sizes
li_n = {}
for m in li.model.unique():
    li_n[m] = None
T['n_levelid'] = li.attrs.get('n', None)

# ============================================================ 8 Checklist and comparison
CHECKS = [
    # (location, paper wording, key, paper value, tolerance, grade, script, file, match key)
    ('摘要/三(一)', '学生数 12,385', 'n_stu', 12385, 0, 'A', '01_extract_qci.py', 'data/qci_school.parquet', 'len(df)'),
    ('摘要/三(一)', '学校数 374', 'n_sch', 374, 0, 'A', '01_extract_qci.py', 'data/qci_school.parquet', 'CNTSCHID nunique'),

    ('表1', '科学素养均值 597.12', 'd_PV1SCIE_mean', 597.12, .005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'PV1SCIE 加权均值'),
    ('表1', '科学素养标准差 87.53', 'd_PV1SCIE_sd', 87.53, .005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'PV1SCIE 加权标准差'),
    ('表1', '家庭 ESCS 均值 −0.52', 'd_ESCS_mean', -0.52, .005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'ESCS 加权均值'),
    ('表1', '家庭 ESCS 标准差 1.00', 'd_ESCS_sd', 1.00, .005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'ESCS 加权标准差'),
    ('表1', 'ESCS 缺失率 0.5', 'd_ESCS_miss', 0.5, .05, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'ESCS isna 占比'),
    ('表1', '设备使用均值 −1.46', 'd_ICTWKDY_mean', -1.46, .005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'ICTWKDY'),
    ('表1', '设备使用标准差 1.53', 'd_ICTWKDY_sd', 1.53, .005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'ICTWKDY'),
    ('表1', '家庭 ICT 资源均值 0.55', 'd_ICTHOME_mean', 0.55, .005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'ICTHOME'),
    ('表1', 'AI 使用强度均值 2.42', 'd_AI_total_mean', 2.42, .005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'AI_total'),
    ('表1', 'AI 使用强度标准差 1.06', 'd_AI_total_sd', 1.06, .005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'AI_total'),
    ('表1', '科学兴趣均值 0.60', 'd_JOYSCIE_mean', 0.60, .005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'JOYSCIE'),
    ('表1', '科学自我效能均值 0.43', 'd_EFFSCIE_mean', 0.43, .005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'EFFSCIE'),
    ('表1', '自我调节均值 0.31', 'd_SELFREG_mean', 0.31, .005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'SELFREG'),
    ('表1', '认知激活均值 0.67', 'd_COGACSC_mean', 0.67, .005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'COGACSC'),
    ('表1', '教师支持均值 0.62', 'd_TEACHSUP_mean', 0.62, .005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'TEACHSUP'),
    ('表1', '纪律氛围均值 0.80', 'd_DISCLISCI_mean', 0.80, .005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'DISCLISCI'),
    ('表1', '校均 ESCS −0.24', 'sc_SCH_MEAN_ESCS_mean', -0.24, .005, 'A', '13_school_constructs.py', 'school_constructs.csv', 'SCH_MEAN_ESCS'),
    ('表1', '校均 ESCS 标准差 0.66', 'sc_SCH_MEAN_ESCS_sd', 0.66, .005, 'A', '13_school_constructs.py', 'school_constructs.csv', 'SCH_MEAN_ESCS'),
    ('表1', '数字基础条件 3.22', 'sc_A_DIGCOND_mean', 3.22, .005, 'A', '13_school_constructs.py', 'school_constructs.csv', 'A_DIGCOND'),
    ('表1', '数字基础条件标准差 0.67', 'sc_A_DIGCOND_sd', 0.67, .005, 'A', '13_school_constructs.py', 'school_constructs.csv', 'A_DIGCOND'),
    ('表1', '数字基础条件缺失 0.5', 'sc_DIGCOND_miss', 0.5, .05, 'A', '13_school_constructs.py', 'school_constructs.csv', '(374-n)/374'),
    ('表1', '数字使用规范 0.89', 'sc_D_RULES_mean', 0.89, .005, 'A', '13_school_constructs.py', 'school_constructs.csv', 'D_RULES'),
    ('表1', '数字使用规范缺失 1.3', 'sc_RULES_miss', 1.3, .05, 'A', '13_school_constructs.py', 'school_constructs.csv', '(374-n)/374'),
    ('表1', '手机禁令 0.81', 'sc_D_BAN_mean', 0.81, .005, 'A', '13_school_constructs.py', 'school_constructs.csv', 'D_BAN'),
    ('表1', '手机禁令缺失 1.1', 'sc_BAN_miss', 1.1, .05, 'A', '13_school_constructs.py', 'school_constructs.csv', '(374-n)/374'),
    ('表1 注', '女性占 47.9%', 'd_female', 0.479, .0005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'ST004D01T==1 加权'),
    ('表1 注', '留级占 3.9%', 'd_repeat', 0.039, .0005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'REPEAT>0 加权'),

    ('四(二)', '每天·总结 3.3%', 'd_daily_ST438Q01DA', 0.033, .0005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'ST438Q01DA==5 加权'),
    ('四(二)', '每天·初步研究 3.1%', 'd_daily_ST438Q02DA', 0.031, .0005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'ST438Q02DA==5'),
    ('四(二)', '每天·起草 3.5%', 'd_daily_ST438Q03DA', 0.035, .0005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'ST438Q03DA==5'),
    ('四(二)', '每天·帮助学习 9.8%', 'd_daily_ST438Q04DA', 0.098, .0005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'ST438Q04DA==5'),

    ('三(二)', '量表 α = .876', 'ai_alpha', 0.876, .0005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'ST438Q01-04 Cronbach'),
    ('三(二)', '组合信度 .878', 'ai_CR', 0.878, .001, 'A', '25_paper_audit.py', 'data/qci_school.parquet', '单因子 CFA 标准化载荷'),
    ('三(二)', '平均方差抽取量 .643', 'ai_AVE', 0.643, .001, 'A', '25_paper_audit.py', 'data/qci_school.parquet', '单因子 CFA'),
    ('三(二)', '题间相关 .59–.74 下限', 'ai_r_min', 0.59, .006, 'A', '25_paper_audit.py', 'data/qci_school.parquet', '四题相关矩阵'),
    ('三(二)', '题间相关 .59–.74 上限', 'ai_r_max', 0.74, .006, 'A', '25_paper_audit.py', 'data/qci_school.parquet', '四题相关矩阵'),
    ('三(二)', '数字基础条件 α = .953', 'digbase_alpha', 0.953, .0005, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'SC155Q01-Q07 Cronbach'),
    ('三(二)', '氛围分量校均值信度下限 .64', 'rel_min', 0.64, .005, 'A', '25_paper_audit.py', 'school_mean_reliability.csv', '四分量最小 λ'),

    ('表2', 'M0 τ00 = 3157.0', 'm_M0_tau00', 3157.0, .05, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M0/__var__/tau00'),
    ('表2', 'M0 σ² = 4259.1', 'm_M0_sigma2', 4259.1, .05, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M0/__var__/sigma2'),
    ('表2', 'M0 ICC = 0.426', 'm_M0_icc', 0.426, .0005, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M0/__var__/icc'),
    ('表2', 'M1 τ00 = 2689.0', 'm_M1_tau00', 2689.0, .05, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M1/__var__/tau00'),
    ('表2', 'M1 σ² = 4144.2', 'm_M1_sigma2', 4144.2, .05, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M1/__var__/sigma2'),
    ('表2', 'M1 ICC = 0.394', 'm_M1_icc', 0.394, .0005, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M1/__var__/icc'),
    ('表2', 'M2 τ00 = 2197.8', 'm_M2_tau00', 2197.8, .05, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M2/__var__/tau00'),
    ('表2', 'M2 σ² = 3876.4', 'm_M2_sigma2', 3876.4, .05, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M2/__var__/sigma2'),
    ('表2', 'M2 ICC = 0.362', 'm_M2_icc', 0.362, .0005, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M2/__var__/icc'),
    ('表2', 'M3 τ00 = 2206.3', 'm_M3_tau00', 2206.3, .05, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M3/__var__/tau00'),
    ('表2', 'M3 σ² = 3832.1', 'm_M3_sigma2', 3832.1, .05, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M3/__var__/sigma2'),
    ('表2', 'M3 ICC = 0.365', 'm_M3_icc', 0.365, .0005, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M3/__var__/icc'),
    ('表2', 'M3 τ11 = 46.02', 'm_M3_tau11', 46.02, .005, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M3/__var__/tau11'),
    ('表2', 'M4 τ00 = 2094.5', 'm_M4_tau00', 2094.5, .05, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M4/__var__/tau00'),
    ('表2', 'M4 σ² = 3832.2', 'm_M4_sigma2', 3832.2, .05, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M4/__var__/sigma2'),
    ('表2', 'M4 ICC = 0.353', 'm_M4_icc', 0.353, .0005, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M4/__var__/icc'),
    ('表2', 'M4 τ11 = 46.01', 'm_M4_tau11', 46.01, .005, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M4/__var__/tau11'),
    ('表2', 'M5 τ00 = 2092.6', 'm_M5_tau00', 2092.6, .05, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M5/__var__/tau00'),
    ('表2', 'M5 σ² = 3832.4', 'm_M5_sigma2', 3832.4, .05, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M5/__var__/sigma2'),
    ('表2', 'M5 ICC = 0.353', 'm_M5_icc', 0.353, .0005, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M5/__var__/icc'),
    ('表2', 'M5 τ11 = 42.35', 'm_M5_tau11', 42.35, .005, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M5/__var__/tau11'),
    ('表2', 'M2 AI 斜率 −2.99', 'slope_mean', -2.99, .005, 'A', '14_mlm_ladder.py', 'mlm_ladder.csv', 'M3/term=AIz/b'),
    ('四(三)', '斜率标准差 6.8', 'slope_sd', 6.8, .05, 'B', '25_paper_audit.py', 'mlm_ladder.csv', 'sqrt(τ11=46.02)'),
    ('四(三)', '斜率下降 8.0%', 'tau11_drop_pct', 8.0, .05, 'A', '25_paper_audit.py', 'mlm_ladder.csv', '(46.02−42.35)/46.02'),
    ('四(三)', 'LRT 最小 22.77', 'lrt_min', 22.77, .005, 'A', '14_mlm_ladder.py', 'mlm_lrt_slope.csv', 'min(LR)'),
    ('四(三)', 'LRT 最大 46.37', 'lrt_max', 46.37, .005, 'A', '14_mlm_ladder.py', 'mlm_lrt_slope.csv', 'max(LR)'),

    ('四(二)', '校均 ESCS 26.82', 's_escs_b', 26.82, .005, 'A', '15_school_models.py', 'school_main.csv', 'MAIN/SCH_MEAN_ESCS_z'),
    ('四(二)', '校均 ESCS SE 2.72', 's_escs_se', 2.72, .005, 'A', '15_school_models.py', 'school_main.csv', '同上'),
    ('四(二)', '学术氛围 13.05', 's_clim_b', 13.05, .005, 'A', '15_school_models.py', 'school_main.csv', 'MAIN/SCH_CLIM_z'),
    ('四(二)', '学术氛围 SE 3.33', 's_clim_se', 3.33, .005, 'A', '15_school_models.py', 'school_main.csv', '同上'),
    ('表3', '数字基础条件 3.62', 's_base_b', 3.62, .005, 'B', '23_figures_fix.py', 'fig9_coefs.csv', 'DIG/DIGBASE_z'),
    ('表3', '数字基础条件 SE 2.57', 's_base_se', 2.57, .005, 'B', '23_figures_fix.py', 'fig9_coefs.csv', '同上'),
    ('表3', '数字基础条件 p=.159', 's_base_p', 0.159, .0005, 'B', '23_figures_fix.py', 'fig9_coefs.csv', '同上'),
    ('表3', '使用取向 0.99', 's_purp_b', 0.99, .005, 'A', '15_school_models.py', 'school_main.csv', 'MAIN/C_DIGPURP_z'),
    ('表3', '使用取向 SE 6.19', 's_purp_se', 6.19, .005, 'A', '15_school_models.py', 'school_main.csv', '同上'),
    ('表3', '使用取向 p=.873', 's_purp_p', 0.873, .0005, 'A', '15_school_models.py', 'school_main.csv', '同上'),
    ('表3', '使用规范 1.02', 's_rules_b', 1.02, .005, 'A', '15_school_models.py', 'school_main.csv', 'MAIN/D_RULES_z'),
    ('表3', '使用规范 SE 2.34', 's_rules_se', 2.34, .005, 'A', '15_school_models.py', 'school_main.csv', '同上'),
    ('表3', '使用规范 p=.664', 's_rules_p', 0.664, .0005, 'A', '15_school_models.py', 'school_main.csv', '同上'),
    ('表3', '自主学习优先度 −0.61', 's_srl_b', -0.61, .005, 'A', '15_school_models.py', 'school_main.csv', 'MAIN/F_SRL_z'),
    ('表3', '自主学习优先度 SE 2.35', 's_srl_se', 2.35, .005, 'A', '15_school_models.py', 'school_main.csv', '同上'),
    ('表3', '自主学习优先度 p=.794', 's_srl_p', 0.794, .0005, 'A', '15_school_models.py', 'school_main.csv', '同上'),
    ('表3', '过程追踪 −0.06', 's_mon_b', -0.06, .005, 'A', '15_school_models.py', 'school_main.csv', 'MAIN/H_MONITOR_z'),
    ('表3', '过程追踪 SE 2.31', 's_mon_se', 2.31, .005, 'A', '15_school_models.py', 'school_main.csv', '同上'),
    ('表3', '过程追踪 p=.978', 's_mon_p', 0.978, .0005, 'A', '15_school_models.py', 'school_main.csv', '同上'),
    ('表3', '促自主学习 0.58', 's_enab_b', 0.58, .005, 'A', '15_school_models.py', 'school_main.csv', 'MAIN/I_ENABLE_z'),
    ('表3', '促自主学习 SE 6.03', 's_enab_se', 6.03, .005, 'A', '15_school_models.py', 'school_main.csv', '同上'),
    ('表3', '促自主学习 p=.923', 's_enab_p', 0.923, .0005, 'A', '15_school_models.py', 'school_main.csv', '同上'),
    ('四(二)', '数字教学组织形态 −6.07', 's_org_b', -6.07, .005, 'A', '15_school_models.py', 'school_main.csv', 'MAIN/G_DIGORG_z'),
    ('四(二)', '数字教学组织形态 p=.015', 's_org_p', 0.015, .0005, 'A', '15_school_models.py', 'school_main.csv', '同上'),
    ('四(二)', '数字条件/教师能力 r = .80', 'r_AB', 0.80, .005, 'B', '25_paper_audit.py', 'data/qci_school.parquet', '校层 A_DIGCOND~B_TCHCAP'),
    ('四(二)', '单入模数字条件 4.8', 'sup_A_sep', 4.8, .05, 'B', '25_paper_audit.py', 'suppression_check.csv', 'A_only/A_DIGCOND_z'),
    ('四(二)', '单入模数字条件 p=.057', 'sup_A_sep_p', 0.057, .0005, 'B', '25_paper_audit.py', 'suppression_check.csv', '同上'),
    ('四(二)', '单入模教师能力 −0.2', 'sup_B_sep', -0.2, .05, 'B', '25_paper_audit.py', 'suppression_check.csv', 'B_only/B_TCHCAP_z'),
    ('四(二)', '同时入模数字条件 13.2', 'sup_A_both', 13.2, .05, 'B', '25_paper_audit.py', 'suppression_check.csv', 'AB/A_DIGCOND_z'),
    ('四(二)', '同时入模数字条件 p<.001', 'sup_A_both_p', 0.001, .001, 'B', '25_paper_audit.py', 'suppression_check.csv', '同上'),
    ('四(二)', '同时入模教师能力 −10.6', 'sup_B_both', -10.65, .05, 'B', '25_paper_audit.py', 'suppression_check.csv', 'AB/B_TCHCAP_z'),
    ('四(二)', '同时入模教师能力 p=.003', 'sup_B_both_p', 0.003, .0005, 'B', '25_paper_audit.py', 'suppression_check.csv', '同上'),
    ('四(二)', '组织形态单入模 −3.35', 'org_sep', -3.35, .005, 'B', '25_paper_audit.py', 'suppression_check.csv', 'ORG_only/G_DIGORG_z'),
    ('四(二)', '组织形态单入模 p=.128', 'org_sep_p', 0.128, .0005, 'B', '25_paper_audit.py', 'suppression_check.csv', '同上'),
    ('四(二)', '氛围单变量降 26.5%', 'mv_drop_pct', 26.5, .05, 'A', '22_figures_school.py', 'mlm_variance.csv', '(2173.23−1597.48)/2173.23'),
    ('四(二)', '氛围 ICC 0.361→0.293', 'mv_icc_rand', 0.361, .0005, 'A', '22_figures_school.py', 'mlm_variance.csv', '+AI 随机斜率/icc'),
    ('四(二)', '氛围 ICC 0.361→0.293（后）', 'mv_icc_clim', 0.293, .0005, 'A', '22_figures_school.py', 'mlm_variance.csv', '+学校氛围/icc'),

    ('四(四)', '候选数 29', 'mod_n', 29, 0, 'A', '16_moderator_scan.py', 'moderator_scan.csv', 'outcome=SCIE 且属 29 集'),
    ('四(四)', '交互系数下限 −2.34', 'mod_bmin', -2.34, .005, 'A', '16_moderator_scan.py', 'moderator_scan.csv', 'min(b_int)'),
    ('四(四)', '交互系数上限 2.12', 'mod_bmax', 2.12, .005, 'A', '16_moderator_scan.py', 'moderator_scan.csv', 'max(b_int)'),
    ('四(四)', '最小 p=.018', 'mod_pmin', 0.018, .0005, 'A', '16_moderator_scan.py', 'moderator_scan.csv', 'min(p_int)'),
    ('四(四)', '最小 FDR q=.275', 'mod_qmin', 0.275, .0005, 'A', '16_moderator_scan.py', 'moderator_scan.csv', 'min(p_fdr)'),
    ('四(四)', '负向交互 17 个', 'mod_neg', 17, 0, 'A', '16_moderator_scan.py', 'moderator_scan.csv', 'count(b_int<0)'),
    ('四(四)', '交互 SE 下限 0.88', 'mod_se_min', 0.88, .005, 'A', '16_moderator_scan.py', 'moderator_scan.csv', 'min(se_int)'),
    ('四(四)', '交互 SE 上限 1.88', 'mod_se_max', 1.877, .005, 'A', '16_moderator_scan.py', 'moderator_scan.csv', 'max(se_int)'),
    ('表4', '科学教师培训 −2.34', 'mod_tr_b', -2.34, .005, 'A', '16_moderator_scan.py', 'moderator_scan.csv', 'L 科学教师培训'),
    ('表4', '科学教师培训 SE 0.99', 'mod_tr_se', 0.99, .005, 'A', '16_moderator_scan.py', 'moderator_scan.csv', '同上'),
    ('表4', '科学教师培训 p=.018', 'mod_tr_p', 0.018, .0005, 'A', '16_moderator_scan.py', 'moderator_scan.csv', '同上'),
    ('表4', '手机禁令 −1.56', 'mod_ban_b', -1.56, .005, 'A', '16_moderator_scan.py', 'moderator_scan.csv', 'J 手机禁令'),
    ('表4', '手机禁令 p=.136', 'mod_ban_p', 0.136, .0005, 'A', '16_moderator_scan.py', 'moderator_scan.csv', '同上'),
    ('四(四)', '教师能力时间 1.07', 'mod_tch_b', 1.07, .005, 'A', '16_moderator_scan.py', 'moderator_scan.csv', 'B 教师能力时间'),
    ('四(四)', '教师能力时间 p=.315', 'mod_tch_p', 0.315, .0005, 'A', '16_moderator_scan.py', 'moderator_scan.csv', '同上'),
    ('四(四)', '数字条件 0.81', 'mod_prep_b', 0.81, .005, 'A', '16_moderator_scan.py', 'moderator_scan.csv', 'MAINPREP'),
    ('四(四)', '数字条件 p=.467', 'mod_prep_p', 0.467, .0005, 'A', '16_moderator_scan.py', 'moderator_scan.csv', '同上'),
    ('四(四)', '规范共定 −0.60', 'mod_rules_b', -0.60, .005, 'A', '16_moderator_scan.py', 'moderator_scan.csv', 'D 规范共定'),
    ('四(四)', '规范共定 p=.686', 'mod_rules_p', 0.686, .0005, 'A', '16_moderator_scan.py', 'moderator_scan.csv', '同上'),

    ('表5', '单层强度 −3.94', 'lv_a_int_b', -3.94, .005, 'A', '21_level_id_correct.py', 'level_id.csv', 'A 单层(强度)/AIz'),
    ('表5', '单层强度 SE 1.12', 'lv_a_int_se', 1.12, .005, 'A', '21_level_id_correct.py', 'level_id.csv', '同上'),
    ('表5', '单层每天 −50.4', 'lv_a_day_b', -50.4, .05, 'A', '21_level_id_correct.py', 'level_id.csv', 'A 单层(每天)/AI_daily'),
    ('表5', '单层每天 SE 5.6', 'lv_a_day_se', 5.6, .05, 'A', '21_level_id_correct.py', 'level_id.csv', '同上'),
    ('表5', 'Mundlak 校内强度 −4.95', 'lv_bw_int_b', -4.95, .005, 'A', '21_level_id_correct.py', 'level_id.csv', 'B Mundlak(强度)/AI_within'),
    ('表5', 'Mundlak 校内强度 SE 1.02', 'lv_bw_int_se', 1.02, .005, 'A', '21_level_id_correct.py', 'level_id.csv', '同上'),
    ('表5', 'Mundlak 校内每天 −46.1', 'lv_bw_day_b', -46.1, .05, 'A', '21_level_id_correct.py', 'level_id.csv', 'B Mundlak(每天)/D_within'),
    ('表5', 'Mundlak 校内每天 SE 5.7', 'lv_bw_day_se', 5.7, .05, 'A', '21_level_id_correct.py', 'level_id.csv', '同上'),
    ('表5', 'Mundlak 校间强度 3.57', 'lv_bb_int_b', 3.57, .005, 'A', '21_level_id_correct.py', 'level_id.csv', 'B Mundlak(强度)/AI_between_z'),
    ('表5', 'Mundlak 校间强度 SE 2.76', 'lv_bb_int_se', 2.76, .005, 'A', '21_level_id_correct.py', 'level_id.csv', '同上'),
    ('表5', 'Mundlak 校间每天 −147.1', 'lv_bb_day_b', -147.1, .05, 'A', '21_level_id_correct.py', 'level_id.csv', 'B Mundlak(每天)/D_between'),
    ('表5', 'Mundlak 校间每天 SE 72.5', 'lv_bb_day_se', 72.5, .05, 'A', '21_level_id_correct.py', 'level_id.csv', '同上'),
    ('表5', '校内FE强度 −2.89', 'lv_c_int_b', -2.89, .005, 'A', '21_level_id_correct.py', 'level_id.csv', 'C 校内FE(强度)/AI_within'),
    ('表5', '校内FE强度 SE 0.92', 'lv_c_int_se', 0.92, .005, 'A', '21_level_id_correct.py', 'level_id.csv', '同上'),
    ('表5', '校内FE每天 −34.4', 'lv_c_day_b', -34.4, .05, 'A', '21_level_id_correct.py', 'level_id.csv', 'C 校内FE(每天)/D_within'),
    ('表5', '校内FE每天 SE 5.3', 'lv_c_day_se', 5.3, .05, 'A', '21_level_id_correct.py', 'level_id.csv', '同上'),

    ('表6', '总结 单层 −10.75', 'use_sum_single_b', -10.75, .005, 'A', '21_level_id_correct.py', 'uses_within.csv', 'E1/ciz_ST438Q01DA'),
    ('表6', '总结 单层 SE 1.33', 'use_sum_single_se', 1.33, .005, 'A', '21_level_id_correct.py', 'uses_within.csv', '同上'),
    ('表6', '总结 校内 −10.30', 'use_sum_fe_b', -10.30, .005, 'A', '21_level_id_correct.py', 'uses_within.csv', 'E3/ciz_ST438Q01DA'),
    ('表6', '总结 校内 SE 1.31', 'use_sum_fe_se', 1.31, .005, 'A', '21_level_id_correct.py', 'uses_within.csv', '同上'),
    ('表6', '初步研究 单层 −3.99', 'use_res_single_b', -3.99, .005, 'A', '21_level_id_correct.py', 'uses_within.csv', 'E1/ciz_ST438Q02DA'),
    ('表6', '初步研究 校内 −2.71', 'use_res_fe_b', -2.71, .005, 'A', '21_level_id_correct.py', 'uses_within.csv', 'E3/ciz_ST438Q02DA'),
    ('表6', '起草 单层 3.38', 'use_draft_single_b', 3.38, .005, 'A', '21_level_id_correct.py', 'uses_within.csv', 'E1/ciz_ST438Q03DA'),
    ('表6', '起草 校内 3.96', 'use_draft_fe_b', 3.96, .005, 'A', '21_level_id_correct.py', 'uses_within.csv', 'E3/ciz_ST438Q03DA'),
    ('表6', '帮助学习 单层 5.59', 'use_help_single_b', 5.59, .005, 'A', '21_level_id_correct.py', 'uses_within.csv', 'E1/ciz_ST438Q04DA'),
    ('表6', '帮助学习 校内 5.92', 'use_help_fe_b', 5.92, .005, 'A', '21_level_id_correct.py', 'uses_within.csv', 'E3/ciz_ST438Q04DA'),

    ('表7', '单层交互 −2.50', 'eco_d1_b', -2.50, .005, 'A', '21_level_id_correct.py', 'level_id.csv', 'D1 交互 单层/AIzxCLIM'),
    ('表7', '单层交互 SE 1.02', 'eco_d1_se', 1.02, .005, 'A', '21_level_id_correct.py', 'level_id.csv', '同上'),
    ('表7', '单层交互 p=.015', 'eco_d1_p', 0.015, .0005, 'A', '21_level_id_correct.py', 'level_id.csv', '同上'),
    ('表7', 'Mundlak 交互 −0.13', 'eco_d2_b', -0.13, .005, 'A', '21_level_id_correct.py', 'level_id.csv', 'D2 交互 Mundlak/AIxCLIM'),
    ('表7', 'Mundlak 交互 SE 0.79', 'eco_d2_se', 0.79, .005, 'A', '21_level_id_correct.py', 'level_id.csv', '同上'),
    ('表7', 'Mundlak 交互 p=.867', 'eco_d2_p', 0.867, .0005, 'A', '21_level_id_correct.py', 'level_id.csv', '同上'),
    ('表7', 'FE 交互 −0.19', 'eco_d3_b', -0.19, .005, 'A', '21_level_id_correct.py', 'level_id.csv', 'D3 交互 校内FE/AIxCLIM'),
    ('表7', 'FE 交互 SE 0.78', 'eco_d3_se', 0.78, .005, 'A', '21_level_id_correct.py', 'level_id.csv', '同上'),
    ('表7', 'FE 交互 p=.807', 'eco_d3_p', 0.807, .0005, 'A', '21_level_id_correct.py', 'level_id.csv', '同上'),

    ('四(四)', '全用途饱和 86.6%', 'sat_pct', 86.6, .05, 'A', '25_paper_audit.py', 'data/qci_school.parquet', '四题全=5 / 每天组'),
    ('四(四)', '饱和分子 343', 'sat_n', 343, 0, 'A', '25_paper_audit.py', 'data/qci_school.parquet', '同上'),
    ('四(四)', '饱和分母 396', 'sat_den', 396, 0, 'A', '25_paper_audit.py', 'data/qci_school.parquet', '同上'),
    ('四(六)', '校均AI~校均ESCS r=.48', 'r_AI_ESCS', 0.48, .005, 'B', '25_paper_audit.py', 'data/qci_school.parquet', '校层相关'),
    ('四(六)', '校均AI~校均认知激活 r=.48', 'r_AI_COG', 0.484, .005, 'B', '25_paper_audit.py', 'data/qci_school.parquet', '校层相关'),
    ('五(二)', '89.4% 有规范', 'sc_D_RULES_mean', 0.894, .0005, 'A', '13_school_constructs.py', 'school_constructs.csv', 'D_RULES mean'),
    ('五(二)', '81.4% 有手机禁令', 'sc_D_BAN_mean', 0.814, .0005, 'A', '13_school_constructs.py', 'school_constructs.csv', 'D_BAN mean'),
    ('六(三)', '4 所学校 ≤5 人', 'sch_le5', 4, 0, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'groupby size<=5'),
    ('六(三)', '最小校规模 1', 'sch_min', 1, 0, 'A', '25_paper_audit.py', 'data/qci_school.parquet', 'min size'),
    ('三(三)', '交互 SE 中位数 1.17', 'se_med', 1.17, .005, 'B', '25_paper_audit.py', 'moderator_scan.csv', 'median(se_int)'),
    ('三(三)', '中位检验力下可检出 3.3 分', 'mde_med', 3.3, .05, 'B', '25_paper_audit.py', 'moderator_scan.csv', '2.802*median(se_int)'),
    ('三(三)', '最弱候选检出 5.3 分', 'mde_max', 5.3, .05, 'B', '25_paper_audit.py', 'moderator_scan.csv', '2.802*max(se_int)'),
    ('三(二)', '氛围合成后校均值 α=.864', 'clim_alpha_mean', 0.864, .0005, 'B', '25_paper_audit.py', 'data/qci_school.parquet', '四校均值 Cronbach'),
]

rows = []
for pos, desc, key, paper, tol, grade, script, fname, keydesc in CHECKS:
    if key not in T or T[key] is None:
        rows.append(dict(位置=pos, 论文表述=desc, 论文值=paper, 复算值=None,
                         差值=None, 判定='缺少复算', 等级=grade, 来源脚本=script,
                         来源文件=fname, 匹配键=keydesc))
        continue
    val = float(T[key])
    diff = val - paper
    ok = abs(diff) <= tol + 1e-12
    rows.append(dict(位置=pos, 论文表述=desc, 论文值=paper, 复算值=round(val, 6),
                     差值=round(diff, 6), 判定='一致' if ok else '偏差',
                     等级=grade, 来源脚本=script, 来源文件=fname, 匹配键=keydesc))

aud = pd.DataFrame(rows)
aud.to_csv(os.path.join(OUT, 'paper_number_audit.csv'), index=False, encoding='utf-8-sig')

bad = aud[aud.判定 != '一致']
print('checks:', len(aud), '| match:', (aud.判定 == '一致').sum(),
      '| deviation:', (aud.判定 == '偏差').sum(), '| missing:', (aud.判定 == '缺少复算').sum())
if len(bad):
    print('\n=== Items that need correction ===')
    print(bad[['位置', '论文表述', '论文值', '复算值', '差值', '来源文件']].to_string(index=False))

print('\n=== Key recomputed values (reference) ===')
for k in ['slope_sd', 'slope_mean', 'slope_lo_95', 'slope_hi_95', 'slope_lo_1sd', 'slope_hi_1sd',
          'mde', 'rel_min', 'ai_alpha', 'ai_CR', 'ai_AVE', 'digbase_alpha', 'n_digbase',
          'r_AB', 'r_AI_ESCS', 'r_AI_COG', 'sat_pct', 'mod_neg', 'mod_se_min', 'mod_se_max']:
    print(f'  {k:16s} = {T[k]:.4f}' if isinstance(T.get(k), float) else f'  {k:16s} = {T.get(k)}')
