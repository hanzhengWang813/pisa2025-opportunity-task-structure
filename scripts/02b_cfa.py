# -*- coding: utf-8 -*-
"""02b AI 使用结构：单因子 vs 双因子 CFA 比较"""
import pandas as pd, numpy as np, semopy, os, warnings
warnings.filterwarnings('ignore')
from paths import BASE  # repo root (override with PISA_BASE)
q = pd.read_parquet(os.path.join(BASE,'data','qci_analysis.parquet'))
AI = ['ST438Q01DA','ST438Q02DA','ST438Q03DA','ST438Q04DA']
d = q[AI].dropna()

models = {
 'M1 单因子': 'F =~ ST438Q01DA + ST438Q02DA + ST438Q03DA + ST438Q04DA',
 'M2 双因子(代写/辅助)': 'F1 =~ ST438Q03DA\nF2 =~ ST438Q01DA + ST438Q02DA + ST438Q04DA',
 'M3 双因子(文本/探究)': 'F1 =~ ST438Q01DA + ST438Q03DA\nF2 =~ ST438Q02DA + ST438Q04DA',
}
rows = {}
for name, desc in models.items():
    m = semopy.Model(desc)
    m.fit(d)
    v = semopy.calc_stats(m).iloc[0].to_dict()
    rows[name] = {k: v.get(k) for k in ['chi2','DoF','CFI','TLI','RMSEA','SRMR','GFI','AIC','BIC']}
tab = pd.DataFrame(rows).T
tab['ΔCFI(vs M1)'] = tab['CFI'] - tab.loc['M1 单因子','CFI']
print('=== CFA 模型比较 (n=%d) ===' % len(d))
print(tab.round(3).to_string())

print('\n=== M1 标准化载荷 ===')
m1 = semopy.Model(models['M1 单因子']); m1.fit(d)
ins = semopy.inspect(m1)
print(ins[ins.op=='~'][['lval','rval','Est. Std']].round(3).to_string(index=False))

print('\n=== M3 因子相关（若两因子高度相关则支持单因子）===')
m3 = semopy.Model(models['M3 双因子(文本/探究)']); m3.fit(d)
ins3 = semopy.inspect(m3)
print(ins3[ins3.op=='~~'][['lval','rval','Est. Std']].round(3).to_string(index=False))
tab.round(4).to_csv(os.path.join(BASE,'outputs','cfa_compare.csv'), encoding='utf-8-sig')
print('\n已保存 outputs/cfa_compare.csv')
