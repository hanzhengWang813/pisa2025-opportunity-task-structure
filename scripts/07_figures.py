# -*- coding: utf-8 -*-
"""Generate the figures and the results summary tables.
"""
import pandas as pd, numpy as np, os, json, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
warnings.filterwarnings('ignore')
rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']
rcParams['axes.unicode_minus']=False
rcParams['figure.dpi']=150

from paths import BASE  # repo root (override with PISA_BASE)
OUT=os.path.join(BASE,'outputs'); os.makedirs(OUT,exist_ok=True)
r=pd.read_csv(os.path.join(OUT,'regression_main.csv'))

# ---------- Figure 1: threshold effect ----------
a2=r[(r.model=='A2 AI分组(全模型)')&(r.outcome=='SCIE')]
g=a2[a2.term.isin(['AI_g2','AI_g3','AI_g4','AI_g5'])].copy()
labs=['每年\n1-2次','每月\n1-2次','每周\n1-2次','每天或\n几乎每天']
g=g.set_index('term').loc[['AI_g2','AI_g3','AI_g4','AI_g5']]
fig,ax=plt.subplots(figsize=(6.4,3.6))
x=np.arange(len(g)); b=g.b.values; se=g.se.values
cols=['#888780']*3+['#A32D2D']
ax.bar(x,b,width=.62,color=cols,zorder=3)
ax.errorbar(x,b,yerr=1.96*se,fmt='none',ecolor='#2C2C2A',capsize=4,lw=1.1,zorder=4)
ax.axhline(0,color='#5F5E5A',lw=.9)
ax.set_xticks(x); ax.set_xticklabels(labs,fontsize=9)
ax.set_ylabel('科学素养差异（分）\n参照组：从不或几乎从不使用',fontsize=9)
ax.set_title('AI 使用的阈值效应：仅“每天使用”出现断崖',fontsize=10.5,pad=10)
for i,(bb,ss) in enumerate(zip(b,se)):
    ax.text(i,bb-2.5 if bb<0 else bb+1.5,f'{bb:.1f}',ha='center',va='top' if bb<0 else 'bottom',fontsize=9,
            color='#A32D2D' if i==3 else '#444441',fontweight='500')
ax.set_ylim(-52,10); ax.grid(axis='y',alpha=.25,zorder=0)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.text(.99,.03,'误差线为 95% 置信区间；已控制人口学、家庭SES、科学倾向、课堂与学校层变量',
        transform=ax.transAxes,ha='right',fontsize=7.2,color='#5F5E5A')
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig1_threshold.png'),bbox_inches='tight'); plt.close()

# ---------- Figure 2: net effect of use composition ----------
items={'概括既定文本':'−18.2','查资料':'−9.5','写初稿':'+6.0','帮学习':'+15.0'}
vals=[-18.23,-9.48,6.04,14.95]; ses_=[2.11,1.68,1.53,1.87]
fig,ax=plt.subplots(figsize=(6.4,3.4))
x=np.arange(4); cols=['#639922' if v<0 else '#E24B4A' for v in vals]
ax.barh(x,vals,height=.6,color=cols,zorder=3)
ax.errorbar(vals,x,xerr=[1.96*s for s in ses_],fmt='none',ecolor='#2C2C2A',capsize=4,lw=1.1,zorder=4)
ax.axvline(0,color='#5F5E5A',lw=.9)
ax.set_yticks(x); ax.set_yticklabels(list(items.keys()),fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel('科学素养净效应（分 / 1 个标准差）',fontsize=9)
ax.set_title('AI 用途构成的净效应梯度',fontsize=10.5,pad=10)
for i,v in enumerate(vals):
    ax.text(v+(1.2 if v>0 else -1.2),i,f'{v:+.1f}',va='center',ha='left' if v>0 else 'right',fontsize=9,
            fontweight='500',color='#A32D2D' if v>0 else '#3B6D11')
ax.set_xlim(-24,20); ax.grid(axis='x',alpha=.25,zorder=0)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.text(.99,.02,'四题同时入模、互为控制；绿色=负向，红色=正向',
        transform=ax.transAxes,ha='right',fontsize=7.2,color='#5F5E5A')
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig2_composition.png'),bbox_inches='tight'); plt.close()

# ---------- Figure 3: moderation ----------
fig,axes=plt.subplots(1,2,figsize=(7.6,3.2))
def panel(ax,mod_term,mod_name,title,txt):
    row=r[(r.model==mod_term)]
    cx=row[row.term=='AI_total_z'].iloc[0]
    ci=row[row.term=='AIx'+mod_name].iloc[0]
    xs=np.array([-1,0,1])
    specs=[(-1,'--','#888780','调节变量 低 (−1SD)'),(0,'-','#2C2C2A','处于均值'),
           (1,'--','#A32D2D','调节变量 高 (+1SD)')]
    for lvl,ls,col,lab in specs:
        ax.plot(xs,cx.b+ci.b*lvl*xs,ls,color=col,lw=1.6,label=lab)
    ax.axhline(0,color='#B4B2A9',lw=.7)
    ax.set_xticks(xs); ax.set_xticklabels(['低','中','高'],fontsize=9)
    ax.set_xlabel('AI 使用强度',fontsize=9)
    ax.set_ylabel('科学素养',fontsize=9)
    ax.set_title(title,fontsize=10)
    ax.legend(frameon=False,fontsize=7.5,loc='lower left')
    for s in ['top','right']: ax.spines[s].set_visible(False)
    ax.text(.98,.03,txt,transform=ax.transAxes,ha='right',fontsize=7,color='#5F5E5A')
try:
    panel(axes[0],'B4 AI×科学教师培训','TRN','学校科学教师培训的调节','交互 −2.1，p=.019')
    panel(axes[1],'B1 AI×数字素养','DL','学生数字素养的调节','交互 −1.6，p=.068')
    plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig3_moderation.png'),bbox_inches='tight')
except Exception as e:
    print('fig3 skipped:',e)
plt.close()

# ---------- Summary table ----------
def star(p): return '***' if p<.001 else '**' if p<.01 else '*' if p<.05 else ''
tab=r[r.outcome=='SCIE'].copy()
tab['sig']=tab.p.map(star)
piv=tab.pivot_table(index='term',columns='model',values='b')
piv.to_csv(os.path.join(OUT,'table_coefs_wide.csv'),encoding='utf-8-sig')
print('Figures written: fig1_threshold.png, fig2_composition.png, fig3_moderation.png')
print('Summary table: table_coefs_wide.csv')
print('\n=== Key coefficients (outcome: scientific literacy) ===')
key=['AI_g5','AI_high','AI_total_z','AI_sq','ST438Q01DA','ST438Q02DA','ST438Q03DA','ST438Q04DA',
     'AI_deep_z','AI_sub_z','AIxDL','AIxSES','AIxTRN','AIxPOL','SCTRAIN_z','ICTINFO_z']
for m in r.model.unique():
    s=r[(r.model==m)&(r.outcome=='SCIE')&(r.term.isin(key))]
    if len(s)==0: continue
    print(f'\n[{m}]')
    for _,w in s.iterrows():
        print(f'   {w.term:14s} {w.b:8.2f} ({w.se:5.2f}) p={w.p:.4f}{star(w.p)}')
