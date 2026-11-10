# -*- coding: utf-8 -*-
"""Re-draw the figures with corrected label placement and colour scheme.
"""
import pandas as pd, numpy as np, os, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
warnings.filterwarnings('ignore')
rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']
rcParams['axes.unicode_minus']=False
rcParams['figure.dpi']=150

from paths import BASE  # repo root (override with PISA_BASE)
OUT=os.path.join(BASE,'outputs')
r=pd.read_csv(os.path.join(OUT,'regression_main.csv'))

# ---------- Figure 1: threshold effect ----------
a2=r[(r.model=='A2 AI分组(全模型)')&(r.outcome=='SCIE')].set_index('term')
order=['AI_g2','AI_g3','AI_g4','AI_g5']
labs=['每年\n1-2次','每月\n1-2次','每周\n1-2次','每天或\n几乎每天']
b=np.array([a2.loc[k,'b'] for k in order]); se=np.array([a2.loc[k,'se'] for k in order])
fig,ax=plt.subplots(figsize=(6.6,3.7))
x=np.arange(4); cols=['#B4B2A9']*3+['#A32D2D']
ax.bar(x,b,width=.6,color=cols,zorder=3)
ax.errorbar(x,b,yerr=1.96*se,fmt='none',ecolor='#2C2C2A',capsize=4,lw=1.0,zorder=4)
ax.axhline(0,color='#5F5E5A',lw=.9,zorder=5)
ax.set_xticks(x); ax.set_xticklabels(labs,fontsize=9.5)
ax.set_ylabel('科学素养差异（分）',fontsize=9.5)
ax.set_title('AI 使用的阈值效应：仅“每天使用”出现断崖',fontsize=11,pad=12)
for i,(bb,ss) in enumerate(zip(b,se)):
    lo,hi=bb-1.96*ss,bb+1.96*ss
    if bb < -20:
        ax.text(i,bb+5,f'{bb:+.1f}',ha='center',va='top',fontsize=11,
                fontweight='500',color='white')
    else:
        ax.text(i,lo-2.0,f'{bb:+.1f}',ha='center',va='top',fontsize=10,color='#444441')
ax.set_ylim(-52,14); ax.set_xlim(-.6,3.6)
ax.grid(axis='y',alpha=.22,zorder=0)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.text(-.55,-49,'参照组：从不或几乎从不使用  ·  误差线为 95% 置信区间  ·  已控制人口学、家庭 SES、\n科学倾向、课堂与学校层变量',
        fontsize=7.4,color='#5F5E5A',va='bottom')
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig1_threshold.png'),bbox_inches='tight'); plt.close()

# ---------- Figure 2: use composition ----------
names=['概括既定文本','查资料','写初稿','帮学习']
vals=[-18.23,-9.48,6.04,14.95]; ses_=[2.11,1.68,1.53,1.87]
fig,ax=plt.subplots(figsize=(6.6,3.4))
y=np.arange(4)
cols=['#639922' if v<0 else '#E24B4A' for v in vals]
ax.barh(y,vals,height=.58,color=cols,zorder=3)
ax.errorbar(vals,y,xerr=[1.96*s for s in ses_],fmt='none',ecolor='#2C2C2A',capsize=4,lw=1.0,zorder=4)
ax.axvline(0,color='#5F5E5A',lw=.9,zorder=5)
ax.set_yticks(y); ax.set_yticklabels(names,fontsize=10)
ax.invert_yaxis()
ax.set_xlabel('科学素养净效应（分，每 1 个标准差）',fontsize=9.5)
ax.set_title('AI 用途构成的净效应梯度',fontsize=11,pad=12)
for i,(v,s_) in enumerate(zip(vals,ses_)):
    edge=v-1.96*s_ if v<0 else v+1.96*s_
    ax.text(edge+(0.8 if v>0 else -0.8),i,f'{v:+.1f}',va='center',
            ha='left' if v>0 else 'right',fontsize=10,fontweight='500',
            color='#A32D2D' if v>0 else '#3B6D11')
ax.set_xlim(-26,21); ax.set_ylim(3.6,-.6)
ax.grid(axis='x',alpha=.22,zorder=0)
for s in ['top','right']: ax.spines[s].set_visible(False)
ax.text(-25,3.45,'绿色＝负向关联   红色＝正向关联',fontsize=8,color='#5F5E5A',va='bottom')
plt.tight_layout(); plt.savefig(os.path.join(OUT,'fig2_composition.png'),bbox_inches='tight'); plt.close()

# ---------- Figure 3: moderation ----------
fig,axes=plt.subplots(1,2,figsize=(8.0,3.5))
def panel(ax,model,inter,t1):
    row=r[r.model==model]
    cx=row[row.term=='AI_total_z'].iloc[0]; ci=row[row.term==inter].iloc[0]
    xs=np.array([-1.5,0,1.5])
    specs=[(-1,'#888780','调节变量低（−1SD）'),(0,'#2C2C2A','处于均值'),(1,'#A32D2D','调节变量高（+1SD）')]
    hs=[]
    for lvl,col,lab in specs:
        h,=ax.plot(xs,cx.b+ci.b*lvl*xs,color=col,lw=2.0,label=lab,solid_capstyle='round')
        hs.append(h)
    ax.axhline(0,color='#D3D1C7',lw=.8,zorder=0)
    ax.set_xticks(xs); ax.set_xticklabels(['低','中','高'],fontsize=9.5)
    ax.set_xlabel('AI 使用强度',fontsize=9.5)
    ax.set_ylabel('科学素养（相对均值）',fontsize=9.5)
    ax.set_title(f'{t1}\n交互项 {ci.b:+.1f}（p={ci.p:.3f}）',fontsize=10,pad=10,linespacing=1.5)
    for s in ['top','right']: ax.spines[s].set_visible(False)
    return hs
h1=panel(axes[0],'B4 AI×科学教师培训','AIxTRN','学校科学教师培训的调节')
h2=panel(axes[1],'B1 AI×数字素养','AIxDL','学生数字素养的调节')
fig.legend(h1,['调节变量低（−1SD）','处于均值','调节变量高（+1SD）'],loc='lower center',
           ncol=3,frameon=False,fontsize=8.6,bbox_to_anchor=(0.5,-0.04),handlelength=1.8)
plt.tight_layout(rect=[0,0.06,1,1])
plt.savefig(os.path.join(OUT,'fig3_moderation.png'),bbox_inches='tight'); plt.close()
print('All three figures redrawn')
