# -*- coding: utf-8 -*-
"""08 图4：每天使用组 vs 其余组 特征对比"""
import pandas as pd, numpy as np, os, warnings
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
warnings.filterwarnings('ignore')
rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']
rcParams['axes.unicode_minus']=False
rcParams['figure.dpi']=150

from paths import BASE  # repo root (override with PISA_BASE)
q=pd.read_parquet(os.path.join(BASE,'data','qci_analysis.parquet'))
q['AI_grp']=q.AI_total.round()
s=q.dropna(subset=['AI_grp','ESCS']).copy()
d5=s[s.AI_grp==5]; rest=s[s.AI_grp<5]

def wmean(df,col):
    x=df[col]; m=x.notna()
    return np.average(x[m].values, weights=df.loc[m,'W_FSTUWT'].values)

specs=[('ESCS','家庭社会经济地位'),('SELFREG','自我调节'),('JOYSCIE','科学兴趣'),
       ('PERSEV','坚持性'),('ICTWKDY','平日数字设备使用'),('EFFSCIE','科学自我效能')]
rows=[]
for c,lab in specs:
    rows.append((lab, wmean(d5,c), wmean(rest,c)))
tab=pd.DataFrame(rows,columns=['var','daily','rest'])
tab['diff']=tab.daily-tab.rest
tab=tab.sort_values('diff')

fig,ax=plt.subplots(figsize=(6.8,3.5))
y=np.arange(len(tab))
ax.hlines(y,0,tab['diff'],color=['#A32D2D' if v>0 else '#639922' for v in tab['diff']],lw=2.2,zorder=3)
ax.scatter(tab['diff'],y,s=64,color=['#A32D2D' if v>0 else '#639922' for v in tab['diff']],zorder=4)
ax.axvline(0,color='#5F5E5A',lw=1.0,zorder=2)
ax.set_yticks(y); ax.set_yticklabels(tab['var'],fontsize=10)
ax.set_xlabel('与其余学生的差值（加权标准化单位）',fontsize=9.5)
ax.set_title('“每天使用 AI”的学生在各项特征上更优',fontsize=11,pad=12)
for i,(v,a,b_) in enumerate(zip(tab['diff'],tab.daily,tab.rest)):
    ax.text(v+(0.035 if v>0 else -0.035),i,f'{v:+.2f}',va='center',
            ha='left' if v>0 else 'right',fontsize=9,color='#444441')
ax.set_xlim(-.18,1.12)
ax.grid(axis='x',alpha=.22,zorder=0)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.text(-.16,-.62,'每天组 n=384，其余组 n=11,515  ·  加权标准化均值差  ·  成绩差 −36 分',
        fontsize=7.6,color='#5F5E5A',va='bottom')
ax.set_ylim(-0.85,len(tab)-0.4)
plt.tight_layout(); plt.savefig(os.path.join(BASE,'outputs','fig4_daily_profile.png'),bbox_inches='tight'); plt.close()
print(tab.round(2).to_string(index=False))
print('已保存 outputs/fig4_daily_profile.png')
