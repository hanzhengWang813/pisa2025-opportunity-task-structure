# -*- coding: utf-8 -*-
"""PISA 2025 中国 B-S-J-Z (QCI) 学生库 + 学校库 抽取与合并"""
import pyreadstat, pandas as pd, numpy as np, os, json, pickle

from paths import BASE  # repo root (override with PISA_BASE)
STU = os.path.join(BASE, 'data', 'CY09_MS_STU_PUF.sav')
SCH = os.path.join(BASE, 'data', 'CY09_MS_SCH_PUF.sav')
OUT = os.path.join(BASE, 'data')
os.makedirs(OUT, exist_ok=True)

META_STU = pickle.load(open(os.path.join(BASE, 'meta.pkl'), 'rb'))
META_SCH = pickle.load(open(os.path.join(BASE, 'meta_sch.pkl'), 'rb'))
AV_STU, AV_SCH = set(META_STU['names']), set(META_SCH['names'])

DESIGN = ['CNT','CNTRYID','CNTSCHID','CNTSTUID','STRATUM','SUBNATIO','OECD','ADMINMODE',
          'ST001D01T','ISCEDP']
WEIGHTS = ['W_FSTUWT','SENWT'] + [f'W_FSTURWT{i}' for i in range(1,81)]
DOMAINS = ['SCIE','MATH','READ','CMPS','CPPK','CMOD','CPRO','SEPS','SEDE','SEID','SENV']
PVS = [f'PV{k}{d}' for d in DOMAINS for k in range(1,11)]

AI = ['ST438Q01DA','ST438Q02DA','ST438Q03DA','ST438Q04DA','AIUSESCH']
DIGLIT = [f'ST436Q{i:02d}DA' for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14]] + ['ST436Q16DA']
ICT = ['ICTSCH','ICTHOME','ICTQUAL','ICTFEED','ICTOUT','ICTWKDY','ICTWKEND','ICTREG',
       'ICTINFO','DIGINTLRN','INTICT','SOIAICT','ICTAVHOME','ICTAVSCH','ICTDISTR',
       'ST322Q01JA','ST322Q02JA','ST322Q06JA','ST322Q07JA','ST437Q01DA','ST437Q02DA',
       'ST253Q01JA','ST254Q01JA','ST254Q02JA','ST254Q03JA','ST254Q04JA','ST254Q05JA']
SES = ['ESCS','HISEI','PAREDINT','HOMEPOS','HISCED','P1ISCED','P2ISCED','BSMJ','EXPECEDU',
       'OCOD3','SISCO','ST255Q01JA','ST403Q01DA']
DEMO = ['ST004D01T','MALE','AGE','GRADE','IMMIG','LANGN','REPEAT','ST021Q01TA','ST022Q01TA',
        'ST333Q01DA','COBN_S','BICT_S']
SCI_DISP = ['JOYSCIE','EFFSCIE','ENGSCIPR','STBELSCIE','PERSEV','CURIO','GOALSET','SELFREG',
            'COGABIL','ENPROBS','ST094Q01NA','ST094Q03NA','ST094Q04NA','ST094Q05NA',
            'ST129Q03TA','ST129Q05TA','ST129Q08DA','ST129Q11DA','ST129Q12DA','ST129Q14DA']
CLASSRM = ['TEACHSUP','COGACSC','DISCLISCI','SCIECOLLR','OPENVLRN','ENVAWARE','ENVAPART',
           'COLLENEFF','ENVCAPCH','ST469Q06DA','ST469Q08DA','ST469Q09DA','ST469Q10DA','ST469Q11DA',
           'ST468Q08DA','ST468Q09DA','ST468Q10DA','ST468Q11DA',
           'ST515Q02DA','ST515Q04DA','ST515Q05DA','ST515Q08DA','ST515Q09DA','ST515Q11DA',
           'ST531Q02DA','ST531Q11DA']
WELLBEING = ['BELONG','FEELSAFE','BULLIED','ST016Q01NA','ST038Q04NA','ST038Q05NA','ST038Q07NA',
             'ST038Q08NA','ST038Q12DA','ST265Q01JA','ST265Q03JA','SKIPPING','TARDYSD','WORKPAY']
PEER = ['ST527Q01DA','ST528Q01DA','ST300Q01JA','ST300Q02JA','ST300Q03JA']
EFFORT = ['EFFORT1','EFFORT2','ST331Q01JA','ST331Q02JA','ST331Q03JA']
TIME = ['ST071Q01NA','ST071Q02NA','ST296Q03JA','ST296Q04JA','IC173Q03JA','IC173Q04JA',
        'ST294Q04JA','ST294Q05JA','ST295Q04JA','ST295Q05JA']

STU_VARS = [v for v in (DESIGN+WEIGHTS+PVS+AI+DIGLIT+ICT+SES+DEMO+SCI_DISP+CLASSRM+WELLBEING+PEER+EFFORT+TIME) if v in AV_STU]
print('学生变量请求/可用:', len(STU_VARS))

SCH_WLE = ['DIGPREP','AVLRSOFT','SCTRAIN','EDUSHORT','STAFFSHORT','FREESCIE','EDULEAD',
           'NEGSCLIM','ALLACTIV','ENCOURPG','SESASSESS','STDTEST','TDTEST','PRIORLRN','DELGLOB','OPPACTENV']
SCH_CORE = ['CNT','CNTSCHID','W_NRASCHBWT','SC001Q01TA','SC002Q01TA','SC003Q01TA','SC048Q01NA']
SCH_Q = []
for pref in ['SC004Q','SC016Q','SC017Q','SC018Q','SC019Q','SC042Q','SC155Q','SC158Q','SC172Q',
             'SC190Q','SC202Q','SC250Q','SC261Q','SC264Q','SC265Q','SC266Q','SC267Q','SC268Q',
             'SC053Q','SC421Q','SC400Q','SC482']:
    SCH_Q += [n for n in META_SCH['names'] if n.startswith(pref)]
SCH_VARS = [v for v in (SCH_CORE + SCH_WLE + SCH_Q) if v in AV_SCH]
print('学校变量请求/可用:', len(SCH_VARS))

print('读取学生库 ...')
stu, _ = pyreadstat.read_sav(STU, usecols=STU_VARS)
print('全库:', stu.shape)
q = stu[stu.CNT == 'QCI'].copy().reset_index(drop=True)
del stu
print('QCI 学生:', len(q), '| 学校:', q.CNTSCHID.nunique())

print('读取学校库 ...')
sch, _ = pyreadstat.read_sav(SCH, usecols=SCH_VARS)
qs = sch[sch.CNT == 'QCI'].copy().reset_index(drop=True)
print('QCI 学校记录:', len(qs))
print('学校层 WLE 有效比例(%):')
print((qs[[c for c in SCH_WLE if c in qs.columns]].notna().mean()*100).round(1).to_string())

sch_use = qs.drop(columns=[c for c in ['CNT'] if c in qs.columns])
m = q.merge(sch_use, on='CNTSCHID', how='left', suffixes=('', '_sch'))
print('\n合并后:', m.shape, '| 匹配到学校层的学生数:', m['DIGPREP'].notna().sum())

q.to_parquet(os.path.join(OUT, 'qci_student.parquet'), index=False)
qs.to_parquet(os.path.join(OUT, 'qci_school.parquet'), index=False)
m.to_parquet(os.path.join(OUT, 'qci_merged.parquet'), index=False)
json.dump({'n_student': len(q), 'n_school': int(q.CNTSCHID.nunique()),
           'stu_vars': STU_VARS, 'sch_vars': [c for c in sch_use.columns]},
          open(os.path.join(OUT, 'qci_meta.json'),'w'), ensure_ascii=False, indent=1)
for f in ['qci_student.parquet','qci_school.parquet','qci_merged.parquet']:
    p = os.path.join(OUT,f); print(f'{f}: {os.path.getsize(p)/1e6:.1f} MB')
