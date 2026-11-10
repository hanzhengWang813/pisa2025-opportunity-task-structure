# Data

The PISA 2025 public-use microdata are distributed by the OECD and are **not**
included in this repository. Download them from the OECD PISA site
(<https://www.oecd.org/pisa/>) and place them under `data/`:

```
data/
├── CY09_MS_STU_PUF.sav        student questionnaire + cognitive scores
└── CY09_MS_SCH_PUF.sav        principal questionnaire (school file)
```

`CY09_MS_TCH_PUF.sav` is not used: schools in the four Chinese jurisdictions did not
sit the PISA teacher survey, which is why teacher preparation is proxied by principal
reports in the paper (§5.5, limitation 5).

Access to the data remains subject to the OECD terms of use. Do not redistribute the
`.sav` files or the derived `.parquet` files; this repository ships only code, result
tables and figures.

## What `01_extract_qci.py` reads

| Block | Variables |
|---|---|
| Design | `CNT`, `CNTRYID`, `CNTSCHID`, `CNTSTUID`, `STRATUM`, `SUBNATIO`, `OECD`, `ADMINMODE`, `ST001D01T`, `ISCEDP` |
| Weights | `W_FSTUWT`, `SENWT`, `W_FSTURWT1`–`W_FSTURWT80` |
| Plausible values | `PV1`–`PV10` for `SCIE`, `MATH`, `READ`, `CMPS`, `CPPK`, `CMOD`, `CPRO`, `SEPS`, `SEDE`, `SEID`, `SENV` |
| AI use | `ST438Q01DA`–`ST438Q04DA` (four purposes), `AIUSESCH` |
| Digital literacy | `ST436Q01DA`–`ST436Q14DA`, `ST436Q16DA` (evaluate AI output) |
| ICT | `ICTSCH`, `ICTHOME`, `ICTQUAL`, `ICTWKDY`, `ICTINFO`, … |
| Socio-economic | `ESCS`, `HISEI`, `PAREDINT`, `HOMEPOS`, `HISCED` |
| Demographics | `ST004D01T`, `AGE`, `GRADE`, `IMMIG`, `REPEAT` |
| Principals | `SC155Q01HA`–`SC155Q07HA` (digital infrastructure), `SC190Q*`, `SC265Q13DA`, … |

The exact list is built at run time by intersecting the requested variables with the
variable names stored in `meta.pkl` (student) and `meta_sch.pkl` (principal), so the
script tolerates small differences between PISA releases.

## Expected analytic files

Running `scripts/run.py 01` then `scripts/run.py 02` creates

```
data/qci_student.parquet    12,385 students, one row each
data/qci_school.parquet     student rows + school constructs (rewritten by script 13)
data/qci_merged.parquet     merged student and principal records
data/qci_analysis.parquet   analysis file with standardised indices (§4.7 uses this)
```

## Sample definition

All 374 schools in Beijing, Shanghai, Jiangsu and Zhejiang matched on `CNTSCHID`; no
school-level case was lost in the merge. Effective sample sizes then vary by model
because of listwise deletion on the required variables (11,734 / 360 for the joint
model, 12,138 for the seven-item digital scale, 11,469 / 368 for §4.7).
