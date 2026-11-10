# Label glossary

The result tables in `outputs/` were written by the analysis pipeline while the
project was being developed in Chinese. Their **column names are English**, but a
number of **cell values are Chinese identifiers**: model names (`model`), term names
(`term`), construct names, and the row labels of the audit tables.

These strings are treated as **stable keys**, not prose. They are matched literally
by `verification/verify_numbers.py`, by `scripts/25_paper_audit.py`, and by the
cross-file joins inside the pipeline, so renaming them would silently break the
audit trail. They are therefore kept verbatim, and translated here.

Everything that a reader has to *understand* — the README, `docs/*.md`, the code,
its comments and its console output — is in English.

---

## 1. Generic values

| Chinese | English |
|---|---|
| `(截距)` | intercept |
| `__var__` | variance components (a synthetic term row; see `tau00`, `tau11`, `sigma2`, `icc`) |
| `一致` | match |
| `同上` | same as above |
| `空模型` | empty model |
| `校层相关` | school-level correlation |
| `校长报告` | principal report |
| `校内实践聚合` | aggregate of in-school practice |
| `学生` / `学生/家庭` / `教师/课程` | student / student-and-family / teacher-and-curriculum (lever level) |
| `教师（可培训）` / `学校（可投入）` / `学生（可教）` | teacher (trainable) / school (resourced) / student (teachable) |

## 2. Model-name grammar

Most `model` values follow `OUTCOME|FAMILY label`, where `OUTCOME` is `SCIE`
(science literacy) or `CMPS` (the composite score used for the estimator benchmark).
Families:

| Family | Meaning |
|---|---|
| `A …` | single-level weighted regression |
| `B …` | Mundlak decomposition (within and between components in one model) |
| `C …` | school fixed effects |
| `D1–D3 …` | academic-climate x AI interaction under the three estimators |
| `E1–E3 …` | four use purposes entered jointly, under the three estimators |
| `S1–S4 …` | school-level ladder (student level / main effects / one interaction / all interactions) |
| `M0–M8 …` | two-level model ladder |
| `F1–F3 …` | climate-focus models |
| `R1–R6 …` | specification-robustness variants for AI use |
| `T0–T2 …` | levers: total use only / composition only / both |
| `L0 …` | full lever model |
| `P0–P3 …` | OECD claim tests |

Segment glosses:

| Chinese | English |
|---|---|
| `单层` | single level |
| `Mundlak` | Mundlak decomposition |
| `校内FE` | within-school fixed effects |
| `强度` | continuous intensity score |
| `每天` | daily-use dummy |
| `每天哑变量` | daily-use dummy |
| `四用途` | four use purposes |
| `含校间` | with between-school terms added |
| `交互` | interaction |
| `学生层` | student level |
| `学校层` / `+学校层主效应` | school level / plus school-level main effects |
| `全部交互` | all interactions |
| `空模型` | empty model |
| `+学生层` | plus student covariates |
| `+AI(固定斜率)` | plus AI use, fixed slope |
| `+AI随机斜率` | plus AI use, random slope |
| `+跨层交互` | plus cross-level interactions |
| `+学校氛围` | plus school academic climate |
| `AI分组(基础)` / `AI分组(全模型)` | AI-use frequency groups, baseline / full covariate set |
| `AI线性+平方` | AI use, linear plus quadratic |
| `AI双维度(全模型)` | AI use split into two dimensions, full model |
| `四分组` | four frequency groups |
| `徐文二分` | Xu & Schleicher (2026) binary coding |
| `连续` | continuous |
| `二分+全控制` | binary, full controls |
| `基准` / `基准(四题均值分组)` | baseline |
| `仅「每天」二分类` | daily-versus-rest binary |
| `线性+二次` | linear plus quadratic |
| `剔除每天组` | excluding daily users |
| `单题(查资料/写初稿)` | single item (look up material / write a first draft) |
| `线性项(正文模型)` | linear term, the model reported in the text |
| `仅总量` / `仅构成对比` / `总量+构成` | total only / composition only / both |
| `全模型` | full model |
| `组内组间分解` | within/between decomposition |
| `分组+交互` | grouping plus interaction |
| `×AI评估机会(频繁)` | interaction with frequent opportunity to evaluate AI |
| `×线上信息评估` | interaction with online-information evaluation |

## 3. School-level constructs

Constructs A–I enter the main models; J–Z are single items that appear only in the
moderator scan.

| Code | Chinese | English |
|---|---|---|
| A | `数字条件` | digital conditions (infrastructure and connectivity), 5 items |
| A (7-item) | `数字基础条件` | digital infrastructure, 7 items (A plus the two teacher-capacity items) |
| B | `教师能力时间` | teacher capacity and time |
| C | `数字使用取向` | orientation of digital use |
| D | `规范共定` | co-determined rules |
| F | `自主学习优先` | priority given to self-regulated learning |
| G | `数字教学组织` | organisation of digital teaching |
| H | `过程追踪(单题)` | process monitoring (single item) |
| I | `赋能取向(单题)` | enabling orientation (single item) |
| J | `手机禁令` | mobile-phone ban |
| K | `数字教学准备度` | readiness for digital teaching |
| L | `科学教师培训` | science-teacher training |
| M | `教育短缺` | educational shortage |
| N | `师资短缺` | teacher shortage |
| O | `软件可用性` | software availability |
| P | `教学领导` | instructional leadership |
| Q | `负面校风` | negative school climate |
| R | `科学合作文化` | culture of science collaboration |
| S | `书面数字设备规定` | written rules on digital devices |
| T | `教师自定使用规则` | teacher-set rules on use |
| U | `师生共定规则` | rules set jointly by teachers and students |
| V | `网络素养课程` | internet-literacy curriculum |
| W | `社交网络政策` | social-media policy |
| X | `教师应对学生用AI` | teachers responding to student AI use |
| Y | `数字教学PD需求` | demand for digital-teaching professional development |
| Z | `自主学习教学支持` | instructional support for self-regulated learning |

Aggregates and climate components:

| Chinese | English |
|---|---|
| `校均AI使用` / `校均AI` | school mean AI use |
| `校内AI离散度` | within-school dispersion of AI use |
| `校均屏幕时间` | school mean screen time |
| `每天使用占比` | share of daily users in the school |
| `学校规模` | school size |
| `校均ESCS` / `校均 ESCS` | school mean ESCS |
| `校均氛围` | school mean academic climate |
| `校均AI×氛围` | school mean AI use x climate |
| `认知激活` | cognitive activation |
| `纪律氛围` | disciplinary climate |
| `自我调节` / `自我调节学习` | self-regulation / self-regulated learning |
| `科学兴趣` | interest in science |
| `科学自我效能` | science self-efficacy |
| `教师支持` | teacher support |
| `探究式科学教学` | inquiry-based science teaching |
| `课堂认知激活教学` | cognitively activating classroom instruction |
| `学生数字信息素养` | student digital-information literacy |
| `学校 ICT 资源质量` | quality of school ICT resources |
| `学校数字教学准备度` | school readiness for digital teaching |
| `学校科学教师培训` | school-level science-teacher training |
| `上学日数字设备使用总量` | total weekday digital-device use |
| `AI 用途构成（输入导向→产出导向）` | AI use composition (input- to output-oriented) |
| `AI 使用强度（线性项）` | AI use intensity, linear term |
| `单因子 CFA` / `单因子 CFA 标准化载荷` | one-factor CFA / its standardised loadings |
| `M1 单因子`, `M2 双因子(代写/辅助)`, `M3 双因子(文本/探究)` | one-factor, two-factor (drafting/assistance), two-factor (text/inquiry) CFA models |
| `四题相关矩阵` / `四题全=5 / 每天组` | correlation matrix of the four items / all four items at the maximum, daily users |
| `六(三)` … `三(一)` | manuscript section numbers (e.g. section 6.3) — only appear in the audit tables |

## 4. Estimator benchmark labels (`outputs/estimator_check.csv`)

| Check | Chinese | English |
|---|---|---|
| `1a` | `固定效应与 statsmodels 最大差` | maximum difference in fixed effects versus statsmodels |
| `1b` | `校间方差 自建` | between-school variance, our estimator |
| `1c` | `校间方差相对差` | relative difference in between-school variance |
| `2` | `对数似然 与直接 MVN 之差` | difference in log-likelihood versus a direct multivariate-normal evaluation |
| `3` | `向量化与循环版似然差` | difference in log-likelihood between the vectorised and loop implementations |
| `4` | `学生最终权重均值（需归一化）` | mean student final weight (must be normalised) |

## 5. Descriptive strings

Some cells are prose rather than identifiers, for example
`中位检验力下可检出 3.3 分` ("3.3 points detectable at median power") or
`斜率下降 8.0%` ("slope shrinks by 8.0%"). These occur only in
`paper_number_audit.csv` (the internal provenance ledger behind
`verification/number_check.csv`) and in the diagnostic tables
`school_aggregate_scan.csv`, `suppression_check.csv` and `school_r2.csv`.
They restate the finding they accompany and carry no numeric value that is not
also stored in a numeric column of the same row.
