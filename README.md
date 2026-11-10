# Opportunity Structure or Task Structure?

**School Conditions, Generative AI Use, and Science Literacy in Four Chinese Provinces and Municipalities**

Replication package for the manuscript. Everything reported in the paper is produced by
the scripts in `scripts/` from the PISA 2025 public-use files and checked against the
manuscript in `verification/`.

- 12,385 students, 374 schools, Beijing–Shanghai–Jiangsu–Zhejiang (B-S-J-Z)
- Weighted two-level random-slope models, Mundlak decomposition, school fixed effects
- 29 candidate school-level moderators, Benjamini–Hochberg correction

---

## 1. Verification at a glance

`verification/number_check.csv` contains one row per number printed in the manuscript
(211 rows), with the value as printed, the value recomputed here, the absolute
difference, a status, and the exact file and column the number comes from.

| status | rows | meaning |
|---|---|---|
| `reproduced` | 203 | matches to the precision printed in the paper |
| `last-digit` | 5 | printed standard error rounds 0.01 away from the stored value |
| `rounding` | 2 | paper rounds a hedged quantity ("about 33", cumulative ICC 7.3) |
| `external` | 1 | OECD Vol. I benchmark, not computable from these data |

No coefficient, standard error, p-value, sample size, variance component or
percentage is unaccounted for. The five `last-digit` rows are all in standard-error
columns; in each case the printed *p*-value matches the stored (unrounded) standard
error, so no inference changes. See `docs/VERIFICATION.md` for the itemised list.

Reproduce the check:

```bash
python verification/verify_numbers.py        # writes verification/number_check.csv
```

---

## 2. Layout

```
├── scripts/                  analysis pipeline (run in numeric order)
│   ├── 01_extract_qci.py     extract + merge student and principal files
│   ├── 02_variables.py       derived variables, plausible-value scales
│   ├── 13_school_constructs.py   eight school constructs, missingness, descriptives
│   ├── 14_mlm_ladder.py      ICC / tau00 ladder (Table 1 Panel A)
│   ├── 15_school_models.py   school-level main effects (Table 1 Panel B)
│   ├── 16_moderator_scan.py  29-moderator cross-level scan (Table 2)
│   ├── 19_level_decomposition.py  within vs between decomposition (§4.6)
│   ├── 20_mundlak_within.py  Mundlak and school-FE comparisons (§4.4)
│   ├── 21_level_id_correct.py    daily-use ladder, three estimators (§4.4)
│   ├── 11_oecd_claims.py     critical-use access analysis (§4.7)
│   ├── 26_estimator_check.py custom estimator benchmark (Appendix A)
│   ├── pisa_engine.py        weighted two-level estimator (PV + BRR-Fay)
│   └── wmlm.py               Woodbury-identity multilevel likelihood
├── outputs/                  result tables (.csv) and figures (.png)
├── verification/
│   ├── verify_numbers.py     recomputes every reported number
│   ├── repro_mundlak_daily.py  two coefficients not stored in outputs/
│   └── number_check.csv      the generated audit table
├── docs/
│   ├── REPRODUCTION.md       script-by-script map to manuscript sections
│   ├── VERIFICATION.md       itemised audit result
│   ├── LABELS.md             English glossary of the Chinese identifiers in outputs/
│   └── DATA.md               how to obtain the PISA 2025 files
├── meta.pkl / meta_sch.pkl   variable dictionaries used by 01_extract_qci.py
└── requirements.txt
```

The PISA microdata are **not** included (OECD terms of use). See `docs/DATA.md`.

Code, comments and documentation are in English. Column names in `outputs/*.csv`
are English; a number of **cell values** kept the Chinese identifiers they were
written with during development (`M0 空模型`, `SCIE|C 校内FE(每天)`, the A–Z
construct codes). Those strings are literal join and audit keys, so renaming them
would break the audit trail. `docs/LABELS.md` translates every one of them.

---

## 3. Quick start

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements.txt

# place the PISA 2025 files in ./data (see docs/DATA.md), then:
python scripts/run.py 01        # extract and merge
python scripts/run.py 02        # derived variables
python scripts/run.py 13        # ... through 26, or:
python scripts/run.py all

python verification/verify_numbers.py
```

`scripts/run.py <id>` runs a single script with the right import paths;
`run.py list` shows the full list. If the data live elsewhere, set `PISA_BASE`:

```bash
export PISA_BASE=/path/to/project-root     # expects $PISA_BASE/data/*.parquet
```

Scripts read `<root>/data/*.parquet` and write `<root>/outputs/*`.

---

## 4. Where the headline numbers come from

| Manuscript | Value | File | Column / term |
|---|---|---|---|
| ICC, empty model | 0.426 | `outputs/mlm_ladder.csv` | `icc`, `M0 空模型` |
| Between-school variance tau00 | 3157.0 | `outputs/mlm_ladder.csv` | `tau00`, `M0` |
| School mean ESCS, per SD | +26.8 | `outputs/school_main.csv` | `SCH_MEAN_ESCS_z` |
| Academic climate, per SD | +13.1 | `outputs/school_main.csv` | `SCH_CLIM_z` |
| Digital infrastructure, 7-item | +3.62 | `outputs/fig9_coefs.csv` | `DIGBASE_z` |
| Digital infrastructure, 5-item, univariate | +4.79 | `outputs/suppression_check.csv` | `A_only`, `A_DIGCOND_z` |
| Digital infrastructure, 5-item, joint | +6.81 | `outputs/school_main.csv` | `A_DIGCOND_z` |
| Mean slope of AI use | −2.99 | `outputs/mlm_ladder.csv` | `b`, `M3`, `AIz` |
| Slope variance | 46.02 | `outputs/mlm_ladder.csv` | `tau11`, `M3` |
| Slope variance, 8 interactions | 42.35 | `outputs/mlm_ladder.csv` | `tau11`, `M5` |
| 29 interactions, range / median | −2.34 … +2.12 / −0.16 | `outputs/moderator_scan.csv` | `b_int` |
| BH-adjusted q | .53 | `outputs/moderator_scan.csv` | `p_int`, 29 tests |
| Daily AI use, school FE | −34.4 | `outputs/level_id.csv` | `D_within`, `C 校内FE(每天)` |
| Daily AI use, single level | −50.4 | `outputs/level_id.csv` | `AI_daily`, `A 单层(每天)` |
| Climate x AI, single level / Mundlak / FE | −2.50 / −0.13 / −0.19 | `outputs/level_id.csv` | `D1` / `D2` / `D3`, `AIxCLIM` |
| Between-school climate x AI | −26.3 | `outputs/level_decomposition.csv` | `校均AI×氛围` |
| Four purposes, within school | −10.30 … +5.92 | `outputs/within_uses.csv` | `ci_ST438Q01DA`–`Q04DA` |
| Critical-use access, at least sometimes | 44.7% | `scripts/11_oecd_claims.py` (stdout) | `LO_AI_any` |
| Estimator benchmark | 1.19e−6 | `outputs/estimator_check.csv` | check `1a` |

`docs/REPRODUCTION.md` maps all 26 scripts to the manuscript sections they generate.

Two coefficients in the paper are **not** stored as rows in `outputs/`; both are
re-derived in `verification/repro_mundlak_daily.py`:

- §4.4 Mundlak within-school component for the daily dummy: **−46.10 (SE 5.66)**.
  `outputs/level_id.csv` stores the continuous-intensity Mundlak (−51.88), which is a
  different quantity.
- §4.1 univariate coefficient for organization of digital teaching: **−3.35 (p = .128)**.

---

## 5. Implementation notes

- **Estimator.** `pisa_engine.py` is a weighted two-level estimator that handles the
  ten plausible values and 80 replicate weights internally (Fay coefficient 0.5),
  pooling with Rubin's rules. `wmlm.py` uses the Woodbury identity so that no
  per-school matrix is inverted. Appendix A of the paper reports the benchmark:
  maximum fixed-effect difference 1.19e−6 against `statsmodels`, between-school
  variance 2945.581 vs 2945.579, identical log-likelihoods.
- **Two-level models** use model-based standard errors pooled across plausible
  values; **single-level weighted models** use design-based BRR standard errors.
- **Weighting.** Student final weights only; there is no school-level weight for
  mainland China because these schools did not sit the teacher survey.
- **Missing data.** Listwise deletion within each model, no imputation. Hence the
  varying sample sizes (11,734 / 360 for the joint model; 12,138 for the seven-item
  scale; 11,469 / 368 for §4.7).
- **`semopy`** is only needed by `02b_cfa.py`, which feeds an appendix CFA table and
  no number in the manuscript. Pin `semopy<=2.3.10`; 2.3.11 removed a module-level
  import that the script relies on.

---

## 6. Data

The PISA 2025 public-use files are distributed by the OECD and are not redistributed
here. `docs/DATA.md` lists the two files the pipeline needs and the variables it
extracts. Any use of the data remains subject to the OECD terms of use.

## 7. Citation

```bibtex
@article{opportunity_task_structure_2026,
  title   = {Opportunity Structure or Task Structure? School Conditions,
             Generative {AI} Use, and Science Literacy in Four {C}hinese
             Provinces and Municipalities},
  year    = {2026},
  note    = {Replication package}
}
```

## 8. License

Code: MIT (`LICENSE`). Result tables and figures: CC BY 4.0.

## 9. Before publishing

Three placeholders still need the author's input:

| Where | What to replace |
|---|---|
| `CITATION.cff`, `LICENSE` | `The authors` with the author list |
| `README.md` §7 | the `@article` key and author fields, once the paper is accepted |

The repository URL is already set to
`https://github.com/hanzhengWang813/pisa2025-opportunity-task-structure`.

Upload with:

The remote `origin` is already configured, so a push is a single command:

```bash
git push -u origin main
```

If the repository has not been created yet, create it first as an empty public repo
named `pisa2025-opportunity-task-structure` (no README, no .gitignore, no licence —
this package already contains them).
