# Reproduction map

Every table, figure and in-text number in the manuscript is produced by the scripts
below. Run them from the repository root:

```bash
python scripts/run.py <id>        # one script
python scripts/run.py all         # 01 → 26 in order
```

Scripts read `<root>/data/*.parquet` and write `<root>/outputs/*`. The order matters
at three points: `13` augments `data/qci_school.parquet` in place (so `01` must run
first), `22` writes `outputs/school_main.csv` and `25` writes `outputs/fig9_coefs.csv`,
and both must run before the verification table is generated.

| Script | Writes | Used for |
|---|---|---|
| `01_extract_qci.py` | `data/qci_student.parquet`, `qci_school.parquet`, `qci_merged.parquet` | §3.1 sample: 12,385 students, 374 schools, 458 principal variables |
| `02_variables.py` | `data/qci_analysis.parquet` | §3.2 derived variables and plausible-value scales |
| `02b_cfa.py` | `outputs/cfa_compare.csv` | appendix CFA comparison; no number in the text |
| `03_main_analysis.py` | `regression_results.csv`, `regression_within_between.csv` | earlier student-level models (superseded by `04`) |
| `04_main_revised.py` | `regression_main.csv` | §5.4.4 (ESCS 25.6 → 4.6; +10.0 / +8.2 / +7.8); §4.4 frequency ladder (−40.9; weekly/monthly/yearly) |
| `05_robustness.py` | `robustness.csv` | §4.1 teacher training +6.35 (SE 1.68); §4.4 −38.4 with school controls |
| `06_diagnose_pattern.py` | — | diagnostics only |
| `07_figures.py`, `07b_figures_fix.py` | `fig1`–`fig3`, `table_coefs_wide.csv` | exploratory figures, not in the manuscript |
| `08_fig4.py` | `fig4_daily_profile.png` | §4.4 heavy-user profile |
| `09_levers.py`, `10_fig5.py` | `intervention_levers.csv`, `levers_regression.csv`, `fig5` | supporting material for §5.4 |
| `11_oecd_claims.py` | `oecd_claim_tests.csv` + stdout | **§4.7** (44.7%, 17.5%, 13.1/24.4, −2.2 pp, 581.0/570.9/613.7, −1.76) |
| `12_replicate_xu.py` | `replicate_xu2026.csv` | §1 literature comparison (qualitative) |
| `13_school_constructs.py` | `school_constructs.csv`, rewrites `data/qci_school.parquet` | §3.2 construct definitions and alpha; §4.2 (372 schools, 3.22, SD 0.67, 89.4%, 81.4%) |
| `14_mlm_ladder.py` | `mlm_ladder.csv`, `mlm_lrt_slope.csv` | **§4.1 Table 1 Panel A** (ICC 0.426→0.353; tau00 3157.0→2094.5); §4.3 slope variance 46.02 → 46.01 → 42.35 |
| `15_school_models.py` | `school_models.csv`, `school_r2.csv` | §4.3 R² 26.7 → 27.8 (ΔR² 0.011, interaction 0.0008); the joint-model sample (11,734 / 360) |
| `16_moderator_scan.py` | `moderator_scan.csv` | **§4.3 Table 2** (29 moderators; range, median, min p, BH q, SE spread, MDES) |
| `17_school_aggregate_scan.py` | `school_aggregate_scan.csv`, `suppression_check.csv` | §4.1 note and §4.2 (4.79 univariate; −0.18; 13.2 / −10.6) |
| `18_climate_focus.py` | `climate_focus.csv`, `climate_components.csv` | §4.6 single-level climate x AI with school means (−2.55) |
| `19_level_decomposition.py` | `level_decomposition.csv`, `climate_components.csv` | §4.6 within +0.12 / between −26.3 |
| `20_mundlak_within.py` | `mundlak.csv`, `within_uses.csv` | §4.4 Mundlak; **§4.5 Table 3** (four purposes, within school) |
| `21_level_id_correct.py` | `level_id.csv`, `uses_within.csv` | §4.3 three slopes (−2.99 / −3.94 / −4.19); §4.4 daily-use ladder (−34.4 / −50.4); §4.6 D1/D2/D3 interactions |
| `22_figures_school.py` | `school_main.csv`, `mlm_variance.csv`, `moderator_all.csv`, `fig6`–`fig9` | **§4.1 Table 1 Panel B** (26.8, 13.1, 6.81, −6.63, −6.07, ≤1.1) |
| `23_figures_fix.py` | `fig6`–`fig10` | figure styling |
| `25_paper_audit.py` | `fig9_coefs.csv`, `school_mean_reliability.csv`, `suppression_check.csv`, `paper_number_audit.csv` | §3.2 component reliabilities 0.64–0.79; §4.1 3.62 (SE 2.57) |
| `26_estimator_check.py` | `estimator_check.csv` | **Appendix A** (1.19e−6; 2945.581 vs 2945.579) |
| `verification/verify_numbers.py` | `verification/number_check.csv` | the 211-row audit |
| `verification/repro_mundlak_daily.py` | stdout | §4.4 Mundlak −46.10 (SE 5.66); §4.1 −3.35 (p = .128) |

`pisa_engine.py` (weighted two-level estimator, PV × BRR-Fay) and `wmlm.py`
(Woodbury-identity multilevel likelihood) are libraries, not scripts.

## Mapping from manuscript sections to the check rows

`verification/number_check.csv` carries a `section` column with the manuscript
location of each number. Filter on it to inspect one section:

```python
import pandas as pd
d = pd.read_csv("verification/number_check.csv")
d[d.section == "4.3"][["claim", "paper_value", "recomputed", "status", "source"]]
```

| Section | Rows | What is checked |
|---|---|---|
| 1 | 1 | OECD benchmark (~20 points), external |
| 3.1 | 17 | sample sizes, weighted descriptives |
| 3.2 | 9 | construct count, covariates, alpha, reliability |
| 4.1 | 42 | ICC/tau00 ladder, Table 1 Panel B, suppression values |
| 4.2 | 10 | scale distribution, coverage, collinearity |
| 4.3 | 51 | slope, variance components, R², 29 moderators, power |
| 4.4 | 24 | three estimators, frequency ladder, heavy-user profile |
| 4.5 | 17 | four purposes, single-level comparison, correlations |
| 4.6 | 17 | three identification strategies for climate x AI |
| 4.7 | 12 | critical-use access and stratified means |
| 5.1 / 5.4.4 / 5.5 / A | 11 | derived quantities, Appendix A benchmarks |
