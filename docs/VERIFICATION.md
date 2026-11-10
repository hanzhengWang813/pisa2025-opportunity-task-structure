# Verification report

Scope: every number printed in the manuscript — coefficients, standard errors,
p-values, variance components, sample sizes, percentages, derived quantities.
Method: recompute each one from the archived result tables or from the microdata and
compare at the precision printed.

Result file: `verification/number_check.csv` (211 rows).
Regenerate: `python verification/verify_numbers.py`.

## Summary

| status | rows |
|---|---|
| `reproduced` — matches to printed precision | 203 |
| `last-digit` — printed SE rounds 0.01 off the stored value | 5 |
| `rounding` — paper rounds a hedged quantity | 2 |
| `external` — OECD benchmark, not in these data | 1 |
| **total** | **211** |

No coefficient, standard error, *p*-value, variance component, sample size, share or
percentage in the manuscript is untraceable, and no substantive estimate is off by
more than a rounding unit.

## Items that are not exact

### Five standard errors, last digit (Table 2 and §4.6)

| Location | Printed SE | Stored SE | Printed p | p implied by stored SE |
|---|---|---|---|---|
| Table 2, digital infrastructure | 1.18 | 1.16805 | .524 | .524 |
| Table 2, teacher capacity and time | 1.07 | 1.06420 | .315 | .315 |
| Table 2, school rules | 1.48 | 1.47059 | .686 | .686 |
| Table 2, readiness for digital teaching | 1.11 | 1.11999 | .467 | .467 |
| §4.6, climate × AI with school means | 1.01 | 1.00493 | .011 | .011 |

In all five cases the printed *p*-value agrees with the **stored** standard error, not
with the printed one, so the standard-error column carries a one-unit last-digit
transcription slip. No test statistic, threshold or conclusion changes. Fix by
rounding the stored values (1.17, 1.06, 1.47, 1.12, 1.00).

### Two rounded hedged quantities

| Location | Printed | Recomputed | Why |
|---|---|---|---|
| §3.1 | "about 33 students per school" | 33.1 | hedged wording |
| §4.1 | "cumulative total of 7.3 percentage points" | 7.23 | 0.426 − 0.353 from the rounded ICCs in Table 1; the unrounded drop is 0.42570 − 0.35340 = 7.23 |

The ICC drop is internally consistent with the table but 0.07 pp high. If the sentence
is kept, "about 7 percentage points" or "7.2" both match the variance components.

### One external value

§1, "a gap of about 20 score points between students who used AI for specific
schoolwork tasks and students who did not" is an OECD (2026) Vol. I figure. The
release package contains no variable that supports or contradicts it.

## Two coefficients that were re-derived

These appear in the manuscript but are **not** stored as rows in `outputs/`, so they
were re-estimated from the microdata (`verification/repro_mundlak_daily.py`):

| Location | Printed | Re-derived | Note |
|---|---|---|---|
| §4.4, Mundlak within-school component for daily use | −46.1 (SE 5.7) | **−46.10 (SE 5.66)** | `outputs/level_id.csv` and `outputs/mundlak.csv` store the continuous-intensity Mundlak (−51.88), a different quantity. The daily variant centres the binary dummy on its unweighted school mean, exactly as §4.4 describes. |
| §4.1, univariate coefficient for organization of digital teaching | −3.35 (p = .128) | **−3.35 (p = .128)** | single school-level predictor plus the 12 student covariates and AI use intensity (n = 12,138) |

Both reproduce. Adding these two rows to `outputs/` would remove the only gap between
the manuscript and the archived tables.

## Wording notes (no arithmetic error)

- **§3.1 descriptives.** "Mean 597.1 (SD = 87.5)" is the first plausible value
  (`PV1SCIE`: 597.12, 87.53). The average over all ten plausible values is 597.0 with
  a pooled SD of 87.8. Either pair is defensible; the paper mixes neither.
- **§4.4 profile.** The five gaps (+0.78, +0.86, +0.95, +1.06, +0.43) are weighted
  mean differences on the original index scales. Each index has a full-sample SD close
  to 1.0 (0.96, 1.03, 0.96, 0.89), so reading them as SD units holds to ±0.05 for four
  of five; perseverance would be +1.19 in exact SD units. Family ESCS is already
  standardized, so +0.43 is exact.
- **§4.3 power paragraph.** "3.3 points" and "5.3 points" are
  (1.96 + 0.8416) × median and maximum interaction SE (1.168, 1.877) → 3.27 and 5.26.

## Pipeline reproducibility

Executed from a relocated root with `PISA_BASE` set (data in a different directory,
outputs written to the repository):

| Script | Comparison with the archived table |
|---|---|
| `26_estimator_check.py` | all six benchmark checks pass; identical values to `outputs/estimator_check.csv` |
| `13_school_constructs.py` | `school_constructs.csv` identical, maximum absolute difference 0.0 |

`outputs/` in this repository is the set of tables the manuscript was written from.
Re-running the full pipeline regenerates them; the two spot checks above confirm the
patched scripts reproduce the archived values exactly.

## What is not checked here

- The OECD benchmark values (20 points; 44.6% for China in Vol. I).
- The literature claims in §1 and §2 (Xu & Schleicher, Li & Xin, Wu et al.,
  Strömberg et al., Lepe & Contreras) — references, not computations.
- Causal interpretation. All estimates are cross-sectional associations, as the
  manuscript states.
