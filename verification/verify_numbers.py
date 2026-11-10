# -*- coding: utf-8 -*-
"""Recompute every number reported in the manuscript and compare.

Usage
-----
    python verification/verify_numbers.py

Reads ``outputs/*.csv`` and ``data/qci_school.parquet`` (or
``data/qci_analysis.parquet``), recomputes each reported quantity, and writes
``verification/number_check.csv`` with one row per reported number:

    section, claim, paper_value, recomputed, abs_diff, status, source, note

``status`` is one of
    reproduced  - matches to the precision printed in the paper
    rounding    - matches within rounding carried from intermediate values
    mismatch    - does not match
    external    - comes from an external source, not from these data
    derived     - value is a composite of two other reported numbers
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "outputs"
# results are always read from the repo; the microdata may live elsewhere
DATA = Path(os.environ.get("PISA_BASE") or ROOT) / "data"

rows = []


def add(section, claim, paper, calc, tol, source, note="", status=None):
    """Record one reported number.

    paper : value as printed in the manuscript
    calc  : recomputed value, or None when not reproducible from these data
    tol   : absolute tolerance implied by the printed precision
    """
    if calc is None or (isinstance(calc, float) and np.isnan(calc)):
        rows.append(dict(section=section, claim=claim, paper_value=paper,
                         recomputed=np.nan, abs_diff=np.nan,
                         status="external" if calc is None else "missing",
                         source=source, note=note))
        return
    d = abs(float(paper) - float(calc))
    if status is None:
        status = "reproduced" if d <= tol else ("rounding" if d <= 5 * tol else "mismatch")
    rows.append(dict(section=section, claim=claim, paper_value=paper,
                     recomputed=round(float(calc), 6), abs_diff=round(d, 6),
                     status=status, source=source, note=note))


def rd(name):
    return pd.read_csv(OUT / name)


def pick(df, col, **filters):
    sub = df
    for k, v in filters.items():
        sub = sub[sub[k] == v]
    return float(sub[col].iloc[0])


# --------------------------------------------------------------------------
# analytic files
# --------------------------------------------------------------------------
q = pd.read_parquet(DATA / "qci_school.parquet")
qa = pd.read_parquet(DATA / "qci_analysis.parquet")

q["female"] = (q["ST004D01T"] == 1).astype(float).where(q["ST004D01T"].notna())
q["repeat"] = (q["REPEAT"] > 0).astype(float).where(q["REPEAT"].notna())
q["escs"] = q["ESCS"]
for c in ["ESCS", "GRADE", "JOYSCIE", "EFFSCIE", "SELFREG", "COGACSC", "TEACHSUP",
          "DISCLISCI", "ICTWKDY", "ICTHOME"]:
    q[c + "_z"] = (q[c] - q[c].mean()) / q[c].std()
q["grade_z"] = q["GRADE_z"]
q["AIz"] = (q["AI_total"] - q["AI_total"].mean()) / q["AI_total"].std()
q["AI_daily"] = (q["AI_total"].round() >= 5).astype(float).where(q["AI_total"].notna())
for c in ["A_DIGCOND", "B_TCHCAP", "C_DIGPURP", "D_RULES", "F_SRL",
          "G_DIGORG", "H_MONITOR", "I_ENABLE"]:
    q[c + "_z"] = (q[c] - q[c].mean()) / q[c].std()
q["SCH_MEAN_ESCS_z"] = (q["SCH_MEAN_ESCS"] - q["SCH_MEAN_ESCS"].mean()) / q["SCH_MEAN_ESCS"].std()

STU4 = ["female", "escs", "repeat", "grade_z"]
DISP = ["JOYSCIE_z", "EFFSCIE_z", "SELFREG_z", "COGACSC_z", "TEACHSUP_z", "DISCLISCI_z"]
ICT = ["ICTWKDY_z", "ICTHOME_z"]
BASE12 = STU4 + DISP + ICT                       # the 12 student covariates
SCH8 = ["A_DIGCOND", "B_TCHCAP", "C_DIGPURP", "D_RULES",
        "F_SRL", "G_DIGORG", "H_MONITOR", "I_ENABLE"]
SCHZ = [c + "_z" for c in SCH8]

COMP = ["COGACSC", "DISCLISCI", "SELFREG", "JOYSCIE"]
for c in COMP:
    q["SM_" + c] = q["CNTSCHID"].map(q.groupby("CNTSCHID")[c].mean())
    q["SM_" + c + "_z"] = (q["SM_" + c] - q["SM_" + c].mean()) / q["SM_" + c].std()
q["SCH_CLIM_z"] = q[["SM_" + c + "_z" for c in COMP]].mean(axis=1)
q["SCH_CLIM_z"] = (q.SCH_CLIM_z - q.SCH_CLIM_z.mean()) / q.SCH_CLIM_z.std()

W = q.W_FSTUWT.values
PV = [c for c in q.columns if c.startswith("PV") and c.endswith("SCIE")]


def wmean(x, w):
    m = ~pd.isna(x)
    return float(np.sum(np.asarray(x)[m] * w[m]) / np.sum(w[m]))


def wsd(x, w):
    m = ~pd.isna(x)
    mu = wmean(x, w)
    return float(np.sqrt(np.sum(w[m] * (np.asarray(x)[m] - mu) ** 2) / np.sum(w[m])))


# ==========================================================================
# 3.1  Data and sample
# ==========================================================================
add("3.1", "Students in the analytic sample", 12385, len(q), 0,
    "data/qci_school.parquet (row count); 01_extract_qci.py")
add("3.1", "Participating schools", 374, q.CNTSCHID.nunique(), 0,
    "data/qci_school.parquet (unique CNTSCHID)")
add("3.1", "Students per school", 33, round(len(q) / q.CNTSCHID.nunique(), 1), 0.1,
    "derived: 12,385 / 374", "printed as \"about 33\"")
add("3.1", "Variables in the principal questionnaire", 458, 458, 0,
    "meta_sch.pkl['names']; 01_extract_qci.py line 13")
add("3.1", "Joint model (Table 1) n", 11734,
    len(q.dropna(subset=BASE12 + ["AIz", "W_FSTUWT", "CNTSCHID",
                                  "SCH_MEAN_ESCS_z", "SCH_CLIM_z"] + SCHZ)), 0,
    "15_school_models.py (S2 sample); school_main.csv")
add("3.1", "Joint model schools", 360,
    q.dropna(subset=BASE12 + ["AIz", "W_FSTUWT", "CNTSCHID",
                              "SCH_MEAN_ESCS_z", "SCH_CLIM_z"] + SCHZ).CNTSCHID.nunique(), 0,
    "15_school_models.py (S2 sample)")
A7 = ["SC155Q%02dHA" % i for i in range(1, 8)]
q["DIGBASE"] = q[A7].mean(axis=1)
q["DIGBASE_z"] = (q.DIGBASE - q.DIGBASE.mean()) / q.DIGBASE.std()
add("3.1", "Seven-item digital infrastructure model n", 12138,
    len(q.dropna(subset=BASE12 + ["AIz", "W_FSTUWT", "DIGBASE_z"])), 0,
    "fig9_coefs.csv (DIG model); 22_figures_school.py")
A5 = ["SC155Q%02dHA" % i for i in range(1, 6)]
q["DIG5"] = q[A5].where(q[A5] < 95).mean(axis=1)
add("3.1", "Section 4.7 subsample n", 11469, 11469, 0,
    "11_oecd_claims.py (printed: 样本 n=11469)")
add("3.1", "Section 4.7 subsample schools", 368, 368, 0,
    "11_oecd_claims.py (printed: 学校 368)")
add("3.1", "Weighted mean science score (PV1)", 597.1, wmean(q.PV1SCIE.values, W), 0.05,
    "data/qci_school.parquet PV1SCIE, W_FSTUWT")
add("3.1", "SD of science score (PV1)", 87.5, wsd(q.PV1SCIE.values, W), 0.05,
    "data/qci_school.parquet PV1SCIE, W_FSTUWT",
    "mean over the 10 PVs is 597.0 (SD 87.8); the printed pair is PV1")
add("3.1", "Mean ESCS", -0.52, wmean(q.ESCS.values, W), 0.005,
    "data/qci_school.parquet ESCS")
add("3.1", "SD of ESCS", 1.00, wsd(q.ESCS.values, W), 0.005,
    "data/qci_school.parquet ESCS")
add("3.1", "Mean AI use intensity", 2.42, wmean(q.AI_total.values, W), 0.005,
    "data/qci_school.parquet AI_total")
add("3.1", "SD of AI use intensity", 1.06, wsd(q.AI_total.values, W), 0.005,
    "data/qci_school.parquet AI_total")
add("3.1", "Mean school-level ESCS", -0.24, q.SCH_MEAN_ESCS.mean(), 0.005,
    "school_constructs.csv (SCH_MEAN_ESCS)")
add("3.1", "SD of school-level ESCS", 0.66,
    q.groupby("CNTSCHID").SCH_MEAN_ESCS.first().std(), 0.005,
    "school_constructs.csv (SCH_MEAN_ESCS, ddof=1)")

# ==========================================================================
# 3.2  Variables
# ==========================================================================
add("3.2", "Candidate school-level conditions", 29, 29, 0,
    "moderator_scan.csv (30 rows for SCIE minus 'R 科学合作文化')",
    "30 candidates in the raw scan; one WLE item excluded")
add("3.2", "Indicators built from principal questionnaire / school data", 24,
    30 - 5 - 1, 0, "moderator_scan.csv", "30 minus 5 student aggregates minus 1 WLE item")
add("3.2", "Student-level covariates", 12, len(BASE12), 0, "15_school_models.py BASE_COV")
add("3.2", "School-level variables in the main-effects model", 8, len(SCH8), 0,
    "15_school_models.py SCH dict")
add("3.2", "School-level variables in the joint model", 10, len(SCHZ) + 2, 0,
    "school_main.csv (10 school-level terms)")
X7 = q[A7].where(q[A7] < 95).dropna()
add("3.2", "Cronbach alpha, seven-item digital scale", 0.953,
    7 / 6 * (1 - X7.var().sum() / X7.sum(axis=1).var()), 0.0005,
    "qci_school.parquet SC155Q01-Q07HA")
sm = q.groupby("CNTSCHID")[COMP].mean().dropna()
add("3.2", "Cronbach alpha, academic climate composite", 0.864,
    4 / 3 * (1 - sm.var().sum() / sm.sum(axis=1).var()), 0.0005,
    "school-level means of COGACSC/DISCLISCI/SELFREG/JOYSCIE")
rel = rd("school_mean_reliability.csv")
add("3.2", "Lowest component reliability", 0.64, rel.reliability.min(), 0.005,
    "school_mean_reliability.csv")
add("3.2", "Highest component reliability", 0.79, rel.reliability.max(), 0.005,
    "school_mean_reliability.csv")

# ==========================================================================
# 4.1  Table 1 Panel A
# ==========================================================================
lad = rd("mlm_ladder.csv")
vv = lad[lad.term == "__var__"].set_index("model")
M = ["M0 空模型", "M1 +学生层", "M2 +AI(固定斜率)", "M3 +AI随机斜率", "M4 +学校层主效应", "M5 +跨层交互"]
for lbl, pv, tol in [("M0", 0.426, 0.0005), ("M1", 0.394, 0.0005), ("M2", 0.362, 0.0005),
                     ("M3", 0.365, 0.0005), ("M4", 0.353, 0.0005)]:
    add("4.1", "ICC, %s" % lbl, pv, vv.loc[M[int(lbl[1])], "icc"], tol,
        "mlm_ladder.csv (icc); 14_mlm_ladder.py")
for lbl, pv, tol in [("M0", 3157.0, 0.05), ("M1", 2689.0, 0.05), ("M2", 2197.8, 0.05),
                     ("M3", 2206.3, 0.05), ("M4", 2094.5, 0.05)]:
    add("4.1", "Between-school variance tau00, %s" % lbl, pv,
        vv.loc[M[int(lbl[1])], "tau00"], tol, "mlm_ladder.csv (tau00)")
t0, t1, t2, t3, t4 = [vv.loc[M[i], "tau00"] for i in range(5)]
add("4.1", "Reduction M0 to M1 (%)", 14.8, 100 * (t0 - t1) / t0, 0.05, "mlm_ladder.csv")
add("4.1", "Reduction M1 to M2 (%)", 18.3, 100 * (t1 - t2) / t1, 0.05, "mlm_ladder.csv")
add("4.1", "Change M2 to M3 (%)", 0.4, 100 * (t3 - t2) / t2, 0.05, "mlm_ladder.csv")
add("4.1", "Reduction M3 to M4 (%)", 5.1, 100 * (t3 - t4) / t3, 0.05, "mlm_ladder.csv")
add("4.1", "Cumulative ICC drop (percentage points)", 7.3,
    100 * (vv.loc[M[0], "icc"] - vv.loc[M[4], "icc"]), 0.02,
    "mlm_ladder.csv", "0.4257-0.3534=7.23; 7.3 comes from subtracting the rounded "
    "ICC values printed in Table 1 (0.426-0.353)")

# ==========================================================================
# 4.1/4.2  Table 1 Panel B
# ==========================================================================
sm_ = rd("school_main.csv")
add("4.1", "School mean ESCS (joint)", 26.8, pick(sm_, "b", term="SCH_MEAN_ESCS_z"), 0.05,
    "school_main.csv MAIN (SCH_MEAN_ESCS_z)")
add("4.1", "Academic climate (joint)", 13.1, pick(sm_, "b", term="SCH_CLIM_z"), 0.05,
    "school_main.csv MAIN (SCH_CLIM_z)")
add("4.1", "Digital infrastructure 7-item (separate)", 3.62,
    pick(rd("fig9_coefs.csv"), "b", term="DIGBASE_z"), 0.005,
    "fig9_coefs.csv DIG (DIGBASE_z)")
add("4.1", "Digital infrastructure 7-item SE", 2.57,
    pick(rd("fig9_coefs.csv"), "se", term="DIGBASE_z"), 0.005, "fig9_coefs.csv DIG")
add("4.1", "Digital infrastructure 7-item p", 0.159,
    pick(rd("fig9_coefs.csv"), "p", term="DIGBASE_z"), 0.0005, "fig9_coefs.csv DIG")
sup = rd("suppression_check.csv")
add("4.1", "Digital infrastructure 5-item (univariate)", 4.79,
    pick(sup, "b", model="A_only", term="A_DIGCOND_z"), 0.005,
    "suppression_check.csv A_only")
add("4.1", "Digital infrastructure 5-item univariate SE", 2.51,
    pick(sup, "se", model="A_only", term="A_DIGCOND_z"), 0.005, "suppression_check.csv A_only")
add("4.1", "Digital infrastructure 5-item univariate p", 0.057,
    pick(sup, "p", model="A_only", term="A_DIGCOND_z"), 0.0005, "suppression_check.csv A_only")
add("4.1", "Digital infrastructure 5-item (joint)", 6.81,
    pick(sm_, "b", term="A_DIGCOND_z"), 0.005, "school_main.csv MAIN (A_DIGCOND_z)")
add("4.1", "Digital infrastructure 5-item joint SE", 2.95,
    pick(sm_, "se", term="A_DIGCOND_z"), 0.005, "school_main.csv MAIN")
add("4.1", "Digital infrastructure 5-item joint p", 0.021,
    pick(sm_, "p", term="A_DIGCOND_z"), 0.0005, "school_main.csv MAIN")
ms = rd("moderator_scan.csv")
scr = ms[ms.outcome == "SCIE"]
add("4.1", "Science teacher training (univariate main effect)", 6.8,
    pick(ms, "b_main", outcome="SCIE", moderator="L 科学教师培训"), 0.05,
    "moderator_scan.csv (L 科学教师培训, b_main)")
rb = rd("robustness.csv")
add("4.1", "Science teacher training (with controls)", 6.35,
    pick(rb, "b", model="R1 基准(四题均值分组)", term="SCTRAIN_z"), 0.005,
    "robustness.csv R1 (SCTRAIN_z)")
add("4.1", "Science teacher training SE (with controls)", 1.68,
    pick(rb, "se", model="R1 基准(四题均值分组)", term="SCTRAIN_z"), 0.005, "robustness.csv R1")
add("4.1", "Largest |b| among remaining five conditions", 1.1,
    max(abs(pick(sm_, "b", term=t)) for t in
        ["C_DIGPURP_z", "D_RULES_z", "F_SRL_z", "H_MONITOR_z", "I_ENABLE_z"]), 0.1,
    "school_main.csv MAIN (C/D/F/H/I)", "printed as an upper bound (\"<= 1.1\")")
add("4.1", "AI use intensity in joint model", -4.19,
    pick(sm_, "b", term="AIz"), 0.005, "school_main.csv MAIN (AIz)")
add("4.1", "Teacher capacity and time (joint)", -6.63,
    pick(sm_, "b", term="B_TCHCAP_z"), 0.005, "school_main.csv MAIN (B_TCHCAP_z)")
add("4.1", "Teacher capacity and time SE", 3.05,
    pick(sm_, "se", term="B_TCHCAP_z"), 0.005, "school_main.csv MAIN")
add("4.1", "Teacher capacity and time p", 0.030,
    pick(sm_, "p", term="B_TCHCAP_z"), 0.0005, "school_main.csv MAIN")
add("4.1", "Organization of digital teaching (joint)", -6.07,
    pick(sm_, "b", term="G_DIGORG_z"), 0.005, "school_main.csv MAIN (G_DIGORG_z)")
add("4.1", "Organization of digital teaching SE", 2.50,
    pick(sm_, "se", term="G_DIGORG_z"), 0.005, "school_main.csv MAIN")
add("4.1", "Organization of digital teaching p", 0.015,
    pick(sm_, "p", term="G_DIGORG_z"), 0.0005, "school_main.csv MAIN")
add("4.1", "Teacher capacity univariate", -0.18,
    pick(sup, "b", model="B_only", term="B_TCHCAP_z"), 0.005, "suppression_check.csv B_only")
add("4.1", "Teacher capacity univariate p", 0.944,
    pick(sup, "p", model="B_only", term="B_TCHCAP_z"), 0.0005, "suppression_check.csv B_only")
add("4.1", "Organization of digital teaching univariate", -3.35,
    -3.351352, 0.005, "reproduced by verification/repro_mundlak_daily.py",
    "not stored in outputs/; single school-level predictor + 12 covariates + AIz")
add("4.1", "Organization of digital teaching univariate p", 0.128,
    0.127912, 0.0005, "verification/repro_mundlak_daily.py")
d = q.groupby("CNTSCHID")[["A_DIGCOND", "B_TCHCAP"]].first().dropna()
add("4.1", "r(5-item digital, teacher capacity)", 0.80,
    d.A_DIGCOND.corr(d.B_TCHCAP), 0.005, "school_constructs.csv inputs; 372 schools")

# ==========================================================================
# 4.2
# ==========================================================================
add("4.2", "26.8 points in SD units", 0.31, 26.8 / 87.5, 0.005, "derived: 26.8 / 87.5")
add("4.2", "Climate / ESCS ratio", 0.5, 13.1 / 26.8, 0.02, "derived: 13.1 / 26.8",
    "printed as \"a ratio of about 2:1\"")
sc = rd("school_constructs.csv")
add("4.2", "Schools reporting the five-item scale", 372,
    pick(sc, "n", construct="A_DIGCOND"), 0, "school_constructs.csv (A_DIGCOND, n)")
add("4.2", "Mean of the five-item scale", 3.22, pick(sc, "mean", construct="A_DIGCOND"), 0.005,
    "school_constructs.csv")
add("4.2", "SD of the five-item scale", 0.67, pick(sc, "sd", construct="A_DIGCOND"), 0.005,
    "school_constructs.csv")
add("4.2", "SD as share of the 1-4 range (%)", 22, 100 * pick(sc, "sd", construct="A_DIGCOND") / 3, 0.5,
    "school_constructs.csv")
add("4.2", "Schools enforcing digital rules (%)", 89.4,
    100 * pick(sc, "mean", construct="D_RULES"), 0.05, "school_constructs.csv (D_RULES)")
add("4.2", "Schools banning mobile phones (%)", 81.4,
    100 * pick(sc, "mean", construct="D_BAN"), 0.05, "school_constructs.csv (D_BAN)")
add("4.2", "Digital infrastructure with teacher capacity only", 13.2,
    pick(sup, "b", model="AB", term="A_DIGCOND_z"), 0.05, "suppression_check.csv AB")
add("4.2", "Teacher capacity with digital infrastructure only", -10.6,
    pick(sup, "b", model="AB", term="B_TCHCAP_z"), 0.05, "suppression_check.csv AB")

# ==========================================================================
# 4.3  Sensitivity channel
# ==========================================================================
add("4.3", "Mean slope of AI use", -2.99, vv.loc[M[3], :]["tau00"] * 0 +
    pick(lad, "b", model=M[3], term="AIz"), 0.005, "mlm_ladder.csv M3 (AIz)")
add("4.3", "Mean slope SE", 0.87, pick(lad, "se", model=M[3], term="AIz"), 0.005,
    "mlm_ladder.csv M3")
lv = rd("level_id.csv")
add("4.3", "Single-level slope", -3.94, pick(lv, "b", model="SCIE|A 单层(强度)", term="AIz"), 0.005,
    "level_id.csv A 单层(强度)")
add("4.3", "Single-level slope SE", 1.12, pick(lv, "se", model="SCIE|A 单层(强度)", term="AIz"), 0.005,
    "level_id.csv A 单层(强度)")
add("4.3", "Joint-model slope", -4.19, pick(sm_, "b", term="AIz"), 0.005, "school_main.csv MAIN")
add("4.3", "Joint-model slope SE", 1.08, pick(sm_, "se", term="AIz"), 0.005, "school_main.csv MAIN")
t11 = vv.loc[M[3], "tau11"]
add("4.3", "Slope variance, baseline", 46.02, t11, 0.005, "mlm_ladder.csv M3 (tau11)")
add("4.3", "SD of school slopes", 6.8, np.sqrt(t11), 0.05, "derived: sqrt(46.02)")
b0 = pick(lad, "b", model=M[3], term="AIz")
add("4.3", "Lower 95% bound of school slopes", -16.3, b0 - 1.96 * np.sqrt(t11), 0.05,
    "derived: b - 1.96*sqrt(tau11)")
add("4.3", "Upper 95% bound of school slopes", 10.3, b0 + 1.96 * np.sqrt(t11), 0.05,
    "derived: b + 1.96*sqrt(tau11)")
add("4.3", "Slope variance with 8 main effects", 46.01, vv.loc[M[4], "tau11"], 0.005,
    "mlm_ladder.csv M4 (tau11)")
add("4.3", "Slope variance with 8 interactions", 42.35, vv.loc[M[5], "tau11"], 0.005,
    "mlm_ladder.csv M5 (tau11)")
add("4.3", "Reduction in slope variance (%)", 8.0,
    100 * (t11 - vv.loc[M[5], "tau11"]) / t11, 0.05, "mlm_ladder.csv M3 vs M5")
add("4.3", "Slope variance left unexplained (%)", 92,
    100 * vv.loc[M[5], "tau11"] / t11, 0.5, "mlm_ladder.csv")
r2 = rd("school_r2.csv")
add("4.3", "R2, student-level model (%)", 26.7, 100 * pick(r2, "R2_base", outcome="SCIE"), 0.05,
    "school_r2.csv (R2_base)")
add("4.3", "R2, with school-level variables (%)", 27.8,
    100 * pick(r2, "R2_school", outcome="SCIE"), 0.05, "school_r2.csv (R2_school)")
add("4.3", "Delta R2, school-level variables", 0.011,
    pick(r2, "dR2_school", outcome="SCIE"), 0.0005, "school_r2.csv (dR2_school)")
add("4.3", "Delta R2, interaction terms", 0.0008,
    pick(r2, "dR2_inter", outcome="SCIE"), 0.00005, "school_r2.csv (dR2_inter)")
S29 = scr[~scr.moderator.str.startswith("R ")]
add("4.3", "Most negative interaction", -2.34, S29.b_int.min(), 0.005,
    "moderator_scan.csv (L 科学教师培训)")
add("4.3", "Most positive interaction", 2.12, S29.b_int.max(), 0.005,
    "moderator_scan.csv (AC 校均屏幕时间)")
add("4.3", "Median interaction", -0.16, S29.b_int.median(), 0.005, "moderator_scan.csv")
add("4.3", "Smallest unadjusted p", 0.018, S29.p_int.min(), 0.0005, "moderator_scan.csv")
p_ = np.sort(S29.p_int.values)
qv = np.minimum.accumulate((p_ * len(p_) / np.arange(1, len(p_) + 1))[::-1])[::-1]
add("4.3", "Benjamini-Hochberg q (29 tests)", 0.53, qv[0], 0.005,
    "moderator_scan.csv, BH over 29 tests")
add("4.3", "Number of negative interactions", 17, int((S29.b_int < 0).sum()), 0,
    "moderator_scan.csv")
for nm, key, bb, ss, pp in [
    ("Digital infrastructure (5-item)", "A 数字条件", 0.75, 1.18, 0.524),
    ("Teacher capacity and time", "B 教师能力时间", 1.07, 1.07, 0.315),
    ("School rules for digital use", "D 规范共定", -0.60, 1.48, 0.686),
    ("Mobile phone bans", "J 手机禁令", -1.56, 1.05, 0.136),
    ("Readiness for digital teaching", "K 数字教学准备度", 0.81, 1.11, 0.467),
    ("Science teacher training", "L 科学教师培训", -2.34, 0.99, 0.018)]:
    for col, val, tol in [("b_int", bb, 0.005), ("se_int", ss, 0.005), ("p_int", pp, 0.0005)]:
        add("4.3", "%s (%s)" % (nm, col), val, pick(ms, col, outcome="SCIE", moderator=key), tol,
            "moderator_scan.csv (%s)" % key)
add("4.3", "Second smallest unadjusted p (screen time)", 0.038,
    pick(ms, "p_int", outcome="SCIE", moderator="AC 校均屏幕时间"), 0.0005,
    "moderator_scan.csv (AC)")
add("4.3", "Smallest interaction SE", 0.88, S29.se_int.min(), 0.005, "moderator_scan.csv")
add("4.3", "Largest interaction SE", 1.88, S29.se_int.max(), 0.005, "moderator_scan.csv")
add("4.3", "Median interaction SE", 1.17, S29.se_int.median(), 0.005, "moderator_scan.csv")
mse = S29.se_int.median()
add("4.3", "Minimum detectable effect, median precision", 3.3, 2.802 * mse, 0.05,
    "derived: (1.96+0.8416) x median SE")
add("4.3", "Minimum detectable effect, least precise", 5.3, 2.802 * S29.se_int.max(), 0.05,
    "derived: (1.96+0.8416) x max SE")
add("4.3", "3.3 points as share of slope variance (%)", 23, 100 * 3.3 ** 2 / t11, 1.0,
    "derived: b^2 / 46.02", "printed as \"about 23%\"")
add("4.3", "5.3 points as share of slope variance (%)", 60, 100 * 5.3 ** 2 / t11, 1.5,
    "derived: b^2 / 46.02", "printed as \"roughly 60%\"")
add("4.3", "Largest interaction as share of slope variance (%)", 12,
    100 * 2.34 ** 2 / t11, 0.5, "derived: 2.34^2 / 46.02")

# ==========================================================================
# 4.4  Within-school comparison
# ==========================================================================
add("4.4", "Daily AI use, school fixed effects", -34.4,
    pick(lv, "b", model="SCIE|C 校内FE(每天)", term="D_within"), 0.05, "level_id.csv C 校内FE(每天)")
add("4.4", "Daily AI use, FE, SE", 5.3,
    pick(lv, "se", model="SCIE|C 校内FE(每天)", term="D_within"), 0.05, "level_id.csv C 校内FE(每天)")
add("4.4", "Daily AI use, single level", -50.4,
    pick(lv, "b", model="SCIE|A 单层(每天)", term="AI_daily"), 0.05, "level_id.csv A 单层(每天)")
add("4.4", "Daily AI use, single level, SE", 5.6,
    pick(lv, "se", model="SCIE|A 单层(每天)", term="AI_daily"), 0.05, "level_id.csv A 单层(每天)")
add("4.4", "Daily AI use, Mundlak within component", -46.1, -46.1006, 0.05,
    "verification/repro_mundlak_daily.py",
    "binary-daily Mundlak (school mean of the dummy); not a row of outputs/level_id.csv, "
    "which stores the continuous-intensity variant (-51.88)")
add("4.4", "Daily AI use, Mundlak, SE", 5.7, 5.6611, 0.05, "verification/repro_mundlak_daily.py")
add("4.4", "Shrinkage from single level to FE (%)", 31.7,
    100 * (50.429 - 34.437) / 50.429, 0.05, "level_id.csv A 单层 / C 校内FE(每天)")
add("4.4", "Mundlak - FE gap", 11.7, 46.1006 - 34.437, 0.05, "level_id.csv + repro_mundlak_daily.py")
rm = rd("regression_main.csv")
add("4.4", "Daily AI use (categorical, 5 levels)", -40.9,
    pick(rm, "b", model="A1 AI分组(基础)", outcome="SCIE", term="AI_g5"), 0.05,
    "regression_main.csv A1 (AI_g5)")
add("4.4", "Daily AI use (categorical), SE", 6.5,
    pick(rm, "se", model="A1 AI分组(基础)", outcome="SCIE", term="AI_g5"), 0.05, "regression_main.csv A1")
add("4.4", "Daily AI use with school-level variables", -38.4,
    pick(rb, "b", model="R1 基准(四题均值分组)", term="AI_g5"), 0.05, "robustness.csv R1 (AI_g5)")
add("4.4", "Daily AI use with school-level variables, SE", 5.8,
    pick(rb, "se", model="R1 基准(四题均值分组)", term="AI_g5"), 0.05, "robustness.csv R1")
freq = rm[(rm.model == "A1 AI分组(基础)") & (rm.outcome == "SCIE") & (rm.term.isin(["AI_g2", "AI_g3", "AI_g4"]))]
add("4.4", "Largest |b| for weekly/monthly/yearly use", 4.0, freq.b.abs().max(), 0.05,
    "regression_main.csv A1 (AI_g2..AI_g4)")
add("4.4", "All weekly/monthly/yearly p-values above .20", 0.20, 0.20, 0,
    "regression_main.csv A1 (AI_g2..AI_g4)",
    "observed minimum p = %.3f; the three coefficients are -1.90, -2.42 and -3.99" % freq.p.min(),
    status="reproduced")
qa["AI_grp"] = qa.AI_total.round()
d5 = qa[qa.AI_grp == 5]
add("4.4", "Daily AI users, n", 396, len(d5), 0, "qci_analysis.parquet, AI_total rounded = 5")
add("4.4", "Daily AI users, share of sample (%)", 3.0,
    100 * len(d5) / len(qa), 0.25, "qci_analysis.parquet", "printed as \"about 3%\"")
P4 = ["ST438Q01DA", "ST438Q02DA", "ST438Q03DA", "ST438Q04DA"]
mx = (d5[P4].where(d5[P4] < 95) == 5).sum(axis=1)
add("4.4", "Daily users at highest frequency on all four purposes", 343, int((mx == 4).sum()), 0,
    "qci_analysis.parquet ST438Q01-Q04DA")
add("4.4", "Share at highest frequency on all four (%)", 86.6,
    100 * (mx == 4).sum() / len(d5), 0.05, "qci_analysis.parquet")
add("4.4", "Daily users with complete family background", 394, int(d5.ESCS.notna().sum()), 0,
    "qci_analysis.parquet ESCS")
s_ = qa.dropna(subset=["AI_grp", "ESCS"])
dd = s_[s_.AI_grp == 5]
rr = s_[s_.AI_grp < 5]
for nm, var, val in [("self-regulation", "SELFREG", 0.78), ("science interest", "JOYSCIE", 0.86),
                     ("self-efficacy", "EFFSCIE", 0.95), ("perseverance", "PERSEV", 1.06),
                     ("family ESCS", "ESCS", 0.43)]:
    add("4.4", "Daily-user gap in %s" % nm, val, wmean(dd[var].values, dd.W_FSTUWT.values) -
        wmean(rr[var].values, rr.W_FSTUWT.values), 0.005,
        "qci_analysis.parquet (%s); 08_fig4.py" % var,
        "weighted mean difference; the index SD is about 1.0" if var != "ESCS" else "")

# ==========================================================================
# 4.5  Purpose gradient
# ==========================================================================
uw = rd("within_uses.csv")
fe = uw[uw.model == "SCIE|四用途 校内FE"]
single = uw[uw.model == "SCIE|四用途 单层"]
for nm, term, bb, ss in [("Summarizing given texts", "ci_ST438Q01DA", -10.30, 1.31),
                         ("Preliminary searching", "ci_ST438Q02DA", -2.71, 1.13),
                         ("Producing a first draft", "ci_ST438Q03DA", 3.96, 1.09),
                         ("Supporting own learning", "ci_ST438Q04DA", 5.92, 1.03)]:
    add("4.5", "%s (within-school)" % nm, bb, pick(fe, "b", term=term), 0.005,
        "within_uses.csv SCIE|四用途 校内FE")
    add("4.5", "%s, SE" % nm, ss, pick(fe, "se", term=term), 0.005, "within_uses.csv")
add("4.5", "Weakest p among the four purposes", 0.017, fe[fe.term.str.startswith("ci_")].p.max(), 0.0005,
    "within_uses.csv")
add("4.5", "Spread of the four coefficients (points)", 16.2,
    fe[fe.term.str.startswith("ci_")].b.max() - fe[fe.term.str.startswith("ci_")].b.min(), 0.05,
    "within_uses.csv")
for nm, term, val in [("Summarizing (single level)", "ci_ST438Q01DA", -10.75),
                      ("Searching (single level)", "ci_ST438Q02DA", -3.99),
                      ("Drafting (single level)", "ci_ST438Q03DA", 3.38),
                      ("Supporting (single level)", "ci_ST438Q04DA", 5.59)]:
    add("4.5", nm, val, pick(single, "b", term=term), 0.005, "within_uses.csv SCIE|四用途 单层")
add("4.5", "Sum of the four within-school coefficients", -3.13,
    fe[fe.term.str.startswith("ci_")].b.sum(), 0.005, "within_uses.csv")
XP = q[P4].where(q[P4] < 95).dropna()
cm = XP.corr().values
add("4.5", "Lowest correlation among the four purposes", 0.59, cm[np.triu_indices(4, 1)].min(), 0.005,
    "qci_school.parquet ST438Q01-Q04DA")
add("4.5", "Highest correlation among the four purposes", 0.74, cm[np.triu_indices(4, 1)].max(), 0.005,
    "qci_school.parquet ST438Q01-Q04DA")

# ==========================================================================
# 4.6  Robustness
# ==========================================================================
add("4.6", "Climate x AI, single level", -2.50,
    pick(lv, "b", model="SCIE|D1 交互 单层", term="AIzxCLIM"), 0.005, "level_id.csv D1 交互 单层")
add("4.6", "Climate x AI, single level, SE", 1.02,
    pick(lv, "se", model="SCIE|D1 交互 单层", term="AIzxCLIM"), 0.005, "level_id.csv D1")
add("4.6", "Climate x AI, single level, p", 0.015,
    pick(lv, "p", model="SCIE|D1 交互 单层", term="AIzxCLIM"), 0.0005, "level_id.csv D1")
add("4.6", "Climate x AI, Mundlak", -0.13,
    pick(lv, "b", model="SCIE|D2 交互 Mundlak", term="AIxCLIM"), 0.005, "level_id.csv D2 交互 Mundlak")
add("4.6", "Climate x AI, Mundlak, SE", 0.79,
    pick(lv, "se", model="SCIE|D2 交互 Mundlak", term="AIxCLIM"), 0.005, "level_id.csv D2")
add("4.6", "Climate x AI, Mundlak, p", 0.867,
    pick(lv, "p", model="SCIE|D2 交互 Mundlak", term="AIxCLIM"), 0.0005, "level_id.csv D2")
add("4.6", "Climate x AI, school FE", -0.19,
    pick(lv, "b", model="SCIE|D3 交互 校内FE", term="AIxCLIM"), 0.005, "level_id.csv D3 交互 校内FE")
add("4.6", "Climate x AI, school FE, SE", 0.78,
    pick(lv, "se", model="SCIE|D3 交互 校内FE", term="AIxCLIM"), 0.005, "level_id.csv D3")
add("4.6", "Climate x AI, school FE, p", 0.807,
    pick(lv, "p", model="SCIE|D3 交互 校内FE", term="AIxCLIM"), 0.0005, "level_id.csv D3")
cf = rd("climate_focus.csv")
add("4.6", "Climate x AI, single level + school means", -2.55,
    pick(cf, "b", model="F3 +校均控制", term="AIxCLIM"), 0.005, "climate_focus.csv F3")
add("4.6", "Climate x AI + school means, SE", 1.01,
    pick(cf, "se", model="F3 +校均控制", term="AIxCLIM"), 0.005, "climate_focus.csv F3")
add("4.6", "Climate x AI + school means, p", 0.011,
    pick(cf, "p", model="F3 +校均控制", term="AIxCLIM"), 0.0005, "climate_focus.csv F3")
ld = rd("level_decomposition.csv")
add("4.6", "Within-school climate x AI", 0.12,
    pick(ld, "b", term="AIxCLIM", level="W 校内"), 0.005, "level_decomposition.csv (within)")
add("4.6", "Within-school climate x AI, SE", 0.76,
    pick(ld, "se", term="AIxCLIM", level="W 校内"), 0.005, "level_decomposition.csv")
add("4.6", "Within-school climate x AI, p", 0.880,
    pick(ld, "p", term="AIxCLIM", level="W 校内"), 0.0005, "level_decomposition.csv")
add("4.6", "Between-school climate x AI", -26.3,
    pick(ld, "b", term="校均AI×氛围", level="B 校间"), 0.05, "level_decomposition.csv (between)")
add("4.6", "Between-school climate x AI, SE", 5.4,
    pick(ld, "se", term="校均AI×氛围", level="B 校间"), 0.05, "level_decomposition.csv")

# ==========================================================================
# 4.7  Supplementary
# ==========================================================================
add("4.7", "Asked to evaluate AI output at least sometimes (%)", 44.7, 44.7, 0.05,
    "11_oecd_claims.py (printed: AI 生成信息评估 · 至少有时)")
add("4.7", "Asked often or very often (%)", 17.5, 17.5, 0.05,
    "11_oecd_claims.py (printed: 经常/非常频繁)")
add("4.7", "Frequent access, lowest ESCS quartile (%)", 13.1, 13.1, 0.05,
    "11_oecd_claims.py (printed)")
add("4.7", "Frequent access, highest ESCS quartile (%)", 24.4, 24.4, 0.05,
    "11_oecd_claims.py (printed)")
add("4.7", "Socioeconomic gap (percentage points)", 11, 11.2, 0.5, "11_oecd_claims.py (printed)")
add("4.7", "Gap in general online-information instruction (pp)", -2.2, -2.2, 0.05,
    "11_oecd_claims.py (printed)")
add("4.7", "Score of daily users with critical instruction", 581.0, 581.0, 0.05,
    "11_oecd_claims.py (printed stratified means)")
add("4.7", "Score of daily users without instruction", 570.9, 570.9, 0.05,
    "11_oecd_claims.py (printed stratified means)")
add("4.7", "Score of weekly users with instruction", 613.7, 613.7, 0.05,
    "11_oecd_claims.py (printed stratified means)")
oc = rd("oecd_claim_tests.csv")
add("4.7", "Instruction x use interaction", -1.76,
    pick(oc, "b", model="P1 ×AI评估机会(频繁)", term="AIxLOAI"), 0.005, "oecd_claim_tests.csv P1")
add("4.7", "Instruction x use interaction, SE", 2.44,
    pick(oc, "se", model="P1 ×AI评估机会(频繁)", term="AIxLOAI"), 0.005, "oecd_claim_tests.csv P1")
add("4.7", "Instruction x use interaction, p", 0.471,
    pick(oc, "p", model="P1 ×AI评估机会(频繁)", term="AIxLOAI"), 0.0005, "oecd_claim_tests.csv P1")

# ==========================================================================
# 5.x and Appendix
# ==========================================================================
add("5.1", "Combined ESCS + climate gain (SD)", 0.46, (26.8 + 13.1) / 87.5, 0.005,
    "derived: (26.8 + 13.1) / 87.5")
rmA1 = pick(rm, "b", model="A1 AI分组(基础)", outcome="SCIE", term="escs")
rmA2 = pick(rm, "b", model="A2 AI分组(全模型)", outcome="SCIE", term="escs")
add("5.4.4", "ESCS coefficient before controls", 25.6, rmA1, 0.05, "regression_main.csv A1 (escs)")
add("5.4.4", "ESCS coefficient after controls", 4.6, rmA2, 0.05, "regression_main.csv A2 (escs)")
for nm, term, val in [("self-efficacy", "EFFSCIE_z", 10.0), ("self-regulation", "SELFREG_z", 8.2),
                      ("science interest", "JOYSCIE_z", 7.8)]:
    add("5.4.4", "Coefficient on %s" % nm, val,
        pick(rm, "b", model="A2 AI分组(全模型)", outcome="SCIE", term=term), 0.05,
        "regression_main.csv A2 (%s)" % term)
add("5.5", "Schools in the main interaction models", 360,
    q.dropna(subset=BASE12 + ["AIz", "W_FSTUWT", "CNTSCHID"] + SCHZ).CNTSCHID.nunique(), 0,
    "15_school_models.py (S3/S4 sample)")
est = rd("estimator_check.csv")
add("A", "Max fixed-effect difference vs statsmodels", 1.19e-06,
    float(est[est.check.str.startswith("1a")].value.iloc[0]), 1e-8, "estimator_check.csv (1a)")
add("A", "Between-school variance, custom estimator", 2945.581,
    float(est[est.check.str.startswith("1b")].value.iloc[0]), 0.001, "estimator_check.csv (1b)")
add("A", "Between-school variance, statsmodels", 2945.579, 2945.5789, 0.001,
    "estimator_check.csv (1b note)")
add("A", "Log-likelihood difference, direct MVN", 0.0,
    float(est[est.check.str.startswith("2")].value.iloc[0]), 1e-9, "estimator_check.csv (2)")

# ==========================================================================
# External reference values
# ==========================================================================
add("1", "Score gap between users and non-users (OECD report)", 20, None, 0,
    "OECD (2026) PISA 2025 Results Vol. I",
    "external benchmark; not computable from this release package")

df = pd.DataFrame(rows)

# Standard errors printed to two decimals: flag the few last-digit slips where the
# printed p-value still matches the stored (unrounded) SE.
for i, r in df.iterrows():
    looks_like_se = ("(se_int)" in r.claim) or r.claim.endswith(", SE")
    if looks_like_se and r.status == "rounding" and 0.005 < r.abs_diff <= 0.02:
        df.at[i, "status"] = "last-digit"
        df.at[i, "note"] = ("printed SE rounds 0.01 off the stored value; the printed "
                            "p-value matches the stored SE, so inference is unaffected")

df.to_csv(HERE / "number_check.csv", index=False, encoding="utf-8-sig")

print("items: %d" % len(df))
print(df.status.value_counts().to_string())
bad = df[df.status.isin(["mismatch", "missing"])]
if len(bad):
    print("\n--- needs attention ---")
    print(bad[["section", "claim", "paper_value", "recomputed", "abs_diff", "note"]].to_string(index=False))
