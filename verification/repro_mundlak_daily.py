# -*- coding: utf-8 -*-
"""Reproduce the two published coefficients that are not stored in outputs/.

1. Section 4.4: the Mundlak within-school component for the *binary* daily-use
   indicator (-46.1, SE 5.7).  ``outputs/level_id.csv`` and ``outputs/mundlak.csv``
   store the continuous-intensity variant (-51.88); the daily variant is
   re-estimated here with the same covariate set and replicate weights.

2. Section 4.1 / Table 1 note: the univariate coefficient for "organization of
   digital teaching" (-3.35, p = .128).  This is a single school-level predictor
   entered alongside the 12 student covariates and AI use intensity.

Usage
-----
    python verification/repro_mundlak_daily.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))

from pisa_engine import PISAEngine  # noqa: E402

DATA = Path(os.environ.get("PISA_BASE") or ROOT) / "data"

q = pd.read_parquet(DATA / "qci_school.parquet")
q["female"] = (q["ST004D01T"] == 1).astype(float).where(q["ST004D01T"].notna())
q["repeat"] = (q["REPEAT"] > 0).astype(float).where(q["REPEAT"].notna())
q["escs"] = q["ESCS"]
for c in ["ESCS", "GRADE", "JOYSCIE", "EFFSCIE", "SELFREG", "COGACSC",
          "TEACHSUP", "DISCLISCI", "ICTWKDY", "ICTHOME"]:
    q[c + "_z"] = (q[c] - q[c].mean()) / q[c].std()
q["grade_z"] = q["GRADE_z"]
q["AIz"] = (q["AI_total"] - q["AI_total"].mean()) / q["AI_total"].std()
q["AI_daily"] = (q["AI_total"].round() >= 5).astype(float).where(q["AI_total"].notna())

STU = ["female", "escs", "repeat", "grade_z", "JOYSCIE_z", "EFFSCIE_z", "SELFREG_z",
       "COGACSC_z", "TEACHSUP_z", "DISCLISCI_z", "ICTWKDY_z", "ICTHOME_z"]

# ---------- 1. Mundlak for the daily-use dummy ----------
q["D_between"] = q["CNTSCHID"].map(q.groupby("CNTSCHID")["AI_daily"].mean())
q["D_within"] = q["AI_daily"] - q["D_between"]
q["D_between_z"] = (q["D_between"] - q["D_between"].mean()) / q["D_between"].std()

s = q.dropna(subset=STU + ["D_within", "D_between_z", "W_FSTUWT", "CNTSCHID"])
print("Mundlak (daily dummy) n=%d, schools=%d" % (len(s), s.CNTSCHID.nunique()))
r = PISAEngine(s, "PV", "SCIE").fit(STU + ["D_within", "D_between_z"], "mundlak_daily")
print(r[r.term.isin(["D_within", "D_between_z"])][["term", "b", "se", "p"]].to_string(index=False))

# ---------- 2. Univariate school-level model, organization of digital teaching ----------
q["G_DIGORG_z"] = (q["G_DIGORG"] - q["G_DIGORG"].mean()) / q["G_DIGORG"].std()
s2 = q.dropna(subset=STU + ["AIz", "G_DIGORG_z", "W_FSTUWT", "CNTSCHID"])
print("\nUnivariate G_DIGORG n=%d" % len(s2))
r2 = PISAEngine(s2, "PV", "SCIE").fit(STU + ["AIz", "G_DIGORG_z"], "univariate_G")
print(r2[r2.term == "G_DIGORG_z"][["term", "b", "se", "p"]].to_string(index=False))

print("\nExpected in the manuscript: -46.10 (SE 5.66) and -3.35 (p = .128)")
