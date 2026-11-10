# -*- coding: utf-8 -*-
"""Central path resolution for this replication package.

Analysis scripts read the PISA 2025 public-use files from ``<repo>/data`` and
write results to ``<repo>/outputs``. Set the environment variable ``PISA_BASE``
to point the scripts at a different root (for example, a scratch copy).
"""
import os
from pathlib import Path

BASE = os.environ.get("PISA_BASE") or str(Path(__file__).resolve().parents[1])
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "outputs")
