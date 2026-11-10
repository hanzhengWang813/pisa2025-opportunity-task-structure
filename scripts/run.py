# -*- coding: utf-8 -*-
"""Command-line runner for the analysis pipeline.

Usage, from the repository root:

    python scripts/run.py list      # list every script with its short code
    python scripts/run.py 21        # run 21_level_id_correct.py
    python scripts/run.py 02b       # run 02b_cfa.py
    python scripts/run.py 25        # run 25_paper_audit.py
    python scripts/run.py all       # run 01 -> 27 in order

Why a runner is needed: several scripts import the shared modules
``wmlm`` (TwoLevelML) and ``pisa_engine`` (PISAEngine), which live next to the
scripts. Invoking ``python scripts/21_level_id_correct.py`` directly still works
because Python puts the script's own directory on ``sys.path``; the runner makes
the same guarantee explicit and also lets you run a script by its short code.

All output is written to ``<repo>/outputs``. Set PISA_BASE to read the microdata
from a directory other than ``<repo>/data``.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# ---------- collect the scripts ----------
SCRIPTS = {}
for root, dirs, files in os.walk(HERE):
    dirs[:] = [d for d in dirs if d not in ('lib', '__pycache__')]
    for f in sorted(files):
        if not f.endswith('.py') or f in ('run.py', 'paths.py', 'wmlm.py', 'pisa_engine.py'):
            continue
        key = f.split('_')[0]
        SCRIPTS.setdefault(key, os.path.join(root, f))

# Execution order used by "all". Scripts that only draw figures are kept at the
# end so that every result table they read already exists.
ORDER = ['01', '02', '02b', '03', '04', '05', '06', '07', '07b', '08', '09',
         '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
         '21', '22', '23', '25', '26']


def show():
    print('Available scripts, in execution order:\n')
    for k in sorted(SCRIPTS):
        rel = os.path.relpath(SCRIPTS[k], HERE).replace('\\', '/')
        print('  %-4s %s' % (k, rel))
    print('\nUsage: python scripts/run.py <code>   e.g. python scripts/run.py 21')


def run(key):
    if key not in SCRIPTS:
        print('No script with code %r; run "python scripts/run.py list" for the list.' % key)
        return 1
    path = SCRIPTS[key]
    print('=' * 70)
    print('Running %s' % os.path.relpath(path, HERE).replace('\\', '/'))
    print('=' * 70)
    sys.argv = [path]
    runpy.run_path(path, run_name='__main__')
    return 0


if __name__ == '__main__':
    arg = sys.argv[1].strip() if len(sys.argv) > 1 else 'list'
    if arg in ('list', '-l', '--list'):
        show()
        sys.exit(0)
    if arg == 'all':
        for k in [k for k in ORDER if k in SCRIPTS]:
            rc = run(k)
            if rc:
                sys.exit(rc)
        sys.exit(0)
    sys.exit(run(arg))
