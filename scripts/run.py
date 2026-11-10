# -*- coding: utf-8 -*-
"""
运行器：把 05_代码/lib 加入模块搜索路径后执行指定脚本。

用法（在解压后的包根目录）：
    python 05_代码/run.py 21        # 运行 21_level_id_correct.py
    python 05_代码/run.py 02b       # 运行 02b_cfa.py
    python 05_代码/run.py 25        # 运行 25_paper_audit.py
    python 05_代码/run.py list      # 列出全部脚本与编号
    python 05_代码/run.py all       # 依次运行 01 -> 27（不含打包脚本 28）

为什么需要它：部分脚本使用 `from wmlm import TwoLevelML` / `from pisa_engine import
PISAEngine`，而这两个模块位于 05_代码/lib/。直接 `python 某脚本.py` 时，脚本自身会
把所在目录加入 path，但不会加入 lib/，因而会 ModuleNotFoundError。
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, 'lib')
if LIB not in sys.path:
    sys.path.insert(0, LIB)

# ---------- 收集脚本 ----------
SCRIPTS = {}
for root, dirs, files in os.walk(HERE):
    dirs[:] = [d for d in dirs if d not in ('lib', '__pycache__')]
    for f in sorted(files):
        if not f.endswith('.py') or f == 'run.py':
            continue
        key = f.split('_')[0]
        SCRIPTS.setdefault(key, os.path.join(root, f))


def show():
    print('可用脚本（按执行顺序）：\n')
    for k in sorted(SCRIPTS):
        rel = os.path.relpath(SCRIPTS[k], HERE).replace('\\', '/')
        print('  %-4s %s' % (k, rel))
    print('\n用法：python run.py <编号>   例如 python run.py 21')


def run(key):
    if key not in SCRIPTS:
        print('找不到脚本编号 %r，运行 python run.py list 查看清单' % key)
        return 1
    path = SCRIPTS[key]
    print('=' * 70)
    print('运行 %s' % os.path.relpath(path, HERE).replace('\\', '/'))
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
        order = [k for k in ['01', '02', '02b', '03', '04', '05', '06', '09', '11', '12',
                             '13', '14', '15', '16', '17', '18', '19', '20', '21',
                             '22', '23', '24', '25', '26', '27'] if k in SCRIPTS]
        for k in order:
            rc = run(k)
            if rc:
                sys.exit(rc)
        sys.exit(0)
    sys.exit(run(arg))
