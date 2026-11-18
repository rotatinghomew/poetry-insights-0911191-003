"""隔离模式启动器：python3 -I run.py <子命令> ...

-I 会忽略 PYTHONPATH 与当前目录，因此这里在代码内显式插入本仓库的绝对路径，
保证解释器不会从不信任的数据目录加载任何模块。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from workshop.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
