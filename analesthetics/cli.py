"""命令行入口。"""
from __future__ import annotations

import sys

from .core import main as _main


def main(argv: list[str] | None = None) -> int:
    return _main() if argv is None else _main()


if __name__ == "__main__":
    sys.exit(main())
