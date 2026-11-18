from __future__ import annotations

import argparse
from pathlib import Path

from tests.helpers import write_archives


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    write_archives(parser.parse_args().raw_root)


if __name__ == "__main__":
    main()
