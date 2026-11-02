#!/usr/bin/env python3
"""Cut the tau-truncated ERMA clips from the released videos.

Usage:
    python tools/prepare_erma.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import erma
from utils.config import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--taus", type=float, nargs="+", default=list(erma.TAUS))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cfg = load()
    for tau in args.taus:
        n, failed = erma.prepare(cfg, tau, args.workers)
        print(f"tau={tau:g}: {n} clips -> {erma.tau_dir(cfg, tau)}" + (f" ({len(failed)} failed)" if failed else ""))


if __name__ == "__main__":
    main()
