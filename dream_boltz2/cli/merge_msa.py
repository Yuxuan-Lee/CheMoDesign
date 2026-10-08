#!/usr/bin/env python3
"""Merge a receptor a3m with synthetic paired rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dream_boltz2.msa.merge import merge_receptor_msas_to_csv  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real-a3m", required=True, type=Path, help=' receptor a3m(.a3m or.a3m.gz)')
    p.add_argument("--fake-csv", required=True, type=Path, help="fake receptor CSV(columns: sequence,key)")
    p.add_argument("--out", required=True, type=Path, help='output merged CSV')
    p.add_argument("--max-real", type=int, default=None, help=' a3m sequence(defaultnot)')
    p.add_argument("--max-fake", type=int, default=None, help=' fake sequence(defaultnot)')
    p.add_argument("--no-dedup", action="store_true", help='not(defaultsequence)')
    args = p.parse_args()

    out = merge_receptor_msas_to_csv(
        real_a3m=args.real_a3m,
        fake_csv=args.fake_csv,
        out_csv=args.out,
        max_real=args.max_real,
        max_fake=args.max_fake,
        dedup_by_sequence=not args.no_dedup,
    )
    print("[OK] wrote merged receptor MSA CSV:", out)


if __name__ == "__main__":
    main()


