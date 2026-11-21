"""Verified Findings offline CLI.

Usage: python3 plearun_cli.py REPORT RECORDS_DIR > findings.json
Only Python's standard library is required. Source text is never executed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from claims import extract_claims
from evidence import load_records, read_text
from verifier import verify


def run(report_path: str | Path, records_path: str | Path) -> dict:
    report_text = read_text(Path(report_path))
    records = load_records(Path(records_path))
    return verify(extract_claims(report_text), records)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify report claims against incident records (offline).")
    parser.add_argument("report", type=Path)
    parser.add_argument("records_dir", type=Path)
    args = parser.parse_args()
    try:
        result = run(args.report, args.records_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
