#!/usr/bin/env python3
"""Stage 0A source preflight. Dry-run planning is the default."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oscardp.stage0.preflight import PreflightOptions, run


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 0A source preflight; no source mutation by default.")
    parser.add_argument("--input-root", type=Path, default=Path("/mnt/g/datasets/oscar_movie"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/stage0_source_preflight"))
    parser.add_argument("--apply-renames", action="store_true", help="Perform only collision-safe canonical filename renames.")
    parser.add_argument("--quarantine", action="store_true", help="Move planned disposable/auxiliary files; never deletes.")
    parser.add_argument("--delete-planned", action="store_true", help="Permanently delete only planned image/release-metadata/auxiliary files.")
    parser.add_argument("--quarantine-root", type=Path, default=Path("/mnt/g/datasets/oscar_movie_cleanup_quarantine"))
    parser.add_argument("--include-unknown-quarantine", action="store_true", help="Also quarantine UNKNOWN files (off by default).")
    parser.add_argument("--delete-quarantine", action="store_true", help="Reserved explicit destructive action; intentionally not implemented.")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.delete_quarantine:
        parser.error("--delete-quarantine is intentionally unavailable; quarantine contents require separate manual review.")
    if args.quarantine and args.delete_planned:
        parser.error("--quarantine and --delete-planned cannot be used together.")
    if args.quarantine and not args.apply_renames:
        print("Note: --quarantine moves only planned cleanup files; canonical renames remain plans unless --apply-renames is also supplied.")
    rows, _plan, summary = run(PreflightOptions(input_root=args.input_root.resolve(), report_dir=args.report_dir.resolve(), apply_renames=args.apply_renames, quarantine=args.quarantine, delete_planned=args.delete_planned, quarantine_root=args.quarantine_root.resolve(), include_unknown_quarantine=args.include_unknown_quarantine, limit=args.limit))
    print(f"Stage 0A complete: {len(rows)} movie directories; reports: {args.report_dir.resolve()}")
    print(f"Manual review: {summary['manual_review_count']}; planned renames: {summary['planned_rename_count']}; quarantine eligible: {summary['quarantine_eligible_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
