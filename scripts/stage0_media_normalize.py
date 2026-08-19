#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permit the documented command to run from a source checkout without first
# installing the package.  Production installs continue to use the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from oscardp.stage0.pipeline import NormalizeOptions, run


def find_inventory(input_root: Path) -> Path | None:
    candidates = [Path("reports/oscar_movie_media_inventory.csv"), input_root / "oscar_movie_media_inventory.csv"]
    return next((path.resolve() for path in candidates if path.is_file()), None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 0 safe media normalization; dry-run by default.")
    parser.add_argument("--inventory", type=Path, help="Existing source inventory CSV (read as provenance; never overwritten)")
    parser.add_argument("--input-root", type=Path, default=Path("/mnt/g/datasets/oscar_movie"))
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/g/datasets/oscar_movie_standardized"))
    parser.add_argument("--execute", action="store_true", help="Actually create standardized processing copies")
    parser.add_argument("--movie-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true", help="Allow replacing an invalid existing output; source is never changed")
    parser.add_argument("--cq", type=int, default=25)
    parser.add_argument("--max-size-gib", type=float, default=4.5)
    args = parser.parse_args()
    if args.inventory and not args.inventory.is_file(): parser.error(f"Inventory does not exist: {args.inventory}")
    inventory = args.inventory.resolve() if args.inventory else find_inventory(args.input_root)
    if inventory: print(f"Using existing inventory as provenance: {inventory}")
    rows = run(NormalizeOptions(input_root=args.input_root.resolve(), output_root=args.output_root.resolve(), inventory=inventory, execute=args.execute, movie_id=args.movie_id, limit=args.limit, force=args.force, cq=args.cq, max_size_gib=args.max_size_gib))
    print(f"Stage 0 complete: {len(rows)} movie files; reports: {args.output_root / 'stage0_media_inventory.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
