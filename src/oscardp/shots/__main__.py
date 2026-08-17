from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .discovery import discover_movies
from .export import export_shots_csv
from .pipeline import ProcessOptions, process_one
from .schema import json_dumps
from .validation import validate_output_root

DEFAULT_INPUT_ROOT = Path("/mnt/g/datasets/oscar_movie")
DEFAULT_OUTPUT_ROOT = Path("/mnt/g/datasets/oscar_movie_processed")


def _add_input_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m oscardp.shots")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("discover", help="discover supported source movies")
    _add_input_root(discover)
    discover.add_argument("--limit", type=int)
    discover.add_argument("--movie-key")
    discover.add_argument("--dry-run", action="store_true")

    one = subparsers.add_parser("process-one", help="process exactly one movie")
    one.add_argument("--video", type=Path, required=True)
    _add_input_root(one)
    one.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    one.add_argument("--weights", type=Path, required=True)
    one.add_argument("--threshold", type=float, default=0.5)
    one.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    behavior = one.add_mutually_exclusive_group()
    behavior.add_argument("--resume", action="store_true", default=True)
    behavior.add_argument("--overwrite", action="store_true")
    one.add_argument("--dry-run", action="store_true")
    one.add_argument("--save-all-boundary-frames", action="store_true")
    one.add_argument("--save-raw-predictions", action="store_true")
    one.add_argument(
        "--no-progress",
        action="store_true",
        help="disable stage and frame progress output on stderr",
    )

    validate = subparsers.add_parser("validate", help="validate published movie outputs")
    validate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    validate.add_argument("--movie-key")

    export_csv = subparsers.add_parser(
        "export-csv", help="export an existing shots.jsonl on demand"
    )
    export_csv.add_argument("--movie-dir", type=Path, required=True)
    export_csv.add_argument("--output", type=Path)
    return parser


def handle_discover(args: argparse.Namespace) -> int:
    movies = discover_movies(args.input_root, limit=args.limit, movie_key=args.movie_key)
    for movie in movies:
        print(json_dumps(movie))
    return 0


def handle_process_one(args: argparse.Namespace) -> int:
    options = ProcessOptions(
        input_root=args.input_root,
        output_root=args.output_root,
        weights=args.weights,
        threshold=args.threshold,
        device=args.device,
        resume=args.resume and not args.overwrite,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        save_all_boundary_frames=args.save_all_boundary_frames,
        save_raw_predictions=args.save_raw_predictions,
        progress=not args.no_progress,
    )
    print(json_dumps(process_one(args.video, options), pretty=True))
    return 0


def handle_validate(args: argparse.Namespace) -> int:
    results = validate_output_root(args.output_root, args.movie_key)
    if not results:
        print("No movie outputs found", file=sys.stderr)
        return 1
    passed = True
    for movie_key, result in results.items():
        print(json_dumps({"movie_key": movie_key, **vars(result)}))
        passed = passed and result.passed
    return 0 if passed else 1


def handle_export_csv(args: argparse.Namespace) -> int:
    output = export_shots_csv(args.movie_dir, args.output)
    print(output)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "discover":
            return handle_discover(args)
        if args.command == "process-one":
            return handle_process_one(args)
        if args.command == "validate":
            return handle_validate(args)
        if args.command == "export-csv":
            return handle_export_csv(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
