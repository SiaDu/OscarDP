from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .discovery import derive_movie_key, discover_movies, load_movie_titles
from .pipeline import ProcessOptions, process_one
from .schema import json_dumps
from .validation import validate_output_root

DEFAULT_INPUT_ROOT = Path("/media/sdu/SiyaoDu/datasets/oscar_movie")
DEFAULT_OUTPUT_ROOT = Path("/media/sdu/SiyaoDu/datasets/oscar_movie_analysis")


def _processing_options(parser: argparse.ArgumentParser, *, require_titles: bool = False) -> None:
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--titles", type=Path, required=require_titles,
                        help="movie_titles.json supplies both batch selection and verified titles")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    behavior = parser.add_mutually_exclusive_group()
    behavior.add_argument("--resume", action="store_true", default=True)
    behavior.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--qc", action="store_true", help="save sampled and suspicious boundary pairs")
    parser.add_argument("--save-all-boundary-frames", action="store_true")
    parser.add_argument("--save-raw-predictions", action="store_true")
    parser.add_argument("--no-progress", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m oscardp.shots")
    commands = parser.add_subparsers(dest="command", required=True)
    discover = commands.add_parser("discover")
    discover.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    discover.add_argument("--movie-id")
    discover.add_argument("--limit", type=int)
    discover.add_argument("--titles", type=Path)
    one = commands.add_parser("process-one")
    _processing_options(one)
    one.add_argument("--video", type=Path, required=True)
    batch = commands.add_parser("process")
    _processing_options(batch, require_titles=True)
    batch.add_argument("--movie-id")
    batch.add_argument("--limit", type=int)
    validate = commands.add_parser("validate")
    validate.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    validate.add_argument("--movie-id")
    validate.add_argument("--frame-profile", choices=("auto", "legacy", "three_sample"), default="auto")
    upgrade = commands.add_parser("frames-upgrade", help="frozen-selected image-only upgrade; default dry-run")
    for name in ("input-root", "output-root", "frozen-manifest", "report-root"):
        upgrade.add_argument(f"--{name}", type=Path, required=True)
    upgrade.add_argument("--expected-manifest-sha256", required=True)
    upgrade.add_argument("--expect-movies", type=int, required=True)
    selection = upgrade.add_mutually_exclusive_group()
    selection.add_argument("--movie-id")
    selection.add_argument("--exclude-movie-id")
    mode = upgrade.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser


def _options(args: argparse.Namespace) -> ProcessOptions:
    return ProcessOptions(
        input_root=args.input_root, output_root=args.output_root,
        weights=args.weights, threshold=args.threshold, device=args.device,
        resume=not args.overwrite, overwrite=args.overwrite, dry_run=args.dry_run,
        save_all_boundary_frames=args.save_all_boundary_frames,
        save_raw_predictions=args.save_raw_predictions, progress=not args.no_progress,
        titles=args.titles, qc=args.qc,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "frames-upgrade":
            from .frames_upgrade import frames_upgrade
            print(json_dumps(frames_upgrade(**{k: v for k, v in vars(args).items() if k != "command"}), pretty=True))
            return 0
        if args.command == "discover":
            titles = load_movie_titles(args.titles) if args.titles else None
            if titles is not None and args.movie_id and args.movie_id not in titles:
                raise ValueError(f"Movie ID is not in title mapping: {args.movie_id}")
            for movie in discover_movies(args.input_root, limit=args.limit, movie_key=args.movie_id,
                                         allowed_ids=set(titles) if titles else None,
                                         require_all=titles is not None):
                print(json_dumps(movie))
            return 0
        if args.command == "process-one":
            # Discovery catches ambiguous sources before publishing a movie ID.
            movie = args.video.resolve()
            refs = discover_movies(args.input_root, movie_key=derive_movie_key(movie, args.input_root))
            if movie not in {ref.source_path for ref in refs}:
                raise ValueError(f"Video was not discovered under input root: {movie}")
            print(json_dumps(process_one(movie, _options(args)), pretty=True))
            return 0
        if args.command == "process":
            titles = load_movie_titles(args.titles)
            if args.movie_id and args.movie_id not in titles:
                raise ValueError(f"Movie ID is not in title mapping: {args.movie_id}")
            failures = 0
            refs = discover_movies(args.input_root, limit=args.limit, movie_key=args.movie_id,
                                   allowed_ids=set(titles), require_all=True)
            for movie in refs:
                try:
                    print(json_dumps(process_one(movie.source_path, _options(args))))
                except (OSError, ValueError, RuntimeError) as exc:
                    failures += 1
                    print(json_dumps({"movie_id": movie.movie_key, "status": "failed", "error": str(exc)}), file=sys.stderr)
            return 1 if failures else 0
        if args.command == "validate":
            results = validate_output_root(args.output_root, args.movie_id, frame_profile=args.frame_profile)
            if not results:
                print("No movie outputs found", file=sys.stderr)
                return 1
            for movie_id, result in results.items():
                print(json_dumps({"movie_id": movie_id, **vars(result)}))
            return 0 if all(result.passed for result in results.values()) else 1
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
