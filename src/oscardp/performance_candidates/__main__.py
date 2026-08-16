from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oscardp.shots.schema import json_dumps

from .pipeline import MiningOptions, mine
from .review import evaluate_review, prepare_review_sample
from .validation import validate_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m oscardp.performance_candidates")
    commands = parser.add_subparsers(dest="command", required=True)
    mining = commands.add_parser("mine", help="mine shot-first performance candidates from a frozen Stage 2 release")
    mining.add_argument("--release-manifest", type=Path, required=True)
    mining.add_argument("--output-root", type=Path, required=True)
    mining.add_argument("--nominees-file", type=Path, required=True)
    mining.add_argument("--movie-key", default="tt12300742")
    mining.add_argument("--performer-id")
    mining.add_argument("--performer-name")
    mining.add_argument("--face-model", type=Path, required=True)
    mining.add_argument("--face-model-sha256")
    mining.add_argument("--semantic-threshold", type=float, default=0.35)
    mining.add_argument("--semantic-override-threshold", type=float, default=0.75)
    mining.add_argument("--max-event-duration-sec", type=float, default=30.0)
    behavior = mining.add_mutually_exclusive_group()
    behavior.add_argument("--resume", action="store_true", default=True)
    behavior.add_argument("--overwrite", action="store_true")
    mining.add_argument("--dry-run", action="store_true")
    validate = commands.add_parser("validate", help="validate an existing Stage 3 run")
    validate.add_argument("--run-dir", type=Path, required=True)
    review = commands.add_parser("prepare-review-sample", help="build deterministic shot/event review packages")
    review.add_argument("--run-dir", type=Path, required=True)
    review.add_argument("--shot-count", type=int, default=20)
    review.add_argument("--event-count", type=int, default=20)
    evaluate = commands.add_parser("evaluate-review", help="evaluate completed human review labels")
    evaluate.add_argument("--shot-sample", type=Path, required=True)
    evaluate.add_argument("--event-sample", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "mine":
            options = MiningOptions(
                release_manifest=args.release_manifest, output_root=args.output_root,
                nominees_file=args.nominees_file, performer_id=args.performer_id, performer_name=args.performer_name,
                face_model=args.face_model, face_model_sha256=args.face_model_sha256,
                movie_key=args.movie_key, semantic_threshold=args.semantic_threshold,
                semantic_override_threshold=args.semantic_override_threshold,
                max_event_duration_sec=args.max_event_duration_sec,
                resume=args.resume and not args.overwrite, overwrite=args.overwrite, dry_run=args.dry_run,
            )
            print(json_dumps(mine(options), pretty=True)); return 0
        if args.command == "validate":
            report = validate_run(args.run_dir)
            print(json_dumps(report, pretty=True)); return 0 if report.passed else 1
        if args.command == "prepare-review-sample":
            print(json_dumps(prepare_review_sample(args.run_dir, args.shot_count, args.event_count), pretty=True)); return 0
        if args.command == "evaluate-review":
            result = evaluate_review(args.shot_sample, args.event_sample)
            print(json_dumps(result, pretty=True)); return 0 if result["passed"] else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr); return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
