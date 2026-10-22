import argparse
import sys
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description="Stage05.2 face detection and shot-local tracking (research prototype)")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("detect-track", help="Process only explicitly selected shots, every source frame")
    d.add_argument("--movie-id", required=True, help="Canonical IMDb movie ID")
    d.add_argument("--analysis-root", type=Path, required=True, help="Parent of movie-first analysis directories")
    d.add_argument("--input-root", type=Path, default=Path("data/movies"), help="Video discovery root; ignored with --video")
    d.add_argument("--video", type=Path, help="Explicit source video when discovery is ambiguous")
    d.add_argument("--shot-ids", nargs="+", required=True, help="Space-separated explicit Stage01 shot IDs; no whole-movie default")
    d.add_argument("--detector", choices=["yunet", "scrfd"], default="yunet", help="YuNet implemented; SCRFD reserved")
    d.add_argument("--model", type=Path, required=True, help="Local YuNet ONNX model path")
    d.add_argument("--device", default="cpu", choices=["cpu"], help="Explicit OpenCV DNN CPU backend")
    d.add_argument("--score-threshold", type=float, default=.8, help="Minimum YuNet confidence, default 0.8")
    d.add_argument("--inference-max-size", type=int, default=0, help="Optional longest inference edge; 0 uses original resolution")
    d.add_argument("--max-missed", type=int, default=3, help="Maximum intervening missed frames retained for association")
    d.add_argument("--min-hits", type=int, default=3, help="Observed frames required to confirm a track; tentative tracks remain in output")
    d.add_argument("--discontinuity-threshold", type=float, default=.18, help="Normalized thumbnail difference that conservatively breaks associations; does not change shot bounds")
    d.add_argument("--output-root", type=Path, default=Path("reports/stage05_2"), help="Reports parent; appends movie ID")
    d.add_argument("--boundary-sidecar", type=Path, help="Explicit audited boundary correction JSON; original Stage01 is never changed")
    d.add_argument("--allow-experimental-boundaries", action="store_true", help="Opt in to model-reviewed, unapproved draft ranges in a separate experimental output directory")
    d.add_argument("--dry-run", action="store_true", help="Validate discovery, bounds, timeline hints and real model inference; write no reports")
    mode = d.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", help="Skip completed shots only when input/configuration match")
    mode.add_argument("--overwrite", action="store_true", help="Explicitly replace this implementation's prior outputs")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        from .pipeline import run
        return run(args)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"Stage05.2 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
