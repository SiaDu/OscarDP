import argparse
import sys
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description="Stage05 face tracking and cross-shot identity association (research prototype)")
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
    i = sub.add_parser("associate-identities", help="Stage05.3: sample corrected real tracks and associate ArcFace identity candidates")
    i.add_argument("--movie-id", required=True, help="IMDb movie ID of the corrected Stage05.2 run")
    i.add_argument("--source-run", type=Path, required=True, help="Corrected Stage05.2 movie run directory; original ranges are rejected")
    i.add_argument("--shot-ids", nargs="+", required=True, help="Explicit pilot shots; no whole-movie default")
    i.add_argument("--model", type=Path, required=True, help="Existing local InsightFace ArcFace 112x112/512D ONNX weights; never downloaded automatically")
    i.add_argument("--provider", choices=["CPUExecutionProvider","CUDAExecutionProvider"], default="CPUExecutionProvider", help="Explicit ONNX Runtime provider; unavailable provider fails")
    i.add_argument("--output-dir", type=Path, required=True, help="New independent pilot directory; reruns never overwrite")
    i.add_argument("--max-samples", type=int, default=16, help="Maximum temporal candidate frames per track, 2–16")
    i.add_argument("--min-detection-score", type=float, default=.85, help="Crop quality confidence floor; not an identity threshold")
    i.add_argument("--min-face-size", type=float, default=80., help="Minimum native face box edge in pixels")
    i.add_argument("--min-sharpness", type=float, default=30., help="Minimum aligned-112 Laplacian variance")
    i.add_argument("--same-cosine", type=float, default=.55, help="Uncalibrated research same-candidate threshold, plus margin")
    i.add_argument("--different-cosine", type=float, default=.25, help="Uncalibrated research different-candidate threshold, minus margin")
    i.add_argument("--margin", type=float, default=.05, help="Minimum cosine distance from each decision threshold; uncertain band rejected")
    i.add_argument("--same-sample-q10", type=float, default=.45, help="Minimum lower-decile cross-sample cosine for same candidates")
    i.add_argument("--consistency-median", type=float, default=.65, help="Minimum within-track median pair cosine for association eligibility")
    i.add_argument("--consistency-q10", type=float, default=.50, help="Minimum within-track lower-decile cosine")
    i.add_argument("--min-samples", type=int, default=3, help="Minimum quality-passing samples per eligible track")
    i.add_argument("--dry-run", action="store_true", help="Validate corrected provenance/model/candidate selection without source-frame decode or outputs")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "associate-identities":
            from .identity_pipeline import run
        else:
            from .pipeline import run
        return run(args)
    except (Exception, KeyboardInterrupt) as exc:
        stage = "Stage05.3" if args.command == "associate-identities" else "Stage05.2"
        print(f"{stage} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
