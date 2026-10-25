import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Stage05.6 local evidence preparation and independent human review")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--event-run", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--review-dir", type=Path, required=True)
    build.add_argument("--pilot-size", type=int, default=48)
    check = sub.add_parser("verify")
    check.add_argument("run_dir", type=Path)
    server = sub.add_parser("serve")
    server.add_argument("run_dir", type=Path)
    server.add_argument("--review-dir", type=Path)
    server.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if args.command == "build":
        from .pipeline import build
        print(json.dumps(build(args.event_run, args.output_dir, args.review_dir, args.pilot_size), indent=2))
    elif args.command == "verify":
        from .pipeline import verify
        print(json.dumps(verify(args.run_dir), indent=2))
    else:
        from .server import serve
        serve(args.run_dir, args.review_dir, args.port)


if __name__ == "__main__":
    main()
