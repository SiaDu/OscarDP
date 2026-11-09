import argparse
import json

from .inputs import load_sources, selection
from .pipeline import run


def main():
    parser=argparse.ArgumentParser(description="Bugonia Stage06 local pilot and interactive-annotation assembly; no external API")
    parser.add_argument("command",choices=["inspect","prepare","run"])
    parser.add_argument("--prepared",action="store_true",help="Use an existing local preparation without overwriting media/audio")
    parser.add_argument("--analysis-dir",help="Real saved interactive-assistant annotation JSONs")
    parser.add_argument("--pilot-input",default="reports/stage05_6/tt12300742/pilot_001")
    parser.add_argument("--output-dir",default="reports/stage06/tt12300742/pilot_001")
    parser.add_argument("--start",type=float)
    parser.add_argument("--end",type=float)
    parser.add_argument("--ser-model",default="superb/wav2vec2-base-superb-er")
    parser.add_argument("--model-dir",default="models/stage06")
    parser.add_argument("--video-fps",type=float,default=2.)
    args=parser.parse_args()
    if args.prepared and args.command != "run":
        parser.error("--prepared is only valid with run")
    if not 0.5 <= args.video_fps <= 10:
        parser.error("--video-fps must be between 0.5 and 10")
    if args.command == "run" and not args.analysis_dir:
        parser.error("run requires --analysis-dir; use prepare for local media/audio only")
    if args.command == "inspect":
        value=selection(load_sources(args.pilot_input),args.start,args.end)
        print(json.dumps({k:value[k] for k in ("movie_id","scene_id","start","end","duration","character_candidate","scene_candidates")},indent=2))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
