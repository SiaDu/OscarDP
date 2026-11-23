"""Build the Bugonia v1.2 look-state pilot and attach reviewed frame observations."""

import argparse
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from oscardp.shots.transnet import sha256_file
from oscardp.stage06_2.evidence_replay import write_json
from oscardp.stage06_2.inputs import read
from .__main__ import MOVIE, SEQUENCE, load_tracks
from .inference import gaze_states
from .look_states import make_states

VERSION = "stage06.3.look_state.v1.2"


def numeric(full, qc_root, baseline1, baseline2):
    timeline = full/"sequence_gaze_timeline.json"
    mapping_path = baseline1/"character_track_mapping.json"
    orientation_path = baseline2/"camera_orientation_constraints.json"
    previous_path = baseline2/"gaze_target_candidates.json"
    qc_path = qc_root/"eye_observability_frames.parquet"
    source, mapping, orientation, previous = (read(p) for p in (timeline,mapping_path,orientation_path,previous_path))
    if any((x["movie_id"],x["sequence_id"])!=(MOVIE,SEQUENCE) for x in (source,mapping,orientation,previous)):
        raise ValueError("Movie or sequence mismatch")
    shots = source["shots"]
    expected = [f"shot_{n:06d}" for n in range(211,224)]
    if [s["shot_id"] for s in shots]!=expected:
        raise ValueError("Expected all 13 Shot records")
    ids = {m["track_id"]:m for m in mapping["tracks"]}
    qc = defaultdict(list)
    for row in pq.read_table(qc_path).to_pylist():
        qc[row["track_id"]].append(row)
    if set(qc)!={m["track_id"] for m in mapping["tracks"] if m["measured"]}:
        raise ValueError("QC Track coverage changed")
    provenance = {str(p):sha256_file(p) for p in (timeline,mapping_path,orientation_path,previous_path,qc_path)}
    states, events = [],[]
    for shot in shots:
        for t in shot["tracks"]:
            tid = t["track_id"]
            m = ids[tid]
            if m["shot_id"]!=shot["shot_id"] or m["measured"]!=("stats" in t):
                raise ValueError("Baseline identity / measurement coverage mismatch")
            if not m["measured"]:
                continue
            rows,gaze,prior,motion,_,stats = load_tracks(full,qc_root,shot,t,qc[tid],provenance)
            adjunct = read(Path(stats["motion_source"]))["direction_motion_features"]
            blinks = [e for e in motion if e["type"]=="blink_candidate"]
            if blinks:
                rows = [{**r,"gaze_direction":None,"head_facing_direction":None} if any(
                    e["start_frame"]<=r["global_frame_index"]<e["end_frame"] for e in blinks) else r for r in rows]
                gaze = gaze_states(rows)
            s,e = make_states(rows,gaze,[v for v in prior if v["type"]=="eye_position_stable"],motion,adjunct,m)
            states.extend(s)
            events.extend(e)
    return source,mapping,orientation,previous,states,events,qc,provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("full-sequence","eye-qc","baseline-v1","baseline-v2","layout","output-dir"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--observations",type=Path)
    args = parser.parse_args()
    protected = (args.full_sequence,args.eye_qc,args.baseline_v1,args.baseline_v2,args.layout.parent)
    if any(args.output_dir.resolve().is_relative_to(p.resolve()) or p.resolve().is_relative_to(args.output_dir.resolve())
           for p in protected):
        raise ValueError("Output must be independent of prior stages and Pilots")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    data = numeric(args.full_sequence,args.eye_qc,args.baseline_v1,args.baseline_v2)
    if args.observations is None:
        from .look_review import sample_visual
        args.output_dir.mkdir(parents=True,exist_ok=False)
        sample_visual(args.output_dir,*data[:2],data[4],data[5],data[6])
        print(f"{len(data[4])} Look States; original-frame packet at {args.output_dir}")
    else:
        from .look_targets import publish
        publish(args,*data)


if __name__=="__main__":
    main()
