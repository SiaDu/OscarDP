"""Read-only source/phase integrity checks and deterministic numerical replay."""
import argparse
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq

from oscardp.stage05.event_config import load_config,config_hash
from oscardp.stage05.event_detection import detect,validate_events,review_queue,summary
from oscardp.stage05.event_inputs import load_measurements
from oscardp.stage05.event_records import assign_ids
from oscardp.stage05.identity_inputs import file_hash


def require(ok,message):
    if not ok: raise ValueError(message)


def verify(root):
    run = json.loads((root/"run_config.json").read_text())
    document = json.loads((root/"performance_events.json").read_text())
    for name,sha in json.loads((root/"artifact_sha256.json").read_text()).items():
        require(file_hash(root/name)==sha,f"Pilot artifact SHA mismatch: {name}")
    config = load_config(root/"thresholds.json")
    require(config_hash(config)==run["thresholds_sha256"]==document["thresholds_sha256"],"Threshold fingerprint mismatch")
    by,comparison,source = load_measurements(Path(run["source_provenance"]["measurement_run"]),run["movie_id"],run["selected_shot_ids"])
    require(source==run["source_provenance"]==document["source_provenance"],"Input provenance changed")
    for name,sha in run["source_code_sha256"].items():
        path = Path(__file__).resolve().parents[1]/"src/oscardp/stage05"/name
        require(file_hash(path)==sha,"Detector source changed since original run; preserve original implementation to replay")
    events,rejected,traces,sensitivity,eye_info = detect(by,comparison,config,run["compare_dynamic"])
    measurement_qc = json.loads((Path(source["measurement_run"])/"qc_report.json").read_text())
    for event in events:
        event.update(thresholds_sha256=config_hash(config),thresholds_file="thresholds.json",boundary_status=source["boundary_sidecar"]["mode"])
        event["evidence"]["measurement_run"] = source["measurement_run"]
        for track in measurement_qc["tracks"]:
            if track["track_id"]!=event["track_id"]: continue
            for jump in track["jump_flags"]:
                if (event["evidence"]["evidence_start_frame"]<=jump["global_frame_index"]<event["evidence"]["evidence_end_frame"]
                        and (event["feature"]==jump["field"] or event["feature"]=="eye_geometry")):
                    event["evidence"].setdefault("source_measurement_jump_flags",[]).append({"mode":track["extraction_mode"],**jump})
                    event["qc_flags"].append("source_measurement_spike_review")
    events = assign_ids(events)
    validate_events(events,by)
    require(events==document["events"],"Numerical replay/event IDs differ")
    jsonl = [json.loads(line) for line in (root/"event_candidates.jsonl").read_text().splitlines()]
    require(jsonl==events,"Event JSON/JSONL mismatch")
    require(pq.read_table(root/"processed_timeseries.parquet").to_pylist()==traces,"Raw/processed trace replay mismatch")
    require(json.loads((root/"rejected_candidates.json").read_text())["intervals"]==rejected,"Rejection replay differs")
    alternative = json.loads((root/"sensitivity_analysis.json").read_text())
    require(alternative["dynamic_comparisons"]==sensitivity and alternative["eye_geometry"]==eye_info,"Sensitivity replay differs")
    expected = review_queue(events,rejected,by,config,eye_info)
    with (root/"event_review_queue.csv").open() as stream: actual = list(csv.DictReader(stream))
    require(actual==[{k:"" if v is None else str(v) for k,v in row.items()} for row in expected],"Review queue mismatch")
    preview = json.loads((root/"stage06_event_preview.json").read_text())
    require(preview["annotation_generated"] is False,"Stage06 annotation unexpectedly generated")
    for track in preview["tracks"]:
        require(track["event_ids"]==[e["event_id"] for e in events if e["track_id"]==track["track_id"]],"Event preview reference mismatch")
    qc = json.loads((root/"event_qc_report.json").read_text())
    stats = summary(events,rejected)
    for key in ["event_count","by_type","rejected_interval_count","qc_flagged_event_count","phase_observability"]:
        require(stats[key]==qc[key],f"QC statistics mismatch: {key}")
    return {"status":"passed","event_count":len(events),"source_hashes_unchanged":True,
            "processed_trace_rows":len(traces),"deterministic_replay_identical":True,"human_verified":False,
            "checks":["all original artifact fingerprints", "raw measurement/provenance/mask validation",
                      "source algorithm/config fingerprints", "deterministic event IDs and numerical replay",
                      "invalid frames absent from event phases", "JSON/JSONL/Parquet/review references",
                      "static-only formal stream and separate dynamic sensitivity", "no Stage06 annotation"]}


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot_dir",type=Path)
    parser.add_argument("--output",type=Path,help="New verification report; existing files refused")
    args = parser.parse_args()
    if args.output and args.output.exists(): parser.error("Verification output already exists")
    payload = json.dumps(verify(args.pilot_dir),indent=2)+"\n"
    if args.output:
        with args.output.open("x") as stream: stream.write(payload)
    print(payload)
