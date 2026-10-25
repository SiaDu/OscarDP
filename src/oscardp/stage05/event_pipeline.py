"""Stage05.5 candidates from frozen measurements; never decode or call models."""
import csv
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .event_config import load_config, config_hash
from .event_inputs import load_measurements
from .event_detection import detect, validate_events, review_queue, summary
from .event_records import assign_ids
from .event_plots import timelines
from .identity_inputs import file_hash
from .output import atomic_json, load_json


def known_review(by,traces,events,eye_info):
    # User-requested audit window only; this code does not participate in detection.
    tid = "shot_000217_track_001"
    if (tid,"static") not in by: return {"status":"requested audit Track not selected"}
    geometries = {t["global_frame_index"]:t for t in traces if t["track_id"]==tid and t["feature"]=="eye_geometry"}
    rows = []
    other = {r["global_frame_index"]:r for r in by[tid,"dynamic"]}
    for r in by[tid,"static"]:
        n = r["global_frame_index"]
        if 35132<=n<=35160:
            t = geometries[n]
            rows.append({"global_frame_index":n,"movie_timestamp_sec":r["movie_timestamp_sec"],
                         "static_AU45_r":r["AU45_r"],"static_AU45_c":r["AU45_c"],
                         "dynamic_AU45_r":other[n]["AU45_r"],"dynamic_AU45_c":other[n]["AU45_c"],
                         "raw_ear":t["raw_values"],"processed_relative_ear":t["processed_values"],"geometry_valid":t["valid"]})
    return {"status":"independent_review_evidence_only","track_id":tid,"window_inclusive":[35132,35160],
            "eye_reference":eye_info[tid],"frames":rows,
            "overlapping_eye_event_ids":[e["event_id"] for e in events if e["track_id"]==tid and e["event_type"]=="eye_closure_candidate"
                                         and e["start_frame"]<=35160 and e["end_frame"]>35132],
            "reviewer":None,"review_status":"pending_independent_review","human_verified":False}


def run(args):
    started = time.perf_counter()
    root = args.output_dir
    if root.exists(): raise FileExistsError(f"Event pilot already exists; choose a new output directory: {root}")
    if root.resolve().is_relative_to(args.measurement_run.resolve()): raise ValueError("Event output must be independent of measurements")
    if args.au_mode!="static": raise ValueError("Only static AU can be the formal event baseline")
    config = load_config(args.config)
    by,comparison,source = load_measurements(args.measurement_run,args.movie_id,args.shot_ids)
    fingerprint = config_hash(config)
    run_config = {"schema_version":"stage05.5.events.v1","movie_id":args.movie_id,"selected_shot_ids":sorted(args.shot_ids),
                  "source_provenance":source,"algorithm_version":config["algorithm_version"],"thresholds_sha256":fingerprint,
                  "au_mode":"static","compare_dynamic":args.compare_dynamic,"human_verified":False,
                  "identity_status":"tentative","boundary_status":source["boundary_sidecar"]["mode"],
                  "no_model_inference_or_video_decode":True,"no_interpolation":True,
                  "source_code_sha256":{p.name:file_hash(p) for p in Path(__file__).parent.glob("event_*.py")}}
    if args.dry_run:
        print(json.dumps({"status":"preflight_passed","tracks":len(by)//2,"source_rows":sum(map(len,by.values())),
                          "thresholds_sha256":fingerprint,"config":config},indent=2))
        return 0
    root.mkdir(parents=True,exist_ok=False)
    (root/"previews").mkdir()
    atomic_json(root/"run_config.json",run_config)
    atomic_json(root/"thresholds.json",config)
    try:
        detection_started = time.perf_counter()
        events,rejected,traces,sensitivity,eye_info = detect(by,comparison,config,args.compare_dynamic)
        measurements_qc = load_json(args.measurement_run/"qc_report.json")
        for event in events:
            event["thresholds_sha256"] = fingerprint
            event["thresholds_file"] = "thresholds.json"
            event["boundary_status"] = source["boundary_sidecar"]["mode"]
            event["evidence"]["measurement_run"] = source["measurement_run"]
            for track in measurements_qc["tracks"]:
                if track["track_id"]!=event["track_id"]: continue
                for jump in track["jump_flags"]:
                    if (event["evidence"]["evidence_start_frame"]<=jump["global_frame_index"]<event["evidence"]["evidence_end_frame"]
                            and (event["feature"]==jump["field"] or event["feature"]=="eye_geometry")):
                        event["evidence"].setdefault("source_measurement_jump_flags",[]).append({"mode":track["extraction_mode"],**jump})
                        event["qc_flags"].append("source_measurement_spike_review")
        events = assign_ids(events)
        validate_events(events,by)
        detection_sec = time.perf_counter()-detection_started
        queue = review_queue(events,rejected,by,config,eye_info)
        atomic_json(root/"performance_events.json",{"schema_version":"stage05.5.events.v1","movie_id":args.movie_id,
                    "source_provenance":source,"thresholds_sha256":fingerprint,"primary_au_mode":"static",
                    "human_verified":False,"events":events})
        with (root/"event_candidates.jsonl").open("x") as stream:
            for event in events: stream.write(json.dumps(event,allow_nan=False,sort_keys=True)+"\n")
        atomic_json(root/"rejected_candidates.json",{"status":"algorithm_rejections_not_ground_truth","intervals":rejected})
        atomic_json(root/"sensitivity_analysis.json",{"formal_event_stream":False,"dynamic_comparisons":sensitivity,
                    "eye_geometry":eye_info,"human_verified":False})
        atomic_json(root/"known_interval_review.json",known_review(by,traces,events,eye_info))
        with (root/"event_review_queue.csv").open("x",newline="") as stream:
            writer = csv.DictWriter(stream,fieldnames=list(queue[0]) if queue else ["review_id","event_id","review_status"])
            writer.writeheader()
            writer.writerows(queue)
        pq.write_table(pa.Table.from_pylist(traces),root/"processed_timeseries.parquet",compression="zstd")
        atomic_json(root/"stage06_event_preview.json",{"schema_version":"stage05.5.event_references.v1","annotation_generated":False,
                    "source":"performance_events.json","tracks":[{"shot_id":data[0]["shot_id"],"scene_id":data[0].get("scene_id"),
                         "track_id":tid,"identity_cluster_id":data[0]["identity_cluster_id"],"identity_status":data[0]["identity_status"],
                         "event_ids":[e["event_id"] for e in events if e["track_id"]==tid]}
                         for (tid,mode),data in sorted(by.items()) if mode=="static"]})
        timelines(root,by,traces,events,eye_info,config,args.compare_dynamic)
        unchanged = all(file_hash(Path(path))==sha for path,sha in source["protected_source_sha256"].items())
        if not unchanged: raise ValueError("Protected Stage01–05.4 input changed during event detection")
        report = summary(events,rejected)
        # Include selected tracks even if no rule passes.
        for tid,mode in by:
            if mode=="static": report["by_track"].setdefault(tid,{})
        report.update(status="completed_algorithm_candidates",source_hashes_unchanged=unchanged,review_queue_rows=len(queue),
                      unique_source_frames=sum(len(data) for (tid,mode),data in by.items() if mode=="static"),
                      timing={"total_sec":time.perf_counter()-started,"detection_sec":detection_sec},
                      limitations=["No human event ground truth or accuracy calibration", "Descriptive baseline is not neutral facial expression",
                                   "Eye closures are relative geometry candidates, not verified blinks", "Head/gaze are camera-relative heuristic-intrinsics model estimates",
                                   "Net motion displacement rule can miss return-to-origin motions", "Static/dynamic sensitivity is not a second formal event stream"],
                      human_verified=False)
        atomic_json(root/"event_qc_report.json",report)
        artifacts = {str(p.relative_to(root)):file_hash(p) for p in root.rglob("*") if p.is_file()}
        atomic_json(root/"artifact_sha256.json",artifacts)
        print(json.dumps(report,indent=2))
    except (Exception,KeyboardInterrupt) as exc:
        atomic_json(root/"failure.json",{"status":"failed","error":f"{type(exc).__name__}: {exc}"})
        raise
    return 0
