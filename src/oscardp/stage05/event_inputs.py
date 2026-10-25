"""Validate actual measurement rows against raw CSV, masks and earlier provenance."""
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .identity_inputs import file_hash
from .output import load_json
from .performance_inputs import load_inputs, snapshot
from .performance_csv import parse_csv, columns_by_modality
from .performance_qc import assess

MASKS = ["openface_row_present","openface_success","tracking_usable","au_valid","head_pose_valid",
         "gaze_valid","landmarks_2d_valid","landmarks_3d_valid"]


def require(ok,message):
    if not ok: raise ValueError(message)


def validate_rows(rows,status,config,measurements,qc,tracks,identities,root,observations):
    require(len(rows)==len(status)==qc["measurement_rows"],"Measurement/status row count mismatch")
    by,seen = {},set()
    for index,(row,state) in enumerate(zip(rows,status)):
        require(row["parquet_row"]==state["parquet_row"]==index,"Parquet reference mismatch")
        for key in ["movie_id","shot_id","track_id","identity_cluster_id","identity_status","identity_human_verified",
                    "global_frame_index","movie_timestamp_sec","source_next_timestamp_sec","openface_local_frame","extraction_mode"]:
            require(row[key]==state[key],f"Measurement/status mismatch: {key}")
        for key in MASKS:
            require(type(row[key]) is bool and row[key]==state["reliability"][key],f"Source QC mask mismatch: {key}")
        require(row["movie_id"]==config["movie_id"],"Measurement movie mismatch")
        tid,mode = row["track_id"],row["extraction_mode"]
        require(tid in tracks and mode in config["au_modes"],"Unknown Track/mode")
        require(all(row[k]==v for k,v in identities[tid].items()),"Identity upgraded or changed")
        key = (tid,mode,row["global_frame_index"])
        require(key not in seen,"Duplicate measurement frame")
        seen.add(key)
        by.setdefault((tid,mode),[]).append(row)
    maps = {sid:load_json(root/f"frame_maps/{sid}.json") for sid in config["selected_shot_ids"]}
    summaries = {(r["track_id"],r["extraction_mode"]):r for r in measurements["tracks"]}
    for tid,track in tracks.items():
        sid = track["shot_id"]
        t = track["effective_processing_temporal"]
        observation_lookup = {r["frame_index"]:r for r in observations[tid]}
        for mode in config["au_modes"]:
            data = by.get((tid,mode),[])
            require([r["global_frame_index"] for r in data]==list(range(t["start_frame"],t["end_frame"])),"Missing/noncontinuous/reordered source frame")
            times = np.array([r["movie_timestamp_sec"] for r in data])
            require(np.isfinite(times).all() and np.all(np.diff(times)>0),"Nonmonotonic source movie timestamps")
            require(all(r["source_next_timestamp_sec"]>r["movie_timestamp_sec"] for r in data),"Invalid source next PTS")
            require(all(abs(a["source_next_timestamp_sec"]-b["movie_timestamp_sec"])<1e-6 for a,b in zip(data,data[1:])),"Source timestamp boundary mismatch")
            summary = summaries[tid,mode]
            for key in ["original_temporal","effective_processing_temporal"]:
                require(summary[key]==track[key]==maps[sid][key],"Measurement range/provenance mismatch")
            path = root/summary["raw_csv"]
            require(file_hash(path)==summary["execution"]["raw_csv_sha256"],"Raw OpenFace CSV SHA256 mismatch")
            header,parsed = parse_csv(path,len(data),config["camera_fps_for_validation"])
            groups = columns_by_modality(header)
            for local,row in enumerate(data,1):
                mapped = maps[sid]["frames"][local-1]
                require(row["openface_local_frame"]==local and row["shot_id"]==sid,"Local/source Shot frame mismatch")
                require(row["global_frame_index"]==mapped["global_frame_index"] and row["movie_timestamp_sec"]==mapped["movie_timestamp_sec"],"Source frame-map mismatch")
                raw = parsed.get(local)
                for name in header:
                    if name in {"frame","face_id","timestamp","confidence","success"}: continue
                    require(row[name]==(raw[name] if raw else None),f"Parquet/raw CSV value mismatch: {name}")
                flags = assess(raw,groups,observation_lookup.get(row["global_frame_index"]),config["qc_parameters"]["tracking_confidence"])
                require(all(row[k]==flags[k] for k in MASKS),"QC mask does not match original measurement rule/evidence")
            for flag,count in summary["valid_counts"].items():
                require(sum(r[flag] for r in data)==count,"Track source QC count mismatch")
        a,b = by[tid,"static"],by[tid,"dynamic"]
        for left,right in zip(a,b):
            require(all(left[k]==right[k] for k in ["global_frame_index","movie_timestamp_sec","source_next_timestamp_sec"]),"Static/dynamic frame identity mismatch")
            require(all(left[n]==right[n] for n in left if n.startswith(("pose_","gaze_"))),"Duplicate Pose/Gaze measurement modes disagree")
    return by


def load_measurements(root,movie_id,shot_ids):
    required = ["frame_features.parquet","frame_status.jsonl","track_measurements.json","qc_report.json","run_config.json",
                "comparisons/au_static_vs_dynamic.csv"]
    hashes = {name:file_hash(root/name) for name in required}
    config,measurements,qc = [load_json(root/name) for name in ["run_config.json","track_measurements.json","qc_report.json"]]
    require(config["schema_version"]==measurements["schema_version"]=="stage05.4.frame_measurements.v1","Unsupported Stage05.4 schema")
    require(config["movie_id"]==measurements["movie_id"]==movie_id,"Measurement movie mismatch")
    require(shot_ids and len(set(shot_ids))==len(shot_ids) and set(shot_ids)<=set(config["selected_shot_ids"]),"Explicit valid unique shots required")
    require(config["au_modes"]==["static","dynamic"] and config["main_au_mode"]=="static","Pilot requires separate static primary and dynamic comparison measurements")
    source = config["source_provenance"]
    original,tracks,observations,identities,provenance = load_inputs(Path(source["tracking_run"]),Path(source["identity_run"]),movie_id,config["selected_shot_ids"])
    require(source==provenance,"Stage05.2/05.3 provenance mismatch")
    require(config["boundary_sidecar"]==measurements["boundary_sidecar"]==qc["boundary_sidecar"]==original["boundary_sidecar"],"Boundary metadata mismatch")
    require(config["boundary_sidecar"]["mode"]=="experimental_model_reviewed_pending_human" and not config["boundary_sidecar"]["human_verified"],"Boundary review state changed")
    require(qc["source_hashes_unchanged"] and qc["status"]=="completed_frame_measurements_with_qc","Unfinished or invalid measurement run")
    table = pq.read_table(root/"frame_features.parquet")
    require(table.schema.metadata[b"schema_version"].decode()==config["schema_version"],"Parquet schema version mismatch")
    require(json.loads(table.schema.metadata[b"source_provenance"])==provenance,"Parquet provenance mismatch")
    require(table.schema.metadata[b"main_au_mode"]==b"static","Parquet primary mode mismatch")
    rows = table.to_pylist()
    status = [json.loads(line) for line in (root/"frame_status.jsonl").read_text().splitlines()]
    validated_config = dict(config,camera_fps_for_validation=original["fps"])
    by = validate_rows(rows,status,validated_config,measurements,qc,tracks,identities,root,observations)
    with (root/"comparisons/au_static_vs_dynamic.csv").open() as stream:
        comparison = {(r["track_id"],r["au"]):r for r in csv.DictReader(stream)}
    expected_comparisons = {(tid,n) for tid in tracks for n in measurements["modalities"]["au_intensity"]}
    require(set(comparison)==expected_comparisons,"Incomplete mode comparison table")
    require(qc["tracks"]==measurements["tracks"] and qc["human_verified"] is False,"Measurement QC summary mismatch")
    for (tid,feature),stats in comparison.items():
        static,dynamic = by[tid,"static"],by[tid,"dynamic"]
        pairs = [(a[feature],b[feature]) for a,b in zip(static,dynamic) if a["au_valid"] and b["au_valid"]]
        require(len(pairs)==int(stats["paired_valid_count"]),"Mode comparison count mismatch")
        if pairs:
            a,b = np.array(pairs).T
            require(abs(float(stats["paired_rmse"])-float(np.sqrt(np.mean((a-b)**2))))<1e-8,"Mode comparison RMSE mismatch")
            if stats["paired_pearson"]:
                require(abs(float(stats["paired_pearson"])-float(np.corrcoef(a,b)[0,1]))<1e-8,"Mode correlation mismatch")
    protected = dict(source["protected_source_sha256"])
    protected.update(snapshot(root.rglob("*")))
    selected = {(tid,mode):data for (tid,mode),data in by.items() if tracks[tid]["shot_id"] in shot_ids}
    return selected,comparison,{"measurement_run":str(root.resolve()),"measurement_files_sha256":hashes,
            "measurement_schema":config["schema_version"],"boundary_sidecar":config["boundary_sidecar"],
            "earlier_provenance":source,"protected_source_sha256":protected,
            "hash_verification":"Current input hashes recorded; raw CSV hashes and full Parquet/mask/source referential integrity checked. No signed release manifest exists."}
