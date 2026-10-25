"""Candidate contract, source evidence and deterministic identifiers."""
import hashlib
import json

import numpy as np


def candidate(rows, indices, a, b, event_type, feature, validity, config, apex=None):
    first,last = rows[indices[a]],rows[indices[b-1]]
    left,right = a==0,b==len(indices)
    onset = None if left else first["global_frame_index"]
    offset = None if right else rows[indices[b]]["global_frame_index"]
    event = {"movie_id":first["movie_id"],"scene_id":first.get("scene_id"),"shot_id":first["shot_id"],
             "track_id":first["track_id"],"identity_cluster_id":first["identity_cluster_id"],
             "identity_status":"tentative" if first["identity_status"]=="tentative_identity_cluster" else "unknown",
             "event_type":event_type,"feature":feature,"extraction_mode":first["extraction_mode"],
             "start_frame":first["global_frame_index"],"end_frame":last["global_frame_index"]+1,
             "onset_frame":onset,"apex_frame":rows[indices[apex]]["global_frame_index"] if apex is not None else None,
             "offset_frame":offset,"start_sec":first["movie_timestamp_sec"],"end_sec":last["source_next_timestamp_sec"],
             "left_censored":left,"right_censored":right,"apex_plateau":None,
             "measurements":{},"evidence":{"source_feature":feature,"source_identity_status":first["identity_status"],
               "source_measurement_qc":{"mask":validity,"all_event_frames_valid":True,
                   "parquet_rows":[rows[i]["parquet_row"] for i in indices[a:b]]},
               "evidence_start_frame":rows[indices[max(0,a-1)]]["global_frame_index"],
               "evidence_end_frame":rows[indices[min(len(indices)-1,b)]]["global_frame_index"]+1,
               "processed_trace_key":f"{first['track_id']}/{first['extraction_mode']}/{feature}"},
             "qc_flags":[],"rule_support":{"is_probability":False,"description":"Configured numerical rules passed; no event accuracy probability"},
             "algorithm_version":config["algorithm_version"],"review_status":"algorithm_candidate","review_needed":True,
             "human_verified":False}
    if left or right: event["qc_flags"].append("censored_event")
    boundary = config["series"]["boundary_review_sec"]
    if first["movie_timestamp_sec"]-rows[0]["movie_timestamp_sec"]<=boundary or rows[-1]["source_next_timestamp_sec"]-last["source_next_timestamp_sec"]<=boundary:
        event["qc_flags"].append("near_shot_boundary")
    if any(not r[validity] and min(abs(r["movie_timestamp_sec"]-first["movie_timestamp_sec"]),
                                 abs(r["movie_timestamp_sec"]-last["movie_timestamp_sec"]))<=boundary for r in rows):
        event["qc_flags"].append("low_quality_neighbor_or_invalid_segment_edge")
    return event


def rejection(rows,indices,a,b,event_type,feature,reasons):
    return {"shot_id":rows[indices[a]]["shot_id"],"track_id":rows[indices[a]]["track_id"],"event_type":event_type,
            "feature":feature,"start_frame":rows[indices[a]]["global_frame_index"],"end_frame":rows[indices[b-1]]["global_frame_index"]+1,
            "reasons":reasons,"review_status":"algorithm_rejected","human_verified":False}


def trace(rows,feature,raw,processed,mask,derived=None):
    def clean(value):
        array = np.atleast_1d(value)
        return [float(x) if x is not None and np.isfinite(x) else None for x in array]
    return [{"track_id":r["track_id"],"shot_id":r["shot_id"],"extraction_mode":r["extraction_mode"],"feature":feature,
             "global_frame_index":r["global_frame_index"],"movie_timestamp_sec":r["movie_timestamp_sec"],
             "source_parquet_row":r["parquet_row"],"valid":bool(mask[i]),"raw_values":clean(raw[i]),
             "processed_values":clean(processed[i]) if mask[i] else None,
             "derived_value":float(derived[i]) if derived is not None and mask[i] else None} for i,r in enumerate(rows)]


def assign_ids(events):
    events = sorted(events,key=lambda e:(e["shot_id"],e["track_id"],e["start_frame"],e["event_type"],e["feature"],e["end_frame"]))
    for i,event in enumerate(events,1):
        event["event_id"] = f"event_{i:06d}"
        event["qc_flags"] = sorted(set(event["qc_flags"]))
        payload = {k:v for k,v in event.items() if k not in {"event_id","content_sha256"}}
        event["content_sha256"] = hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False,separators=(",",":")).encode()).hexdigest()
    return events
