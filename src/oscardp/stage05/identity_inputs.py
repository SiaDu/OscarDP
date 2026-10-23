"""Read-only validation of the corrected Stage05.2 run and its actual observations."""
import hashlib
import json
import math

from .output import identity, load_json


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_corrected(root, movie_id, shot_ids):
    if not shot_ids or len(set(shot_ids)) != len(shot_ids):
        raise ValueError("Explicit unique pilot shot IDs required")
    required = ["face_tracks.json", "face_detections.jsonl", "run_config.json", "qc_report.json"]
    inputs = {name: {**identity(root/name), "sha256": file_hash(root/name)} for name in required}
    config = load_json(root/"run_config.json")
    tracks_file = load_json(root/"face_tracks.json")
    qc = load_json(root/"qc_report.json")
    boundary = config.get("boundary_sidecar", {})
    if (config["movie_id"] != movie_id or tracks_file["movie_id"] != movie_id
            or boundary.get("mode") != "experimental_model_reviewed_pending_human"
            or boundary.get("human_verified") is not False):
        raise ValueError("Stage05.3 pilot requires the experimental corrected Stage05.2 provenance, not the original run")
    if tracks_file.get("boundary_sidecar") != boundary or qc.get("boundary_sidecar") != boundary:
        raise ValueError("Corrected provenance differs between config, tracks and QC")
    from pathlib import Path
    sidecar = Path(boundary["path"])
    if file_hash(sidecar) != boundary["sha256"]:
        raise ValueError("Boundary sidecar changed since Stage05.2")
    for review in boundary["reviews"].values():
        if review["human_verified"] or review["approved_for_processing"] or review["status"] != "model_reviewed_pending_human":
            raise ValueError("Boundary review status differs from the expected experimental draft")
    if not set(shot_ids) <= set(boundary["applied_shot_ids"]):
        raise ValueError("Selected shot does not have an explicitly applied correction")
    source = Path(config["stage01"]["path"])
    if file_hash(source) != config["stage01_sha256"]:
        raise ValueError("Stage01 source SHA256 mismatch")
    video = Path(config["video"]["path"])
    if identity(video) != config["video"]:
        raise ValueError("Source video identity changed")
    shots = {s["shot_id"]: s for s in config["shots"]}
    ranges = {s["shot_id"]: s for s in tracks_file.get("shot_ranges", [])}
    selected = set(shot_ids)
    tracks = {t["track_id"]: t for t in tracks_file["tracks"] if t["shot_id"] in selected}
    if len(tracks) != sum(t["shot_id"] in selected for t in tracks_file["tracks"]):
        raise ValueError("Duplicate Track IDs")
    qc_shots = {s["shot_id"]: s for s in qc["shots"]}
    for sid in shot_ids:
        if sid not in shots or sid not in ranges or qc_shots[sid]["status"] != "complete":
            raise ValueError(f"Missing/incomplete corrected shot: {sid}")
        shot = shots[sid]
        if shot["temporal"] != shot["effective_processing_temporal"] or shot["original_temporal"] == shot["effective_processing_temporal"]:
            raise ValueError(f"Original range accidentally supplied instead of corrected range: {sid}")
        for key in ["original_temporal","effective_processing_temporal"]:
            if ranges[sid][key] != shot[key] or qc_shots[sid][key] != shot[key]:
                raise ValueError(f"Shot provenance mismatch: {sid}")
    records = {tid: [] for tid in tracks}
    seen = set()
    with (root/"face_detections.jsonl").open() as stream:
        for line in stream:
            r = json.loads(line)
            if r["shot_id"] not in selected:
                continue
            tid = r["track_id"]
            if tid not in tracks or r["shot_id"] != tracks[tid]["shot_id"] or r["detection_status"] != "observed":
                raise ValueError("Detection not an observed corrected Track")
            shot = shots[r["shot_id"]]
            for key in ["original_temporal","effective_processing_temporal"]:
                if r[key] != shot[key] or tracks[tid][key] != shot[key]:
                    raise ValueError("Detection/Track corrected range mismatch")
            t = shot["effective_processing_temporal"]
            if type(r["frame_index"]) is not int or not t["start_frame"] <= r["frame_index"] < t["end_frame"]:
                raise ValueError("Observation outside effective range")
            if not math.isfinite(r["timestamp_sec"]) or not t["start"]-1e-6 <= r["timestamp_sec"] < t["end"]:
                raise ValueError("Observation outside effective timestamps")
            if config["timestamp_rule"] == "frame_index_cfr" and abs(r["timestamp_sec"]-r["frame_index"]/config["fps"]) > 1e-6:
                raise ValueError("Observation CFR timestamp mismatch")
            key = (tid,r["frame_index"])
            if key in seen:
                raise ValueError("Duplicate observation")
            seen.add(key)
            records[tid].append(r)
    for tid,t in tracks.items():
        rows = sorted(records[tid],key=lambda r:r["frame_index"])
        if not rows or len(rows) != t["detected_frame_count"]:
            raise ValueError(f"Track observations missing or summary mismatch: {tid}")
        if rows[0]["frame_index"] != t["start_frame"] or rows[-1]["frame_index"]+1 != t["end_frame"]:
            raise ValueError("Track observation endpoints mismatch")
        records[tid] = rows
    if not tracks:
        raise ValueError("No actual face tracks selected")
    return config, tracks, records, inputs
