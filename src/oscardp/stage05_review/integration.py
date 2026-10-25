"""Read actual prototype structures and retain separate temporal/identity authorities."""
import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from oscardp.stage05.identity_inputs import file_hash


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def load_sources(event_run):
    document = read(event_run/"performance_events.json")
    run = read(event_run/"run_config.json")
    source = run["source_provenance"]
    if document["source_provenance"] != source:
        raise ValueError("Event/run source provenance mismatch")
    protected = dict(source["protected_source_sha256"])
    for name, expected in read(event_run/"artifact_sha256.json").items():
        protected[str((event_run/name).resolve())] = expected
    for name, expected in source["measurement_files_sha256"].items():
        protected[str(Path(source["measurement_run"])/name)] = expected
    for path, expected in protected.items():
        if file_hash(Path(path)) != expected:
            raise ValueError("Source fingerprint mismatch: "+path)
    for path in event_run.rglob("*"):
        if path.is_file():
            protected[str(path.resolve())] = file_hash(path)
    stage = {}
    for n in range(1, 5):
        paths = [Path(p) for p in protected if Path(p).name == f"Stage0{n}_{document['movie_id']}.json"]
        if len(paths) != 1:
            raise ValueError("Missing/ambiguous Stage01–04 provenance")
        stage[n] = read(paths[0])
        if stage[n]["movie_id"] != document["movie_id"]:
            raise ValueError("Stage movie mismatch")
    earlier = source["earlier_provenance"]
    tracking = read(Path(earlier["tracking_run"])/"face_tracks.json")
    tracking_config = read(Path(earlier["tracking_run"])/"run_config.json")
    identities = read(Path(earlier["identity_run"])/"identity_clusters.json")
    video = Path(earlier["video"]["path"])
    if video.stat().st_size != earlier["video"]["size"]:
        raise ValueError("Source movie size changed")
    protected[str(video)] = file_hash(video)
    events = document["events"]
    tracks = {t["track_id"]: t for t in tracking["tracks"]}
    membership = {}
    for cluster in identities["clusters"]:
        for tid in cluster["track_ids"]:
            if tid in membership:
                raise ValueError("Duplicate identity membership")
            membership[tid] = cluster["cluster_id"]
    seen = set()
    for e in events:
        if e["event_id"] in seen:
            raise ValueError("Duplicate Event ID")
        seen.add(e["event_id"])
        t = tracks[e["track_id"]]
        if (t["shot_id"] != e["shot_id"] or membership[e["track_id"]] != e["identity_cluster_id"]
                or e["identity_status"] != "tentative" or e["human_verified"] is not False
                or e["review_status"] != "algorithm_candidate"
                or not t["start_frame"] <= e["start_frame"] < e["end_frame"] <= t["end_frame"]
                or e["boundary_status"] != source["boundary_sidecar"]["mode"]):
            raise ValueError("Event Track/Cluster/range/status integrity failure")
    with (event_run/"event_review_queue.csv").open(newline="") as stream:
        queue = list(csv.DictReader(stream))
    if len({r["review_id"] for r in queue}) != len(queue):
        raise ValueError("Duplicate Review ID")
    event_map = {e["event_id"]: e for e in events}
    for r in queue:
        t = tracks[r["track_id"]]
        if (r["shot_id"] != t["shot_id"] or r["reviewer"] or r["human_verified"] != "False"
                or not t["start_frame"] <= int(r["start_frame"]) < int(r["end_frame"]) <= t["end_frame"]):
            raise ValueError("Queue Track/range/review integrity failure")
        if r["event_id"]:
            e = event_map[r["event_id"]]
            if any(str(e[k]) != r[k] for k in ["shot_id", "track_id", "event_type", "feature", "start_frame", "end_frame"]):
                raise ValueError("Queue Event mismatch")
    if set(event_map) != {r["event_id"] for r in queue if r["event_id"]}:
        raise ValueError("Queue omits event references")
    preview = read(event_run/"stage06_event_preview.json")
    if preview["annotation_generated"] is not False:
        raise ValueError("Unexpected generated Stage06 annotation")
    for t in preview["tracks"]:
        if t["event_ids"] != [e["event_id"] for e in events if e["track_id"] == t["track_id"]]:
            raise ValueError("Stage06 preview references mismatch")
    traces = pq.read_table(event_run/"processed_timeseries.parquet").to_pylist()
    # Read required QC and threshold inputs, retaining their immutable references.
    qc, thresholds = read(event_run/"event_qc_report.json"), read(event_run/"thresholds.json")
    if qc["event_count"] != len(events) or qc["review_queue_rows"] != len(queue):
        raise ValueError("Stage05.5 QC counts mismatch")
    return dict(document=document, run=run, stage=stage, tracking=tracking, tracking_config=tracking_config,
                identities=identities, membership=membership, tracks=tracks, queue=queue, traces=traces,
                protected=protected, source_hash=digest(protected), video=video, qc=qc, thresholds=thresholds,
                event_run=str(event_run.resolve()))


def build_packets(data):
    stage = data["stage"]
    s2 = {s["shot_id"]: s for s in stage[2]["shots"]}
    s4 = {s["shot_id"]: s for s in stage[4]["shots"]}
    s1 = {s["shot_id"]: s for s in stage[1]["shots"]}
    s3 = {s["shot_id"]: s for s in stage[3]["shots"]}
    scenes = {s["scene_id"]: s for s in stage[4]["scenes"]}
    output_scenes = {}
    for shot in data["tracking"]["shot_ranges"]:
        sid = shot["shot_id"]
        if sid not in data["run"]["selected_shot_ids"]:
            continue
        context = s4[sid]
        scene_id = context["scene_id"]
        # Stage04 prototype uses per-shot scene_id, NOT a proposal shot_ids array.
        scene = output_scenes.setdefault(scene_id, {"scene_id": scene_id, "context": scenes[scene_id], "shots": []})
        effective = shot["effective_processing_temporal"]
        tracks = []
        for tid, t in data["tracks"].items():
            if t["shot_id"] != sid:
                continue
            event_list = [e for e in data["document"]["events"] if e["track_id"] == tid]
            invalid = {}
            for row in data["traces"]:
                if row["track_id"] == tid and not row["valid"]:
                    invalid.setdefault(row["feature"], []).append(row["global_frame_index"])
            tracks.append({"track_id": tid, "identity_cluster": {"cluster_id": data["membership"][tid],
                "identity_status": "tentative", "character_mapping_status": "unverified",
                "candidate_mapping": {"candidates": context.get("character_candidates", []),
                    "source": "Stage04 shot-local candidates; not Dialogue Speaker",
                    "track_to_stage03_subject_match": "unverified", "human_verified": False}},
                "event_ids": [e["event_id"] for e in event_list],
                "human_review_status": "pending_independent_review", "human_verified": False,
                "invalid_frames_by_feature": invalid,
                "temporal_unknowns": ["Invalid frames are not interpolated", "No pose/gaze continuity across shots",
                                      "Algorithm-negative intervals are not negative ground truth"]})
        scene["shots"].append({"shot_id": sid, "effective_processing_temporal": effective,
            "original_stage01_temporal": s1[sid]["temporal"], "stage04_context_temporal": context["temporal"],
            "boundary_status": data["run"]["boundary_status"],
            "boundary_review": data["run"]["source_provenance"]["boundary_sidecar"]["reviews"][sid],
            "stage02_dialogue": s2[sid]["dialogue"], "stage02_narrative": s2[sid]["narrative"],
            "stage02_dialogue_temporal_basis": "original Stage01 shot association; not retimed to corrected frames",
            "stage03_reference": {"shot_id": s3[sid]["shot_id"], "source": "Stage03", "role": "context only"},
            "stage04_character_candidates": context.get("character_candidates", []),
            "face_tracks": tracks, "evidence": {"video": f"viewer/media/{sid}.mp4",
                "frame_map": f"viewer/data/{sid}_frames.json", "timeline": f"viewer/data/{sid}_timeline.json"}})
    compact_events = []
    for e in data["document"]["events"]:
        keys = ["event_id", "content_sha256", "shot_id", "track_id", "event_type", "feature", "start_frame",
                "end_frame", "start_sec", "end_sec", "onset_frame", "apex_frame", "offset_frame", "apex_plateau",
                "left_censored", "right_censored", "measurements", "qc_flags", "boundary_status", "review_status", "human_verified"]
        compact_events.append({**{k: e[k] for k in keys},
            "evidence_reference": {"source": data["event_run"]+"/performance_events.json",
                "event_id": e["event_id"], "processed_trace_key": e["evidence"]["processed_trace_key"]}})
    ordered = sorted([s for scene in output_scenes.values() for s in scene["shots"]],
                     key=lambda s: s["effective_processing_temporal"]["start_frame"])
    gaps = []
    for left, right in zip(ordered, ordered[1:]):
        a = left["effective_processing_temporal"]["end_frame"]
        b = right["effective_processing_temporal"]["start_frame"]
        if a < b:
            gaps.append({"frame_range": [a, b], "status": "not_processed_in_pilot", "interpolation": False})
    return {"schema_version": "stage05.6.scene_packet.v1", "movie_id": data["document"]["movie_id"],
        "source_run_sha256": data["source_hash"], "annotation_generated": False,
        "evidence_status": "algorithm_candidate", "human_verified": False,
        "identity_status": "tentative", "character_mapping_status": "unverified",
        "stage04_version": stage[4].get("stage_version"), "stage04_interpretation_policy": stage[4].get("interpretation_policy"),
        "policies": ["Dialogue Speaker is narrative attribution, not visual identity or active-speaker truth",
                     "Stage04 old times never replace effective processing temporal",
                     "No emotion or intent inference; simultaneous AU retain independent event IDs",
                     "Each Shot is its own camera coordinate system; no cross-shot interpolation",
                     "Human review lives in a separate overlay; this packet is an immutable unreviewed snapshot"],
        "scenes": list(output_scenes.values()), "event_candidates": compact_events,
        "temporal_gaps": gaps, "unknowns": ["Most of the scene is outside this five-Shot pilot",
            "Boundary corrections, character mappings and event presence require independent human video review"]}
