"""Load one Track and audit the published Stage05 coverage read-only."""

from collections import Counter
import json
import math

import pyarrow.parquet as pq

from oscardp.shots.transnet import sha256_file

POSE = [f"pose_R{axis}" for axis in "xyz"]
EYES = [[f"gaze_{i}_{axis}" for axis in "xyz"] for i in range(2)]
MASKS = ["openface_row_present", "openface_success", "tracking_usable", "head_pose_valid",
         "gaze_valid", "landmarks_2d_valid", "landmarks_3d_valid"]
KEYS = ["movie_id", "shot_id", "track_id", "identity_cluster_id", "identity_status",
        "identity_human_verified", "global_frame_index", "movie_timestamp_sec",
        "source_next_timestamp_sec", "extraction_mode", "openface_local_frame", "openface_confidence"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(ok, message):
    if not ok:
        raise ValueError(message)


def load(args, extra_columns=()):
    path = args.measurement_run
    summary = read(path / "track_measurements.json")
    track_path = args.tracking_run / "face_tracks.json"
    identity_path = args.identity_run / "identity_clusters.json"
    event_path = args.event_run / "performance_events.json"
    tracks, identities, events = map(read, (track_path, identity_path, event_path))
    scene, sequence = read(args.stage04), read(args.sequences)
    mid = args.movie_id
    check(all(d["movie_id"] == mid for d in (summary, tracks, identities, events, scene, sequence)),
          "Movie IDs differ across inputs")
    check(summary["schema_version"] == "stage05.4.frame_measurements.v1"
          and events["schema_version"] == "stage05.5.events.v1", "Unsupported Stage05 schema")
    selection = next(s for s in sequence["sequence_candidates"] if s["sequence_id"] == args.sequence_id)
    check(args.shot_id in selection["shot_ids"], "Shot is outside the selected sequence")
    mode = summary["main_au_mode"]
    check(mode == events["primary_au_mode"], "Measurement and event primary modes differ")
    selected = [t for t in summary["tracks"] if t["extraction_mode"] == mode and t["shot_id"] == args.shot_id
                and (args.track_id is None or t["track_id"] == args.track_id)]
    check(len(selected) == 1, "Choose exactly one actually measured Track using --track-id if needed")
    measurement = selected[0]
    sid, tid = measurement["shot_id"], measurement["track_id"]
    clusters = {tid: c for c in identities["clusters"] for tid in c["track_ids"]}
    named = {}
    for row in scene["shots"]:
        candidates = row["character_candidates"]
        if len(candidates) == 1 and candidates[0]["status"] in {"candidate", "likely"}:
            named[row["shot_id"]] = candidates[0]["character_name"]

    files = [path / name for name in ("frame_features.parquet", "frame_status.jsonl", "track_measurements.json")]
    hashes = {str(p.resolve()): sha256_file(p) for p in [*files, track_path, identity_path, event_path,
                                                        args.stage04, args.sequences, path/"model_provenance.json"]}
    upstream = summary["source_provenance"]
    for p, provenance in ((track_path, upstream["tracking_files"]), (identity_path, upstream["identity_files"])):
        check(hashes[str(p.resolve())] == provenance[p.name]["sha256"], f"Stage05.4 source mismatch: {p}")
    for p in files:
        check(hashes[str(p.resolve())] == events["source_provenance"]["measurement_files_sha256"][p.name],
              f"Stage05.5 source mismatch: {p}")
    upstream_track = next(t for t in tracks["tracks"] if t["shot_id"] == sid and t["track_id"] == tid)
    bounds = measurement["effective_processing_temporal"]
    check(all(measurement[k] == upstream_track[k] for k in ("original_temporal", "effective_processing_temporal")),
          "Track and measurement frame bounds differ")
    identity = clusters[tid]
    check(identity["cluster_id"] == measurement["identity_cluster_id"], "Identity cluster mismatch")
    # Assign a candidate character name only if there is a single tracked face
    # and a single named visual subject in that shot.
    person = named.get(sid) if sum(t["shot_id"] == sid for t in tracks["tracks"]) == 1 else None
    cols = KEYS + MASKS + POSE + sum(EYES, []) + ["gaze_angle_x", "gaze_angle_y",
            "parquet_row", "landmark_bbox_iou", "five_point_normalized_error", "AU45_r", "AU45_c"]
    rows = pq.read_table(files[0], columns=cols + list(extra_columns), filters=[("shot_id", "=", sid), ("track_id", "=", tid),
                         ("extraction_mode", "=", mode)]).to_pylist()
    states = []
    with files[1].open(encoding="utf-8") as stream:
        for line in stream:
            s = json.loads(line)
            if (s["shot_id"], s["track_id"], s["extraction_mode"]) == (sid, tid, mode):
                states.append(s)
    check(len(rows) == len(states) == measurement["expected_frame_count"], "Incomplete Track data")
    check([r["global_frame_index"] for r in rows] == list(range(bounds["start_frame"], bounds["end_frame"])),
          "Unexpected frame range")
    for i, (row, state) in enumerate(zip(rows, states, strict=True)):
        qc = state["reliability"]
        check(all(row[key] == state[key] for key in KEYS + ["parquet_row"])
              and all(row[key] == qc[key] for key in MASKS), "Parquet and frame status mismatch")
        check(row["movie_id"] == mid and row["identity_cluster_id"] == identity["cluster_id"],
              "Unexpected frame identity")
        start, end = row["movie_timestamp_sec"], row["source_next_timestamp_sec"]
        check(math.isfinite(start) and math.isfinite(end) and end > start
              and (not i or start >= rows[i-1]["source_next_timestamp_sec"] - 1e-6), "Invalid movie timestamps")
        row.update(source_qc_reasons=qc["reasons"], eye_aspect_ratio_proxies=qc.get("eye_aspect_ratio_proxies"),
                   face_bbox_xywh=qc["association"].get("openface_landmark_bbox_xywh"), person=person)
    for key in ("head_pose_valid", "gaze_valid", "tracking_usable"):
        check(sum(r[key] for r in rows) == measurement["valid_counts"][key], "Track QC counts disagree")
    check(any(r["head_pose_valid"] and r["gaze_valid"] for r in rows), "No jointly usable source pose and gaze")

    coverage = []
    for shot in selection["shot_ids"]:
        tids = [t["track_id"] for t in tracks["tracks"] if t["shot_id"] == shot]
        counts = Counter(e["event_type"] for e in events["events"] if e["shot_id"] == shot)
        coverage.append({"shot_id": shot, "tracking": tids,
                         "identities": {tid: clusters[tid]["cluster_id"] for tid in tids if tid in clusters},
                         "character_candidate": named.get(shot) if len(tids) == 1 else None,
                         "measurements": {t["track_id"]: {"frames": t["expected_frame_count"],
                                                         "head": t["valid_counts"]["head_pose_valid"],
                                                         "gaze": t["valid_counts"]["gaze_valid"]}
                                          for t in summary["tracks"] if t["shot_id"] == shot and t["extraction_mode"] == mode},
                         "event_tracks": sorted({e["track_id"] for e in events["events"] if e["shot_id"] == shot}),
                         "event_counts": dict(counts)})
    selected_events = [e for e in events["events"] if (e["shot_id"], e["track_id"]) == (sid, tid)
                       and e["event_type"] in {"head_orientation_change", "camera_relative_gaze_change", "eye_closure_candidate"}]
    return rows, measurement, selected_events, coverage, hashes, upstream["video"]
