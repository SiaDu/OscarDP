"""Build an immutable local review run and compact Stage06 candidate interface."""
import csv
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from oscardp.stage05.identity_inputs import file_hash
from .annotations import blank_record, validate_records
from .integration import read, write, digest, load_sources, build_packets
from .media import make_media, verify_media
from .sampling import select_pilot


def csv_write(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify(root):
    config = read(root/"run_config.json")
    for path, sha in config["input_sha256"].items():
        if file_hash(Path(path)) != sha:
            raise ValueError("Input changed: "+path)
    packet = read(root/"scene_performance_packets.json")
    source = read(Path(config["event_run"])/"performance_events.json")
    events = source["events"]
    full = list(csv.DictReader((root/"review_queue_full.csv").open(newline="")))
    original = list(csv.DictReader((Path(config["event_run"])/"event_review_queue.csv").open(newline="")))
    if full != original:
        raise ValueError("Full queue differs from Stage05.5")
    items = read(root/"review_items.json")
    if [x["source_queue_item"] for x in items["items"]] != original:
        raise ValueError("Review item omissions")
    expected, _ = select_pilot(full, config["sampling"]["target"])
    pilot = list(csv.DictReader((root/"review_queue_pilot.csv").open(newline="")))
    if pilot != expected:
        raise ValueError("Pilot sampling is not reproducible")
    emitted = packet["event_candidates"]
    if [e["event_id"] for e in emitted] != [e["event_id"] for e in events]:
        raise ValueError("Event reference omission")
    for compact, e in zip(emitted, events):
        if any(compact[k] != e[k] for k in compact if k != "evidence_reference"):
            raise ValueError("Original event content changed")
    shots = [s for scene in packet["scenes"] for s in scene["shots"]]
    context_paths = {Path(p).name: Path(p) for p in config["input_sha256"]}
    movie = packet["movie_id"]
    stage02 = {s["shot_id"]: s for s in read(context_paths[f"Stage02_{movie}.json"])["shots"]}
    stage04 = read(context_paths[f"Stage04_{movie}.json"])
    shot04 = {s["shot_id"]: s for s in stage04["shots"]}
    scene04 = {s["scene_id"]: s for s in stage04["scenes"]}
    tracking_path = Path(source["source_provenance"]["earlier_provenance"]["tracking_run"])/"face_tracks.json"
    tracking = read(tracking_path)
    effective = {s["shot_id"]: s["effective_processing_temporal"] for s in tracking["shot_ranges"]}
    identity = read(Path(source["source_provenance"]["earlier_provenance"]["identity_run"])/"identity_clusters.json")
    members = {tid: c["cluster_id"] for c in identity["clusters"] for tid in c["track_ids"]}
    source_traces = pq.read_table(Path(config["event_run"])/"processed_timeseries.parquet").to_pylist()
    for scene in packet["scenes"]:
        if scene["context"] != scene04[scene["scene_id"]]:
            raise ValueError("Stage04 scene context changed")
        for s in scene["shots"]:
            sid = s["shot_id"]
            if (scene["scene_id"] != shot04[sid]["scene_id"]
                    or s["effective_processing_temporal"] != effective[sid]
                    or s["stage04_context_temporal"] != shot04[sid]["temporal"]
                    or s["stage02_dialogue"] != stage02[sid]["dialogue"]
                    or s["stage02_narrative"] != stage02[sid]["narrative"]
                    or s["stage04_character_candidates"] != shot04[sid]["character_candidates"]
                    or s["boundary_status"] != source["source_provenance"]["boundary_sidecar"]["mode"]):
                raise ValueError("Scene/Dialogue/temporal authority mismatch")
            for t in s["face_tracks"]:
                expected_ids = [e["event_id"] for e in events if e["track_id"] == t["track_id"]]
                cluster = t["identity_cluster"]
                if (t["event_ids"] != expected_ids or cluster["identity_status"] != "tentative"
                        or cluster["cluster_id"] != members[t["track_id"]]
                        or cluster["character_mapping_status"] != "unverified"
                        or cluster["candidate_mapping"]["candidates"] != shot04[sid]["character_candidates"]
                        or t["human_verified"] is not False):
                    raise ValueError("Track/identity references changed")
            actual_traces = read(root/"viewer/data"/f"{sid}_traces.json")
            if actual_traces != [r for r in source_traces if r["shot_id"] == sid]:
                raise ValueError("Raw/processed/invalid traces changed")
            mapping = read(root/"viewer/data"/f"{sid}_frames.json")["frames"]
            times = {r["global_frame_index"]: r["movie_timestamp_sec"] for r in actual_traces}
            if any(abs(times[f["global_frame_index"]]-f["movie_timestamp_sec"]) > 1e-6 for f in mapping):
                raise ValueError("Measurement/video time mismatch")
    if digest(config["input_sha256"]) != packet["source_run_sha256"]:
        raise ValueError("Packet source mismatch")
    stage06 = read(root/"stage06_input_pilot.json")
    if stage06["packet"] != packet or stage06["annotation_generated"] is not False:
        raise ValueError("Stage06 interface mismatch")
    media = verify_media(root, shots)
    manifest = root/"artifact_sha256.json"
    if manifest.exists():
        for name, sha in read(manifest).items():
            if file_hash(root/name) != sha:
                raise ValueError("Stage05.6 artifact changed: "+name)
    return {"status": "passed", "event_count": len(events), "full_queue_count": len(full),
            "pilot_queue_count": len(pilot), "scene_ids": [s["scene_id"] for s in packet["scenes"]],
            "review_video_frames": media, "input_hash_count": len(config["input_sha256"]),
            "input_hashes_unchanged": True, "all_original_event_fields_retained_unchanged_or_referenced": True,
            "human_review_completed_count": 0, "human_verified": False,
            "sampling": config["sampling"], "limitations": ["Software integrity checks are not event accuracy validation",
                "Videos are silent compressed previews; exact stepping uses one still per original source frame",
                "Stage02 dialogue remains associated with old shot bounds; no sub-frame dialogue retiming",
                "Stage04 model_reviewed prototype and experimental boundaries retain original statuses"]}


def build(event_run, root, review_dir, pilot_size=48):
    if root.exists():
        raise FileExistsError("Stage05.6 output exists; choose a new run directory")
    root, event_run, review_dir = root.resolve(), event_run.resolve(), review_dir.resolve()
    if root.is_relative_to(event_run) or review_dir.is_relative_to(root) or root.is_relative_to(review_dir):
        raise ValueError("Algorithm outputs, source events and human reviews must be independent directories")
    data = load_sources(event_run)
    pilot, sampling = select_pilot(data["queue"], pilot_size)
    packet = build_packets(data)
    root.mkdir(parents=True)
    viewer = root/"viewer"
    (viewer/"data").mkdir(parents=True)
    run = {"schema_version": "stage05.6.review_run.v1", "movie_id": data["document"]["movie_id"],
           "event_run": str(event_run), "review_directory": str(review_dir),
           "source_run_sha256": data["source_hash"], "input_sha256": data["protected"],
           "sampling": sampling, "human_verified": False,
           "source_code_sha256": {p.name: file_hash(p) for p in Path(__file__).parent.iterdir() if p.is_file()},
           "no_llm_or_detection": True}
    write(root/"run_config.json", run)
    shutil.copyfile(event_run/"event_review_queue.csv", root/"review_queue_full.csv")
    csv_write(root/"review_queue_pilot.csv", pilot)
    items = []
    event_map = {e["event_id"]: e for e in data["document"]["events"]}
    shot_map = {s["shot_id"]: s for scene in packet["scenes"] for s in scene["shots"]}
    scenes = {s["shot_id"]: scene["scene_id"] for scene in packet["scenes"] for s in scene["shots"]}
    for r in data["queue"]:
        s = shot_map[r["shot_id"]]
        item = {"review_id": r["review_id"], "source_queue_item": r,
                "source_event_id": r["event_id"] or None, "source_run_sha256": data["source_hash"],
                "scene_id": scenes[r["shot_id"]], "shot_id": r["shot_id"], "track_id": r["track_id"],
                "event_reference": event_map.get(r["event_id"], {}).get("content_sha256"),
                "human_review": blank_record(r, data["source_hash"]),
                "evidence": s["evidence"], "pilot_selected": r["review_id"] in {x["review_id"] for x in pilot}}
        items.append(item)
    write(root/"review_items.json", {"schema_version": "stage05.6.review_items.v1", "items": items})
    write(root/"scene_performance_packets.json", packet)
    write(root/"stage06_input_pilot.json", {"schema_version": "stage06.candidate_input.v1", "annotation_generated": False,
        "evidence_status": "algorithm_candidate", "human_verified": False,
        "pilot_review_ids": [r["review_id"] for r in pilot], "packet": packet})
    for name in ["index.html", "viewer.js", "viewer.css"]:
        shutil.copyfile(Path(__file__).parent/name, viewer/name)
    write(viewer/"data/index.json", {"source_run_sha256": data["source_hash"], "items": items,
          "packet": packet, "sampling": sampling})
    for sid, s in shot_map.items():
        traces = [r for r in data["traces"] if r["shot_id"] == sid]
        write(viewer/"data"/f"{sid}_traces.json", traces)
        write(viewer/"data"/f"{sid}_timeline.json", {"shot_id": sid,
            "events": [e for e in packet["event_candidates"] if e["shot_id"] == sid],
            "review_intervals": [r for r in data["queue"] if r["shot_id"] == sid]})
    make_media(data, viewer)
    result = verify(root)
    write(root/"qc_report.json", result)
    template = {"schema_version": "stage05.6.human_reviews.v1", "source_run_sha256": data["source_hash"],
                "records": [blank_record(r, data["source_hash"]) for r in data["queue"]]}
    # Create a new source-specific blank overlay only; never overwrite prior human work.
    review_dir.mkdir(parents=True, exist_ok=True)
    review_path = review_dir/f"{data['source_hash']}.json"
    if review_path.exists():
        existing = read(review_path)
        if existing["source_run_sha256"] != data["source_hash"]:
            raise ValueError("Human review source mismatch")
        validate_records(existing["records"], data["queue"], data["source_hash"],
            {sid: [s["effective_processing_temporal"]["start_frame"], s["effective_processing_temporal"]["end_frame"]] for sid, s in shot_map.items()})
    else:
        write(review_path, template)
    write(root/"artifact_sha256.json", {str(p.relative_to(root)): file_hash(p) for p in root.rglob("*") if p.is_file()})
    print(f"Stage05.6 complete: {root}")
    return result
