import fcntl
import hashlib
import math
import os
import re
import sys
import time

from oscardp.shots.discovery import CANONICAL_ID, discover_movies
from oscardp.shots.media import probe_video
from oscardp.shots.schema import json_dumps
from .detectors import YuNetDetector
from .output import atomic_json, identity, load_json
from .qc import DiscontinuityGuard, Preview, report
from .tracker import ShotTracker
from .video_reader import iter_frames


def preflight(args):
    import cv2
    import numpy as np
    if not CANONICAL_ID.fullmatch(args.movie_id):
        raise ValueError("--movie-id must be a canonical IMDb ID")
    if not args.shot_ids or len(set(args.shot_ids)) != len(args.shot_ids):
        raise ValueError("Explicit, unique --shot-ids required")
    if args.max_missed < 0 or args.min_hits < 1 or args.inference_max_size < 0:
        raise ValueError("Invalid tracking or inference parameters")
    if not math.isfinite(args.score_threshold) or not 0 < args.score_threshold <= 1:
        raise ValueError("--score-threshold must be in (0, 1]")
    if not math.isfinite(args.discontinuity_threshold) or not 0 < args.discontinuity_threshold <= 1:
        raise ValueError("--discontinuity-threshold must be in (0, 1]")
    source = args.analysis_root / args.movie_id / f"Stage01_{args.movie_id}.json"
    data = load_json(source)
    if data["movie_id"] != args.movie_id:
        raise ValueError("Stage01 movie ID mismatch")
    lookup = {s["shot_id"]: s for s in data["shots"]}
    if len(lookup) != len(data["shots"]):
        raise ValueError("Duplicate Stage01 shot IDs")
    missing = set(args.shot_ids)-lookup.keys()
    if missing:
        raise ValueError(f"Unknown Shot IDs: {sorted(missing)}")
    if args.video:
        video = args.video.resolve(strict=True)
    else:
        refs = discover_movies(args.input_root, movie_key=args.movie_id)
        if len(refs) != 1:
            raise ValueError(f"Expected one discovered video for {args.movie_id}; found {len(refs)}. Use --video.")
        video = refs[0].source_path
    metadata, _ = probe_video(video, video.name)
    if min(metadata.width, metadata.height) <= 0:
        raise ValueError("Invalid input video dimensions")
    cfr = all(abs(s["temporal"][key]-s["temporal"][frame_key]/metadata.fps) < .0001
              for s in data["shots"] for key, frame_key in [("start", "start_frame"), ("end", "end_frame")])
    shots = sorted([lookup[s] for s in args.shot_ids], key=lambda s: s["temporal"]["start_frame"])
    last_end = -1
    for shot in shots:
        if not re.fullmatch(r"shot_\d{6}", shot["shot_id"]):
            raise ValueError("Unsafe/noncanonical shot ID")
        t = shot["temporal"]
        a, b = t["start_frame"], t["end_frame"]
        if type(a) is not int or type(b) is not int or not 0 <= a < b:
            raise ValueError(f"Invalid inclusive/exclusive frame bounds: {shot['shot_id']}")
        if a < last_end or (metadata.frame_count and b > metadata.frame_count):
            raise ValueError("Overlapping shots or bounds beyond video")
        last_end = b
        if not all(math.isfinite(t[k]) for k in ["start", "end"]) or not 0 <= t["start"] < t["end"] <= metadata.duration_sec+.1:
            raise ValueError("Invalid Stage01 time bounds")
    if args.detector != "yunet":
        raise NotImplementedError("Only YuNet is implemented; SCRFD is reserved")
    detector = YuNetDetector(args.model, args.score_threshold, args.inference_max_size, args.device)
    # Exercise the network, not just the ONNX file's existence.
    detector.detect(np.zeros((320, 320, 3), np.uint8))
    config = {"schema_version": "stage05.2.v1", "implementation_version": 2,
              "movie_id": args.movie_id, "stage01": identity(source),
              "stage01_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "video": identity(video), "model": identity(args.model),
              "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "detector": "YuNet", "device": args.device, "opencv_version": cv2.__version__,
              "score_threshold": args.score_threshold, "nms_threshold": .3, "top_k": 5000,
              "inference_max_size": args.inference_max_size,
              "max_missed": args.max_missed, "min_hits": args.min_hits, "ambiguity_margin": .12,
              "discontinuity_threshold": args.discontinuity_threshold,
              "sample_stride": 1, "shot_ids": [s["shot_id"] for s in shots],
              "shots": shots, "width": metadata.width, "height": metadata.height, "fps": metadata.fps,
              "timestamp_rule": "frame_index_cfr" if cfr else "decoded_pts_relative_to_first_frame",
              "decode_method": "sequential FFmpeg global frame select; no seeking; selected boundary PTS checked",
              "frame_boundaries": "zero-based start inclusive, end exclusive", "preview_fps": 6}
    return video, shots, metadata, cfr, detector, config


def aggregate(root, shots, config):
    results = []
    destination = root / "face_detections.jsonl"
    temporary = destination.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for shot in shots:
            sid = shot["shot_id"]
            path = root / "checkpoints" / f"{sid}.json"
            result = load_json(path) if path.exists() else {"shot_id": sid, "status": "incomplete"}
            results.append(result)
            if result["status"] == "complete":
                with (root / "checkpoints" / f"{sid}.jsonl").open(encoding="utf-8") as shard:
                    for line in shard:
                        output.write(line)
    temporary.replace(destination)
    atomic_json(root / "face_tracks.json", {"schema_version": "stage05.2.v1", "movie_id": config["movie_id"],
                "frame_boundaries": config["frame_boundaries"], "tracks": [t for r in results for t in r.get("tracks", [])]})
    atomic_json(root / "qc_report.json", report(results, {k: config[k] for k in [
        "detector", "device", "opencv_version", "model", "model_sha256", "score_threshold", "nms_threshold", "inference_max_size"]}))
    return results


def run(args):
    import cv2
    # Keep CPU work bounded and reproducible; no CUDA/ORT configuration is touched.
    cv2.setNumThreads(4)
    video, shots, metadata, cfr, detector, config = preflight(args)
    if args.dry_run:
        print(json_dumps(dict(config, dry_run=True, selected_frame_count=sum(
            s["temporal"]["end_frame"]-s["temporal"]["start_frame"] for s in shots)), pretty=True))
        return 0
    root = args.output_root / args.movie_id
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another run holds {root}/.run.lock") from exc
        return execute(args, root, video, shots, metadata, cfr, detector, config)


def execute(args, root, video, shots, metadata, cfr, detector, config):
    old = root / "run_config.json"
    if old.exists():
        if not (args.resume or args.overwrite):
            raise FileExistsError("Output exists; use --resume for matching inputs or explicit --overwrite")
        if args.resume and load_json(old) != config:
            raise ValueError("Resume refused: inputs or configuration changed")
    elif any(p.name != ".run.lock" for p in root.iterdir()):
        raise FileExistsError("Unowned nonempty output directory")
    (root / "checkpoints").mkdir(exist_ok=True)
    (root / "previews").mkdir(exist_ok=True)
    if args.overwrite:
        # Only files owned by this implementation; source inputs are never touched.
        for sid in load_json(old).get("shot_ids", []) if old.exists() else []:
            if not re.fullmatch(r"shot_\d{6}", sid):
                raise ValueError("Invalid prior checkpoint name")
            for path in [root/"checkpoints"/f"{sid}.json", root/"checkpoints"/f"{sid}.jsonl", root/"previews"/f"{sid}.mp4"]:
                path.unlink(missing_ok=True)
    atomic_json(old, config)
    pending = []
    for s in shots:
        checkpoint = root/"checkpoints"/f"{s['shot_id']}.json"
        if args.resume and checkpoint.exists() and load_json(checkpoint)["status"] == "complete":
            receipt = load_json(checkpoint)
            shard = root/"checkpoints"/f"{s['shot_id']}.jsonl"
            preview = root/"previews"/f"{s['shot_id']}.mp4"
            if not shard.exists() or not preview.exists() or shard.stat().st_size != receipt["detection_file_bytes"] or preview.stat().st_size != receipt["preview_file_bytes"]:
                raise ValueError(f"Completed checkpoint artifacts missing/changed: {s['shot_id']}")
            print(f"Skip completed {s['shot_id']}", flush=True)
        else:
            pending.append(s)
    if not pending:
        aggregate(root, shots, config)
        return 0
    started = time.perf_counter()
    print(f"Exact sequential decode through global frame {pending[-1]['temporal']['end_frame']}; prefix decode may take time.", flush=True)
    frames = iter_frames(video, pending, metadata, cfr)
    interrupted = False
    try:
        for shot in pending:
            sid, temporal = shot["shot_id"], shot["temporal"]
            tracker = ShotTracker(sid, args.max_missed, args.min_hits)
            guard = DiscontinuityGuard(args.discontinuity_threshold)
            shard = root/"checkpoints"/f"{sid}.jsonl"
            temp = shard.with_suffix(".jsonl.tmp")
            preview_path = root/"previews"/f"{sid}.mp4"
            preview = Preview(preview_path, metadata.width, metadata.height)
            count = low_quality = 0
            detection_sec = decode_sec = 0.
            shot_start = time.perf_counter()
            result = {"shot_id": sid, "status": "failed"}
            try:
                with temp.open("w", encoding="utf-8") as output:
                    total = temporal["end_frame"]-temporal["start_frame"]
                    for expected in range(temporal["start_frame"], temporal["end_frame"]):
                        before = time.perf_counter()
                        frame = next(frames)
                        decode_sec += time.perf_counter()-before
                        if frame.index != expected:
                            raise ValueError(f"Global frame mismatch: {frame.index} != {expected}")
                        if count == 0 and abs(frame.timestamp-temporal["start"]) > .002:
                            raise ValueError(f"Stage01 start timestamp mismatch: {sid}")
                        if expected == temporal["end_frame"]-1 and abs(frame.next_timestamp-temporal["end"]) > .002:
                            raise ValueError(f"Stage01 exclusive end timestamp mismatch: {sid}")
                        if shot is pending[-1] and expected == temporal["end_frame"]-1:
                            if next(frames, None) is not None:
                                raise ValueError("Unexpected frames beyond selected shots")
                        before = time.perf_counter()
                        detections = detector.detect(frame.image)
                        detection_sec += time.perf_counter()-before
                        difference = guard.update(frame.image)
                        if difference is not None:
                            tracker.break_links(frame.index, difference)
                        records = tracker.update(detections, frame.index, frame.timestamp, frame.next_timestamp)
                        for record in records:
                            output.write(json_dumps(record)+"\n")
                        low_quality += sum(r["detection_score"] < .9 or min(r["bbox_xywh"][2:]) < 24 for r in records)
                        preview.add(frame, records, sid)
                        count += 1
                        if count == 1 or count % 24 == 0 or count == total:
                            elapsed = time.perf_counter()-shot_start
                            print(f"\r{sid} [{'#'*int(20*count/total):20s}] {count}/{total} frames {count/max(elapsed,1e-9):.2f} FPS", end="", file=sys.stderr, flush=True)
                    output.flush()
                    os.fsync(output.fileno())
                preview.close()
                temp.replace(shard)
                result = {"shot_id": sid, "status": "complete", "decoded_frame_count": count,
                          "face_track_count": len(tracker.tracks), "tracks": tracker.summaries(),
                          "tracking_ambiguities": tracker.ambiguities,
                          "low_quality_detection_count": low_quality,
                          "low_quality_rule": "score < 0.9 or min(original bbox width,height) < 24 pixels",
                          "decode_wait_sec": decode_sec, "detection_sec": detection_sec,
                          "elapsed_sec": time.perf_counter()-shot_start,
                          "preview": str(preview_path), "detection_file_bytes": shard.stat().st_size,
                          "preview_file_bytes": preview_path.stat().st_size}
            except (Exception, KeyboardInterrupt) as exc:
                preview.close()
                result.update(error=f"{type(exc).__name__}: {exc}", decoded_frame_count=count,
                              decode_wait_sec=decode_sec, detection_sec=detection_sec)
                interrupted = True
            finally:
                atomic_json(root/"checkpoints"/f"{sid}.json", result)
                print(file=sys.stderr)
            aggregate(root, shots, config)
            if interrupted:
                print(f"Shot failed: {result.get('error')}; resume will retry it", file=sys.stderr)
                break
    finally:
        frames.close()
        results = aggregate(root, shots, config)
        atomic_json(root/"run_metrics.json", {"wall_sec": time.perf_counter()-started,
                    "sequential_prefix_through_frame": pending[-1]["temporal"]["end_frame"],
                    "requested_frame_count": sum(s["temporal"]["end_frame"]-s["temporal"]["start_frame"] for s in pending),
                    "note": "Decode wait includes sequential prefix/gaps on first access. Only requested frames enter detection. Metrics cover this invocation."})
    return 1 if any(r["status"] != "complete" for r in results) else 0
