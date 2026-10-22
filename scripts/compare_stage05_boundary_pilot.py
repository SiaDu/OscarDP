"""Measure boundary-only pilot changes and prepare explicit visual-review artifacts."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from oscardp.stage05.output import atomic_json, load_json


def summarize(root):
    config, qc, metrics = [load_json(root/name) for name in ["run_config.json", "qc_report.json", "run_metrics.json"]]
    records = collections.defaultdict(list)
    for line in (root/"face_detections.jsonl").read_text().splitlines():
        r = json.loads(line)
        assert r["detection_status"] == "observed"
        records[r["shot_id"]].append(r)
    result = {}
    for shot in config["shots"]:
        sid, t = shot["shot_id"], shot["temporal"]
        s = next(s for s in qc["shots"] if s["shot_id"] == sid)
        assert s["status"] == "complete"
        rows = records[sid]
        frames = {r["frame_index"] for r in rows}
        expected = set(range(t["start_frame"], t["end_frame"]))
        assert frames <= expected
        unique = {(r["frame_index"], r["track_id"]) for r in rows}
        assert len(unique) == len(rows)
        tracks = s["tracks"]
        for track in tracks:
            assert sum(r["track_id"] == track["track_id"] for r in rows) == track["detected_frame_count"]
        warnings = s["tracking_ambiguities"]
        result[sid] = {"temporal": t, "frame_count": s["decoded_frame_count"], "detections": len(rows),
                       "face_track_count": len(tracks), "single_frame_track_count": sum(t["detected_frame_count"] == 1 for t in tracks),
                       "detected_frame_coverage": len(frames)/(t["end_frame"]-t["start_frame"]),
                       "track_valid_detection_ratios": {t["track_id"]: t["valid_detection_ratio"] for t in tracks},
                       "tracking_ambiguity_count": sum(w["reason"] != "visual_discontinuity_association_reset" for w in warnings),
                       "visual_discontinuity_count": sum(w["reason"] == "visual_discontinuity_association_reset" for w in warnings),
                       "detection_score_mean": sum(r["detection_score"] for r in rows)/len(rows) if rows else None,
                       "detection_score_min": min((r["detection_score"] for r in rows), default=None),
                       "elapsed_sec": s["elapsed_sec"], "decode_wait_sec": s["decode_wait_sec"], "detection_sec": s["detection_sec"],
                       "processed_fps_including_decode": s["decoded_frame_count"]/s["elapsed_sec"],
                       "detector_fps": s["decoded_frame_count"]/s["detection_sec"]}
    return config, result, metrics


def preview_frames(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot read QC video: {path}")
    declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    while True:
        ok, image = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    cap.release()
    assert len(frames) == declared and frames
    return frames


def visual_artifacts(before, after, shot_ids, out):
    canvas = Image.new("RGB", (2000, len(shot_ids)*315), "#181b20")
    draw = ImageDraw.Draw(canvas)
    columns = ["BEFORE FIRST", "BEFORE BODY SAMPLE", "AFTER FIRST", "AFTER MIDDLE", "AFTER LAST"]
    preview_counts = {}
    (out/"filmstrips").mkdir()
    for row, sid in enumerate(shot_ids):
        old = preview_frames(before/"previews"/f"{sid}.mp4")
        new = preview_frames(after/"previews"/f"{sid}.mp4")
        preview_counts[sid] = len(new)
        selected = [old[0], old[min(1,len(old)-1)], new[0], new[len(new)//2], new[-1]]
        draw.text((5,row*315+4), sid, fill="white")
        for col, (image, label) in enumerate(zip(selected, columns)):
            draw.text((col*400+5,row*315+22), label, fill="#ffe278")
            im = image.copy()
            im.thumbnail((400,267))
            canvas.paste(im, (col*400,row*315+42))
        strip = Image.new("RGB", (1920, math.ceil(len(new)/8)*180), "#181b20")
        labels = ImageDraw.Draw(strip)
        for i, image in enumerate(new):
            image = image.copy()
            image.thumbnail((240,160))
            x,y = (i%8)*240, (i//8)*180
            strip.paste(image, (x,y+18))
            labels.text((x+3,y+3), f"{sid} QC frame {i}", fill="white")
        strip.save(out/"filmstrips"/f"{sid}_after.jpg", quality=94)
    canvas.save(out/"before_after_contact_sheet.jpg", quality=95)
    return preview_counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--visual-review", type=Path, help="Finalize only after actual visual inspection; review is never inferred from track counts")
    args = p.parse_args()
    out = args.after/"comparison"
    old_config, old, old_metrics = summarize(args.before)
    new_config, new, new_metrics = summarize(args.after)
    for key in ["movie_id", "stage01_sha256", "video", "model_sha256", "device", "opencv_version", "score_threshold",
                "nms_threshold", "top_k", "inference_max_size", "max_missed", "min_hits", "ambiguity_margin",
                "discontinuity_threshold", "sample_stride", "preview_fps"]:
        assert old_config[key] == new_config[key], f"Not a boundary-only comparison: {key}"
    assert set(old) == set(new)
    for s in new_config["shots"]:
        assert s["original_temporal"] == old[s["shot_id"]]["temporal"]
        assert s["effective_processing_temporal"] == new[s["shot_id"]]["temporal"]
    comparison = {"before": str(args.before.resolve()), "after": str(args.after.resolve()),
                  "boundary_mode": new_config["boundary_sidecar"]["mode"],
                  "human_verified": False, "before_shots": old, "after_shots": new,
                  "before_run_wall_sec": old_metrics["wall_sec"], "after_run_wall_sec": new_metrics["wall_sec"],
                  "before_end_to_end_fps": sum(s["frame_count"] for s in old.values())/old_metrics["wall_sec"],
                  "after_end_to_end_fps": sum(s["frame_count"] for s in new.values())/new_metrics["wall_sec"],
                  "comparison_config_sha256": {"before": hashlib.sha256((args.before/"run_config.json").read_bytes()).hexdigest(),
                                               "after": hashlib.sha256((args.after/"run_config.json").read_bytes()).hexdigest()},
                  "neighbor_conflicts": new_config["boundary_sidecar"]["neighbor_conflicts"],
                  "limits": "Frame coverage and track detection ratios are not identity accuracy. Preview review samples 6 FPS; every original frame was processed. Runtime comparison is descriptive, not a controlled benchmark."}
    if args.visual_review is None:
        out.mkdir(exist_ok=False)
        counts = visual_artifacts(args.before, args.after, list(new), out)
        comparison.update(pilot_status="awaiting_visual_review", preview_frame_counts=counts)
        atomic_json(out/"before_after_metrics.json", comparison)
        print(out/"before_after_contact_sheet.jpg")
        return
    existing = load_json(out/"before_after_metrics.json")
    assert existing["comparison_config_sha256"] == comparison["comparison_config_sha256"]
    if (out/"pilot_validation.json").exists():
        raise FileExistsError("Visual validation already exists; preserve this experiment")
    review = load_json(args.visual_review)
    assert review["human_verified"] is False
    assert set(review["shots"]) == set(new)
    for sid, decision in review["shots"].items():
        assert decision["preview_frames_reviewed"] == existing["preview_frame_counts"][sid]
    visual_pass = all(r["first_frame_contamination_removed"] and not r["new_obvious_tracking_errors_observed"] for r in review["shots"].values())
    measured_pass = all(s["single_frame_track_count"] == 0 and s["tracking_ambiguity_count"] == 0 and s["visual_discontinuity_count"] == 0
                        and s["detected_frame_coverage"] == 1. for s in new.values())
    status = "pilot_passed_with_experimental_boundary_correction" if visual_pass and measured_pass else "pilot_needs_review"
    validation = {"pilot_status": status, "formal_stage01_fix": False, "human_verified": False,
                  "boundary_approval_status": "model_reviewed_pending_human", "boundary_sidecar": new_config["boundary_sidecar"],
                  "visual_review": review, "measurement_summary": comparison, "limits": comparison["limits"]}
    atomic_json(out/"pilot_validation.json", validation)
    existing.update(pilot_status=status, visual_review_file=str(args.visual_review))
    atomic_json(out/"before_after_metrics.json", existing)
    print(status)


if __name__ == "__main__":
    main()
