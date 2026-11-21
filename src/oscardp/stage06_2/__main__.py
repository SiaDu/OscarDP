"""Run one Stage06.2 pilot from saved Stage05 measurements."""

import argparse
import json
from pathlib import Path
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq

from .inputs import load, read
from .review import extract_review_frames, markdown, plot, qc_summary
from .temporal import (BINOCULAR_CHANGE_LIMIT_DEG, BINOCULAR_LIMIT_DEG, JITTER_MEDIAN_LIMIT_DEG,
                       MAX_GAP, MEDIAN_RADIUS, SMALL_DEG, THRESHOLDS, detect, prepare_frames)

VERSION = "stage06.2.gaze_direction.v1"


def frame_table(rows):
    table = pa.Table.from_pylist(rows)
    for key in ("head_direction", "gaze_direction", "left_eye_relative_direction",
                "right_eye_relative_direction", "eye_relative_direction",
                "gaze_direction_qc_rejected", "eye_relative_direction_qc_rejected"):
        table = table.set_column(table.schema.get_field_index(key), key,
                                 pa.array([r[key] for r in rows], type=pa.list_(pa.float64())))
    return table.replace_schema_metadata({
        "schema_version": VERSION, "time_axis": "source movie seconds; inclusive starts and exclusive next PTS",
        "camera_axes": "+X right, +Y down, +Z away from camera",
        "head_pose": "OpenFace XYZ radians; model to camera, R = Rx @ Ry @ Rz",
        "eye_axes": "eye 0 = OpenFace left; eye 1 = OpenFace right; local direction = R.T @ camera direction",
    })


def run(args):
    if args.output_dir.exists():
        raise ValueError(f"Output directory already exists: {args.output_dir}")
    rows, measure, source_events, coverage, files, video_record = load(args)
    rows = prepare_frames(rows, source_events)
    events = detect(rows, source_events)
    qc = qc_summary(rows)
    model = read(args.measurement_run / "model_provenance.json")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pairs, video_error = [], None
    if args.review_frames:
        try:
            pairs = extract_review_frames(Path(args.video) if args.video else Path(video_record["path"]),
                                          video_record, rows, events, args.output_dir/"review_frames.jpg")
        except subprocess.CalledProcessError as exc:
            video_error = f"原片解码失败（{exc.returncode}）。"
        except (OSError, ValueError) as exc:
            video_error = str(exc)
    pq.write_table(frame_table(rows), args.output_dir/"gaze_frames.parquet")
    result = {
        "schema_version": VERSION, "status": "model_candidate", "movie_id": args.movie_id,
        "sequence_id": args.sequence_id, "shot_id": args.shot_id, "track_id": rows[0]["track_id"],
        "person": rows[0]["person"], "identity_cluster_id": rows[0]["identity_cluster_id"],
        "identity_human_verified": rows[0]["identity_human_verified"], "extraction_mode": rows[0]["extraction_mode"],
        "original_temporal": measure["original_temporal"],
        "effective_processing_temporal": measure["effective_processing_temporal"],
        "input_sha256": files, "coverage": coverage,
        "openface_provenance": {"source_commit": model["source_build"]["git_commit"],
                                "source_cmake_version": model["source_build"]["cmake_declared_version"],
                                "binary_sha256": model["binary_sha256"]},
        "settings": {"motion_thresholds_deg": THRESHOLDS, "median_radius_s": MEDIAN_RADIUS, "max_gap_s": MAX_GAP,
                     "small_change_deg": SMALL_DEG, "binocular_limit_deg": BINOCULAR_LIMIT_DEG,
                     "binocular_change_mismatch_limit_deg": BINOCULAR_CHANGE_LIMIT_DEG,
                     "residual_limit_deg": JITTER_MEDIAN_LIMIT_DEG, "min_frames": 3, "min_duration_s": .12},
        "qc": qc,
        "stage05_reference_events": [{k: e[k] for k in ("event_id", "event_type", "start_frame", "end_frame",
                                      "start_sec", "end_sec")} for e in source_events],
        "review_frames": pairs, "review_frames_error": video_error, "events": events,
    }
    with (args.output_dir/"gaze_events.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    plot(rows, events, args.output_dir/"gaze_review.png")
    with (args.output_dir/"review.md").open("x", encoding="utf-8") as stream:
        stream.write(markdown(rows, measure, coverage, events, qc, files, pairs, video_error))
    print(f"{rows[0]['track_id']}: {len(rows)} frames, {qc['eye_relative_usable']} usable binocular frames, "
          f"{len(events)} movement candidates → {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movie-id", default="tt12300742")
    parser.add_argument("--sequence-id", default="seq_006")
    parser.add_argument("--shot-id", default="shot_000217")
    parser.add_argument("--track-id")
    for name in ("stage04", "sequences", "tracking-run", "identity-run", "measurement-run", "event-run", "output-dir"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--review-frames", action="store_true")
    parser.add_argument("--video", type=Path, help="Local copy matching Stage05 media metadata")
    args = parser.parse_args()
    if args.video and not args.review_frames:
        parser.error("--video requires --review-frames")
    try:
        run(args)
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        parser.exit(1, f"stage06_2: {exc}\n")


if __name__ == "__main__":
    main()
