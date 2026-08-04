from __future__ import annotations

import csv
import json
from pathlib import Path

from oscardp.shots.export import SHOT_CSV_FIELDS, export_shots_csv


def _shot() -> dict:
    return {
        "movie_key": "sample",
        "shot_id": "shot_000001",
        "source_video_relpath": "sample.mp4",
        "start_frame": 0,
        "end_frame": 10,
        "frame_count": 10,
        "start_time": "00:00:00.000",
        "end_time": "00:00:01.000",
        "start_sec": 0.0,
        "end_sec": 1.0,
        "duration_sec": 1.0,
        "keyframe_frame": 4,
        "keyframe_time_sec": 0.4,
        "keyframe_relpath": "keyframes/shot_000001.jpg",
        "boundary_before_confidence": None,
        "boundary_after_confidence": None,
        "shot_scale": None,
        "camera_movement": None,
        "model": {"name": "TransNetV2", "threshold": 0.5},
    }


def test_csv_export_preserves_record_count_and_fields(tmp_path: Path) -> None:
    rows = [_shot(), {**_shot(), "shot_id": "shot_000002", "start_frame": 10}]
    (tmp_path / "shots.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = export_shots_csv(tmp_path)
    with output.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(rows)
    assert list(csv_rows[0]) == SHOT_CSV_FIELDS
    assert csv_rows[0]["shot_id"] == rows[0]["shot_id"]
    assert json.loads(csv_rows[0]["model"]) == rows[0]["model"]
