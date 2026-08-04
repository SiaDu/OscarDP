from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import ValidationResult


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def validate_movie(movie_dir: Path) -> ValidationResult:
    errors: list[str] = []
    metadata_path = movie_dir / "video_metadata.json"
    shots_path = movie_dir / "shots.jsonl"
    if not metadata_path.is_file():
        return ValidationResult(False, ["missing video_metadata.json"], 0, 0)
    if not shots_path.is_file():
        return ValidationResult(False, ["missing shots.jsonl"], 0, 0)
    try:
        metadata = _load_json(metadata_path)
        shots = []
        with shots_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        shots.append(json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value))))
                    except (ValueError, json.JSONDecodeError) as exc:
                        errors.append(f"invalid shots.jsonl line {line_number}: {exc}")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return ValidationResult(False, [f"invalid JSON: {exc}"], 0, 0)
    if not shots:
        errors.append("at least one shot is required")
        return ValidationResult(False, errors, 0, 0)

    required_metadata = {
        "source_video_relpath", "duration_sec", "width", "height", "codec_name",
        "fps", "frame_count", "is_vfr", "timestamp_source",
    }
    missing_metadata = sorted(required_metadata - set(metadata))
    if missing_metadata:
        errors.append(f"missing metadata fields: {', '.join(missing_metadata)}")
    expected_ids = [f"shot_{index:06d}" for index in range(1, len(shots) + 1)]
    actual_ids = [shot.get("shot_id") for shot in shots]
    if actual_ids != expected_ids:
        errors.append("shot IDs must be unique and sequential")

    missing_keyframes = 0
    previous_end: int | None = None
    previous_start_sec: float | None = None
    previous_end_sec: float | None = None
    for index, shot in enumerate(shots):
        start = shot.get("start_frame")
        end = shot.get("end_frame")
        if not isinstance(start, int) or not isinstance(end, int) or start >= end:
            errors.append(f"shot {index + 1} has invalid frame range")
            continue
        if shot.get("frame_count") != end - start:
            errors.append(f"shot {index + 1} frame_count is inconsistent")
        if previous_end is not None and start != previous_end:
            errors.append(f"gap or overlap before shot {index + 1}")
        previous_end = end
        start_sec = shot.get("start_sec")
        end_sec = shot.get("end_sec")
        if not isinstance(start_sec, (int, float)) or not isinstance(end_sec, (int, float)) or start_sec > end_sec:
            errors.append(f"shot {index + 1} has invalid timestamps")
        elif previous_start_sec is not None and start_sec < previous_start_sec:
            errors.append(f"timestamps are not monotonic at shot {index + 1}")
        else:
            previous_start_sec = float(start_sec)
            if previous_end_sec is not None and float(start_sec) < previous_end_sec:
                errors.append(f"timestamp overlap before shot {index + 1}")
            previous_end_sec = float(end_sec)
        keyframe = shot.get("keyframe_frame")
        if not isinstance(keyframe, int) or not start <= keyframe < end:
            errors.append(f"shot {index + 1} keyframe is outside its shot")
        keyframe_relpath = shot.get("keyframe_relpath")
        keyframe_path = movie_dir / keyframe_relpath if isinstance(keyframe_relpath, str) else None
        if keyframe_path is None or not keyframe_path.is_file():
            missing_keyframes += 1

    if shots[0].get("start_frame") != 0:
        errors.append("first shot must start at frame 0")
    frame_count = metadata.get("frame_count")
    if not isinstance(frame_count, int) or frame_count < 1:
        errors.append("metadata frame_count must be a positive decoded-frame count")
    elif shots[-1].get("end_frame") != frame_count:
        errors.append("final shot does not reach the final decoded frame")
    if missing_keyframes:
        errors.append(f"missing {missing_keyframes} keyframe files")
    return ValidationResult(not errors, errors, len(shots), missing_keyframes)


def validate_output_root(output_root: Path, movie_key: str | None = None) -> dict[str, ValidationResult]:
    if movie_key:
        candidates = [output_root / movie_key]
    else:
        candidates = sorted(
            path for path in output_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ) if output_root.is_dir() else []
    return {path.name: validate_movie(path) for path in candidates}
