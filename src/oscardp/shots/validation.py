from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from .schema import ValidationResult

MOVIE_ID = re.compile(r"tt\d{7,10}")
IMAGE_NAME = re.compile(r"shot_\d{6}_\d{2}-\d{2}-\d{2}-\d{3}\.jpg")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def validate_movie(movie_dir: Path, *, frame_count: int | None = None, official_layout: bool = True) -> ValidationResult:
    errors: list[str] = []
    movie_id = movie_dir.name
    stage_path = movie_dir / f"Stage01_{movie_id}.json"
    if not MOVIE_ID.fullmatch(movie_id):
        return ValidationResult(False, ["movie directory must use an IMDb ID"], 0, 0)
    if not stage_path.is_file():
        return ValidationResult(False, [f"missing {stage_path.name}"], 0, 0)
    if official_layout and {path.name for path in movie_dir.iterdir()} != {stage_path.name, "shot_img"}:
        errors.append("official movie directory must contain only Stage 01 JSON and shot_img")
    try:
        data = _load_json(stage_path)
    except (OSError, ValueError, TypeError) as exc:
        return ValidationResult(False, [f"invalid Stage 01 JSON: {exc}"], 0, 0)
    if not isinstance(data, dict) or set(data) != {"movie_id", "movie_name", "shots"}:
        return ValidationResult(False, ["invalid movie-level schema"], 0, 0)
    if data["movie_id"] != movie_id:
        errors.append("movie_id does not match directory")
    if data["movie_name"] is not None and (not isinstance(data["movie_name"], str) or not data["movie_name"].strip()):
        errors.append("movie_name must be a nonempty string or null")
    shots = data["shots"]
    if not isinstance(shots, list) or not shots:
        return ValidationResult(False, errors + ["at least one shot is required"], 0, 0)
    missing = 0
    previous_end = 0
    previous_time = 0.0
    image_dir = movie_dir / "shot_img"
    images = list(image_dir.glob("*.jpg")) if image_dir.is_dir() else []
    image_by_id: dict[str, list[Path]] = {}
    for image in images:
        if not IMAGE_NAME.fullmatch(image.name):
            errors.append(f"invalid image filename: {image.name}")
        image_by_id.setdefault(image.name[:11], []).append(image)
    for index, shot in enumerate(shots, 1):
        expected_id = f"shot_{index:06d}"
        if not isinstance(shot, dict) or set(shot) != {"shot_id", "temporal"} or shot.get("shot_id") != expected_id:
            errors.append(f"shot {index} has invalid schema or ID")
            continue
        temporal = shot["temporal"]
        if not isinstance(temporal, dict) or set(temporal) != {"start", "end", "start_frame", "end_frame"}:
            errors.append(f"shot {index} has invalid temporal schema")
            continue
        start, end = temporal["start_frame"], temporal["end_frame"]
        start_sec, end_sec = temporal["start"], temporal["end"]
        if not all(type(value) is int for value in (start, end)) or start < 0 or start >= end:
            errors.append(f"shot {index} has invalid frame range")
            continue
        if start != previous_end:
            errors.append(f"gap or overlap before shot {index}")
        previous_end = end
        if not all(type(value) in (float, int) for value in (start_sec, end_sec)) or start_sec < previous_time - 1e-6 or start_sec >= end_sec:
            errors.append(f"shot {index} has invalid timestamps")
        elif abs(start_sec - previous_time) > 1e-5:
            errors.append(f"timestamp gap before shot {index}")
        previous_time = end_sec
        matches = image_by_id.get(expected_id, [])
        if len(matches) != 1:
            missing += 1
            errors.append(f"shot {index} must have exactly one image")
        else:
            try:
                with Image.open(matches[0]) as image:
                    image.verify()
            except (OSError, ValueError) as exc:
                missing += 1
                errors.append(f"shot {index} has corrupt image: {exc}")
    if len(images) != len(shots):
        errors.append("image count does not equal shot count")
    if frame_count is not None and previous_end != frame_count:
        errors.append("final shot does not reach the final decoded frame")
    return ValidationResult(not errors, errors, len(shots), missing)


def validate_output_root(output_root: Path, movie_key: str | None = None) -> dict[str, ValidationResult]:
    if movie_key:
        candidates = [output_root / movie_key]
    else:
        candidates = sorted(path for path in output_root.iterdir() if path.is_dir() and MOVIE_ID.fullmatch(path.name)) if output_root.is_dir() else []
    results = {}
    for path in candidates:
        metadata_path = output_root / ".internal" / path.name / "video_metadata.json"
        fingerprint_path = output_root / ".internal" / path.name / "fingerprint.json"
        if not metadata_path.is_file() or not fingerprint_path.is_file():
            results[path.name] = ValidationResult(False, ["missing internal processing metadata"], 0, 0)
            continue
        frame_count = None
        if metadata_path.is_file():
            try:
                frame_count = _load_json(metadata_path).get("frame_count")
            except (ValueError, OSError, AttributeError):
                results[path.name] = ValidationResult(False, ["invalid internal processing metadata"], 0, 0)
                continue
        if type(frame_count) is not int or frame_count < 1:
            results[path.name] = ValidationResult(False, ["invalid decoded frame count"], 0, 0)
            continue
        results[path.name] = validate_movie(path, frame_count=frame_count)
    return results
