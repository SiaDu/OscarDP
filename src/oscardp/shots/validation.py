from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .schema import ValidationResult
from .frame_samples import POLICY, image_info, load_json, physical_samples
from .transnet import sha256_file

MOVIE_ID = re.compile(r"tt\d{7,10}")
IMAGE_NAME = re.compile(r"shot_\d{6}_\d{2}-\d{2}-\d{2}-\d{3}\.jpg")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def validate_movie(movie_dir: Path, *, frame_count: int | None = None, official_layout: bool = True,
                   frame_profile: str = "auto", samples_path: Path | None = None) -> ValidationResult:
    errors: list[str] = []
    movie_id = movie_dir.name
    stage_path = movie_dir / f"Stage01_{movie_id}.json"
    if not MOVIE_ID.fullmatch(movie_id):
        return ValidationResult(False, ["movie directory must use an IMDb ID"], 0, 0)
    if not stage_path.is_file():
        return ValidationResult(False, [f"missing {stage_path.name}"], 0, 0)
    if official_layout:
        required = {stage_path.name, "shot_img"}
        optional = {f"Stage02_{movie_id}.json", f"{movie_id}.srt", "stage02_curated"}
        names = {path.name for path in movie_dir.iterdir()}
        if not required <= names or names - required - optional:
            errors.append("official movie directory has missing Stage01 layout entries or unexpected files")
        for name in names & optional:
            path = movie_dir / name
            if path.is_symlink() or (not path.is_dir() if name == "stage02_curated" else not path.is_file()):
                errors.append(f"invalid canonical workspace entry: {name}")
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
    if frame_profile not in {"auto", "legacy", "three_sample"}:
        raise ValueError("Unknown frame profile")
    samples_path = samples_path or movie_dir.parent / ".internal" / movie_id / "frame_samples_v1.json"
    if frame_profile == "auto":
        frame_profile = "three_sample" if samples_path.exists() or any(
            (movie_dir / "shot_img").glob("*_middle.jpg")) else "legacy"
    record = None
    if frame_profile == "three_sample":
        try:
            record = load_json(samples_path)
            if (record["policy"] != POLICY or record["movie_id"] != movie_id
                    or record["stage01_sha256"] != sha256_file(stage_path)
                    or record["status"] != "complete"
                    or set(record["shots"]) != {shot["shot_id"] for shot in shots}):
                raise ValueError("sampling policy, identity, status or shot inventory mismatch")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(f"invalid frame sampling metadata: {exc}")
            record = None
    missing = 0
    previous_end = 0
    previous_time = 0.0
    image_dir = movie_dir / "shot_img"
    if image_dir.is_symlink():
        errors.append("symlink shot_img directory is not valid output")
    images = list(image_dir.iterdir()) if image_dir.is_dir() else []
    expected_images = set()
    image_by_id: dict[str, list[Path]] = {}
    for image in images:
        if frame_profile == "legacy" and not IMAGE_NAME.fullmatch(image.name):
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
        if not all(type(value) in (float, int) and math.isfinite(value) for value in (start_sec, end_sec)) or start_sec < previous_time - 1e-6 or start_sec >= end_sec:
            errors.append(f"shot {index} has invalid timestamps")
        elif abs(start_sec - previous_time) > 1e-5:
            errors.append(f"timestamp gap before shot {index}")
        previous_time = end_sec
        matches = image_by_id.get(expected_id, [])
        if frame_profile == "legacy":
            if len(matches) != 1:
                missing += 1
                errors.append(f"shot {index} must have exactly one image")
            for path in matches:
                try:
                    image_info(path)
                except (OSError, ValueError) as exc:
                    missing += 1
                    errors.append(f"shot {index} has corrupt image: {exc}")
        else:
            expected = physical_samples(expected_id, start, end)
            expected_images.update(Path(item["image_relpath"]).name for item in expected.values())
            try:
                if record is None:
                    raise ValueError("missing verified sample metadata")
                actual = record["shots"][expected_id]
                if set(actual) != set(expected):
                    raise ValueError("sample roles mismatch")
                for role, spec in expected.items():
                    sample = actual[role]
                    if any(sample.get(key) != value for key, value in spec.items()):
                        raise ValueError(f"wrong {role} frame, filename or alias")
                    if image_info(movie_dir / spec["image_relpath"])["sha256"] != sample["sha256"]:
                        raise ValueError(f"changed {role} image hash")
                    timestamp = sample.get("timestamp_sec")
                    if timestamp is not None and (type(timestamp) not in (float, int)
                                                  or not math.isfinite(timestamp)):
                        raise ValueError("invalid sample timestamp")
                for rename in record.get("renames", []):
                    if rename["shot_id"] == expected_id and actual["middle"]["sha256"] != rename["original_sha256"]:
                        raise ValueError("Middle image differs from preserved legacy SHA-256")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                missing += 1
                errors.append(f"shot {index} invalid three_sample images: {exc}")
    if frame_profile == "legacy" and len(images) != len(shots):
        errors.append("image count does not equal shot count")
    if frame_profile == "three_sample" and {p.name for p in images} != expected_images:
        errors.append("missing or unexpected three_sample images")
    if frame_count is not None and previous_end != frame_count:
        errors.append("final shot does not reach the final decoded frame")
    return ValidationResult(not errors, errors, len(shots), missing)


def validate_output_root(output_root: Path, movie_key: str | None = None, *, frame_profile: str = "auto") -> dict[str, ValidationResult]:
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
        results[path.name] = validate_movie(path, frame_count=frame_count, frame_profile=frame_profile)
    return results
