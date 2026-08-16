from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path
from typing import Any

from .io import read_json, read_jsonl, sha256_file
from .schema import ValidationReport

FORBIDDEN_KEYS = {
    "person_count", "person_score", "person_detector", "identity", "face_identity",
    "face_track", "emotion_label", "action_unit", "au", "gaze",
}


def _walk_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | set().union(*(_walk_keys(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_walk_keys(item) for item in value), set())
    return set()


def validate_run(run_dir: Path) -> ValidationReport:
    errors: list[str] = []
    required = ["manifest.json", "performance_shots.jsonl", "performance_events.jsonl", "screening_audit.jsonl", "qc_summary.json"]
    for name in required:
        if not (run_dir / name).is_file():
            errors.append(f"missing output: {name}")
    if errors:
        return ValidationReport(False, errors, 0, 0, 0)
    try:
        manifest = read_json(run_dir / "manifest.json")
        shots = read_jsonl(run_dir / "performance_shots.jsonl")
        events = read_jsonl(run_dir / "performance_events.jsonl")
        audit = read_jsonl(run_dir / "screening_audit.jsonl")
    except (OSError, ValueError) as exc:
        return ValidationReport(False, [str(exc)], 0, 0, 0)
    for name, artifact in manifest.get("outputs", {}).items():
        path = run_dir / name
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            errors.append(f"output hash mismatch: {name}")
    shot_ids = [row.get("performance_shot_id") for row in shots]
    source_ids = [row.get("source_shot_id") for row in shots]
    if len(set(shot_ids)) != len(shot_ids) or None in shot_ids:
        errors.append("performance shot IDs are missing or duplicated")
    if len(set(source_ids)) != len(source_ids) or None in source_ids:
        errors.append("source shot IDs are missing or duplicated")
    indices = [row.get("source_index") for row in shots]
    if indices != sorted(indices):
        errors.append("performance shots are not in source order")
    rank_order = sorted(shots, key=lambda row: (-float(row["shot_score"]), float(row["time"]["start_sec"]), row["performance_shot_id"]))
    if any(row.get("movie_rank") != rank for rank, row in enumerate(rank_order, 1)):
        errors.append("performance shot ranks are invalid")
    for row in shots:
        semantic = float(row.get("semantic", {}).get("semantic_score", -1))
        face = float(row.get("cv", {}).get("face_score", -1))
        context = float(row.get("context_confidence", -1))
        expected = round(min(1.0, 0.65 * semantic + 0.25 * face + 0.10 * context), 6)
        if not math.isclose(float(row.get("shot_score", -1)), expected, abs_tol=1e-6):
            errors.append(f"shot score is not reproducible: {row.get('performance_shot_id')}")
        if row.get("selection_basis") not in {"semantic_and_cv", "semantic_override"}:
            errors.append(f"invalid selection basis: {row.get('performance_shot_id')}")
        start, end = float(row["time"]["start_sec"]), float(row["time"]["end_sec"])
        if not start < end:
            errors.append(f"invalid shot time range: {row.get('performance_shot_id')}")
        for frame in row.get("cv", {}).get("sample_frames", []):
            if not start <= float(frame["time_sec"]) < end:
                errors.append(f"sample timestamp outside shot: {row.get('performance_shot_id')}")
            if not (run_dir / frame["path"]).is_file():
                errors.append(f"missing sampled frame: {frame.get('path')}")
        forbidden = _walk_keys(row) & FORBIDDEN_KEYS
        if forbidden:
            errors.append(f"forbidden CV/biometric fields in shot row: {sorted(forbidden)}")
    selected_by_id = {row["performance_shot_id"]: row for row in shots}
    maximum_duration = float(manifest.get("fingerprint", {}).get("max_event_duration_sec", 30.0))
    event_ids: set[str] = set()
    event_rank_order = sorted(events, key=lambda row: (-float(row["event_score"]), float(row["time"]["start_sec"]), row["event_id"]))
    if any(row.get("movie_rank") != rank for rank, row in enumerate(event_rank_order, 1)):
        errors.append("performance event ranks are invalid")
    for event in events:
        event_id = event.get("event_id")
        if not event_id or event_id in event_ids:
            errors.append("performance event IDs are missing or duplicated")
        event_ids.add(event_id)
        members = [selected_by_id.get(value) for value in event.get("performance_shot_ids", [])]
        if not members or any(row is None for row in members):
            errors.append(f"event has missing performance-shot references: {event_id}")
            continue
        member_rows = [row for row in members if row is not None]
        member_indices = [int(row["source_index"]) for row in member_rows]
        if any(right != left + 1 for left, right in pairwise(member_indices)):
            errors.append(f"event members are not consecutive: {event_id}")
        scenes = {(row.get("scene") or {}).get("scene_id") for row in member_rows}
        if len(scenes) != 1:
            errors.append(f"event crosses scene boundaries: {event_id}")
        duration = float(event["time"]["duration_sec"])
        if duration > maximum_duration and event.get("duration_limit_exception") != "single_source_shot":
            errors.append(f"event exceeds duration limit: {event_id}")
    counts = manifest.get("counts", {})
    if counts.get("performance_shot_count") != len(shots) or counts.get("performance_event_count") != len(events):
        errors.append("manifest primary-output counts do not match")
    return ValidationReport(not errors, errors, len(shots), len(events), len(audit))
