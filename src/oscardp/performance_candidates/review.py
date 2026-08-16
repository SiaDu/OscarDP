from __future__ import annotations

import random
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from oscardp.shots.media import ExternalToolError, require_media_tools

from .io import read_json, read_jsonl, write_json, write_jsonl


def _take_unique(target: list[tuple[str, dict[str, Any]]], label: str, rows: list[dict[str, Any]], count: int, key: str) -> None:
    existing = {row.get(key) for _source, row in target}
    for row in rows:
        identity = row.get(key)
        if identity in existing:
            continue
        target.append((label, row))
        existing.add(identity)
        if sum(source == label for source, _row in target) >= count:
            return


def _deterministic_random(rows: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    result = list(rows)
    random.Random(seed).shuffle(result)
    return result


def _shot_sample(shots: list[dict[str, Any]], audit: list[dict[str, Any]], count: int, movie_id: str) -> list[dict[str, Any]]:
    target: list[tuple[str, dict[str, Any]]] = []
    quota = max(1, count // 5)
    selected_by_score = sorted(shots, key=lambda row: (-float(row["shot_score"]), int(row["source_index"])))
    _take_unique(target, "top_ranked", selected_by_score, quota, "source_shot_id")
    threshold_rows = sorted(shots, key=lambda row: (abs(float(row["semantic"]["semantic_score"]) - 0.35), int(row["source_index"])))
    _take_unique(target, "threshold_near", threshold_rows, quota, "source_shot_id")
    overrides = [row for row in shots if row.get("selection_basis") == "semantic_override"]
    _take_unique(target, "semantic_override", overrides, quota, "source_shot_id")
    _take_unique(target, "random_selected", _deterministic_random(shots, movie_id + ":shots"), quota, "source_shot_id")
    rejected = [row for row in audit if row.get("status") == "rejected"]
    _take_unique(target, "random_rejected", _deterministic_random(rejected, movie_id + ":rejected"), quota, "source_shot_id")
    pool = [("selected", row) for row in selected_by_score] + [("rejected", row) for row in rejected]
    seen = {row.get("source_shot_id") for _label, row in target}
    for label, row in pool:
        if len(target) >= count:
            break
        if row.get("source_shot_id") not in seen:
            target.append((label, row)); seen.add(row.get("source_shot_id"))
    result = []
    selected_ids = {row["source_shot_id"] for row in shots}
    for ordinal, (stratum, row) in enumerate(target[:count], 1):
        result.append({
            "review_id": f"shot_review_{ordinal:04d}", "movie_id": movie_id,
            "stratum": stratum, "source_population": "selected" if row["source_shot_id"] in selected_ids else "rejected",
            "performance_shot_id": row.get("performance_shot_id"),
            "source_shot_id": row["source_shot_id"], "time": row.get("time"),
            "shot_score": row.get("shot_score"), "semantic": row.get("semantic"), "cv": row.get("cv"),
            "human_decision": None, "reviewer_notes": None,
        })
    return result


def _event_sample(events: list[dict[str, Any]], count: int, movie_id: str) -> list[dict[str, Any]]:
    target: list[tuple[str, dict[str, Any]]] = []
    quota = max(1, count // 5)
    ranked = sorted(events, key=lambda row: (-float(row["event_score"]), row["event_id"]))
    _take_unique(target, "top_ranked", ranked, quota, "event_id")
    _take_unique(target, "single_shot", [row for row in ranked if len(row["performance_shot_ids"]) == 1], quota, "event_id")
    _take_unique(target, "multi_shot", [row for row in ranked if len(row["performance_shot_ids"]) > 1], quota, "event_id")
    near_limit = sorted(events, key=lambda row: (abs(float(row["time"]["duration_sec"]) - 30.0), row["event_id"]))
    _take_unique(target, "near_duration_limit", near_limit, quota, "event_id")
    _take_unique(target, "random", _deterministic_random(events, movie_id + ":events"), quota, "event_id")
    seen = {row["event_id"] for _label, row in target}
    for row in ranked:
        if len(target) >= count:
            break
        if row["event_id"] not in seen:
            target.append(("ranked_fill", row)); seen.add(row["event_id"])
    return [{
        "review_id": f"event_review_{ordinal:04d}", "movie_id": movie_id,
        "stratum": stratum, **row, "boundary_decision": None,
        "performance_decision": None, "reviewer_notes": None,
    } for ordinal, (stratum, row) in enumerate(target[:count], 1)]


def _first_frame(row: dict[str, Any]) -> str | None:
    frames = (row.get("cv") or {}).get("sample_frames") or []
    return frames[0].get("path") if frames else None


def _contact_sheet(rows: list[dict[str, Any]], paths: list[str | None], run_dir: Path, output: Path, title_key: str) -> None:
    width, cell_height = 320, 220
    columns = 4
    line_count = max(1, (len(rows) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * width, line_count * cell_height), "white")
    for index, (row, relative) in enumerate(zip(rows, paths, strict=True)):
        x, y = (index % columns) * width, (index // columns) * cell_height
        if relative and (run_dir / relative).is_file():
            with Image.open(run_dir / relative) as image:
                image = image.convert("RGB"); image.thumbnail((width, 180))
                sheet.paste(image, (x + (width - image.width) // 2, y))
        draw = ImageDraw.Draw(sheet)
        draw.text((x + 6, y + 184), str(row.get(title_key) or row.get("source_shot_id")), fill="black")
        draw.text((x + 6, y + 200), str(row.get("stratum", "")), fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=90)


def _preview(video: Path, event: dict[str, Any], output: Path) -> None:
    require_media_tools()
    start = float(event["time"]["start_sec"])
    duration = float(event["time"]["duration_sec"])
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-v", "error", "-ss", f"{start:.6f}", "-t", f"{duration:.6f}",
        "-dn", "-sn", "-i", str(video), "-map", "0:v:0", "-map", "0:a:0?",
        "-map_metadata", "-1", "-map_chapters", "-1",
        "-vf", "scale='min(1280,iw)':-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-profile:v", "high", "-level:v", "4.1", "-preset", "veryfast", "-crf", "28",
        "-c:a", "aac", "-ac", "2", "-b:a", "128k", "-movflags", "+faststart", str(output),
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    if process.returncode:
        raise ExternalToolError(process.stderr.strip() or f"Failed to create event preview: {event['event_id']}")


def prepare_review_sample(run_dir: Path, shot_count: int = 20, event_count: int = 20) -> dict[str, Any]:
    if shot_count < 0 or event_count < 0 or shot_count + event_count == 0:
        raise ValueError("Review sample counts must be non-negative and not both zero")
    manifest = read_json(run_dir / "manifest.json")
    shots = read_jsonl(run_dir / "performance_shots.jsonl")
    events = read_jsonl(run_dir / "performance_events.jsonl")
    audit = read_jsonl(run_dir / "screening_audit.jsonl")
    movie_id = manifest["movie_id"]
    shot_sample = _shot_sample(shots, audit, min(shot_count, len(shots) + len(audit)), movie_id)
    event_sample = _event_sample(events, min(event_count, len(events)), movie_id)
    review_dir = run_dir / "review"
    write_jsonl(review_dir / "performance_shots.review.jsonl", shot_sample)
    write_jsonl(review_dir / "performance_events.review.jsonl", event_sample)
    shot_paths = [_first_frame(row) for row in shot_sample]
    shot_by_id = {row["performance_shot_id"]: row for row in shots}
    event_paths = [_first_frame(shot_by_id[row["performance_shot_ids"][0]]) for row in event_sample]
    _contact_sheet(shot_sample, shot_paths, run_dir, review_dir / "performance_shots.contact_sheet.jpg", "source_shot_id")
    _contact_sheet(event_sample, event_paths, run_dir, review_dir / "performance_events.contact_sheet.jpg", "event_id")
    video = Path(manifest["inputs"]["video"]["path"])
    for event in event_sample:
        preview = review_dir / "previews" / f"{event['event_id']}.mp4"
        _preview(video, event, preview)
        event["preview_path"] = preview.relative_to(run_dir).as_posix()
    write_jsonl(review_dir / "performance_events.review.jsonl", event_sample)
    summary = {
        "movie_id": movie_id, "shot_sample_count": len(shot_sample), "event_sample_count": len(event_sample),
        "shot_labels": {"human_decision": ["keep", "reject", "borderline"]},
        "event_labels": {"boundary_decision": ["accept", "reject", "borderline"], "performance_decision": ["keep", "reject", "borderline"]},
    }
    write_json(review_dir / "review_manifest.json", summary)
    return summary


def evaluate_review(shot_sample: Path, event_sample: Path) -> dict[str, Any]:
    shots, events = read_jsonl(shot_sample), read_jsonl(event_sample)
    invalid_shots = [row.get("review_id") for row in shots if row.get("human_decision") not in {"keep", "reject", "borderline"}]
    invalid_events = [row.get("review_id") for row in events if row.get("boundary_decision") not in {"accept", "reject", "borderline"}]
    if invalid_shots or invalid_events:
        raise ValueError(f"Unlabelled or invalid review rows: shots={invalid_shots[:5]} events={invalid_events[:5]}")
    retained = [row for row in shots if row.get("source_population") == "selected" and row["human_decision"] != "borderline"]
    rejected = [row for row in shots if row.get("source_population") == "rejected" and row["human_decision"] != "borderline"]
    boundary = [row for row in events if row["boundary_decision"] != "borderline"]
    retained_precision = sum(row["human_decision"] == "keep" for row in retained) / len(retained) if retained else None
    rejected_false_negative = sum(row["human_decision"] == "keep" for row in rejected) / len(rejected) if rejected else None
    boundary_acceptance = sum(row["boundary_decision"] == "accept" for row in boundary) / len(boundary) if boundary else None
    gates = {
        "retained_precision": retained_precision is not None and retained_precision >= 0.80,
        "rejected_false_negative_rate": rejected_false_negative is not None and rejected_false_negative <= 0.20,
        "event_boundary_acceptance": boundary_acceptance is not None and boundary_acceptance >= 0.80,
    }
    return {
        "shot_review_count": len(shots), "event_review_count": len(events),
        "retained_precision": retained_precision, "rejected_false_negative_rate": rejected_false_negative,
        "event_boundary_acceptance": boundary_acceptance, "gates": gates, "passed": all(gates.values()),
    }
