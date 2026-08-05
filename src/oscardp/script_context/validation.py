from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import read_jsonl


@dataclass(frozen=True)
class ContextValidationResult:
    passed: bool
    errors: list[str]
    scene_count: int
    alignment_count: int
    shot_count: int


def validate_data(context: dict[str, Any], alignments: list[dict[str, Any]], shot_context: list[dict[str, Any]], shots: list[dict[str, Any]]) -> ContextValidationResult:
    errors: list[str] = []
    scenes = context.get("script_scenes", [])
    scene_ids = [scene.get("scene_id") for scene in scenes]
    if len(scene_ids) != len(set(scene_ids)):
        errors.append("scene_id values must be unique")
    blocks = [block for scene in scenes for block in scene.get("script_blocks", [])]
    block_ids = [block.get("block_id") for block in blocks]
    if len(block_ids) != len(set(block_ids)):
        errors.append("block_id values must be unique")
    known_scenes, known_blocks = set(scene_ids), set(block_ids)
    for scene in scenes:
        orders = [block.get("source_order") for block in scene.get("script_blocks", [])]
        if orders != list(range(1, len(orders) + 1)):
            errors.append(f"unstable block order in {scene.get('scene_id')}")
        pages = scene.get("script_pages", {})
        if not isinstance(pages.get("start"), int) or not isinstance(pages.get("end"), int) or pages["start"] < 1 or pages["end"] < pages["start"]:
            errors.append(f"invalid page range in {scene.get('scene_id')}")
        for block in scene.get("script_blocks", []):
            if not str(block.get("text", "")).strip():
                errors.append(f"empty text in {block.get('block_id')}")
            if block.get("block_type") == "dialogue" and not block.get("speaker"):
                errors.append(f"dialogue lacks speaker: {block.get('block_id')}")
    for page in context.get("broken_pages", []):
        if not any(scene["script_pages"]["start"] <= page <= scene["script_pages"]["end"] and scene["parsing"]["needs_review"] for scene in scenes):
            errors.append(f"broken page {page} is not marked needs_review")
    subtitle_ids = [row.get("subtitle_id") for row in alignments]
    if len(subtitle_ids) != len(set(subtitle_ids)):
        errors.append("subtitle_id values must be unique")
    times = [row.get("time", {}).get("start_sec") for row in alignments]
    if times != sorted(times):
        errors.append("alignments must be sorted by subtitle time")
    last_scene_index = -1
    scene_order = {scene_id: index for index, scene_id in enumerate(scene_ids)}
    for row in alignments:
        scene_id = row.get("scene_id")
        if scene_id is not None and scene_id not in known_scenes:
            errors.append(f"unknown alignment scene_id: {scene_id}")
        if scene_id is not None and row.get("alignment", {}).get("status") in {"auto_aligned", "llm_aligned"}:
            current = scene_order[scene_id]
            if current < last_scene_index:
                errors.append(f"alignment sequence retreats at {row.get('subtitle_id')}")
            last_scene_index = max(last_scene_index, current)
        for match in row.get("script_matches", []):
            if match.get("block_id") not in known_blocks:
                errors.append(f"unknown alignment block_id: {match.get('block_id')}")
            score = match.get("combined_score")
            if score is not None and (not isinstance(score, (int, float)) or not 0 <= score <= 1):
                errors.append(f"invalid combined_score in {row.get('subtitle_id')}")
        status, review = row.get("alignment", {}).get("status"), row.get("alignment", {}).get("needs_review")
        if status == "needs_review" and review is not True:
            errors.append(f"needs_review status mismatch in {row.get('subtitle_id')}")
        if status == "auto_aligned" and review is not False:
            errors.append(f"auto_aligned status mismatch in {row.get('subtitle_id')}")
    if len(shot_context) != len(shots):
        errors.append("shot context row count differs from shots.jsonl")
    for index, (mapped, original) in enumerate(zip(shot_context, shots), 1):
        if mapped.get("shot_id") != original.get("shot_id"):
            errors.append(f"shot order differs at row {index}")
        expected_frame = {key: original[key] for key in ("start_frame", "end_frame", "frame_count")}
        if mapped.get("frame_range") != expected_frame:
            errors.append(f"frame range changed for {original.get('shot_id')}")
        expected_time = {"start": original["start_time"], "end": original["end_time"], "start_sec": original["start_sec"], "end_sec": original["end_sec"], "duration_sec": original["duration_sec"]}
        if mapped.get("time") != expected_time:
            errors.append(f"timestamps changed for {original.get('shot_id')}")
        if any(mapped.get(key) is not None for key in ("visible_characters", "shot_scale", "camera_movement")):
            errors.append(f"visual analysis fields must remain null for {original.get('shot_id')}")
        for sub in mapped.get("subtitles", []):
            if sub.get("subtitle_id") not in set(subtitle_ids):
                errors.append(f"unknown subtitle reference in {original.get('shot_id')}")
            if any(not 0 <= float(sub.get(key, -1)) <= 1 for key in ("shot_coverage", "subtitle_coverage")) or float(sub.get("overlap_sec", -1)) < 0:
                errors.append(f"invalid overlap in {original.get('shot_id')}")
        scene = mapped.get("scene")
        if scene and scene.get("scene_id") not in known_scenes:
            errors.append(f"unknown shot scene in {original.get('shot_id')}")
        for match in mapped.get("script_matches", []):
            if match.get("block_id") not in known_blocks:
                errors.append(f"unknown shot block in {original.get('shot_id')}")
    return ContextValidationResult(not errors, errors, len(scenes), len(alignments), len(shot_context))


def validate_files(context_path: Path, alignment_path: Path, shot_context_path: Path, shots_path: Path) -> ContextValidationResult:
    import json
    with context_path.open("r", encoding="utf-8") as handle:
        context = json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    return validate_data(context, read_jsonl(alignment_path), read_jsonl(shot_context_path), read_jsonl(shots_path))
