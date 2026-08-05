from __future__ import annotations

import math
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
    scene_order = {scene_id: index for index, scene_id in enumerate(scene_ids)}
    scene_by_id = {scene["scene_id"]: scene for scene in scenes}
    block_lookup = {block["block_id"]: (scene["scene_id"], block) for scene in scenes for block in scene.get("script_blocks", [])}
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
    last_script_order = -1
    for row in alignments:
        scene_id = row.get("scene_id")
        if scene_id is not None and scene_id not in known_scenes:
            errors.append(f"unknown alignment scene_id: {scene_id}")
        if scene_id is not None and row.get("alignment", {}).get("reliable_anchor") is True:
            current = scene_order[scene_id]
            if current < last_scene_index:
                errors.append(f"alignment sequence retreats at {row.get('subtitle_id')}")
            last_scene_index = max(last_scene_index, current)
            script_order = row.get("alignment", {}).get("script_order_start")
            if not isinstance(script_order, int) or script_order < last_script_order:
                errors.append(f"alignment screenplay order retreats at {row.get('subtitle_id')}")
            else:
                last_script_order = script_order
        for match in row.get("script_matches", []):
            if match.get("block_id") not in known_blocks:
                errors.append(f"unknown alignment block_id: {match.get('block_id')}")
            score = match.get("combined_score")
            if score is not None and (not isinstance(score, (int, float)) or not 0 <= score <= 1):
                errors.append(f"invalid combined_score in {row.get('subtitle_id')}")
            token_start, token_end = match.get("matched_script_token_start"), match.get("matched_script_token_end")
            if token_start is not None and (not isinstance(token_start, int) or not isinstance(token_end, int) or token_start < 0 or token_end <= token_start):
                errors.append(f"invalid script token span in {row.get('subtitle_id')}")
        status, review = row.get("alignment", {}).get("status"), row.get("alignment", {}).get("needs_review")
        if status == "needs_review" and review is not True:
            errors.append(f"needs_review status mismatch in {row.get('subtitle_id')}")
        if status == "auto_aligned" and review is not False:
            errors.append(f"auto_aligned status mismatch in {row.get('subtitle_id')}")
    if len(shot_context) != len(shots):
        errors.append("shot context row count differs from shots.jsonl")
    known_subtitle_ids = set(subtitle_ids)
    for index, (mapped, original) in enumerate(zip(shot_context, shots), 1):
        if mapped.get("shot_id") != original.get("shot_id"):
            errors.append(f"shot order differs at row {index}")
        expected_frame = {key: original[key] for key in ("start_frame", "end_frame", "frame_count")}
        if mapped.get("frame_range") != expected_frame:
            errors.append(f"frame range changed for {original.get('shot_id')}")
        expected_time = {"start": original["start_time"], "end": original["end_time"], "start_sec": original["start_sec"], "end_sec": original["end_sec"], "duration_sec": original["duration_sec"]}
        if mapped.get("time") != expected_time:
            errors.append(f"timestamps changed for {original.get('shot_id')}")
        expected_keyframe = {"frame": original["keyframe_frame"], "time_sec": original["keyframe_time_sec"], "path": original["keyframe_relpath"]}
        if mapped.get("keyframe") != expected_keyframe:
            errors.append(f"keyframe changed for {original.get('shot_id')}")
        if any(mapped.get(key) is not None for key in ("visible_characters", "shot_scale", "camera_movement")):
            errors.append(f"visual analysis fields must remain null for {original.get('shot_id')}")
        for sub in mapped.get("subtitles", []):
            if sub.get("subtitle_id") not in known_subtitle_ids:
                errors.append(f"unknown subtitle reference in {original.get('shot_id')}")
            if any(not 0 <= float(sub.get(key, -1)) <= 1 for key in ("shot_coverage", "subtitle_coverage")) or float(sub.get("overlap_sec", -1)) < 0:
                errors.append(f"invalid overlap in {original.get('shot_id')}")
        scene = mapped.get("scene")
        if scene and scene.get("scene_id") not in known_scenes:
            errors.append(f"unknown shot scene in {original.get('shot_id')}")
        if scene and scene.get("scene_id") in scene_by_id and scene.get("screenplay_scene_id") != scene_by_id[scene["scene_id"]].get("screenplay_scene_id"):
            errors.append(f"shot screenplay scene ID mismatch in {original.get('shot_id')}")
        match_scenes: set[str] = set()
        for match in mapped.get("script_matches", []):
            if match.get("block_id") not in known_blocks:
                errors.append(f"unknown shot block in {original.get('shot_id')}")
            else:
                match_scenes.add(block_lookup[match["block_id"]][0])
        local = mapped.get("local_script_context", {})
        local_ids = [block_id for key in ("action_before", "action_during", "action_after") for block_id in local.get(key, [])]
        for block_id in local_ids:
            if block_id not in block_lookup:
                errors.append(f"unknown local action block in {original.get('shot_id')}: {block_id}")
                continue
            action_scene, block = block_lookup[block_id]
            if block.get("block_type") != "action":
                errors.append(f"local context references non-action block in {original.get('shot_id')}: {block_id}")
            if scene and action_scene != scene.get("scene_id"):
                errors.append(f"local action crosses primary scene in {original.get('shot_id')}: {block_id}")
        if "scene_transition" in mapped or "scene_candidates" in mapped:
            transition = mapped.get("scene_transition")
            if not isinstance(transition, bool):
                errors.append(f"invalid scene_transition in {original.get('shot_id')}")
                transition = False
            candidates = mapped.get("scene_candidates")
            if not isinstance(candidates, list):
                errors.append(f"invalid scene_candidates in {original.get('shot_id')}")
                candidates = []
            candidate_ids = [candidate.get("scene_id") for candidate in candidates if isinstance(candidate, dict)]
            if len(candidate_ids) != len(set(candidate_ids)) or any(scene_id not in known_scenes for scene_id in candidate_ids):
                errors.append(f"invalid scene candidate IDs in {original.get('shot_id')}")
            if candidate_ids != sorted(candidate_ids, key=lambda scene_id: scene_order.get(scene_id, len(scene_order))):
                errors.append(f"scene candidates are not in screenplay order: {original.get('shot_id')}")
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                candidate_scene = scene_by_id.get(candidate.get("scene_id"))
                if candidate_scene and candidate.get("screenplay_scene_id") != candidate_scene.get("screenplay_scene_id"):
                    errors.append(f"scene candidate screenplay ID mismatch in {original.get('shot_id')}")
                overlap, confidence = candidate.get("overlap_sec"), candidate.get("confidence")
                if not isinstance(overlap, (int, float)) or isinstance(overlap, bool) or not math.isfinite(overlap) or overlap < 0:
                    errors.append(f"invalid scene candidate overlap in {original.get('shot_id')}")
                if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    errors.append(f"invalid scene candidate confidence in {original.get('shot_id')}")
            method = scene.get("method") if scene else None
            if scene is None:
                if candidates != [] or transition is not False:
                    errors.append(f"unaligned shot has scene evidence in {original.get('shot_id')}")
            elif method == "same_scene_interpolation":
                if candidates != []:
                    errors.append(f"interpolated shot has scene candidates in {original.get('shot_id')}")
                if transition is not False:
                    errors.append(f"interpolated shot is marked scene transition in {original.get('shot_id')}")
                if mapped.get("subtitles") != [] or mapped.get("script_matches") != []:
                    errors.append(f"interpolated shot has direct alignment evidence in {original.get('shot_id')}")
                if mapped.get("alignment") != {"status": "interpolated", "needs_review": False}:
                    errors.append(f"invalid interpolated alignment metadata in {original.get('shot_id')}")
            elif method == "subtitle_script_alignment":
                if not candidates:
                    errors.append(f"direct shot has no scene candidates in {original.get('shot_id')}")
                if not match_scenes.issubset(set(candidate_ids)):
                    errors.append(f"scene candidates omit matched scenes in {original.get('shot_id')}")
                if scene.get("scene_id") not in candidate_ids:
                    errors.append(f"scene candidates omit primary scene in {original.get('shot_id')}")
                if any(not isinstance(candidate, dict) or not isinstance(candidate.get("overlap_sec"), (int, float)) or isinstance(candidate.get("overlap_sec"), bool) or not math.isfinite(candidate["overlap_sec"]) or candidate["overlap_sec"] <= 0 for candidate in candidates):
                    errors.append(f"direct scene candidate lacks positive overlap in {original.get('shot_id')}")
                if candidates and abs(sum(float(candidate.get("confidence", 0)) for candidate in candidates if isinstance(candidate, dict)) - 1.0) > 1e-5:
                    errors.append(f"scene candidate confidence is not normalized in {original.get('shot_id')}")
                expected_transition = len(match_scenes) > 1 if match_scenes else len(candidate_ids) > 1
                if transition != expected_transition:
                    errors.append(f"scene transition flag mismatch in {original.get('shot_id')}")
                if expected_transition and (mapped.get("alignment", {}).get("status") != "scene_transition" or mapped.get("alignment", {}).get("needs_review") is not True):
                    errors.append(f"multi-scene shot is not marked for review: {original.get('shot_id')}")
            else:
                errors.append(f"unknown shot scene method in {original.get('shot_id')}: {method}")
    return ContextValidationResult(not errors, errors, len(scenes), len(alignments), len(shot_context))


def validate_files(context_path: Path, alignment_path: Path, shot_context_path: Path, shots_path: Path) -> ContextValidationResult:
    import json
    with context_path.open("r", encoding="utf-8") as handle:
        context = json.load(handle, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    return validate_data(context, read_jsonl(alignment_path), read_jsonl(shot_context_path), read_jsonl(shots_path))
