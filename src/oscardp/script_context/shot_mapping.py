from __future__ import annotations

from typing import Any


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def map_shots(
    shots: list[dict[str, Any]], alignments: list[dict[str, Any]], context: dict[str, Any],
    movie_key: str, interpolation_max_gap: float = 10.0,
) -> list[dict[str, Any]]:
    scene_by_id = {scene["scene_id"]: scene for scene in context["script_scenes"]}
    block_by_id = {block["block_id"]: (scene, block) for scene in context["script_scenes"] for block in scene["script_blocks"]}
    rows: list[dict[str, Any]] = []
    subtitle_cursor = 0
    for shot in shots:
        start, end = float(shot["start_sec"]), float(shot["end_sec"])
        while subtitle_cursor < len(alignments) and float(alignments[subtitle_cursor]["time"]["end_sec"]) <= start:
            subtitle_cursor += 1
        overlaps: list[tuple[dict[str, Any], float]] = []
        probe = subtitle_cursor
        while probe < len(alignments) and float(alignments[probe]["time"]["start_sec"]) < end:
            alignment = alignments[probe]
            amount = _overlap(start, end, float(alignment["time"]["start_sec"]), float(alignment["time"]["end_sec"]))
            if amount > 0:
                overlaps.append((alignment, amount))
            probe += 1
        scene_scores: dict[str, float] = {}
        for alignment, amount in overlaps:
            if alignment.get("scene_id"):
                scene_scores[alignment["scene_id"]] = scene_scores.get(alignment["scene_id"], 0.0) + amount
        chosen_scene = max(scene_scores, key=lambda key: (scene_scores[key], key)) if scene_scores else None
        subtitles = []
        match_map: dict[str, dict[str, Any]] = {}
        speakers: list[str] = []
        review = False
        for alignment, amount in overlaps:
            sub_duration = float(alignment["time"]["end_sec"]) - float(alignment["time"]["start_sec"])
            subtitles.append({
                "subtitle_id": alignment["subtitle_id"], "text": alignment["text"],
                "overlap_sec": round(amount, 6),
                "shot_coverage": round(amount / (end - start), 6) if end > start else 0.0,
                "subtitle_coverage": round(amount / sub_duration, 6) if sub_duration > 0 else 0.0,
            })
            review = review or bool(alignment["alignment"]["needs_review"])
            for match in alignment["script_matches"]:
                match_map.setdefault(match["block_id"], {
                    "block_id": match["block_id"], "speaker": match["speaker"], "match_score": match["combined_score"],
                })
                if match["speaker"] not in speakers:
                    speakers.append(match["speaker"])
        local = {"action_before": [], "action_during": [], "action_after": []}
        if match_map:
            ordered = sorted((block_by_id[block_id][1] for block_id in match_map), key=lambda b: b["source_order"])
            scene = block_by_id[ordered[0]["block_id"]][0]
            actions = [b for b in scene["script_blocks"] if b["block_type"] == "action"]
            before = [b for b in actions if b["source_order"] < ordered[0]["source_order"]]
            after = [b for b in actions if b["source_order"] > ordered[-1]["source_order"]]
            if before:
                local["action_before"] = [before[-1]["block_id"]]
            if after:
                local["action_after"] = [after[0]["block_id"]]
        scene_value = None
        if chosen_scene:
            scene = scene_by_id[chosen_scene]
            confidence = min(1.0, scene_scores[chosen_scene] / max(end - start, 1e-9))
            scene_value = {"scene_id": chosen_scene, "screenplay_scene_id": scene["screenplay_scene_id"], "method": "subtitle_script_alignment", "confidence": round(confidence, 6)}
        rows.append({
            "movie_id": movie_key, "shot_id": shot["shot_id"],
            "frame_range": {"start_frame": shot["start_frame"], "end_frame": shot["end_frame"], "frame_count": shot["frame_count"]},
            "time": {"start": shot["start_time"], "end": shot["end_time"], "start_sec": shot["start_sec"], "end_sec": shot["end_sec"], "duration_sec": shot["duration_sec"]},
            "keyframe": {"frame": shot["keyframe_frame"], "time_sec": shot["keyframe_time_sec"], "path": shot["keyframe_relpath"]},
            "scene": scene_value, "subtitles": subtitles, "script_matches": list(match_map.values()),
            "local_script_context": local, "dialogue_speakers": speakers,
            "visible_characters": None, "shot_scale": None, "camera_movement": None,
            "alignment": {"status": "needs_review" if review else ("aligned" if chosen_scene else "unaligned"), "needs_review": review},
        })
    _interpolate_scenes(rows, scene_by_id, interpolation_max_gap)
    return rows


def _interpolate_scenes(rows: list[dict[str, Any]], scenes: dict[str, dict[str, Any]], max_gap: float) -> None:
    anchors = [index for index, row in enumerate(rows) if row["scene"] is not None]
    for left_index, right_index in zip(anchors, anchors[1:]):
        if right_index == left_index + 1:
            continue
        left, right = rows[left_index], rows[right_index]
        scene_id = left["scene"]["scene_id"]
        if right["scene"]["scene_id"] != scene_id:
            continue
        gap = float(right["time"]["start_sec"]) - float(left["time"]["end_sec"])
        if gap > max_gap:
            continue
        scene = scenes[scene_id]
        confidence = min(float(left["scene"]["confidence"]), float(right["scene"]["confidence"]))
        for index in range(left_index + 1, right_index):
            if rows[index]["scene"] is None and not rows[index]["subtitles"]:
                rows[index]["scene"] = {"scene_id": scene_id, "screenplay_scene_id": scene["screenplay_scene_id"], "method": "same_scene_interpolation", "confidence": round(confidence, 6)}
                rows[index]["alignment"] = {"status": "interpolated", "needs_review": False}
