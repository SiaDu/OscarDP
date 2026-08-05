from __future__ import annotations

from typing import Any, Protocol


class LLMResolver(Protocol):
    def resolve(self, request: dict[str, Any]) -> dict[str, Any]: ...


def build_review_requests(context: dict[str, Any], alignments: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    dialogues = [
        {"scene_id": scene["scene_id"], "block_id": block["block_id"], "speaker": block["speaker"], "text": block["text"]}
        for scene in context["script_scenes"] for block in scene["script_blocks"] if block["block_type"] == "dialogue"
    ]
    by_block = {row["block_id"]: index for index, row in enumerate(dialogues)}
    alignment_requests = []
    for index, alignment in enumerate(alignments):
        if not alignment["alignment"]["needs_review"]:
            continue
        matched = alignment["script_matches"]
        center = by_block.get(matched[0]["block_id"], 0) if matched else 0
        alignment_requests.append({
            "request_id": f"alignment_review_{len(alignment_requests) + 1:06d}",
            "task": "select_existing_blocks_or_no_match",
            "subtitle": {"subtitle_id": alignment["subtitle_id"], "text": alignment["text"], "time": alignment["time"]},
            "subtitle_context": [
                {"subtitle_id": row["subtitle_id"], "text": row["text"]}
                for row in alignments[max(0, index - 2): index + 3]
            ],
            "dialogue_candidates": dialogues[max(0, center - 5): center + 6],
            "automatic_alignment": {"scene_id": alignment["scene_id"], "script_matches": matched, "alignment": alignment["alignment"]},
        })
    page_requests = [
        {"request_id": f"page_repair_{index:06d}", "task": "repair_page_structure", "script_page": page}
        for index, page in enumerate(context.get("broken_pages", []), 1)
    ]
    scene_requests = [
        {"request_id": f"scene_annotation_{index:06d}", "task": "annotate_scene", "scene_id": scene["scene_id"], "slugline": scene["slugline"]}
        for index, scene in enumerate(context["script_scenes"], 1)
        if scene["parsing"]["needs_review"]
    ]
    return {"page_repair_requests": page_requests, "alignment_requests": alignment_requests, "scene_annotation_requests": scene_requests}


def apply_alignment_responses(alignments: list[dict[str, Any]], responses: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    block_lookup = {
        block["block_id"]: (scene["scene_id"], block)
        for scene in context["script_scenes"] for block in scene["script_blocks"] if block["block_type"] == "dialogue"
    }
    subtitle_lookup = {row["subtitle_id"]: row for row in alignments}
    for response in responses:
        subtitle_id = response.get("subtitle_id")
        if subtitle_id not in subtitle_lookup:
            raise ValueError(f"LLM response references unknown subtitle_id: {subtitle_id}")
        block_ids = response.get("block_ids", [])
        if response.get("decision") == "no_match":
            block_ids = []
        if not isinstance(block_ids, list) or any(block_id not in block_lookup for block_id in block_ids):
            raise ValueError(f"LLM response references unknown block_id: {block_ids}")
        scenes = {block_lookup[block_id][0] for block_id in block_ids}
        if len(scenes) > 1:
            raise ValueError("LLM response cannot select blocks from multiple scenes")
        row = subtitle_lookup[subtitle_id]
        original = {"scene_id": row["scene_id"], "script_matches": row["script_matches"], "alignment": dict(row["alignment"])}
        row["scene_id"] = next(iter(scenes)) if scenes else None
        row["script_matches"] = [{
            "block_id": block_id, "speaker": block_lookup[block_id][1]["speaker"],
            "matched_text": block_lookup[block_id][1]["text"], "lexical_score": None,
            "semantic_score": None, "combined_score": float(response.get("confidence", 1.0)),
        } for block_id in block_ids]
        row["alignment"] = {
            "method": "llm_resolved", "status": "llm_aligned" if block_ids else "no_match",
            "candidate_margin": original["alignment"].get("candidate_margin", 0.0), "needs_review": False,
            "llm_resolution": {
                "resolver": response.get("resolver"), "model": response.get("model"),
                "request_id": response.get("request_id"), "response_id": response.get("response_id"),
                "resolved_at": response.get("resolved_at"), "original_automatic": original,
            },
        }
    return alignments
