from __future__ import annotations

from typing import Any, Protocol

from .alignment import build_script_units, tokenize


class LLMResolver(Protocol):
    def resolve(self, request: dict[str, Any]) -> dict[str, Any]: ...


def _anchor_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {
        "subtitle_id": row["subtitle_id"], "scene_id": row["scene_id"],
        "block_ids": [match["block_id"] for match in row["script_matches"]],
        "screenplay_order": [row["alignment"]["script_order_start"], row["alignment"]["script_order_end"]],
        "method": row["alignment"]["method"], "score": min(match["combined_score"] for match in row["script_matches"]),
    }


def _candidate_interval(
    before: dict[str, Any] | None, after: dict[str, Any] | None,
    units: list[Any], window: int,
) -> tuple[list[Any], bool, str]:
    if before is None and after is None:
        return [], True, "no_reliable_anchor"
    if before is not None and after is not None:
        left = int(before["alignment"]["script_order_end"])
        right = int(after["alignment"]["script_order_start"])
        left_scene, right_scene = units[left].scene_index, units[right].scene_index
        if left > right or abs(right_scene - left_scene) > 1:
            return [], True, "untrustworthy_anchor_interval"
        allowed_scenes = {left_scene} if left_scene == right_scene else {left_scene, right_scene}
        return [unit for unit in units[left:right + 1] if unit.scene_index in allowed_scenes], False, "between_reliable_anchors"
    if before is not None:
        left = int(before["alignment"]["script_order_end"])
        scene = units[left].scene_index
        return [unit for unit in units[left:min(len(units), left + window + 1)] if unit.scene_index in {scene, scene + 1}], False, "after_reliable_anchor"
    right = int(after["alignment"]["script_order_start"])
    scene = units[right].scene_index
    return [unit for unit in units[max(0, right - window):right + 1] if unit.scene_index in {scene - 1, scene}], False, "before_reliable_anchor"


def build_review_requests(
    context: dict[str, Any], alignments: list[dict[str, Any]], *,
    local_window: int = 40, max_group_size: int = 8, max_time_gap_sec: float = 5.0,
) -> dict[str, list[dict[str, Any]]]:
    units = build_script_units(context)
    anchors = [index for index, row in enumerate(alignments) if row["alignment"].get("reliable_anchor")]
    unresolved = [index for index, row in enumerate(alignments) if row["alignment"]["needs_review"]]
    prepared: list[dict[str, Any]] = []
    for index in unresolved:
        before_index = next((anchor for anchor in reversed(anchors) if anchor < index), None)
        after_index = next((anchor for anchor in anchors if anchor > index), None)
        before = alignments[before_index] if before_index is not None else None
        after = alignments[after_index] if after_index is not None else None
        candidates, insufficient, reason = _candidate_interval(before, after, units, local_window)
        prepared.append({
            "index": index, "before": before, "after": after, "candidates": candidates,
            "candidate_key": tuple(unit.block["block_id"] for unit in candidates),
            "insufficient": insufficient, "reason": reason,
        })

    groups: list[list[dict[str, Any]]] = []
    for item in prepared:
        if groups:
            previous = groups[-1][-1]
            previous_row, row = alignments[previous["index"]], alignments[item["index"]]
            adjacent = item["index"] == previous["index"] + 1
            temporal = float(row["time"]["start_sec"]) - float(previous_row["time"]["end_sec"]) <= max_time_gap_sec
            same_interval = item["candidate_key"] == previous["candidate_key"] and item["insufficient"] == previous["insufficient"]
            if adjacent and temporal and same_interval and len(groups[-1]) < max_group_size:
                groups[-1].append(item)
                continue
        groups.append([item])

    alignment_requests: list[dict[str, Any]] = []
    for ordinal, group in enumerate(groups, 1):
        first = group[0]
        rows = [alignments[item["index"]] for item in group]
        candidates = first["candidates"]
        alignment_requests.append({
            "request_id": f"alignment_review_{ordinal:06d}",
            "task": "select_existing_blocks_or_no_match",
            "subtitle_ids": [row["subtitle_id"] for row in rows],
            "subtitles": [{"subtitle_id": row["subtitle_id"], "text": row["text"], "time": row["time"]} for row in rows],
            "previous_anchor": _anchor_summary(first["before"]), "next_anchor": _anchor_summary(first["after"]),
            "candidate_scenes": list(dict.fromkeys(unit.block["scene_id"] for unit in candidates)),
            "dialogue_candidates": [{
                "scene_id": unit.block["scene_id"], "block_id": unit.block["block_id"],
                "screenplay_order": unit.block_index, "speaker": unit.block["speaker"], "text": unit.block["text"],
            } for unit in candidates],
            "automatic_candidate_mappings": [{
                "subtitle_id": row["subtitle_id"], "scene_id": row["scene_id"],
                "matches": row["script_matches"], "alignment": row["alignment"],
            } for row in rows],
            "reason_for_review": sorted(set(row["alignment"].get("review_reason", "low_confidence") for row in rows)),
            "candidate_interval_reason": first["reason"], "insufficient_candidates": first["insufficient"],
        })
    page_requests = [{"request_id": f"page_repair_{index:06d}", "task": "repair_page_structure", "script_page": page} for index, page in enumerate(context.get("broken_pages", []), 1)]
    scene_requests = [{"request_id": f"scene_annotation_{index:06d}", "task": "annotate_scene", "scene_id": scene["scene_id"], "slugline": scene["slugline"]} for index, scene in enumerate(context["script_scenes"], 1) if scene["parsing"]["needs_review"]]
    return {"page_repair_requests": page_requests, "alignment_requests": alignment_requests, "scene_annotation_requests": scene_requests}


def validate_review_requests(requests: list[dict[str, Any]], context: dict[str, Any]) -> list[str]:
    units = build_script_units(context)
    known = {unit.block["block_id"]: unit for unit in units}
    errors: list[str] = []
    for request in requests:
        orders: list[int] = []
        scenes = set(request.get("candidate_scenes", []))
        for candidate in request.get("dialogue_candidates", []):
            block_id = candidate.get("block_id")
            if block_id not in known:
                errors.append(f"{request.get('request_id')} references invalid candidate {block_id}")
                continue
            unit = known[block_id]
            orders.append(unit.block_index)
            if unit.block["scene_id"] not in scenes:
                errors.append(f"{request.get('request_id')} candidate scene mismatch")
        if orders != sorted(orders) or len(orders) != len(set(orders)):
            errors.append(f"{request.get('request_id')} candidates are not ordered and unique")
        if request.get("insufficient_candidates") and request.get("dialogue_candidates"):
            errors.append(f"{request.get('request_id')} insufficient request has candidates")
    return errors


def build_alignment_diagnostics(context: dict[str, Any], alignments: list[dict[str, Any]], requests: list[dict[str, Any]]) -> dict[str, Any]:
    methods: dict[str, int] = {}
    statuses: dict[str, int] = {}
    for row in alignments:
        method, status = row["alignment"]["method"], row["alignment"]["status"]
        methods[method] = methods.get(method, 0) + 1
        statuses[status] = statuses.get(status, 0) + 1
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in alignments:
        groups.setdefault(row["alignment_group_id"], []).append(row)
    n_to_one = sum(len(rows) > 1 and len({match["block_id"] for row in rows for match in row["script_matches"]}) == 1 for rows in groups.values())
    one_to_m = sum(len(row["script_matches"]) > 1 for row in alignments)
    n_to_m = sum(len(rows) > 1 and len({match["block_id"] for row in rows for match in row["script_matches"]}) > 1 for rows in groups.values())
    confirmed = [row for row in alignments if row["alignment"]["status"] == "auto_aligned"]
    violations = sum(
        int(right["alignment"]["script_order_start"] < left["alignment"]["script_order_start"])
        for left, right in zip(confirmed, confirmed[1:])
    )
    deletion_gaps = sum(max(0, int(right["alignment"]["script_order_start"]) - int(left["alignment"]["script_order_end"]) - 1) for left, right in zip(confirmed, confirmed[1:]))
    request_errors = validate_review_requests(requests, context)
    substring_false_no_match = 0
    for request in requests:
        candidate_tokens = [tokenize(candidate["text"]).tokens for candidate in request.get("dialogue_candidates", [])]
        mapping = {item["subtitle_id"]: item for item in request.get("automatic_candidate_mappings", [])}
        for subtitle in request.get("subtitles", []):
            automatic = mapping.get(subtitle["subtitle_id"], {})
            if automatic.get("alignment", {}).get("status") != "no_match":
                continue
            tokens = tokenize(subtitle["text"]).tokens
            if len(tokens) >= 4 and any(any(block[index:index + len(tokens)] == tokens for index in range(len(block) - len(tokens) + 1)) for block in candidate_tokens):
                substring_false_no_match += 1
    return {
        "schema_version": "1.0", "total_subtitles": len(alignments),
        "auto_aligned": statuses.get("auto_aligned", 0), "needs_review": statuses.get("needs_review", 0),
        "no_match": statuses.get("no_match", 0), "method_counts": dict(sorted(methods.items())),
        "exact_matches": sum(count for method, count in methods.items() if "exact" in method),
        "substring_fragment_matches": sum(count for method, count in methods.items() if "substring" in method),
        "fuzzy_matches": sum(count for method, count in methods.items() if "rapidfuzz" in method),
        "n_subtitles_to_one_block_count": n_to_one, "one_subtitle_to_m_blocks_count": one_to_m,
        "n_to_m_count": n_to_m, "screenplay_deletion_gaps": deletion_gaps,
        "improvised_no_match_candidates": statuses.get("no_match", 0),
        "insufficient_candidate_groups": sum(bool(request["insufficient_candidates"]) for request in requests),
        "review_request_group_count": len(requests),
        "unresolved_local_substring_false_no_match": substring_false_no_match,
        "monotonicity_violations": violations, "scene_candidate_window_violations": len(request_errors),
        "candidate_window_errors": request_errors,
    }


def apply_alignment_responses(alignments: list[dict[str, Any]], responses: list[dict[str, Any]], context: dict[str, Any]) -> list[dict[str, Any]]:
    block_lookup = {block["block_id"]: (scene["scene_id"], block) for scene in context["script_scenes"] for block in scene["script_blocks"] if block["block_type"] == "dialogue"}
    subtitle_lookup = {row["subtitle_id"]: row for row in alignments}
    for response in responses:
        subtitle_id = response.get("subtitle_id")
        if subtitle_id not in subtitle_lookup:
            raise ValueError(f"LLM response references unknown subtitle_id: {subtitle_id}")
        block_ids = [] if response.get("decision") == "no_match" else response.get("block_ids", [])
        if not isinstance(block_ids, list) or any(block_id not in block_lookup for block_id in block_ids):
            raise ValueError(f"LLM response references unknown block_id: {block_ids}")
        scenes = {block_lookup[block_id][0] for block_id in block_ids}
        if len(scenes) > 1:
            raise ValueError("LLM response cannot select blocks from multiple scenes")
        row = subtitle_lookup[subtitle_id]
        original = {"scene_id": row["scene_id"], "script_matches": row["script_matches"], "alignment": dict(row["alignment"])}
        row["scene_id"] = next(iter(scenes)) if scenes else None
        row["script_matches"] = [{"block_id": block_id, "speaker": block_lookup[block_id][1]["speaker"], "matched_text": block_lookup[block_id][1]["text"], "lexical_score": None, "semantic_score": None, "combined_score": float(response.get("confidence", 1.0))} for block_id in block_ids]
        row["alignment"] = {"method": "llm_resolved", "status": "llm_aligned" if block_ids else "no_match", "candidate_margin": original["alignment"].get("candidate_margin", 0.0), "needs_review": False, "reliable_anchor": True, "script_order_start": original["alignment"].get("script_order_start"), "script_order_end": original["alignment"].get("script_order_end"), "llm_resolution": {"resolver": response.get("resolver"), "model": response.get("model"), "request_id": response.get("request_id"), "response_id": response.get("response_id"), "resolved_at": response.get("resolved_at"), "original_automatic": original}}
    return alignments
