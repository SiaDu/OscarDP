from __future__ import annotations

from typing import Any, Protocol

from .alignment import _fuzzy, build_script_units, tokenize


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
    units: list[Any], local_window: int, fallback_window: int, progress: float,
) -> tuple[list[Any], bool, str, tuple[int, int] | None]:
    if not units:
        return [], True, "empty_screenplay", None
    if before is None and after is None:
        center = round(max(0.0, min(1.0, progress)) * (len(units) - 1))
        half = max(1, fallback_window // 2)
        low, high = max(0, center - half), min(len(units) - 1, center + half)
        return [], True, "fallback_timeline_estimate", (low, high)
    if before is not None and after is not None:
        left = int(before["alignment"]["script_order_end"])
        right = int(after["alignment"]["script_order_start"])
        if left > right:
            center = round(max(0.0, min(1.0, progress)) * (len(units) - 1))
            half = max(1, fallback_window // 2)
            low, high = max(0, center - half), min(len(units) - 1, center + half)
            return [], True, "fallback_nonmonotonic_anchor_interval", (low, high)
        left_scene, right_scene = units[left].scene_index, units[right].scene_index
        if abs(right_scene - left_scene) > 1:
            return [], True, "fallback_between_reliable_anchors", (left, right)
        allowed_scenes = {left_scene} if left_scene == right_scene else {left_scene, right_scene}
        candidates = [unit for unit in units[left:right + 1] if unit.scene_index in allowed_scenes]
        return candidates, False, "between_reliable_anchors", (left, right)
    if before is not None:
        left = int(before["alignment"]["script_order_end"])
        high = min(len(units) - 1, left + max(local_window, fallback_window))
        return [], True, "fallback_after_reliable_anchor", (left, high)
    right = int(after["alignment"]["script_order_start"])
    low = max(0, right - max(local_window, fallback_window))
    return [], True, "fallback_before_reliable_anchor", (low, right)


class _SemanticRetriever:
    def __init__(self, model_name: str, units: list[Any]):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError('semantic candidate retrieval requires pip install -e ".[semantic]"') from exc
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [unit.block["text"] for unit in units], convert_to_numpy=True, normalize_embeddings=True,
        )

    def scores(self, text: str, low: int, high: int) -> dict[int, float]:
        query = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return {index: max(0.0, float(self.embeddings[index] @ query)) for index in range(low, high + 1)}


def _contains(needle: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    return bool(needle) and any(haystack[index:index + len(needle)] == needle for index in range(len(haystack) - len(needle) + 1))


def _lexical_evidence(subtitle_text: str, unit: Any) -> tuple[float, str]:
    subtitle_tokens = tokenize(subtitle_text).tokens
    block_tokens = unit.token_text.tokens
    if not subtitle_tokens or not block_tokens:
        return 0.0, "none"
    if subtitle_tokens == block_tokens:
        return 1.0, "normalized_exact"
    if _contains(subtitle_tokens, block_tokens):
        return 0.99, "normalized_substring"
    if _contains(block_tokens, subtitle_tokens):
        coverage = len(block_tokens) / max(1, len(subtitle_tokens))
        return min(0.95, 0.88 + 0.07 * coverage), "script_substring"
    return _fuzzy(subtitle_tokens, block_tokens), "rapidfuzz"


def _retrieve_candidates(
    rows: list[dict[str, Any]], units: list[Any], bounds: tuple[int, int],
    candidate_limit: int, semantic: _SemanticRetriever | None,
    *, minimum_score: float = 0.22, top_per_subtitle: int = 8,
    required_indices: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    low, high = bounds
    evidence: dict[int, dict[str, Any]] = {}
    for index, method in (required_indices or {}).items():
        if low <= index <= high:
            evidence[index] = {
                "lexical_score": 0.0, "semantic_score": None,
                "retrieval_score": 0.0, "methods": {method}, "required": True,
            }
    preferred_cursor = low
    for row in rows:
        semantic_scores = {} if semantic is None else semantic.scores(row["text"], low, high)
        ranked: list[tuple[float, float, int, str, float | None]] = []
        for index in range(low, high + 1):
            lexical_score, method = _lexical_evidence(row["text"], units[index])
            semantic_score = semantic_scores.get(index)
            retrieval_score = max(lexical_score, semantic_score or 0.0)
            backward_penalty = 0.25 if index < preferred_cursor else 0.0
            ranked.append((retrieval_score - backward_penalty, retrieval_score, index, method, semantic_score))
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
        usable = [item for item in ranked if item[1] >= minimum_score][:top_per_subtitle]
        if usable:
            preferred_cursor = max(preferred_cursor, usable[0][2])
        for _, retrieval_score, index, method, semantic_score in usable:
            item = evidence.setdefault(index, {"lexical_score": 0.0, "semantic_score": None, "retrieval_score": 0.0, "methods": set(), "required": False})
            lexical_score, _ = _lexical_evidence(row["text"], units[index])
            item["lexical_score"] = max(item["lexical_score"], lexical_score)
            if semantic_score is not None:
                item["semantic_score"] = max(item["semantic_score"] or 0.0, semantic_score)
                item["methods"].add("semantic")
            item["retrieval_score"] = max(item["retrieval_score"], retrieval_score)
            item["methods"].add(method)

    for index in list(evidence):
        for adjacent in (index - 1, index + 1):
            if low <= adjacent <= high and adjacent not in evidence:
                evidence[adjacent] = {"lexical_score": 0.0, "semantic_score": None, "retrieval_score": 0.0, "methods": {"adjacent_dialogue"}, "required": False}
    required = sorted(index for index, item in evidence.items() if item.get("required"))
    ranked_optional = sorted(
        (index for index in evidence if index not in required),
        key=lambda index: (-float(evidence[index]["retrieval_score"]), index),
    )
    selected = required[:candidate_limit] + ranked_optional[:max(0, candidate_limit - len(required))]
    return [{
        "unit": units[index],
        "lexical_score": round(float(evidence[index]["lexical_score"]), 6),
        "semantic_score": None if evidence[index]["semantic_score"] is None else round(float(evidence[index]["semantic_score"]), 6),
        "retrieval_score": round(float(evidence[index]["retrieval_score"]), 6),
        "retrieval_methods": sorted(evidence[index]["methods"]),
    } for index in sorted(selected)]


def _same_candidate_interval(left: dict[str, Any], right: dict[str, Any], local_window: int) -> bool:
    if not left["fallback"] or not right["fallback"]:
        return left["candidate_key"] == right["candidate_key"]
    if left["reason"] != right["reason"] or left["bounds"] is None or right["bounds"] is None:
        return False
    left_center = sum(left["bounds"]) / 2
    right_center = sum(right["bounds"]) / 2
    overlap = min(left["bounds"][1], right["bounds"][1]) - max(left["bounds"][0], right["bounds"][0])
    return overlap >= 0 and abs(left_center - right_center) <= local_window


def build_review_requests(
    context: dict[str, Any], alignments: list[dict[str, Any]], *,
    local_window: int = 40, fallback_window: int = 240, candidate_limit: int = 36,
    max_group_size: int = 8, max_time_gap_sec: float = 5.0, semantic_model: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    if local_window < 1 or fallback_window < local_window or not 1 <= candidate_limit <= 100:
        raise ValueError("Require fallback_window >= local_window >= 1 and 1 <= candidate_limit <= 100")
    units = build_script_units(context)
    semantic = None if semantic_model is None else _SemanticRetriever(semantic_model, units)
    anchors = [index for index, row in enumerate(alignments) if row["alignment"].get("reliable_anchor")]
    unresolved = [index for index, row in enumerate(alignments) if row["alignment"]["needs_review"]]
    prepared: list[dict[str, Any]] = []
    for index in unresolved:
        before_index = next((anchor for anchor in reversed(anchors) if anchor < index), None)
        after_index = next((anchor for anchor in anchors if anchor > index), None)
        before = alignments[before_index] if before_index is not None else None
        after = alignments[after_index] if after_index is not None else None
        start_sec = float(alignments[0]["time"]["start_sec"]) if alignments else 0.0
        end_sec = float(alignments[-1]["time"]["end_sec"]) if alignments else 1.0
        progress = (float(alignments[index]["time"]["start_sec"]) - start_sec) / max(1e-6, end_sec - start_sec)
        candidates, fallback, reason, bounds = _candidate_interval(before, after, units, local_window, fallback_window, progress)
        prepared.append({
            "index": index, "before": before, "after": after, "candidates": candidates,
            "candidate_key": (reason, bounds) if fallback else tuple(unit.block["block_id"] for unit in candidates),
            "fallback": fallback, "reason": reason, "bounds": bounds,
        })

    groups: list[list[dict[str, Any]]] = []
    for item in prepared:
        if groups:
            previous = groups[-1][-1]
            previous_row, row = alignments[previous["index"]], alignments[item["index"]]
            adjacent = item["index"] == previous["index"] + 1
            temporal = float(row["time"]["start_sec"]) - float(previous_row["time"]["end_sec"]) <= max_time_gap_sec
            same_interval = _same_candidate_interval(previous, item, local_window)
            if adjacent and temporal and same_interval and len(groups[-1]) < max_group_size:
                groups[-1].append(item)
                continue
        groups.append([item])

    alignment_requests: list[dict[str, Any]] = []
    for ordinal, group in enumerate(groups, 1):
        first = group[0]
        rows = [alignments[item["index"]] for item in group]
        if first["fallback"]:
            required_indices: dict[int, str] = {}
            if (
                first["bounds"] is not None
                and (first["before"] is not None or first["after"] is not None)
                and first["bounds"][1] - first["bounds"][0] <= 2
            ):
                required_indices[first["bounds"][0]] = "anchor_boundary"
                required_indices[first["bounds"][1]] = "anchor_boundary"
            if first["before"] is not None or first["after"] is not None:
                for row in rows:
                    for match in row.get("script_matches", []):
                        order = match.get("screenplay_order")
                        if isinstance(order, int):
                            required_indices[order] = "automatic_mapping"
            retrieved = [] if first["bounds"] is None else _retrieve_candidates(
                rows, units, first["bounds"], candidate_limit, semantic,
                required_indices=required_indices,
            )
        else:
            local_units = first["candidates"]
            if len(local_units) <= candidate_limit:
                retrieved = [{"unit": unit, "lexical_score": None, "semantic_score": None, "retrieval_score": None, "retrieval_methods": ["anchor_window"]} for unit in local_units]
            else:
                retrieved = _retrieve_candidates(rows, units, first["bounds"], candidate_limit, semantic)
        insufficient = not retrieved
        candidate_units = [item["unit"] for item in retrieved]
        bounds = first["bounds"]
        estimated_range = None if bounds is None else {
            "start_screenplay_order": bounds[0], "end_screenplay_order": bounds[1],
            "start_scene_id": units[bounds[0]].block["scene_id"], "end_scene_id": units[bounds[1]].block["scene_id"],
        }
        alignment_requests.append({
            "request_id": f"alignment_review_{ordinal:06d}",
            "task": "select_existing_blocks_or_no_match",
            "subtitle_ids": [row["subtitle_id"] for row in rows],
            "subtitles": [{"subtitle_id": row["subtitle_id"], "text": row["text"], "time": row["time"]} for row in rows],
            "previous_anchor": _anchor_summary(first["before"]), "next_anchor": _anchor_summary(first["after"]),
            "candidate_scenes": list(dict.fromkeys(unit.block["scene_id"] for unit in candidate_units)),
            "dialogue_candidates": [{
                "scene_id": item["unit"].block["scene_id"], "block_id": item["unit"].block["block_id"],
                "screenplay_order": item["unit"].block_index, "speaker": item["unit"].block["speaker"], "text": item["unit"].block["text"],
                "parenthetical": item["unit"].block.get("parenthetical"),
                "retrieval_methods": item["retrieval_methods"], "lexical_score": item["lexical_score"],
                "semantic_score": item["semantic_score"], "retrieval_score": item["retrieval_score"],
            } for item in retrieved],
            "automatic_candidate_mappings": [{
                "subtitle_id": row["subtitle_id"], "scene_id": row["scene_id"],
                "matches": row["script_matches"], "alignment": row["alignment"],
            } for row in rows],
            "reason_for_review": sorted(set(row["alignment"].get("review_reason", "low_confidence") for row in rows)),
            "candidate_interval_reason": first["reason"], "fallback_used": first["fallback"],
            "retrieval_methods": sorted({method for item in retrieved for method in item["retrieval_methods"]}),
            "estimated_screenplay_range": estimated_range, "candidate_limit": candidate_limit,
            "insufficient_candidates": insufficient,
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
        estimated = request.get("estimated_screenplay_range")
        candidate_limit = request.get("candidate_limit")
        for candidate in request.get("dialogue_candidates", []):
            block_id = candidate.get("block_id")
            if block_id not in known:
                errors.append(f"{request.get('request_id')} references invalid candidate {block_id}")
                continue
            unit = known[block_id]
            orders.append(unit.block_index)
            if unit.block.get("block_type") != "dialogue":
                errors.append(f"{request.get('request_id')} candidate is not dialogue")
            if unit.block["scene_id"] not in scenes:
                errors.append(f"{request.get('request_id')} candidate scene mismatch")
            if estimated and not int(estimated["start_screenplay_order"]) <= unit.block_index <= int(estimated["end_screenplay_order"]):
                errors.append(f"{request.get('request_id')} candidate outside estimated screenplay range")
            for score_name in ("lexical_score", "semantic_score", "retrieval_score"):
                score = candidate.get(score_name)
                if score is not None and (not isinstance(score, (int, float)) or not 0 <= score <= 1):
                    errors.append(f"{request.get('request_id')} invalid {score_name}")
        if orders != sorted(orders) or len(orders) != len(set(orders)):
            errors.append(f"{request.get('request_id')} candidates are not ordered and unique")
        if isinstance(candidate_limit, int) and len(orders) > candidate_limit:
            errors.append(f"{request.get('request_id')} exceeds candidate limit")
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
    parsing_audit = context.get("parsing_audit", {})
    return {
        "schema_version": "1.1", "total_subtitles": len(alignments),
        "auto_aligned": statuses.get("auto_aligned", 0), "needs_review": statuses.get("needs_review", 0),
        "no_match": statuses.get("no_match", 0), "method_counts": dict(sorted(methods.items())),
        "exact_matches": sum(count for method, count in methods.items() if "exact" in method),
        "substring_fragment_matches": sum(count for method, count in methods.items() if "substring" in method),
        "fuzzy_matches": sum(count for method, count in methods.items() if "rapidfuzz" in method),
        "n_subtitles_to_one_block_count": n_to_one, "one_subtitle_to_m_blocks_count": one_to_m,
        "n_to_m_count": n_to_m, "screenplay_deletion_gaps": deletion_gaps,
        "improvised_no_match_candidates": statuses.get("no_match", 0),
        "insufficient_candidate_groups": sum(bool(request["insufficient_candidates"]) for request in requests),
        "zero_candidate_groups": sum(not request.get("dialogue_candidates") for request in requests),
        "fallback_retrieval_groups": sum(bool(request.get("fallback_used")) for request in requests),
        "candidate_count_distribution": dict(sorted({
            str(count): sum(len(request.get("dialogue_candidates", [])) == count for request in requests)
            for count in {len(request.get("dialogue_candidates", [])) for request in requests}
        }.items(), key=lambda item: int(item[0]))),
        "review_request_group_count": len(requests),
        "unresolved_local_substring_false_no_match": substring_false_no_match,
        "monotonicity_violations": violations, "scene_candidate_window_violations": len(request_errors),
        "candidate_window_errors": request_errors,
        "action_like_dialogue_count": int(parsing_audit.get("action_like_dialogue_count", 0)),
        "editorial_label_speaker_count": int(parsing_audit.get("editorial_label_speaker_count", 0)),
        "fragmented_parenthetical_count": int(parsing_audit.get("fragmented_parenthetical_count", 0)),
        "screenplay_parsing_audit": parsing_audit,
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
