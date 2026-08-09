from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from .alignment import _fuzzy, build_script_units, tokenize
from .schema import read_jsonl


class LLMResolver(Protocol):
    def resolve(self, request: dict[str, Any]) -> dict[str, Any]: ...


_GLOBAL_RESCUE_FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "course", "do", "for", "from",
    "had", "has", "have", "he", "her", "here", "him", "his", "how", "i", "if", "in", "is",
    "it", "its", "me", "my", "no", "not", "of", "on", "or", "our", "she", "so", "that", "the",
    "their", "them", "then", "there", "they", "this", "to", "up", "us", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "with", "yes", "you", "your",
}


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


def _global_rescue_evidence_v3(subtitle_text: str, unit: Any) -> tuple[float, str]:
    """Return conservative screenplay-wide evidence for fragments and short replies.

    Unlike the local retriever, this helper intentionally requires at least one
    content token.  Two-token subtitles are admitted only by normalized
    containment; longer subtitles can additionally use partial-string or
    unordered content-token coverage.  This keeps generic function-word replies
    from becoming screenplay-wide candidates while recovering legitimate
    expanded/contracted dialogue fragments.
    """
    subtitle_tokens = tokenize(subtitle_text).tokens
    block_tokens = unit.token_text.tokens
    content_tokens = tuple(token for token in subtitle_tokens if token not in _GLOBAL_RESCUE_FUNCTION_WORDS)
    if not subtitle_tokens or not block_tokens or not content_tokens:
        return 0.0, "none"
    score, method = _lexical_evidence(subtitle_text, unit)
    if method != "rapidfuzz" or score >= 0.80:
        return score, method
    if len(subtitle_tokens) < 3:
        return 0.0, "none"

    from rapidfuzz.fuzz import partial_ratio

    subtitle_text_normalized = " ".join(subtitle_tokens)
    block_text_normalized = " ".join(block_tokens)
    block_token_set = set(block_tokens)
    content_coverage = sum(token in block_token_set for token in content_tokens) / len(content_tokens)
    partial_score = partial_ratio(subtitle_text_normalized, block_text_normalized) / 100.0
    if partial_score >= 0.86 and content_coverage >= 2 / 3:
        return min(0.98, partial_score), "global_partial_fragment"

    if len(set(content_tokens)) >= 2 and content_coverage == 1.0:
        return 0.92, "global_content_token_coverage"
    return 0.0, "none"


def _global_rescue_evidence_v4(subtitle_text: str, unit: Any) -> tuple[float, str]:
    """Recover distinctive contracted fragments without admitting generic short replies."""
    score, method = _global_rescue_evidence_v3(subtitle_text, unit)
    block_content = {
        token for token in unit.token_text.tokens
        if token not in _GLOBAL_RESCUE_FUNCTION_WORDS and token != "s"
    }
    generic_script_substring = method == "script_substring" and (
        len(unit.token_text.tokens) < 2 or not block_content
    )
    if method != "none" and not generic_script_substring:
        return score, method
    subtitle_tokens = tokenize(subtitle_text).tokens
    block_tokens = unit.token_text.tokens
    content_tokens = tuple(
        token for token in subtitle_tokens
        if token not in _GLOBAL_RESCUE_FUNCTION_WORDS and token != "s"
    )
    if len(subtitle_tokens) < 4 or len(set(content_tokens)) < 4 or not block_tokens:
        return 0.0, "none"
    from rapidfuzz.fuzz import partial_ratio

    block_token_set = set(block_tokens)
    coverage = sum(token in block_token_set for token in content_tokens) / len(content_tokens)
    partial_score = partial_ratio(" ".join(subtitle_tokens), " ".join(block_tokens)) / 100.0
    if coverage >= 0.60 and partial_score >= 0.70:
        return min(0.94, max(0.86, partial_score)), "global_relaxed_distinctive_fragment"
    return 0.0, "none"


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
            if not retrieved and first["bounds"] is not None and (
                first["before"] is not None or first["after"] is not None
            ):
                low, high = first["bounds"]
                if first["before"] is not None:
                    for index in range(low, min(high, low + 2) + 1):
                        required_indices[index] = "anchor_boundary"
                if first["after"] is not None:
                    for index in range(max(low, high - 2), high + 1):
                        required_indices[index] = "anchor_boundary"
                retrieved = _retrieve_candidates(
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


def augment_review_requests_global_lexical(
    requests_path: Path, context_path: Path, output_path: Path, *, max_rescue_candidates: int = 12,
    retrieval_version: str = "global_lexical_rescue_v2", alignment_path: Path | None = None,
    context_radius: int = 2,
) -> dict[str, Any]:
    """Write a versioned request set with strong screenplay-wide lexical rescue candidates."""
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite global lexical rescue requests")
    if not 1 <= max_rescue_candidates <= 36:
        raise ValueError("max_rescue_candidates must be between 1 and 36")
    if retrieval_version not in {"global_lexical_rescue_v2", "global_lexical_rescue_v3", "global_lexical_rescue_v4"}:
        raise ValueError("unsupported global lexical rescue version")
    if retrieval_version == "global_lexical_rescue_v4" and alignment_path is None:
        raise ValueError("global lexical rescue v4 requires deterministic alignment for nearby subtitle retrieval")
    if not 1 <= context_radius <= 10:
        raise ValueError("context_radius must be between 1 and 10")
    context = json.loads(context_path.read_text(encoding="utf-8"))
    requests = [json.loads(line) for line in requests_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    units = build_script_units(context)
    alignments = read_jsonl(alignment_path) if alignment_path is not None else []
    alignment_positions = {row.get("subtitle_id"): index for index, row in enumerate(alignments)}
    if alignment_path is not None and (None in alignment_positions or len(alignment_positions) != len(alignments)):
        raise ValueError("alignment must contain unique non-empty subtitle IDs")
    rescued_requests = rescued_targets = added_total = context_added_total = 0
    output: list[dict[str, Any]] = []
    for source in requests:
        request = json.loads(json.dumps(source))
        existing = {row["block_id"]: row for row in request.get("dialogue_candidates", [])}
        rescue: dict[int, tuple[float, str, str]] = {}
        target_hits: set[str] = set()
        query_subtitles = [(subtitle, "target_subtitle") for subtitle in request.get("subtitles", [])]
        context_ids: set[str] = set()
        if retrieval_version == "global_lexical_rescue_v4":
            target_ids = list(request.get("subtitle_ids", []))
            if not target_ids or any(subtitle_id not in alignment_positions for subtitle_id in target_ids):
                raise ValueError(f"{request.get('request_id')} target subtitle is absent from alignment")
            positions = [alignment_positions[subtitle_id] for subtitle_id in target_ids]
            first, last = min(positions), max(positions)
            nearby = alignments[max(0, first - context_radius):first] + alignments[last + 1:last + 1 + context_radius]
            context_ids = {row["subtitle_id"] for row in nearby}
            query_subtitles.extend((row, "nearby_subtitle_context") for row in nearby)
        anchor_orders = [
            order
            for anchor_name in ("previous_anchor", "next_anchor")
            for order in ((request.get(anchor_name) or {}).get("screenplay_order") or [])
            if isinstance(order, int)
        ]
        for subtitle, query_source in query_subtitles:
            if query_source == "nearby_subtitle_context" and target_hits:
                continue
            tokens = tokenize(subtitle.get("text", "")).tokens
            content_tokens = {token for token in tokens if token not in _GLOBAL_RESCUE_FUNCTION_WORDS}
            if not content_tokens or (retrieval_version == "global_lexical_rescue_v2" and len(tokens) < 3):
                continue
            ranked: list[tuple[float, float, int, str]] = []
            for index, unit in enumerate(units):
                if unit.block["block_id"] in existing:
                    continue
                if retrieval_version == "global_lexical_rescue_v4":
                    score, method = _global_rescue_evidence_v4(subtitle["text"], unit)
                    accepted = method != "none"
                elif retrieval_version == "global_lexical_rescue_v3":
                    score, method = _global_rescue_evidence_v3(subtitle["text"], unit)
                    accepted = method != "none"
                else:
                    score, method = _lexical_evidence(subtitle["text"], unit)
                    accepted = method != "rapidfuzz" or score >= 0.80
                if accepted:
                    anchor_bonus = (
                        0.12 if retrieval_version == "global_lexical_rescue_v4"
                        and anchor_orders and min(abs(index - order) for order in anchor_orders) <= 8
                        else 0.0
                    )
                    ranked.append((score + anchor_bonus, score, index, method))
            ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
            if ranked and query_source == "target_subtitle":
                target_hits.add(subtitle["subtitle_id"])
            for _selection_score, score, index, method in ranked[:2]:
                previous = rescue.get(index)
                ranking_score = score + (0.1 if query_source == "target_subtitle" else 0.0)
                if previous is None or ranking_score > previous[0]:
                    rescue[index] = (ranking_score, method, query_source)
        ranked_rescue = sorted(rescue.items(), key=lambda item: (-item[1][0], item[0]))[:max_rescue_candidates]
        additions = []
        for index, (ranking_score, method, query_source) in ranked_rescue:
            unit = units[index]
            score = ranking_score - (0.1 if query_source == "target_subtitle" else 0.0)
            additions.append({
                "scene_id": unit.block["scene_id"], "block_id": unit.block["block_id"],
                "screenplay_order": unit.block_index, "speaker": unit.block.get("speaker"),
                "text": unit.block["text"], "parenthetical": unit.block.get("parenthetical"),
                "retrieval_methods": ["global_lexical_rescue", method, query_source],
                "lexical_score": round(score, 6), "semantic_score": None, "retrieval_score": round(score, 6),
            })
        limit = int(request.get("candidate_limit") or 36)
        if additions:
            protected = [row for row in existing.values() if "automatic_mapping" in row.get("retrieval_methods", [])]
            optional = [row for row in existing.values() if row not in protected]
            optional.sort(key=lambda row: (float(row.get("retrieval_score") or 0.0), -int(row["screenplay_order"])), reverse=True)
            keep_count = max(0, limit - len(protected) - len(additions))
            combined = protected + optional[:keep_count] + additions
            by_id = {row["block_id"]: row for row in combined}
            request["dialogue_candidates"] = sorted(by_id.values(), key=lambda row: int(row["screenplay_order"]))[:limit]
            request["candidate_scenes"] = list(dict.fromkeys(row["scene_id"] for row in request["dialogue_candidates"]))
            orders = [int(row["screenplay_order"]) for row in request["dialogue_candidates"]]
            if orders:
                request["estimated_screenplay_range"] = {
                    "start_screenplay_order": min(orders), "end_screenplay_order": max(orders),
                    "start_scene_id": request["dialogue_candidates"][0]["scene_id"],
                    "end_scene_id": request["dialogue_candidates"][-1]["scene_id"],
                }
            rescued_requests += 1; rescued_targets += len(target_hits); added_total += len(additions)
            context_added_total += sum("nearby_subtitle_context" in row["retrieval_methods"] for row in additions)
        if additions:
            request["retrieval_version"] = retrieval_version
            request["global_lexical_rescue_target_ids"] = sorted(target_hits)
            request["global_lexical_rescue_candidate_count"] = len(additions)
            if retrieval_version == "global_lexical_rescue_v4":
                request["global_lexical_rescue_context_subtitle_ids"] = sorted(context_ids)
        output.append(request)
    errors = validate_review_requests(output, context)
    if errors:
        raise ValueError("invalid augmented review requests: " + "; ".join(errors[:20]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in output:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    result = {"schema_version": "1.0", "request_count": len(output),
              "rescued_request_count": rescued_requests,
              "rescued_target_count": rescued_targets, "added_candidate_count": added_total,
              "nearby_context_added_candidate_count": context_added_total,
              "retrieval_version": retrieval_version, "max_rescue_candidates": max_rescue_candidates,
              "context_radius": context_radius if retrieval_version == "global_lexical_rescue_v4" else None,
              "source_requests": requests_path.resolve().as_posix(),
              "source_requests_sha256": hashlib.sha256(requests_path.read_bytes()).hexdigest(),
              "screenplay_context": context_path.resolve().as_posix(),
              "screenplay_context_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
              "source_alignment": alignment_path.resolve().as_posix() if alignment_path is not None else None,
              "source_alignment_sha256": hashlib.sha256(alignment_path.read_bytes()).hexdigest() if alignment_path is not None else None,
              "function_word_policy_sha256": hashlib.sha256(
                  json.dumps(sorted(_GLOBAL_RESCUE_FUNCTION_WORDS), separators=(",", ":")).encode("utf-8")
              ).hexdigest(),
              "output": output_path.resolve().as_posix(),
              "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
              "generated_at": datetime.now(UTC).isoformat()}
    fd, temporary = tempfile.mkstemp(prefix=manifest_path.name + ".", suffix=".tmp", dir=manifest_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, manifest_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return result


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
