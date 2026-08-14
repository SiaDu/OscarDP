from __future__ import annotations

import hashlib
import json
import re
import tempfile
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .alignment import tokenize
from .openai_review import apply_validated_responses
from .pipeline import _write_json, _write_jsonl
from .schema import read_jsonl
from .stage253 import validate_resolution_v3
from .validation import validate_files


_LEXICAL_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "do", "for", "from", "got",
    "had", "has", "have", "he", "her", "here", "him", "his", "i", "in", "is", "it", "just", "me",
    "my", "of", "on", "or", "our", "out", "she", "so", "that", "the", "their", "them", "there",
    "they", "this", "to", "up", "us", "was", "we", "were", "what", "when", "with", "you", "your",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_fingerprint(path: Path) -> dict[str, int]:
    """Small immutable-source guard used when a prior full hash is recorded."""
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _verify_protected_file(path: Path, expected_sha256: str, *, stat_fingerprint: dict[str, Any] | None = None) -> dict[str, Any]:
    """Verify normally by SHA-256, or by a recorded stat guard for large media.

    Stat-only verification is deliberately opt-in: callers must supply both the
    historical SHA-256 and the exact size/mtime fingerprint captured alongside it.
    """
    if stat_fingerprint is not None:
        expected_stat = {
            "size": stat_fingerprint.get("size"),
            "mtime_ns": stat_fingerprint.get("mtime_ns"),
        }
        actual_stat = _stat_fingerprint(path) if path.is_file() else None
        unchanged = actual_stat == expected_stat
        return {
            "path": path.as_posix(), "expected_sha256": expected_sha256,
            "actual_sha256": None, "expected_stat_fingerprint": expected_stat,
            "actual_stat_fingerprint": actual_stat, "unchanged": unchanged,
            "verification_method": "recorded_sha256_plus_stat_fingerprint",
        }
    actual = _sha(path) if path.is_file() else None
    return {
        "path": path.as_posix(), "expected_sha256": expected_sha256,
        "actual_sha256": actual, "unchanged": actual == expected_sha256,
        "verification_method": "sha256",
    }


def _v3_validation_view(response: dict[str, Any]) -> dict[str, Any]:
    """Recover raw candidate-task-v3 resolutions from apply-normalized input."""
    view = deepcopy(response)
    for resolution in view.get("resolutions", []):
        raw = resolution.get("openai_resolution")
        if isinstance(raw, dict) and raw.get("decision") in {"match", "no_candidate_match"}:
            resolution.update(raw)
        elif resolution.get("decision") == "no_match":
            # Backward-compatible fallback for normalized records predating
            # raw-resolution provenance.
            resolution["decision"] = "no_candidate_match"
    return view


def _compact_alignment(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "subtitle_id": row["subtitle_id"], "text": row["text"], "time": row["time"],
        "scene_id": row.get("scene_id"), "status": row["alignment"]["status"],
        "block_ids": [match["block_id"] for match in row.get("script_matches", [])],
        "screenplay_span": [row["alignment"].get("script_order_start"), row["alignment"].get("script_order_end")],
    }


def _multi_speaker(text: str) -> bool:
    return len(re.findall(r"(?:^|\n|\s)-\s+", text)) >= 2 or len([line for line in text.splitlines() if line.lstrip().startswith("-")]) >= 2


def _insert_like(text: str) -> bool:
    stripped = text.strip()
    return bool(re.match(r"^(?:\[.*\]|(?:title|card|super|sign|chapter|telegram|letter|text)\s*:)", stripped, re.I))


def _target_lexical_evidence(text: str, candidates: list[dict[str, Any]]) -> tuple[float, int, bool]:
    target_tokens = tokenize(text).tokens
    target_content = {token for token in target_tokens if token not in _LEXICAL_STOPWORDS}
    if not target_tokens:
        return 0.0, 0, False
    normalized_target = " ".join(target_tokens)
    best_score, best_count, exact_phrase = 0.0, 0, False
    for row in candidates:
        candidate_tokens = tokenize(row.get("text", "")).tokens
        normalized_candidate = " ".join(candidate_tokens)
        phrase = len(target_tokens) >= 2 and normalized_target in normalized_candidate
        candidate_content = {token for token in candidate_tokens if token not in _LEXICAL_STOPWORDS}
        count = len(target_content & candidate_content)
        score = count / len(target_content) if target_content else 0.0
        best_score, best_count, exact_phrase = max(best_score, score), max(best_count, count), exact_phrase or phrase
    return best_score, best_count, exact_phrase


def _target_token_counts(text: str) -> tuple[int, int]:
    tokens = tokenize(text).tokens
    return len(tokens), len({token for token in tokens if token not in _LEXICAL_STOPWORDS})


def _anchor_order_bounds(request: dict[str, Any]) -> tuple[int | None, int | None]:
    previous = request.get("previous_anchor") or {}
    following = request.get("next_anchor") or {}
    previous_orders = previous.get("screenplay_order") or []
    following_orders = following.get("screenplay_order") or []
    lower = previous_orders[-1] if previous_orders and isinstance(previous_orders[-1], int) else None
    upper = following_orders[0] if following_orders and isinstance(following_orders[0], int) else None
    return lower, upper


def _within_anchor_envelope(order: Any, lower: int | None, upper: int | None, *, margin: int = 4) -> bool:
    if not isinstance(order, int):
        return False
    if lower is None and upper is None:
        return False
    if lower is None:
        return abs(order - upper) <= margin
    if upper is None:
        return abs(order - lower) <= margin
    return order >= lower - margin and order <= upper + margin


def build_production_high_risk_audit_v3(
    requests_path: Path, responses_path: Path, alignment_path: Path, shot_context_path: Path,
    context_path: Path, output_path: Path, summary_path: Path, *, low_confidence_threshold: float = 0.8,
    hard_validation_contract_version: str = "candidate_task_v3_structure_v2",
) -> dict[str, Any]:
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to overwrite production high-risk audit artifacts")
    requests, responses = read_jsonl(requests_path), read_jsonl(responses_path)
    alignments, shots = read_jsonl(alignment_path), read_jsonl(shot_context_path)
    context = json.loads(context_path.read_text(encoding="utf-8"))
    response_by_id = {row["request_id"]: row for row in responses}
    if len(response_by_id) != len(responses):
        raise ValueError("production responses contain duplicate request IDs")
    alignment_index = {row["subtitle_id"]: index for index, row in enumerate(alignments)}
    if len(alignment_index) != len(alignments):
        raise ValueError("reviewed alignment contains duplicate subtitle IDs")
    shot_refs: dict[str, list[dict[str, Any]]] = {}
    for shot in shots:
        for subtitle in shot.get("subtitles", []):
            shot_refs.setdefault(subtitle["subtitle_id"], []).append({
                "shot_id": shot["shot_id"], "keyframe_path": shot.get("keyframe", {}).get("path"),
            })
    dialogue: list[dict[str, Any]] = []
    scene_warning: dict[str, bool] = {}
    for scene in context["script_scenes"]:
        parsing = scene.get("parsing", {})
        scene_warning[scene["scene_id"]] = bool(parsing.get("needs_review") or parsing.get("status") not in {None, "parsed"})
        for block in scene["script_blocks"]:
            if block["block_type"] == "dialogue":
                dialogue.append({
                    "screenplay_order": len(dialogue), "scene_id": scene["scene_id"],
                    "block_id": block["block_id"], "speaker": block.get("speaker"), "text": block["text"],
                })
    audit: list[dict[str, Any]] = []
    classification_counts: Counter[str] = Counter()
    for request in requests:
        response = response_by_id.get(request["request_id"])
        if response is None:
            raise ValueError(f"missing response for high-risk audit: {request['request_id']}")
        errors, _diagnostics = validate_resolution_v3(
            response, request, hard_validation_contract_version,
        )
        if errors:
            raise ValueError(f"invalid v3 response for high-risk audit {request['request_id']}: {errors}")
        subtitle_by_id = {row["subtitle_id"]: row for row in request["subtitles"]}
        automatic_by_id = {row["subtitle_id"]: row for row in request.get("automatic_candidate_mappings", [])}
        candidates = request.get("dialogue_candidates", [])
        candidate_by_id = {row["block_id"]: row for row in candidates}
        lower_anchor_order, upper_anchor_order = _anchor_order_bounds(request)
        validation_events = response.get("validation_diagnostics", {}).get("sequence_quality", {}).get("events", [])
        candidate_limit = request.get("candidate_limit")
        saturated = isinstance(candidate_limit, int) and candidate_limit > 0 and len(candidates) >= candidate_limit
        parser_warning = any(scene_warning.get(scene_id, False) for scene_id in request.get("candidate_scenes", []))
        for resolution in response["resolutions"]:
            subtitle_id = resolution["subtitle_id"]
            subtitle = subtitle_by_id[subtitle_id]
            text = subtitle["text"]
            reasons: list[str] = []
            if float(resolution["confidence"]) < low_confidence_threshold: reasons.append("low_confidence")
            if resolution["decision"] == "match" and len(resolution["block_ids"]) > 1: reasons.append("multi_block_selection")
            if _multi_speaker(text): reasons.append("multi_speaker_subtitle")
            if saturated: reasons.append("candidate_limit_saturated")
            if request.get("fallback_used"): reasons.append("fallback_retrieval")
            if _insert_like(text): reasons.append("graphic_or_insert_like_text")
            token_count = len(tokenize(text).tokens)
            if token_count <= 3: reasons.append("very_short_subtitle")
            if resolution.get("decision_basis") == "vocative_attachment": reasons.append("vocative_fragment")
            if resolution.get("decision_basis") == "repeated_or_reordered_dialogue": reasons.append("repeated_or_reordered_mapping")
            if parser_warning: reasons.append("parser_structural_warning")
            events = [
                event for event in validation_events
                if subtitle_id in {event.get("subtitle_id"), event.get("previous_subtitle_id")}
            ]
            for event in events:
                reason = event.get("reason")
                if reason in {"large_backward_mapping", "bounded_backward_mapping"}: reasons.append("backward_sequence_jump")
                if event.get("severity") == "high_risk" and reason == "large_backward_mapping": reasons.append("large_sequence_jump")
                if reason == "cross_scene_sequence_jump": reasons.append("cross_scene_sequence_event")
                if reason == "repeated_block_mapping": reasons.append("repeated_or_reordered_mapping")
            no_match_classification = None
            no_match_evidence: list[str] = []
            outside_matches: list[dict[str, Any]] = []
            recall_matches: list[dict[str, Any]] = []
            displayed_outside_matches: list[dict[str, Any]] = []
            if resolution["decision"] == "no_candidate_match":
                automatic_matches = automatic_by_id.get(subtitle_id, {}).get("matches", [])
                local_candidates = [
                    row for row in candidates
                    if _within_anchor_envelope(
                        row.get("screenplay_order"), lower_anchor_order, upper_anchor_order,
                    )
                ]
                lexical, lexical_overlap_count, lexical_phrase = _target_lexical_evidence(text, local_candidates)
                target_token_count, target_content_count = _target_token_counts(text)
                semantic = max((float(row.get("semantic_score") or 0.0) for row in automatic_matches), default=0.0)
                strong_supplied_candidate = (
                    (lexical_phrase and target_token_count >= 4)
                    or (lexical_overlap_count >= 2 and lexical >= 2 / 3)
                )
                moderate_lexical = lexical_overlap_count >= 2 and lexical >= 0.50
                supplied_ids = set(candidate_by_id)
                for block in dialogue:
                    if block["block_id"] in supplied_ids:
                        continue
                    score, count, phrase = _target_lexical_evidence(text, [block])
                    if phrase or (count >= 2 and score >= 0.75):
                        outside_matches.append({**block, "target_content_overlap": score, "exact_phrase": phrase})
                outside_matches.sort(key=lambda row: (not row["exact_phrase"], -row["target_content_overlap"], row["screenplay_order"]))
                nearby_outside_matches = [
                    row for row in outside_matches
                    if _within_anchor_envelope(
                        row.get("screenplay_order"), lower_anchor_order, upper_anchor_order,
                    )
                    and (
                        (row["exact_phrase"] and target_token_count >= 3)
                        or target_content_count >= 2
                    )
                ]
                distinctive_distant_matches = [
                    row for row in outside_matches
                    if row not in nearby_outside_matches and (
                        (row["exact_phrase"] and target_token_count >= 4)
                        or (row["target_content_overlap"] >= 0.75 and target_content_count >= 4)
                    )
                ]
                recall_matches = nearby_outside_matches + distinctive_distant_matches
                displayed_outside_matches = outside_matches[:5]
                for row in recall_matches:
                    if row not in displayed_outside_matches:
                        displayed_outside_matches.append(row)
                if strong_supplied_candidate:
                    reasons.append("strong_lexical_overlap_no_candidate_match")
                    no_match_evidence.append("strong_supplied_candidate_in_anchor_envelope")
                if semantic >= 0.75:
                    reasons.append("high_semantic_score_no_candidate_match")
                if automatic_matches:
                    no_match_evidence.append("automatic_mapping_present")
                if displayed_outside_matches:
                    reasons.append("strong_screenplay_text_outside_candidate_window")
                if recall_matches:
                    no_match_evidence.append("strong_screenplay_text_outside_candidate_window")
                if strong_supplied_candidate:
                    no_match_classification = "reviewer_selection_risk"
                    reasons.append("reviewer_selection_risk")
                elif recall_matches:
                    no_match_classification = "candidate_recall_risk"
                    reasons.append("candidate_recall_risk")
                elif (moderate_lexical or semantic >= 0.50) and (
                    saturated or request.get("fallback_used") or _multi_speaker(text) or parser_warning
                ):
                    no_match_classification = "ambiguous_needs_review"
                    reasons.append("ambiguous_no_candidate_match")
                else:
                    no_match_classification = "probable_true_no_match"
                classification_counts[no_match_classification] += 1
            reasons = list(dict.fromkeys(reasons))
            if not reasons:
                continue
            index = alignment_index[subtitle_id]
            nearby = {
                "before": [_compact_alignment(row) for row in alignments[max(0, index - 2):index]],
                "after": [_compact_alignment(row) for row in alignments[index + 1:index + 3]],
            }
            sequence_context = {
                "previous_mapped": next((_compact_alignment(row) for row in reversed(alignments[:index]) if row.get("script_matches")), None),
                "next_mapped": next((_compact_alignment(row) for row in alignments[index + 1:] if row.get("script_matches")), None),
                "response_validation_events": events,
            }
            orders = [int(row["screenplay_order"]) for row in candidates if isinstance(row.get("screenplay_order"), int)]
            local: list[dict[str, Any]] = []
            if orders:
                start, end = max(0, min(orders) - 2), min(len(dialogue), max(orders) + 3)
                local = dialogue[start:end]
            audit.append({
                "schema_version": "1.1", "request_id": request["request_id"], "subtitle_id": subtitle_id,
                "subtitle_text": text, "subtitle_time": subtitle["time"], "nearby_subtitle_context": nearby,
                "previous_anchor": request.get("previous_anchor"), "next_anchor": request.get("next_anchor"),
                "dialogue_candidates": candidates,
                "selected_blocks": [candidate_by_id[block_id] for block_id in resolution["block_ids"]],
                "screenplay_local_context": local, "sequence_context": sequence_context,
                "automatic_mapping": automatic_by_id.get(subtitle_id), "openai_resolution": resolution,
                "matched_shots": shot_refs.get(subtitle_id, []), "inclusion_reasons": reasons,
                "diagnostics": {
                    "fallback_used": bool(request.get("fallback_used")), "candidate_limit": candidate_limit,
                    "candidate_count": len(candidates), "candidate_limit_saturated": saturated,
                    "no_candidate_match_classification": no_match_classification,
                    "no_candidate_match_evidence": no_match_evidence, "parser_structural_warning": parser_warning,
                    "strong_screenplay_matches_outside_candidates": displayed_outside_matches,
                    "recall_relevant_screenplay_matches_outside_candidates": recall_matches,
                    "anchor_order_envelope": {
                        "previous": lower_anchor_order, "next": upper_anchor_order, "margin": 4,
                    },
                },
                "human_decision": None, "human_block_ids": None, "reviewer_notes": None, "review_status": "pending",
            })
    audit.sort(key=lambda row: alignment_index[row["subtitle_id"]])
    _write_jsonl(output_path, audit)
    reason_counts = Counter(reason for row in audit for reason in row["inclusion_reasons"])
    summary = {
        "schema_version": "1.1", "record_count": len(audit), "pending_adjudication_count": len(audit),
        "hard_validation_contract_version": hard_validation_contract_version,
        "request_count": len(requests), "resolution_count": sum(len(row["resolutions"]) for row in responses),
        "inclusion_reason_counts": dict(sorted(reason_counts.items())),
        "no_candidate_match_classification_counts": dict(sorted(classification_counts.items())),
        "required_fields_present": True,
        "source_hashes": {
            "requests": _sha(requests_path), "responses": _sha(responses_path), "reviewed_alignment": _sha(alignment_path),
            "reviewed_shot_context": _sha(shot_context_path), "screenplay_context": _sha(context_path),
        },
        "audit_sha256": _sha(output_path), "generated_at": datetime.now(UTC).isoformat(),
    }
    _write_json(summary_path, summary)
    return summary


def apply_production_evidence_corrections_v3(
    correction_plan_path: Path, diagnosis_path: Path, risk_audit_path: Path,
    requests_path: Path, normalized_responses_path: Path, deterministic_alignment_path: Path,
    context_path: Path, shots_path: Path, output_dir: Path, output_tag: str,
    adjudicated_audit_path: Path, adjudicated_summary_path: Path,
) -> dict[str, Any]:
    """Apply source-evidence corrections without representing them as human gold."""
    if not re.fullmatch(r"[a-z0-9_]+", output_tag):
        raise ValueError("output_tag must contain only lowercase letters, digits, and underscores")
    openai_dir = output_dir / "review" / "openai"
    targets = [
        output_dir / f"subtitle_script_alignment.llm_reviewed_{output_tag}.jsonl",
        output_dir / f"shot_script_context.llm_reviewed_{output_tag}.jsonl",
        openai_dir / f"{output_tag}_apply_report.json",
        openai_dir / f"{output_tag}_reviewed_alignment_diagnostics.json",
        openai_dir / f"{output_tag}_evidence_correction_report.json",
        adjudicated_audit_path, adjudicated_summary_path,
    ]
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to overwrite evidence-corrected production artifacts")
    plan = read_jsonl(correction_plan_path)
    diagnoses = read_jsonl(diagnosis_path)
    audit = read_jsonl(risk_audit_path)
    requests = read_jsonl(requests_path)
    responses = read_jsonl(normalized_responses_path)
    request_by_id = {row["request_id"]: row for row in requests}
    response_by_id = {row["request_id"]: deepcopy(row) for row in responses}
    # The production v3 contract uses ``no_candidate_match`` while the
    # historical apply layer consumes ``no_match``.  Normalize every
    # resolution, not only the evidence-corrected subset, and retain the raw
    # v3 resolution for provenance just as production_review does.
    from .production_review import _BASIS_TO_HISTORICAL

    for response in response_by_id.values():
        for resolution in response.get("resolutions", []):
            if resolution.get("decision") not in {"match", "no_candidate_match"}:
                continue
            original = deepcopy(resolution)
            resolution["openai_resolution"] = original
            resolution["decision"] = "no_match" if original["decision"] == "no_candidate_match" else "match"
            basis = original.get("decision_basis")
            if basis in _BASIS_TO_HISTORICAL:
                resolution["decision_basis"] = _BASIS_TO_HISTORICAL[basis]
    diagnosis_by_subtitle = {row["subtitle_id"]: row for row in diagnoses}
    if len(diagnosis_by_subtitle) != len(diagnoses):
        raise ValueError("evidence diagnosis contains duplicate subtitle IDs")
    resolution_lookup = {
        (response["request_id"], resolution["subtitle_id"]): resolution
        for response in response_by_id.values() for resolution in response["resolutions"]
    }
    block_ids = {
        block["block_id"]
        for scene in json.loads(context_path.read_text(encoding="utf-8"))["script_scenes"]
        for block in scene["script_blocks"] if block["block_type"] == "dialogue"
    }
    deterministic_rows = read_jsonl(deterministic_alignment_path)
    if not deterministic_rows or not isinstance(deterministic_rows[0].get("movie_id"), str):
        raise ValueError("deterministic alignment lacks movie_id")
    movie_id = deterministic_rows[0]["movie_id"]
    corrected_ids: set[str] = set()
    for correction in plan:
        request_id, subtitle_id = correction.get("request_id"), correction.get("subtitle_id")
        key = (request_id, subtitle_id)
        if request_id not in request_by_id or key not in resolution_lookup:
            raise ValueError(f"evidence correction references unknown request/subtitle: {request_id}/{subtitle_id}")
        if subtitle_id in corrected_ids:
            raise ValueError(f"duplicate evidence correction for {subtitle_id}")
        corrected_ids.add(subtitle_id)
        decision, selected = correction.get("decision"), correction.get("block_ids")
        if decision not in {"match", "no_match", "uncertain"} or not isinstance(selected, list):
            raise ValueError(f"invalid evidence correction decision for {subtitle_id}")
        if decision == "match" and (not selected or any(block_id not in block_ids for block_id in selected)):
            raise ValueError(f"evidence correction has invalid screenplay blocks for {subtitle_id}")
        if decision != "match" and selected:
            raise ValueError(f"non-match evidence correction has block IDs for {subtitle_id}")
        diagnosis_id = correction.get("diagnosis_subtitle_id", subtitle_id)
        diagnosis = diagnosis_by_subtitle.get(diagnosis_id)
        if diagnosis is None:
            raise ValueError(f"evidence correction lacks diagnosis for {subtitle_id}")
        diagnosis_type = diagnosis.get("diagnosis")
        evidence_ids = set(diagnosis.get("evidence_block_ids") or [])
        supplied_ids = {row["block_id"] for row in request_by_id[request_id].get("dialogue_candidates", [])}
        if decision == "match":
            allowed = supplied_ids if diagnosis_type == "confirmed_reviewer_selection_error" else evidence_ids
            if not set(selected) <= allowed:
                raise ValueError(f"evidence correction selects unsupported blocks for {subtitle_id}")
            if diagnosis_type not in {"confirmed_reviewer_selection_error", "confirmed_candidate_recall_error"}:
                raise ValueError(f"match correction has incompatible diagnosis for {subtitle_id}")
        elif decision == "uncertain" and diagnosis_type != "genuine_ambiguity":
            raise ValueError(f"uncertain correction lacks genuine ambiguity diagnosis for {subtitle_id}")
        original = deepcopy(resolution_lookup[key])
        resolution_lookup[key].update({
            "decision": decision, "block_ids": selected, "confidence": 1.0,
            "decision_basis": "exact_or_near_exact" if decision == "match" else "insufficient_context",
            "openai_resolution": original.get("openai_resolution", original),
            "evidence_correction": {
                "adjudicator_type": "codex_source_evidence_not_human_gold",
                "diagnosis_subtitle_id": diagnosis_id, "diagnosis": diagnosis_type,
                "rationale": correction.get("rationale"),
            },
        })
    if not corrected_ids:
        raise ValueError("evidence correction plan is empty")
    ordered = [response_by_id[row["request_id"]] for row in requests]
    with tempfile.TemporaryDirectory(prefix="oscardp_evidence_corrections_") as directory:
        corrected_responses = Path(directory) / "responses.jsonl"
        _write_jsonl(corrected_responses, ordered)
        applied = apply_validated_responses(
            deterministic_alignment_path, requests_path, corrected_responses, context_path,
            shots_path, output_dir, output_tag, validate_against_historical_schema=False,
        )
    diagnosis_counts: Counter[str] = Counter()
    adjudicated_rows = []
    for row in audit:
        current = deepcopy(row)
        diagnosis = diagnosis_by_subtitle.get(row["subtitle_id"])
        if diagnosis is not None:
            diagnosis_type = diagnosis["diagnosis"]
            diagnosis_counts[diagnosis_type] += 1
            current["evidence_review"] = {
                "adjudicator_type": diagnosis["adjudicator_type"], "diagnosis": diagnosis_type,
                "evidence_block_ids": diagnosis.get("evidence_block_ids", []),
                "rationale": diagnosis["rationale"], "human_gold": False,
            }
            if diagnosis_type == "genuine_ambiguity":
                classification = "ambiguous_needs_review"
            elif diagnosis_type == "diagnostic_false_positive_true_no_match":
                classification = "probable_true_no_match"
            else:
                classification = "resolved_by_evidence_correction"
            current["diagnostics"]["no_candidate_match_classification"] = classification
        adjudicated_rows.append(current)
    _write_jsonl(adjudicated_audit_path, adjudicated_rows)
    classifications = Counter(
        row.get("diagnostics", {}).get("no_candidate_match_classification")
        for row in adjudicated_rows
        if row.get("diagnostics", {}).get("no_candidate_match_classification") is not None
    )
    original_no_match_count = sum(
        resolution.get("decision") in {"no_match", "no_candidate_match"}
        for response in responses for resolution in response.get("resolutions", [])
    )
    unrepresented_probable = original_no_match_count - sum(classifications.values())
    if unrepresented_probable < 0:
        raise ValueError("adjudicated audit classifications exceed original no-match resolutions")
    if unrepresented_probable:
        classifications["probable_true_no_match"] += unrepresented_probable
    reason_counts = Counter(
        reason for row in adjudicated_rows for reason in row.get("inclusion_reasons", [])
    )
    summary = {
        "schema_version": "1.0", "movie_id": movie_id,
        "record_count": len(adjudicated_rows), "audit_sha256": _sha(adjudicated_audit_path),
        "inclusion_reason_counts": dict(sorted(reason_counts.items())),
        "no_candidate_match_classification_counts": dict(sorted(classifications.items())),
        "evidence_diagnosis_counts": dict(sorted(diagnosis_counts.items())),
        "corrected_subtitle_count": len(corrected_ids),
        "human_gold": False, "original_openai_responses_preserved": True,
        "source_hashes": {
            "correction_plan": _sha(correction_plan_path), "diagnosis": _sha(diagnosis_path),
            "risk_audit": _sha(risk_audit_path), "requests": _sha(requests_path),
            "normalized_responses": _sha(normalized_responses_path),
        },
        "generated_at": datetime.now(UTC).isoformat(),
    }
    _write_json(adjudicated_summary_path, summary)
    report = {
        "schema_version": "1.0", "output_tag": output_tag,
        "corrected_subtitle_count": len(corrected_ids), "corrected_subtitle_ids": sorted(corrected_ids),
        "original_openai_responses_preserved": True, "human_gold": False,
        "baseline_files_unchanged": applied["baseline_files_unchanged"],
        "validation_passed": applied["validation_passed"],
        "alignment_output": applied["alignment_output"], "shot_output": applied["shot_output"],
        "adjudicated_audit": adjudicated_audit_path.as_posix(),
        "adjudicated_summary": adjudicated_summary_path.as_posix(),
    }
    _write_json(openai_dir / f"{output_tag}_evidence_correction_report.json", report)
    return report


def finalize_production_movie_v3(
    movie_id: str, inventory_path: Path, status_path: Path, context_path: Path,
    deterministic_alignment_path: Path, deterministic_shot_context_path: Path,
    reviewed_alignment_path: Path, reviewed_shot_context_path: Path, shots_path: Path,
    requests_path: Path, responses_path: Path, reviewer_manifest_path: Path,
    lifecycle_report_paths: list[Path], risk_audit_path: Path, risk_summary_path: Path,
    qc_path: Path, manifest_path: Path, *, max_unresolved_ambiguities: int = 5,
    max_unresolved_candidate_recall_risks: int = 0,
    max_unresolved_reviewer_selection_risks: int = 0,
    deterministic_requests_path: Path | None = None,
) -> dict[str, Any]:
    if qc_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite final production QC artifacts")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    inventory_movie = next(row for row in inventory["movies"] if row["movie_id"] == movie_id)
    status_movie = next(row for row in status["movies"] if row["movie_id"] == movie_id)
    reviewer = json.loads(reviewer_manifest_path.read_text(encoding="utf-8"))
    reviewer_components = reviewer.get("components") if isinstance(reviewer.get("components"), dict) else {}
    hard_validation_contract_version = (
        reviewer.get("hard_validation_contract_version")
        or reviewer_components.get("hard_validation_contract_version")
        or "candidate_task_v3_structure_v2"
    )
    requests, responses = read_jsonl(requests_path), read_jsonl(responses_path)
    request_by_id = {row["request_id"]: row for row in requests}
    response_by_id = {row["request_id"]: row for row in responses}
    errors: list[str] = []
    if set(request_by_id) != set(response_by_id) or len(request_by_id) != len(requests) or len(response_by_id) != len(responses):
        errors.append("production request/response coverage is not exact")
    invalid = 0
    for request_id in sorted(set(request_by_id) & set(response_by_id)):
        current, _diagnostics = validate_resolution_v3(
            _v3_validation_view(response_by_id[request_id]), request_by_id[request_id],
            hard_validation_contract_version,
        )
        if current:
            invalid += 1
    if invalid:
        errors.append(f"{invalid} production responses fail v3 hard validation")
    deterministic_validation = validate_files(context_path, deterministic_alignment_path, deterministic_shot_context_path, shots_path)
    reviewed_validation = validate_files(context_path, reviewed_alignment_path, reviewed_shot_context_path, shots_path)
    if not deterministic_validation.passed:
        errors.extend(f"deterministic: {item}" for item in deterministic_validation.errors)
    if not reviewed_validation.passed:
        errors.extend(f"reviewed: {item}" for item in reviewed_validation.errors)
    lifecycle_reports = []
    for path in lifecycle_report_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        passed = report.get("passed") if "passed" in report else report.get("validation_passed")
        lifecycle_reports.append({"path": path.as_posix(), "sha256": _sha(path), "passed": passed})
        if passed is not True:
            errors.append(f"lifecycle report did not pass: {path}")
    risk_summary = json.loads(risk_summary_path.read_text(encoding="utf-8"))
    if risk_summary.get("audit_sha256") != _sha(risk_audit_path):
        errors.append("high-risk audit hash differs from its summary")
    risk_request_hash = risk_summary.get("source_hashes", {}).get("requests")
    if risk_request_hash is not None and risk_request_hash != _sha(requests_path):
        errors.append("high-risk audit request hash differs from production requests")
    expected_sources = {
        "video": (
            Path(inventory_movie["video"]["path"]), inventory_movie["video"]["sha256"],
            inventory_movie["video"].get("stat_fingerprint"),
        ),
        "screenplay": (Path(inventory_movie["screenplay"]["path"]), inventory_movie["screenplay"]["sha256"], None),
        "shots": (shots_path, inventory_movie["stage1"]["shots_sha256"], None),
    }
    context_sources = json.loads(context_path.read_text(encoding="utf-8"))["source_files"]
    subtitle_expected = status_movie["artifact_hashes"].get("selected_stage2_subtitle", status_movie["artifact_hashes"]["subtitle"])
    expected_sources["stage2_subtitle"] = (Path(context_sources["subtitle"]), subtitle_expected, None)
    protected_hashes: dict[str, dict[str, Any]] = {}
    for label, (path, expected, stat_fingerprint) in expected_sources.items():
        protected_hashes[label] = _verify_protected_file(path, expected, stat_fingerprint=stat_fingerprint)
        if not protected_hashes[label]["unchanged"]:
            errors.append(f"protected {label} hash changed or file is missing")
    deterministic_expected = status_movie["deterministic_output_hashes"]
    deterministic_paths = {
        "screenplay_context": context_path, "alignment": deterministic_alignment_path,
        "shot_context": deterministic_shot_context_path,
        "review_requests": deterministic_requests_path or requests_path,
    }
    for label, path in deterministic_paths.items():
        expected = deterministic_expected[label]
        actual = _sha(path)
        protected_hashes[f"deterministic_{label}"] = {"path": path.as_posix(), "expected_sha256": expected, "actual_sha256": actual, "unchanged": actual == expected}
        if actual != expected:
            errors.append(f"deterministic {label} hash changed")
    reviewed_rows = read_jsonl(reviewed_alignment_path)
    status_counts = Counter(row["alignment"]["status"] for row in reviewed_rows)
    diagnostic_ambiguities = int(risk_summary.get("no_candidate_match_classification_counts", {}).get("ambiguous_needs_review", 0))
    candidate_recall_risks = int(risk_summary.get("no_candidate_match_classification_counts", {}).get("candidate_recall_risk", 0))
    reviewer_selection_risks = int(risk_summary.get("no_candidate_match_classification_counts", {}).get("reviewer_selection_risk", 0))
    needs_review_ids = {
        row["subtitle_id"] for row in reviewed_rows if row["alignment"]["status"] == "needs_review"
    }
    ambiguous_audit_ids = {
        row["subtitle_id"] for row in read_jsonl(risk_audit_path)
        if row.get("diagnostics", {}).get("no_candidate_match_classification") == "ambiguous_needs_review"
    }
    unresolved = (
        len(needs_review_ids | ambiguous_audit_ids)
        if len(ambiguous_audit_ids) == diagnostic_ambiguities
        else len(needs_review_ids) + diagnostic_ambiguities
    )
    if unresolved > max_unresolved_ambiguities:
        errors.append(f"unresolved ambiguity count {unresolved} exceeds allowed isolated maximum {max_unresolved_ambiguities}")
    if candidate_recall_risks > max_unresolved_candidate_recall_risks:
        errors.append(
            f"unresolved candidate-recall risk count {candidate_recall_risks} exceeds allowed maximum "
            f"{max_unresolved_candidate_recall_risks}"
        )
    if reviewer_selection_risks > max_unresolved_reviewer_selection_risks:
        errors.append(
            f"unresolved reviewer-selection risk count {reviewer_selection_risks} exceeds allowed maximum "
            f"{max_unresolved_reviewer_selection_risks}"
        )
    qc = {
        "schema_version": "1.0", "movie_id": movie_id, "production_reviewer_version": reviewer.get("production_reviewer_version"),
        "hard_validation_contract_version": hard_validation_contract_version,
        "request_count": len(requests), "resolution_count": sum(len(row["resolutions"]) for row in responses),
        "invalid_response_count": invalid, "missing_response_count": len(set(request_by_id) - set(response_by_id)),
        "foreign_response_count": len(set(response_by_id) - set(request_by_id)),
        "alignment_count": reviewed_validation.alignment_count, "shot_count": reviewed_validation.shot_count,
        "alignment_status_counts": dict(sorted(status_counts.items())), "unresolved_ambiguity_count": unresolved,
        "max_unresolved_ambiguities": max_unresolved_ambiguities,
        "unresolved_candidate_recall_risk_count": candidate_recall_risks,
        "max_unresolved_candidate_recall_risks": max_unresolved_candidate_recall_risks,
        "unresolved_reviewer_selection_risk_count": reviewer_selection_risks,
        "max_unresolved_reviewer_selection_risks": max_unresolved_reviewer_selection_risks,
        "high_risk_audit_count": risk_summary.get("record_count"), "high_risk_reason_counts": risk_summary.get("inclusion_reason_counts"),
        "deterministic_validation_passed": deterministic_validation.passed, "reviewed_validation_passed": reviewed_validation.passed,
        "protected_hashes": protected_hashes, "lifecycle_reports": lifecycle_reports,
        "errors": errors, "passed": not errors, "generated_at": datetime.now(UTC).isoformat(),
    }
    if errors:
        raise RuntimeError("final production QC failed: " + "; ".join(errors[:20]))
    _write_json(qc_path, qc)
    artifacts = {
        "reviewed_alignment": reviewed_alignment_path, "reviewed_shot_context": reviewed_shot_context_path,
        "validated_responses": responses_path, "high_risk_audit": risk_audit_path,
        "high_risk_summary": risk_summary_path, "qc_report": qc_path, "reviewer_manifest": reviewer_manifest_path,
    }
    manifest = {
        "schema_version": "1.0", "movie_id": movie_id, "status": "COMPLETE",
        "production_reviewer_version": reviewer["production_reviewer_version"],
        "hard_validation_contract_version": hard_validation_contract_version,
        "artifacts": {label: {"path": path.as_posix(), "sha256": _sha(path)} for label, path in artifacts.items()},
        "protected_hashes": protected_hashes, "batch_lifecycle_reports": lifecycle_reports,
        "counts": {
            "requests": len(requests), "resolutions": sum(len(row["resolutions"]) for row in responses),
            "alignment_rows": reviewed_validation.alignment_count, "shot_rows": reviewed_validation.shot_count,
            "high_risk_audit_rows": risk_summary["record_count"], "unresolved_ambiguities": unresolved,
        },
        "frozen_at": datetime.now(UTC).isoformat(),
    }
    _write_json(manifest_path, manifest)
    # The manifest is the completion authority; publish its paths atomically to
    # both goal registries only after every QC gate has passed.
    inventory_review = inventory_movie.setdefault("production_review", {})
    inventory_review.update({
        "status": "COMPLETE", "reviewer_version": reviewer["production_reviewer_version"],
        "hard_validation_contract_version": hard_validation_contract_version,
        "full_request_count": len(requests),
        "full_resolution_count": sum(len(row["resolutions"]) for row in responses),
        "reviewed_alignment": reviewed_alignment_path.as_posix(),
        "reviewed_alignment_sha256": _sha(reviewed_alignment_path),
        "reviewed_shot_context": reviewed_shot_context_path.as_posix(),
        "reviewed_shot_context_sha256": _sha(reviewed_shot_context_path),
        "reviewed_alignment_count": reviewed_validation.alignment_count,
        "reviewed_shot_count": reviewed_validation.shot_count,
        "high_risk_audit": risk_audit_path.as_posix(),
        "high_risk_audit_sha256": _sha(risk_audit_path),
        "final_qc": qc_path.as_posix(), "final_qc_sha256": _sha(qc_path),
        "production_manifest": manifest_path.as_posix(), "production_manifest_sha256": _sha(manifest_path),
        "pending_ambiguity_count": unresolved,
        "candidate_recall_risk_count": candidate_recall_risks,
        "reviewer_selection_risk_count": reviewer_selection_risks,
        "protected_hashes_unchanged": all(row["unchanged"] for row in protected_hashes.values()),
    })
    inventory["updated_at"] = datetime.now(UTC).isoformat()
    status_movie.update({
        "final_qc_status": "COMPLETE", "production_batch_status": "completed_validated_applied_final_qc_passed",
        "production_completed_chunk_count": status_movie.get("production_chunk_count", status_movie.get("production_completed_chunk_count")),
        "reviewed_alignment_path": reviewed_alignment_path.as_posix(),
        "reviewed_shot_context_path": reviewed_shot_context_path.as_posix(),
        "final_qc_path": qc_path.as_posix(), "final_qc_sha256": _sha(qc_path),
        "high_risk_audit_path": risk_audit_path.as_posix(), "high_risk_audit_sha256": _sha(risk_audit_path),
        "high_risk_summary_path": risk_summary_path.as_posix(),
        "pending_ambiguity_count": unresolved,
        "candidate_recall_risk_count": candidate_recall_risks,
        "reviewer_selection_risk_count": reviewer_selection_risks,
    })
    status["updated_at"] = datetime.now(UTC).isoformat()
    _write_json(inventory_path, inventory)
    _write_json(status_path, status)
    return {**qc, "manifest": manifest_path.as_posix(), "manifest_sha256": _sha(manifest_path)}
