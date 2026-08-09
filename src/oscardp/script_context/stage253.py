from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oscardp.shots.schema import json_dumps

from .openai_review import _extract_structured, _submit_validated_batch
from .openai_schema import V3_DECISION_BASES, V3_DECISIONS, V3_SYSTEM_INSTRUCTIONS, alignment_response_schema_v3
from .pipeline import _write_json, _write_jsonl
from .schema import read_jsonl


REQUEST_PREFIX_V3 = "Review this policy-aware candidate-task alignment request:\n"
REQUEST_CONTEXT_VERSION_V31 = "v3.1_nearby_subtitles"


def prepare_review_context_v31(
    requests_path: Path, alignment_path: Path, output_path: Path, radius: int = 2,
) -> dict[str, Any]:
    if radius < 1 or radius > 10:
        raise ValueError("nearby subtitle radius must be between 1 and 10")
    requests = read_jsonl(requests_path)
    alignments = read_jsonl(alignment_path)
    positions = {row.get("subtitle_id"): index for index, row in enumerate(alignments)}
    if len(positions) != len(alignments) or None in positions:
        raise ValueError("alignment must contain unique non-empty subtitle IDs")

    def view(row: dict[str, Any]) -> dict[str, Any]:
        return {"subtitle_id": row["subtitle_id"], "text": row["text"], "time": row["time"]}

    augmented: list[dict[str, Any]] = []
    for request in requests:
        target_ids = list(request.get("subtitle_ids", []))
        if not target_ids or any(subtitle_id not in positions for subtitle_id in target_ids):
            raise ValueError(f"{request.get('request_id')} has target subtitle outside alignment")
        target_positions = [positions[subtitle_id] for subtitle_id in target_ids]
        if target_positions != sorted(target_positions):
            raise ValueError(f"{request.get('request_id')} target subtitles are not ordered")
        first, last = target_positions[0], target_positions[-1]
        context = {
            "version": REQUEST_CONTEXT_VERSION_V31,
            "radius": radius,
            "targets_unchanged": True,
            "non_target_context_only": True,
            "allowed_context_fields": ["subtitle_id", "text", "time"],
            "before": [view(row) for row in alignments[max(0, first - radius):first]],
            "after": [view(row) for row in alignments[last + 1:last + 1 + radius]],
        }
        augmented.append({**request, "review_context": context})

    original_projection = [
        (row.get("request_id"), row.get("subtitle_ids"), [item.get("block_id") for item in row.get("dialogue_candidates", [])])
        for row in requests
    ]
    augmented_projection = [
        (row.get("request_id"), row.get("subtitle_ids"), [item.get("block_id") for item in row.get("dialogue_candidates", [])])
        for row in augmented
    ]
    if original_projection != augmented_projection:
        raise RuntimeError("review context augmentation changed request targets or candidate IDs")
    _write_jsonl(output_path, augmented)
    manifest = {
        "schema_version": "1.0", "request_context_version": REQUEST_CONTEXT_VERSION_V31,
        "radius": radius, "request_count": len(augmented),
        "source_requests": requests_path.resolve().as_posix(),
        "source_requests_sha256": hashlib.sha256(requests_path.read_bytes()).hexdigest(),
        "source_alignment": alignment_path.resolve().as_posix(),
        "source_alignment_sha256": hashlib.sha256(alignment_path.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "target_ids_unchanged": True, "candidate_ids_unchanged": True,
        "gold_labels_included": False, "context_is_not_resolution_target": True,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    _write_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    return manifest


def batch_line_v3(request: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "custom_id": request["request_id"], "method": "POST", "url": "/v1/responses",
        "body": {
            "model": model, "store": False, "instructions": V3_SYSTEM_INSTRUCTIONS,
            "input": REQUEST_PREFIX_V3 + json_dumps(request, pretty=True),
            "text": {"format": {
                "type": "json_schema", "name": "candidate_task_alignment_response_v3", "strict": True,
                "schema": alignment_response_schema_v3(request),
            }},
        },
    }


def _batch_request_v3(row: dict[str, Any]) -> dict[str, Any] | None:
    value = row.get("body", {}).get("input")
    if not isinstance(value, str) or not value.startswith(REQUEST_PREFIX_V3):
        return None
    try:
        parsed = json.loads(value[len(REQUEST_PREFIX_V3):])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def validate_batch_lines_v3(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for line_number, row in enumerate(rows, 1):
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            errors.append(f"line {line_number}: invalid custom_id")
        elif custom_id in seen:
            errors.append(f"line {line_number}: duplicate custom_id {custom_id}")
        seen.add(custom_id)
        if row.get("method") != "POST" or row.get("url") != "/v1/responses":
            errors.append(f"line {line_number}: batch endpoint must be POST /v1/responses")
        request = _batch_request_v3(row)
        if request is None:
            errors.append(f"line {line_number}: embedded v3 request payload is missing or malformed")
        elif request.get("request_id") != custom_id:
            errors.append(f"line {line_number}: custom_id does not match embedded request_id")
        expected_schema = None if request is None else alignment_response_schema_v3(request)
        fmt = row.get("body", {}).get("text", {}).get("format", {})
        if fmt.get("type") != "json_schema" or fmt.get("strict") is not True or fmt.get("schema") != expected_schema:
            errors.append(f"line {line_number}: request-specific v3 schema is missing or changed")
        if row.get("body", {}).get("instructions") != V3_SYSTEM_INSTRUCTIONS:
            errors.append(f"line {line_number}: v3 policy instructions are missing or changed")
    return errors


def prepare_batch_v3(
    requests_path: Path, annotation_policy_path: Path, output_path: Path, model: str,
) -> dict[str, Any]:
    requests = read_jsonl(requests_path)
    ids = [request.get("request_id") for request in requests]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("Requests must have unique non-empty request IDs")
    if not annotation_policy_path.is_file():
        raise ValueError(f"Annotation policy does not exist: {annotation_policy_path}")
    rows = [batch_line_v3(request, model) for request in requests]
    errors = validate_batch_lines_v3(rows)
    if errors:
        raise ValueError("Invalid v3 Batch input: " + "; ".join(errors))
    _write_jsonl(output_path, rows)
    schema_hashes = {
        request["request_id"]: hashlib.sha256(json_dumps(alignment_response_schema_v3(request)).encode("utf-8")).hexdigest()
        for request in requests
    }
    manifest = {
        "schema_version": "1.0", "review_policy_version": "annotation_policy_v1",
        "decision_schema_version": "candidate_task_v3", "model": model,
        "request_count": len(rows), "source_requests": requests_path.resolve().as_posix(),
        "source_requests_sha256": hashlib.sha256(requests_path.read_bytes()).hexdigest(),
        "annotation_policy": annotation_policy_path.resolve().as_posix(),
        "annotation_policy_sha256": hashlib.sha256(annotation_policy_path.read_bytes()).hexdigest(),
        "instructions_sha256": hashlib.sha256(V3_SYSTEM_INSTRUCTIONS.encode("utf-8")).hexdigest(),
        "schema_sha256_by_request": schema_hashes,
        "schema_sha256_aggregate": hashlib.sha256(json_dumps(schema_hashes).encode("utf-8")).hexdigest(),
        "request_payload_data_unchanged": True, "generated_at": datetime.now(UTC).isoformat(),
    }
    context_versions = sorted({
        request.get("review_context", {}).get("version", "v3_no_nearby_subtitle_context")
        for request in requests
    })
    manifest["request_context_versions"] = context_versions
    manifest["request_context_design"] = (
        None if context_versions == ["v3_no_nearby_subtitle_context"] else {
            "nearby_subtitles_are_non_targets": True,
            "fields": ["subtitle_id", "text", "time"],
            "radius_by_request": {
                request["request_id"]: request.get("review_context", {}).get("radius")
                for request in requests
            },
            "gold_labels_included": False,
        }
    )
    _write_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    return manifest


def submit_batch_v3(batch_input: Path, job_file: Path, *, confirm_submit: bool) -> dict[str, Any]:
    if not confirm_submit:
        raise RuntimeError("Refusing paid submission without --confirm-submit")
    rows = read_jsonl(batch_input)
    errors = validate_batch_lines_v3(rows)
    if errors:
        raise ValueError("Invalid v3 Batch input: " + "; ".join(errors))
    return _submit_validated_batch(
        batch_input,
        job_file,
        rows,
        metadata_extra={
            "schema_version": "1.0",
            "decision_schema_version": "candidate_task_v3",
            "review_policy_version": "annotation_policy_v1",
            "batch_input_sha256": hashlib.sha256(batch_input.read_bytes()).hexdigest(),
        },
    )


def _sequence_quality_v3(selections: list[dict[str, Any] | None], request_id: str, threshold: int = 3) -> dict[str, Any]:
    fields = (
        "backward_mapping_count", "bounded_backward_mapping_count", "large_backward_mapping_count",
        "missing_reorder_basis_count", "cross_scene_sequence_jump_count", "large_forward_jump_count",
        "repeated_block_mapping_count", "high_risk_sequence_event_count",
    )
    quality: dict[str, Any] = {field: 0 for field in fields}; quality["events"] = []
    previous: dict[str, Any] | None = None
    for current in selections:
        if current is None:
            continue
        if previous is not None:
            def add_event(reason: str, severity: str, distance: int, **extra: Any) -> None:
                quality["events"].append({
                    "request_id": request_id, "previous_subtitle_id": previous["subtitle_id"],
                    "subtitle_id": current["subtitle_id"],
                    "previous_screenplay_span": [previous["start"], previous["end"]],
                    "current_screenplay_span": [current["start"], current["end"]],
                    "distance": distance, "previous_scene": previous["scene_id"],
                    "current_scene": current["scene_id"], "decision_basis": current["basis"],
                    "severity": severity, "reason": reason, **extra,
                })
                if severity == "high_risk": quality["high_risk_sequence_event_count"] += 1
            if current["scene_id"] != previous["scene_id"]:
                quality["cross_scene_sequence_jump_count"] += 1
                add_event("cross_scene_sequence_jump", "high_risk", 0)
            if (current["start"], current["end"]) == (previous["start"], previous["end"]):
                quality["repeated_block_mapping_count"] += 1
                add_event("repeated_block_mapping", "info", 0)
            if current["start"] < previous["start"] or current["end"] < previous["end"]:
                distance = max(previous["start"] - current["start"], previous["end"] - current["end"])
                large = distance > threshold; missing_basis = current["basis"] != "repeated_or_reordered_dialogue"
                quality["backward_mapping_count"] += 1
                quality["large_backward_mapping_count" if large else "bounded_backward_mapping_count"] += 1
                if missing_basis: quality["missing_reorder_basis_count"] += 1
                severity = "high_risk" if large or current["scene_id"] != previous["scene_id"] else ("warning" if missing_basis else "info")
                add_event("large_backward_mapping" if large else "bounded_backward_mapping", severity, distance, missing_reorder_basis=missing_basis)
            elif current["start"] - previous["end"] > threshold:
                distance = current["start"] - previous["end"]
                quality["large_forward_jump_count"] += 1
                add_event("large_forward_jump", "warning", distance)
        previous = current
    return quality


def validate_resolution_v3(response: dict[str, Any], request: dict[str, Any]) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    diagnostics: dict[str, Any] = {"hard_validation_policy": "candidate_task_v3", "foreign_candidate_output_count": 0}
    if response.get("request_id") != request.get("request_id"):
        errors.append("request_id mismatch")
    resolutions = response.get("resolutions")
    if not isinstance(resolutions, list):
        diagnostics["sequence_quality"] = _sequence_quality_v3([], str(request.get("request_id")))
        return errors + ["resolutions must be an array"], diagnostics
    actual_subtitles = [item.get("subtitle_id") for item in resolutions if isinstance(item, dict)]
    if len(actual_subtitles) != len(set(actual_subtitles)):
        errors.append("duplicate subtitle resolution")
    if actual_subtitles != request["subtitle_ids"]:
        errors.append("subtitle resolutions must exactly match request order")
    candidates = request.get("dialogue_candidates", [])
    candidate_order = {candidate["block_id"]: index for index, candidate in enumerate(candidates)}
    candidate_by_id = {candidate["block_id"]: candidate for candidate in candidates}
    selections: list[dict[str, Any] | None] = []
    for item in resolutions:
        if not isinstance(item, dict):
            errors.append("resolution must be an object"); selections.append(None); continue
        subtitle_id, decision = item.get("subtitle_id"), item.get("decision")
        block_ids, confidence, basis = item.get("block_ids"), item.get("confidence"), item.get("decision_basis")
        if decision not in V3_DECISIONS: errors.append(f"invalid v3 decision for {subtitle_id}")
        if basis not in V3_DECISION_BASES: errors.append(f"invalid v3 decision_basis for {subtitle_id}")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            errors.append(f"invalid confidence for {subtitle_id}")
        if not isinstance(block_ids, list) or any(not isinstance(block_id, str) or not block_id for block_id in block_ids):
            errors.append(f"block_ids must be an array of non-empty strings for {subtitle_id}"); selections.append(None); continue
        if len(block_ids) != len(set(block_ids)):
            errors.append(f"block_ids must be unique for {subtitle_id}"); selections.append(None); continue
        if decision == "match" and not block_ids: errors.append(f"match requires block_ids for {subtitle_id}")
        if decision == "no_candidate_match" and block_ids: errors.append(f"no_candidate_match requires empty block_ids for {subtitle_id}")
        foreign = [block_id for block_id in block_ids if block_id not in candidate_order]
        if foreign:
            diagnostics["foreign_candidate_output_count"] += len(foreign)
            errors.append(f"block outside request candidates for {subtitle_id}"); selections.append(None); continue
        orders = [candidate_order[block_id] for block_id in block_ids]
        if orders != sorted(orders) or any(right != left + 1 for left, right in zip(orders, orders[1:])):
            errors.append(f"selected blocks must be ordered and adjacent for {subtitle_id}")
        scenes = {candidate_by_id[block_id]["scene_id"] for block_id in block_ids}
        if len(scenes) > 1: errors.append(f"selected blocks cross scenes for {subtitle_id}")
        if not block_ids:
            selections.append(None); continue
        screenplay_orders = [int(candidate_by_id[block_id]["screenplay_order"]) for block_id in block_ids]
        selections.append({
            "subtitle_id": subtitle_id, "start": screenplay_orders[0], "end": screenplay_orders[-1],
            "scene_id": next(iter(scenes)) if len(scenes) == 1 else None, "basis": basis,
        })
    diagnostics["sequence_quality"] = _sequence_quality_v3(selections, str(request.get("request_id")))
    return errors, diagnostics


def validate_responses_v3(raw_path: Path, requests_path: Path, output_dir: Path) -> dict[str, Any]:
    requests = read_jsonl(requests_path); request_by_id = {row["request_id"]: row for row in requests}
    valid: list[dict[str, Any]] = []; errors: list[dict[str, Any]] = []; seen: set[str] = set()
    for line_number, row in enumerate(read_jsonl(raw_path), 1):
        custom_id = row.get("custom_id")
        if custom_id not in request_by_id:
            errors.append({"line": line_number, "custom_id": custom_id, "errors": ["unknown custom_id"]}); continue
        if custom_id in seen:
            errors.append({"line": line_number, "custom_id": custom_id, "errors": ["duplicate request response"]}); continue
        seen.add(custom_id)
        api_response = row.get("response", {})
        if row.get("error") or api_response.get("status_code") != 200:
            errors.append({"line": line_number, "custom_id": custom_id, "errors": ["API error or non-200 response"]}); continue
        body = api_response.get("body", {})
        structured, extraction_error = _extract_structured(body)
        if extraction_error:
            errors.append({"line": line_number, "custom_id": custom_id, "errors": [extraction_error]}); continue
        hard_errors, diagnostics = validate_resolution_v3(structured, request_by_id[custom_id])  # type: ignore[arg-type]
        if hard_errors:
            errors.append({"line": line_number, "custom_id": custom_id, "errors": hard_errors, "validation_diagnostics": diagnostics}); continue
        valid.append({**structured, "custom_id": custom_id, "response_id": body.get("id"), "model": body.get("model"), "validation_diagnostics": diagnostics})  # type: ignore[arg-type]
    missing = sorted(set(request_by_id) - seen)
    quality_fields = (
        "backward_mapping_count", "bounded_backward_mapping_count", "large_backward_mapping_count",
        "missing_reorder_basis_count", "cross_scene_sequence_jump_count", "large_forward_jump_count",
        "repeated_block_mapping_count", "high_risk_sequence_event_count",
    )
    quality: dict[str, Any] = {field: 0 for field in quality_fields}; quality["events"] = []
    for response in valid:
        current = response["validation_diagnostics"]["sequence_quality"]
        for field in quality_fields: quality[field] += int(current[field])
        quality["events"].extend(current["events"])
    foreign = sum(int(row.get("validation_diagnostics", {}).get("foreign_candidate_output_count", 0)) for row in [*valid, *errors])
    report = {
        "schema_version": "1.0", "validation_policy": "candidate_task_v3", "request_count": len(requests),
        "resolution_count": sum(len(row["resolutions"]) for row in valid), "valid_count": len(valid),
        "invalid_count": len(errors), "missing_request_ids": missing, "foreign_candidate_output_count": foreign,
        "sequence_quality": quality, "errors": errors, "passed": not errors and not missing,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "validated_responses.jsonl", valid); _write_json(output_dir / "response_validation_report.json", report)
    return report


def _binary_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    confusion = {
        expected: {actual: 0 for actual in ("match", "no_candidate_match", "missing")}
        for expected in ("match", "no_candidate_match")
    }
    candidate_correct = 0; presence_correct = 0; match_total = 0; match_correct = 0; no_candidate_total = 0; no_candidate_correct = 0
    for record in records:
        expected, actual = record["expected"], record["actual"]
        expected_decision = "match" if expected["decision"] == "match" else "no_candidate_match"
        actual_decision = "missing" if actual is None else actual.get("decision", "missing")
        confusion[expected_decision][actual_decision] += 1
        presence = actual is not None and actual_decision == expected_decision
        exact = presence and (expected_decision == "no_candidate_match" or set(actual["block_ids"]) == set(expected["block_ids"]))
        presence_correct += presence; candidate_correct += exact
        if expected_decision == "match":
            match_total += 1; match_correct += exact
        else:
            no_candidate_total += 1; no_candidate_correct += exact
    total = len(records)
    return {
        "resolution_count": total, "candidate_task_correct_count": candidate_correct,
        "candidate_task_accuracy": candidate_correct / total if total else 0.0,
        "candidate_presence_correct_count": presence_correct,
        "candidate_presence_decision_accuracy": presence_correct / total if total else 0.0,
        "match_resolution_count": match_total, "match_exact_block_correct_count": match_correct,
        "match_exact_block_accuracy": match_correct / match_total if match_total else None,
        "no_candidate_match_resolution_count": no_candidate_total, "no_candidate_match_correct_count": no_candidate_correct,
        "no_candidate_match_accuracy": no_candidate_correct / no_candidate_total if no_candidate_total else None,
        "confusion_matrix": confusion, "incorrect_prediction_count": total - candidate_correct,
    }


def evaluate_pilot_v3(
    gold_path: Path, validated_path: Path, manifest_path: Path,
    adjudication_path: Path, output_path: Path,
) -> dict[str, Any]:
    gold, responses = read_jsonl(gold_path), read_jsonl(validated_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    adjudication = read_jsonl(adjudication_path)
    ambiguous_ids = sorted({
        row["subtitle_id"] for row in adjudication
        if row.get("human_adjudication", {}).get("adjudication") == "ambiguous"
    })
    predicted = {(row["request_id"], item["subtitle_id"]): item for row in responses for item in row["resolutions"]}
    selection = {row["request_id"]: row for row in manifest["requests"]}
    records = [{
        "request_id": gold_row["request_id"], "expected": expected,
        "actual": predicted.get((gold_row["request_id"], expected["subtitle_id"])),
        **selection[gold_row["request_id"]],
    } for gold_row in gold for expected in gold_row["resolutions"]]
    resolved = [record for record in records if record["expected"]["subtitle_id"] not in ambiguous_ids]
    all_metrics, resolved_metrics = _binary_metrics(records), _binary_metrics(resolved)
    def grouped(field: str, values: tuple[str, ...]) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        for value in values:
            subset = [record for record in resolved if record[field] == value]
            result[value] = None if not subset else _binary_metrics(subset)["candidate_task_accuracy"]
        return result
    validation_path = validated_path.with_name("response_validation_report.json")
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else {}
    missing_count = sum(record["actual"] is None for record in records)
    checks = {
        "all_30_requests_structurally_valid": validation.get("valid_count") == 30 and validation.get("invalid_count") == 0,
        "zero_missing_predictions": missing_count == 0,
        "zero_foreign_candidates": int(validation.get("foreign_candidate_output_count", 0)) == 0,
        "resolved_candidate_task_accuracy_at_least_0_90": resolved_metrics["candidate_task_accuracy"] >= 0.90,
        "resolved_candidate_presence_accuracy_at_least_0_90": resolved_metrics["candidate_presence_decision_accuracy"] >= 0.90,
    }
    result = {
        "schema_version": "1.0", "decision_schema_version": "candidate_task_v3",
        "gold_status": "provisional_pending_1_ambiguous_case",
        "excluded_ambiguous_count": len(ambiguous_ids), "excluded_ambiguous_subtitle_ids": ambiguous_ids,
        "all_records_metrics": all_metrics, "resolved_gold_metrics": resolved_metrics,
        "resolved_accuracy_by_stratum": grouped("stratum", ("easy", "fuzzy", "multi", "difficult")),
        "resolved_accuracy_by_timeline_region": grouped("timeline_region", ("early", "middle", "late")),
        "structural_invalid_request_count": int(validation.get("invalid_count", 0)),
        "missing_prediction_count": missing_count,
        "foreign_candidate_output_count": int(validation.get("foreign_candidate_output_count", 0)),
        "sequence_quality": validation.get("sequence_quality", {}),
        "acceptance_gate": {"basis": "resolved_gold_metrics", "checks": checks, "passed": all(checks.values())},
    }
    _write_json(output_path, result)
    return result
