from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oscardp.shots.schema import json_dumps

from .openai_review import _extract_structured, _submit_validated_batch
from .openai_schema import V3_DECISION_BASES, V3_DECISIONS, V3_SYSTEM_INSTRUCTIONS, V321_VOCATIVE_SYSTEM_INSTRUCTIONS, V32_POLICY_SYSTEM_INSTRUCTIONS, alignment_response_schema_v3
from .pipeline import _write_json, _write_jsonl
from .schema import read_jsonl


REQUEST_PREFIX_V3 = "Review this policy-aware candidate-task alignment request:\n"
REQUEST_PREFIX_V32_POLICY = "Review this candidate-task alignment request under reviewer policy v3.2:\n"
REQUEST_PREFIX_V33_ACTION_CONTEXT = "Review this candidate-task alignment request under reviewer v3.3 action context:\n"
REQUEST_CONTEXT_VERSION_V31 = "v3.1_nearby_subtitles"
REQUEST_CONTEXT_VERSION_V33 = "v3.3_nearby_screenplay_actions"


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


def prepare_review_action_context_v33(
    requests_path: Path,
    screenplay_context_path: Path,
    output_path: Path,
    radius: int = 8,
    max_actions: int = 24,
) -> dict[str, Any]:
    """Add bounded, non-selectable action context without changing the candidate task."""
    if radius < 1 or radius > 40:
        raise ValueError("screenplay action radius must be between 1 and 40")
    if max_actions < 1 or max_actions > 100:
        raise ValueError("maximum action blocks must be between 1 and 100")
    requests = read_jsonl(requests_path)
    screenplay = json.loads(screenplay_context_path.read_text(encoding="utf-8"))
    scenes = screenplay.get("script_scenes")
    if not isinstance(scenes, list):
        raise ValueError("screenplay context must contain script_scenes")

    block_index: dict[str, tuple[int, dict[str, Any]]] = {}
    scene_blocks: dict[str, list[dict[str, Any]]] = {}
    for scene_index, scene in enumerate(scenes):
        scene_id = scene.get("scene_id")
        blocks = scene.get("script_blocks")
        if not isinstance(scene_id, str) or not isinstance(blocks, list):
            raise ValueError("screenplay scenes require scene_id and script_blocks")
        scene_blocks[scene_id] = blocks
        for block in blocks:
            block_id = block.get("block_id")
            if not isinstance(block_id, str) or block_id in block_index:
                raise ValueError("screenplay blocks require unique non-empty block IDs")
            block_index[block_id] = (scene_index, block)

    augmented: list[dict[str, Any]] = []
    for request in requests:
        candidates = request.get("dialogue_candidates", [])
        candidate_positions: dict[str, list[int]] = {}
        candidate_scene_rank: dict[str, int] = {}
        for candidate_index, candidate in enumerate(candidates):
            block_id = candidate.get("block_id")
            scene_id = candidate.get("scene_id")
            indexed = block_index.get(block_id)
            if indexed is None or indexed[1].get("block_type") != "dialogue":
                raise ValueError(f"{request.get('request_id')} candidate {block_id!r} is not screenplay dialogue")
            if indexed[1].get("block_id") != block_id or scene_id not in scene_blocks:
                raise ValueError(f"{request.get('request_id')} has invalid candidate scene")
            actual_scene_id = scenes[indexed[0]]["scene_id"]
            if scene_id != actual_scene_id:
                raise ValueError(f"{request.get('request_id')} candidate {block_id!r} has wrong scene")
            source_order = indexed[1].get("source_order")
            if not isinstance(source_order, int):
                raise ValueError(f"screenplay block {block_id!r} has invalid source_order")
            candidate_positions.setdefault(scene_id, []).append(source_order)
            candidate_scene_rank.setdefault(scene_id, candidate_index)

        ranked_actions: list[tuple[int, int, int, dict[str, Any]]] = []
        for scene_id, positions in candidate_positions.items():
            for block in scene_blocks[scene_id]:
                if block.get("block_type") != "action":
                    continue
                source_order = block.get("source_order")
                if not isinstance(source_order, int):
                    raise ValueError(f"screenplay action {block.get('block_id')!r} has invalid source_order")
                distance = min(abs(source_order - position) for position in positions)
                if distance <= radius:
                    view = {
                        "block_id": block["block_id"], "scene_id": scene_id,
                        "source_order": source_order, "text": block.get("text", ""),
                        "block_type": "action", "selectable": False,
                    }
                    ranked_actions.append((distance, candidate_scene_rank[scene_id], source_order, view))
        selected = sorted(ranked_actions)[:max_actions]
        selected_views = [item[3] for item in sorted(selected, key=lambda item: (item[1], item[2]))]
        context = {
            "version": REQUEST_CONTEXT_VERSION_V33, "radius": radius,
            "max_action_blocks": max_actions, "dialogue_candidates_unchanged": True,
            "action_blocks_are_non_selectable": True, "gold_labels_included": False,
            "screenplay_action_blocks": selected_views,
        }
        augmented.append({**request, "review_context": context})

    def projection(rows: list[dict[str, Any]]) -> list[tuple[Any, Any, Any]]:
        return [
            (row.get("request_id"), row.get("subtitle_ids"),
             [item.get("block_id") for item in row.get("dialogue_candidates", [])])
            for row in rows
        ]

    if projection(requests) != projection(augmented):
        raise RuntimeError("action context augmentation changed request targets or candidate IDs")
    _write_jsonl(output_path, augmented)
    manifest = {
        "schema_version": "1.0", "reviewer_version": "v3.3-action-context",
        "changed_layer": "request_context_only", "request_context_version": REQUEST_CONTEXT_VERSION_V33,
        "radius": radius, "max_action_blocks": max_actions, "request_count": len(augmented),
        "source_requests": requests_path.resolve().as_posix(),
        "source_requests_sha256": hashlib.sha256(requests_path.read_bytes()).hexdigest(),
        "source_screenplay_context": screenplay_context_path.resolve().as_posix(),
        "source_screenplay_context_sha256": hashlib.sha256(screenplay_context_path.read_bytes()).hexdigest(),
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


def batch_line_v32_policy(request: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "custom_id": request["request_id"], "method": "POST", "url": "/v1/responses",
        "body": {
            "model": model, "store": False, "instructions": V32_POLICY_SYSTEM_INSTRUCTIONS,
            "input": REQUEST_PREFIX_V32_POLICY + json_dumps(request, pretty=True),
            "text": {"format": {
                "type": "json_schema", "name": "candidate_task_alignment_response_v3", "strict": True,
                "schema": alignment_response_schema_v3(request),
            }},
        },
    }


def batch_line_v321_vocative(request: dict[str, Any], model: str) -> dict[str, Any]:
    row = batch_line_v32_policy(request, model)
    row["body"]["instructions"] = V321_VOCATIVE_SYSTEM_INSTRUCTIONS
    return row


def batch_line_v33_action_context(request: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "custom_id": request["request_id"], "method": "POST", "url": "/v1/responses",
        "body": {
            "model": model, "store": False, "instructions": V32_POLICY_SYSTEM_INSTRUCTIONS,
            "input": REQUEST_PREFIX_V33_ACTION_CONTEXT + json_dumps(request, pretty=True),
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


def validate_batch_lines_v32_policy(rows: list[dict[str, Any]]) -> list[str]:
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
        value = row.get("body", {}).get("input")
        request = None
        if isinstance(value, str) and value.startswith(REQUEST_PREFIX_V32_POLICY):
            try:
                parsed = json.loads(value[len(REQUEST_PREFIX_V32_POLICY):])
                request = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                pass
        if request is None:
            errors.append(f"line {line_number}: embedded v3.2 request payload is missing or malformed")
        elif request.get("request_id") != custom_id:
            errors.append(f"line {line_number}: custom_id does not match embedded request_id")
        expected_schema = None if request is None else alignment_response_schema_v3(request)
        fmt = row.get("body", {}).get("text", {}).get("format", {})
        if fmt.get("type") != "json_schema" or fmt.get("strict") is not True or fmt.get("schema") != expected_schema:
            errors.append(f"line {line_number}: request-specific v3 schema is missing or changed")
        if row.get("body", {}).get("instructions") != V32_POLICY_SYSTEM_INSTRUCTIONS:
            errors.append(f"line {line_number}: v3.2 policy instructions are missing or changed")
    return errors


def validate_batch_lines_v321_vocative(rows: list[dict[str, Any]]) -> list[str]:
    normalized = json.loads(json.dumps(rows))
    for row in normalized:
        if row.get("body", {}).get("instructions") == V321_VOCATIVE_SYSTEM_INSTRUCTIONS:
            row["body"]["instructions"] = V32_POLICY_SYSTEM_INSTRUCTIONS
    return validate_batch_lines_v32_policy(normalized)


def validate_batch_lines_v33_action_context(rows: list[dict[str, Any]]) -> list[str]:
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
        value = row.get("body", {}).get("input")
        request = None
        if isinstance(value, str) and value.startswith(REQUEST_PREFIX_V33_ACTION_CONTEXT):
            try:
                parsed = json.loads(value[len(REQUEST_PREFIX_V33_ACTION_CONTEXT):])
                request = parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                pass
        if request is None:
            errors.append(f"line {line_number}: embedded v3.3 request payload is missing or malformed")
        else:
            if request.get("request_id") != custom_id:
                errors.append(f"line {line_number}: custom_id does not match embedded request_id")
            context = request.get("review_context", {})
            actions = context.get("screenplay_action_blocks")
            if context.get("version") != REQUEST_CONTEXT_VERSION_V33 or not isinstance(actions, list):
                errors.append(f"line {line_number}: v3.3 action context is missing or changed")
            elif any(
                action.get("block_type") != "action" or action.get("selectable") is not False
                for action in actions if isinstance(action, dict)
            ) or any(not isinstance(action, dict) for action in actions):
                errors.append(f"line {line_number}: action context must be non-selectable action blocks")
        expected_schema = None if request is None else alignment_response_schema_v3(request)
        fmt = row.get("body", {}).get("text", {}).get("format", {})
        if fmt.get("type") != "json_schema" or fmt.get("strict") is not True or fmt.get("schema") != expected_schema:
            errors.append(f"line {line_number}: request-specific v3 schema is missing or changed")
        if row.get("body", {}).get("instructions") != V32_POLICY_SYSTEM_INSTRUCTIONS:
            errors.append(f"line {line_number}: v3.2 policy instructions are missing or changed")
    return errors


def prepare_batch_v33_action_context(
    requests_path: Path, annotation_policy_path: Path, output_path: Path, model: str,
) -> dict[str, Any]:
    requests = read_jsonl(requests_path)
    ids = [request.get("request_id") for request in requests]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("Requests must have unique non-empty request IDs")
    if not annotation_policy_path.is_file():
        raise ValueError(f"Annotation policy does not exist: {annotation_policy_path}")
    rows = [batch_line_v33_action_context(request, model) for request in requests]
    errors = validate_batch_lines_v33_action_context(rows)
    if errors:
        raise ValueError("Invalid v3.3 Batch input: " + "; ".join(errors))
    _write_jsonl(output_path, rows)
    schema_hashes = {
        request["request_id"]: hashlib.sha256(json_dumps(alignment_response_schema_v3(request)).encode("utf-8")).hexdigest()
        for request in requests
    }
    manifest = {
        "schema_version": "1.0", "reviewer_version": "v3.3-action-context",
        "review_policy_version": "annotation_policy_v1_plus_generic_v3.2_instructions_unchanged",
        "changed_layer": "request_context_only", "decision_schema_version": "candidate_task_v3",
        "request_context_version": REQUEST_CONTEXT_VERSION_V33, "model": model,
        "request_count": len(rows), "source_requests": requests_path.resolve().as_posix(),
        "source_requests_sha256": hashlib.sha256(requests_path.read_bytes()).hexdigest(),
        "annotation_policy": annotation_policy_path.resolve().as_posix(),
        "annotation_policy_sha256": hashlib.sha256(annotation_policy_path.read_bytes()).hexdigest(),
        "instructions_sha256": hashlib.sha256(V32_POLICY_SYSTEM_INSTRUCTIONS.encode("utf-8")).hexdigest(),
        "schema_sha256_by_request": schema_hashes,
        "schema_sha256_aggregate": hashlib.sha256(json_dumps(schema_hashes).encode("utf-8")).hexdigest(),
        "target_and_candidate_ids_unchanged": True, "gold_labels_included": False,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    _write_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    return manifest


def submit_batch_v33_action_context(batch_input: Path, job_file: Path, *, confirm_submit: bool) -> dict[str, Any]:
    if not confirm_submit:
        raise RuntimeError("Refusing paid submission without --confirm-submit")
    rows = read_jsonl(batch_input)
    errors = validate_batch_lines_v33_action_context(rows)
    if errors:
        raise ValueError("Invalid v3.3 Batch input: " + "; ".join(errors))
    return _submit_validated_batch(
        batch_input, job_file, rows,
        metadata_extra={
            "schema_version": "1.0", "reviewer_version": "v3.3-action-context",
            "decision_schema_version": "candidate_task_v3",
            "review_policy_version": "annotation_policy_v1_plus_generic_v3.2_instructions_unchanged",
            "request_context_version": REQUEST_CONTEXT_VERSION_V33,
            "batch_input_sha256": hashlib.sha256(batch_input.read_bytes()).hexdigest(),
        },
    )


def prepare_batch_v32_policy(
    requests_path: Path, annotation_policy_path: Path, output_path: Path, model: str,
) -> dict[str, Any]:
    requests = read_jsonl(requests_path)
    ids = [request.get("request_id") for request in requests]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("Requests must have unique non-empty request IDs")
    if not annotation_policy_path.is_file():
        raise ValueError(f"Annotation policy does not exist: {annotation_policy_path}")
    rows = [batch_line_v32_policy(request, model) for request in requests]
    errors = validate_batch_lines_v32_policy(rows)
    if errors:
        raise ValueError("Invalid v3.2 Batch input: " + "; ".join(errors))
    _write_jsonl(output_path, rows)
    schema_hashes = {
        request["request_id"]: hashlib.sha256(json_dumps(alignment_response_schema_v3(request)).encode("utf-8")).hexdigest()
        for request in requests
    }
    manifest = {
        "schema_version": "1.0", "reviewer_version": "v3.2-policy",
        "review_policy_version": "annotation_policy_v1_plus_generic_v3.2_instructions",
        "changed_layer": "reviewer_prompt_policy_only", "decision_schema_version": "candidate_task_v3",
        "request_context_version": "v3_no_nearby_subtitle_context", "model": model,
        "request_count": len(rows), "source_requests": requests_path.resolve().as_posix(),
        "source_requests_sha256": hashlib.sha256(requests_path.read_bytes()).hexdigest(),
        "annotation_policy": annotation_policy_path.resolve().as_posix(),
        "annotation_policy_sha256": hashlib.sha256(annotation_policy_path.read_bytes()).hexdigest(),
        "instructions_sha256": hashlib.sha256(V32_POLICY_SYSTEM_INSTRUCTIONS.encode("utf-8")).hexdigest(),
        "schema_sha256_by_request": schema_hashes,
        "schema_sha256_aggregate": hashlib.sha256(json_dumps(schema_hashes).encode("utf-8")).hexdigest(),
        "request_payload_data_unchanged": True, "generated_at": datetime.now(UTC).isoformat(),
    }
    _write_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    return manifest


def submit_batch_v32_policy(batch_input: Path, job_file: Path, *, confirm_submit: bool) -> dict[str, Any]:
    if not confirm_submit:
        raise RuntimeError("Refusing paid submission without --confirm-submit")
    rows = read_jsonl(batch_input)
    errors = validate_batch_lines_v32_policy(rows)
    if errors:
        raise ValueError("Invalid v3.2 Batch input: " + "; ".join(errors))
    return _submit_validated_batch(
        batch_input, job_file, rows,
        metadata_extra={
            "schema_version": "1.0", "reviewer_version": "v3.2-policy",
            "decision_schema_version": "candidate_task_v3",
            "review_policy_version": "annotation_policy_v1_plus_generic_v3.2_instructions",
            "batch_input_sha256": hashlib.sha256(batch_input.read_bytes()).hexdigest(),
        },
    )


def prepare_batch_v321_vocative(
    requests_path: Path, annotation_policy_path: Path, output_path: Path, model: str,
) -> dict[str, Any]:
    requests = read_jsonl(requests_path)
    if len(requests) > 30:
        raise ValueError("v3.2.1 vocative candidate is calibration-only and limited to 30 requests")
    ids = [request.get("request_id") for request in requests]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("Requests must have unique non-empty request IDs")
    if not annotation_policy_path.is_file():
        raise ValueError(f"Annotation policy does not exist: {annotation_policy_path}")
    rows = [batch_line_v321_vocative(request, model) for request in requests]
    errors = validate_batch_lines_v321_vocative(rows)
    if errors:
        raise ValueError("Invalid v3.2.1 vocative Batch input: " + "; ".join(errors))
    _write_jsonl(output_path, rows)
    manifest = {
        "schema_version": "1.0", "reviewer_version": "v3.2.1-vocative-candidate",
        "changed_layer": "reviewer_prompt_policy_only", "decision_schema_version": "candidate_task_v3",
        "request_context_version": "v3_no_nearby_subtitle_context", "model": model,
        "request_count": len(rows), "source_requests": requests_path.resolve().as_posix(),
        "source_requests_sha256": hashlib.sha256(requests_path.read_bytes()).hexdigest(),
        "annotation_policy_sha256": hashlib.sha256(annotation_policy_path.read_bytes()).hexdigest(),
        "instructions_sha256": hashlib.sha256(V321_VOCATIVE_SYSTEM_INSTRUCTIONS.encode()).hexdigest(),
        "calibration_only": True, "maximum_request_count": 30,
        "batch_input_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    _write_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    return manifest


def submit_batch_v321_vocative(batch_input: Path, job_file: Path, *, confirm_submit: bool) -> dict[str, Any]:
    if not confirm_submit:
        raise RuntimeError("Refusing paid submission without --confirm-submit")
    rows = read_jsonl(batch_input)
    if len(rows) > 30:
        raise ValueError("v3.2.1 vocative candidate is calibration-only and limited to 30 requests")
    errors = validate_batch_lines_v321_vocative(rows)
    if errors:
        raise ValueError("Invalid v3.2.1 vocative Batch input: " + "; ".join(errors))
    manifest_path = batch_input.with_suffix(batch_input.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("batch_input_sha256") != hashlib.sha256(batch_input.read_bytes()).hexdigest():
        raise ValueError("v3.2.1 vocative Batch hash differs from manifest")
    return _submit_validated_batch(batch_input, job_file, rows, metadata_extra={
        "schema_version": "1.0", "reviewer_version": "v3.2.1-vocative-candidate",
        "decision_schema_version": "candidate_task_v3", "changed_layer": "reviewer_prompt_policy_only",
        "batch_input_sha256": manifest["batch_input_sha256"],
    })


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
        "repeated_block_mapping_count", "non_adjacent_resolution_count",
        "cross_scene_resolution_count", "reversed_block_order_resolution_count",
        "high_risk_sequence_event_count",
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


def validate_resolution_v3(
    response: dict[str, Any], request: dict[str, Any],
    hard_validation_contract_version: str = "candidate_task_v3_structure_v2",
) -> tuple[list[str], dict[str, Any]]:
    if hard_validation_contract_version not in {
        "candidate_task_v3_structure_v2", "candidate_task_v3_structure_v3",
    }:
        raise ValueError("unsupported candidate-task hard validation contract")
    errors: list[str] = []
    diagnostics: dict[str, Any] = {
        "hard_validation_policy": "candidate_task_v3",
        "hard_validation_contract_version": hard_validation_contract_version,
        "foreign_candidate_output_count": 0,
    }
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
        if orders != sorted(orders) and hard_validation_contract_version == "candidate_task_v3_structure_v2":
            errors.append(f"selected blocks must preserve request candidate order for {subtitle_id}")
        scenes = {candidate_by_id[block_id]["scene_id"] for block_id in block_ids}
        if not block_ids:
            selections.append(None); continue
        screenplay_orders = [int(candidate_by_id[block_id]["screenplay_order"]) for block_id in block_ids]
        selections.append({
            "subtitle_id": subtitle_id, "start": min(screenplay_orders), "end": max(screenplay_orders),
            "scene_id": next(iter(scenes)) if len(scenes) == 1 else None, "basis": basis,
            "block_ids": block_ids, "candidate_orders": orders,
            "screenplay_orders": screenplay_orders, "scenes": sorted(scenes),
        })
    diagnostics["sequence_quality"] = _sequence_quality_v3(selections, str(request.get("request_id")))
    quality = diagnostics["sequence_quality"]
    for selection in selections:
        if selection is None:
            continue
        common = {
            "request_id": str(request.get("request_id")),
            "subtitle_id": selection["subtitle_id"],
            "selected_block_ids": selection["block_ids"],
            "selected_candidate_orders": selection["candidate_orders"],
            "selected_screenplay_orders": selection["screenplay_orders"],
            "selected_scenes": selection["scenes"],
            "decision_basis": selection["basis"],
            "severity": "high_risk",
        }
        ordered_candidate_orders = sorted(selection["candidate_orders"])
        if selection["candidate_orders"] != ordered_candidate_orders:
            quality["reversed_block_order_resolution_count"] += 1
            quality["high_risk_sequence_event_count"] += 1
            quality["events"].append({**common, "reason": "reversed_block_order_within_resolution"})
        if any(right != left + 1 for left, right in zip(ordered_candidate_orders, ordered_candidate_orders[1:])):
            quality["non_adjacent_resolution_count"] += 1
            quality["high_risk_sequence_event_count"] += 1
            quality["events"].append({**common, "reason": "non_adjacent_blocks_within_resolution"})
        if len(selection["scenes"]) > 1:
            quality["cross_scene_resolution_count"] += 1
            quality["high_risk_sequence_event_count"] += 1
            quality["events"].append({**common, "reason": "cross_scene_blocks_within_resolution"})
    return errors, diagnostics


def validate_responses_v3(
    raw_path: Path, requests_path: Path, output_dir: Path,
    hard_validation_contract_version: str = "candidate_task_v3_structure_v2",
) -> dict[str, Any]:
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
        hard_errors, diagnostics = validate_resolution_v3(  # type: ignore[arg-type]
            structured, request_by_id[custom_id], hard_validation_contract_version,
        )
        if hard_errors:
            errors.append({"line": line_number, "custom_id": custom_id, "errors": hard_errors, "validation_diagnostics": diagnostics}); continue
        valid.append({**structured, "custom_id": custom_id, "response_id": body.get("id"), "model": body.get("model"), "validation_diagnostics": diagnostics})  # type: ignore[arg-type]
    missing = sorted(set(request_by_id) - seen)
    quality_fields = (
        "backward_mapping_count", "bounded_backward_mapping_count", "large_backward_mapping_count",
        "missing_reorder_basis_count", "cross_scene_sequence_jump_count", "large_forward_jump_count",
        "repeated_block_mapping_count", "non_adjacent_resolution_count",
        "cross_scene_resolution_count", "reversed_block_order_resolution_count",
        "high_risk_sequence_event_count",
    )
    quality: dict[str, Any] = {field: 0 for field in quality_fields}; quality["events"] = []
    for response in valid:
        current = response["validation_diagnostics"]["sequence_quality"]
        for field in quality_fields: quality[field] += int(current[field])
        quality["events"].extend(current["events"])
    foreign = sum(int(row.get("validation_diagnostics", {}).get("foreign_candidate_output_count", 0)) for row in [*valid, *errors])
    report = {
        "schema_version": "1.0", "validation_policy": "candidate_task_v3",
        "hard_validation_contract_version": hard_validation_contract_version, "request_count": len(requests),
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


def validate_independent_calibration_reference(
    reference_path: Path, requests_path: Path, reference_manifest_path: Path, output_path: Path,
) -> dict[str, Any]:
    references = read_jsonl(reference_path)
    requests = read_jsonl(requests_path)
    manifest = json.loads(reference_manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("human_gold") is not False:
        errors.append("reference manifest must explicitly set human_gold=false")
    if manifest.get("frozen_before_reviewer_output") is not True:
        errors.append("reference was not frozen before reviewer output")
    if manifest.get("reference_sha256") != hashlib.sha256(reference_path.read_bytes()).hexdigest():
        errors.append("reference SHA-256 does not match its frozen manifest")
    if manifest.get("source_requests_sha256") != hashlib.sha256(requests_path.read_bytes()).hexdigest():
        errors.append("request SHA-256 does not match the reference manifest")
    by_request = {row.get("request_id"): row for row in requests}
    if len(by_request) != len(requests) or None in by_request:
        errors.append("requests require unique non-empty request IDs")
    seen_requests: set[str] = set()
    resolution_count = 0
    decision_counts = {"match": 0, "no_candidate_match": 0}
    for row in references:
        request_id = row.get("request_id")
        if request_id in seen_requests:
            errors.append(f"duplicate reference request {request_id}")
        seen_requests.add(request_id)
        request = by_request.get(request_id)
        if request is None:
            errors.append(f"reference request {request_id!r} is outside source requests")
            continue
        if row.get("human_gold") is not False:
            errors.append(f"{request_id}: reference must explicitly set human_gold=false")
        if row.get("request") != request:
            errors.append(f"{request_id}: self-contained request differs from frozen source request")
        resolutions = row.get("reference_resolutions")
        if not isinstance(resolutions, list):
            errors.append(f"{request_id}: reference_resolutions must be a list")
            continue
        expected_subtitles = list(request.get("subtitles", []))
        if [item.get("subtitle_id") for item in resolutions] != [item.get("subtitle_id") for item in expected_subtitles]:
            errors.append(f"{request_id}: subtitle IDs are missing, duplicated, or reordered")
        allowed = {item.get("block_id") for item in request.get("dialogue_candidates", [])}
        for index, resolution in enumerate(resolutions):
            resolution_count += 1
            decision = resolution.get("decision")
            block_ids = resolution.get("block_ids")
            if decision not in decision_counts:
                errors.append(f"{request_id}: invalid reference decision {decision!r}")
                continue
            decision_counts[decision] += 1
            if not isinstance(block_ids, list) or len(block_ids) != len(set(block_ids)):
                errors.append(f"{request_id}: block IDs must be a unique list")
                continue
            foreign = sorted(set(block_ids) - allowed)
            if foreign:
                errors.append(f"{request_id}: foreign candidate IDs {foreign}")
            if decision == "match" and not block_ids:
                errors.append(f"{request_id}: match requires at least one block ID")
            if decision == "no_candidate_match" and block_ids:
                errors.append(f"{request_id}: no_candidate_match requires empty block IDs")
            if index < len(expected_subtitles) and resolution.get("subtitle_text") != expected_subtitles[index].get("text"):
                errors.append(f"{request_id}: subtitle text differs from frozen request")
    missing = sorted(set(by_request) - seen_requests)
    if missing:
        errors.append(f"missing reference requests: {missing}")
    if len(references) != manifest.get("request_count"):
        errors.append("reference request count differs from manifest")
    if resolution_count != manifest.get("subtitle_resolution_count"):
        errors.append("reference resolution count differs from manifest")
    if decision_counts != manifest.get("decision_counts"):
        errors.append("reference decision counts differ from manifest")
    result = {
        "schema_version": "1.0", "validation_policy": "independent_calibration_reference_v1",
        "request_count": len(references), "resolution_count": resolution_count,
        "decision_counts": decision_counts, "error_count": len(errors), "errors": errors,
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "requests_sha256": hashlib.sha256(requests_path.read_bytes()).hexdigest(),
        "passed": not errors,
    }
    _write_json(output_path, result)
    return result


def evaluate_independent_calibration_v3(
    reference_path: Path, validated_path: Path, pilot_manifest_path: Path,
    response_validation_path: Path, output_path: Path,
) -> dict[str, Any]:
    references = read_jsonl(reference_path)
    responses = read_jsonl(validated_path)
    pilot_manifest = json.loads(pilot_manifest_path.read_text(encoding="utf-8"))
    validation = json.loads(response_validation_path.read_text(encoding="utf-8"))
    predicted = {(row["request_id"], item["subtitle_id"]): item for row in responses for item in row["resolutions"]}
    selection = {row["request_id"]: row for row in pilot_manifest["requests"]}
    records: list[dict[str, Any]] = []
    for row in references:
        for expected in row["reference_resolutions"]:
            records.append({
                "request_id": row["request_id"], "expected": expected,
                "actual": predicted.get((row["request_id"], expected["subtitle_id"])),
                **selection[row["request_id"]],
            })
    metrics = _binary_metrics(records)
    missing_count = sum(record["actual"] is None for record in records)
    checks = {
        "all_30_requests_structurally_valid": validation.get("valid_count") == 30 and validation.get("invalid_count") == 0,
        "zero_missing_predictions": missing_count == 0,
        "zero_foreign_candidates": int(validation.get("foreign_candidate_output_count", 0)) == 0,
        "candidate_task_accuracy_at_least_0_90": metrics["candidate_task_accuracy"] >= 0.90,
        "candidate_presence_accuracy_at_least_0_90": metrics["candidate_presence_decision_accuracy"] >= 0.90,
    }
    result = {
        "schema_version": "1.0", "decision_schema_version": "candidate_task_v3",
        "evaluation_role": "independent_calibration", "human_gold": False,
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "reference_resolution_count": len(records), "metrics": metrics,
        "structural_invalid_request_count": int(validation.get("invalid_count", 0)),
        "missing_prediction_count": missing_count,
        "foreign_candidate_output_count": int(validation.get("foreign_candidate_output_count", 0)),
        "sequence_quality": validation.get("sequence_quality", {}),
        "numeric_acceptance_gate": {"checks": checks, "passed": all(checks.values())},
        "promotion_requires_error_class_audit": True,
    }
    _write_json(output_path, result)
    return result
