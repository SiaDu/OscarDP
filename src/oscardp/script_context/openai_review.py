from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oscardp.shots.schema import json_dumps

from .llm_review import build_alignment_diagnostics
from .openai_schema import DECISION_BASES, DECISIONS, SYSTEM_INSTRUCTIONS, alignment_response_schema
from .pipeline import _write_atomic, _write_json, _write_jsonl
from .schema import read_jsonl
from .shot_mapping import map_shots
from .validation import validate_data


def _require_model(model: str | None) -> str:
    selected = model or os.environ.get("OPENAI_MODEL")
    if not selected or selected == "REQUIRE_EXPLICIT_MODEL":
        raise ValueError("OpenAI model is required via --model or OPENAI_MODEL")
    return selected


def batch_line(request: dict[str, Any], model: str) -> dict[str, Any]:
    return {
        "custom_id": request["request_id"], "method": "POST", "url": "/v1/responses",
        "body": {
            "model": model, "store": False, "instructions": SYSTEM_INSTRUCTIONS,
            "input": "Review this constrained alignment request:\n" + json_dumps(request, pretty=True),
            "text": {"format": {"type": "json_schema", "name": "alignment_review_response", "strict": True, "schema": alignment_response_schema()}},
        },
    }


def validate_batch_lines(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids: set[str] = set()
    for index, row in enumerate(rows, 1):
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or not custom_id:
            errors.append(f"line {index}: invalid custom_id")
        elif custom_id in ids:
            errors.append(f"line {index}: duplicate custom_id {custom_id}")
        ids.add(custom_id)
        if row.get("method") != "POST" or row.get("url") != "/v1/responses":
            errors.append(f"line {index}: batch endpoint must be POST /v1/responses")
        fmt = row.get("body", {}).get("text", {}).get("format", {})
        if fmt.get("type") != "json_schema" or fmt.get("strict") is not True or fmt.get("schema") != alignment_response_schema():
            errors.append(f"line {index}: strict response schema is missing")
    return errors


def prepare_batch(requests_path: Path, output_path: Path, model: str | None) -> dict[str, Any]:
    selected_model = _require_model(model)
    requests = read_jsonl(requests_path)
    ids = [request.get("request_id") for request in requests]
    if len(ids) != len(set(ids)) or any(not value for value in ids):
        raise ValueError("Requests must have unique non-empty request_id values")
    rows = [batch_line(request, selected_model) for request in requests]
    errors = validate_batch_lines(rows)
    if errors:
        raise ValueError("Invalid batch input: " + "; ".join(errors))
    _write_jsonl(output_path, rows)
    manifest = {
        "schema_version": "1.0", "source_requests": requests_path.resolve().as_posix(),
        "source_requests_sha256": hashlib.sha256(requests_path.read_bytes()).hexdigest(),
        "model": selected_model, "request_count": len(rows),
        "generated_at": datetime.now(UTC).isoformat(), "endpoint": "/v1/responses",
    }
    _write_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    return manifest


def _client() -> Any:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError('OpenAI SDK is required; install with pip install -e ".[openai]"') from exc
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def submit_batch(batch_input: Path, job_file: Path, *, confirm_submit: bool) -> dict[str, Any]:
    if not confirm_submit:
        raise RuntimeError("Refusing paid submission without --confirm-submit")
    rows = read_jsonl(batch_input)
    errors = validate_batch_lines(rows)
    if errors:
        raise ValueError("Invalid batch input: " + "; ".join(errors))
    client = _client()
    with batch_input.open("rb") as handle:
        uploaded = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(input_file_id=uploaded.id, endpoint="/v1/responses", completion_window="24h")
    metadata = {"schema_version": "1.0", "input_file_id": uploaded.id, "batch_id": batch.id, "status": batch.status, "request_count": len(rows), "model": rows[0]["body"]["model"] if rows else None, "submitted_at": datetime.now(UTC).isoformat()}
    _write_json(job_file, metadata)
    return metadata


def check_batch(job_file: Path) -> dict[str, Any]:
    metadata = json.loads(job_file.read_text(encoding="utf-8"))
    batch = _client().batches.retrieve(metadata["batch_id"])
    request_counts = getattr(batch, "request_counts", None)
    if request_counts is not None:
        model_dump = getattr(request_counts, "model_dump", None)
        if not callable(model_dump):
            raise TypeError("SDK batch request_counts does not support model_dump()")
        request_counts = model_dump()
        if not isinstance(request_counts, dict):
            raise TypeError("SDK batch request_counts.model_dump() must return a dictionary")
    return {"batch_id": batch.id, "status": batch.status, "output_file_id": batch.output_file_id, "error_file_id": batch.error_file_id, "request_counts": request_counts}


def _download_text(response: Any) -> str:
    text_method = getattr(response, "text", None)
    if callable(text_method):
        content = text_method()
    else:
        read_method = getattr(response, "read", None)
        if not callable(read_method):
            raise TypeError("SDK file content response supports neither text() nor read()")
        content = read_method()
    if isinstance(content, bytes):
        return content.decode("utf-8")
    if not isinstance(content, str):
        raise TypeError("SDK file content response must return str or bytes")
    return content


def _add_reviewed_status_counts(diagnostics: dict[str, Any], alignments: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = dict(sorted(Counter(row["alignment"]["status"] for row in alignments).items()))
    return {
        **diagnostics,
        "status_counts": status_counts,
        "llm_aligned": status_counts.get("llm_aligned", 0),
        "llm_no_match": status_counts.get("llm_no_match", 0),
        "status_total": sum(status_counts.values()),
    }


def fetch_batch(job_file: Path, output_dir: Path) -> dict[str, Any]:
    metadata = json.loads(job_file.read_text(encoding="utf-8")); client = _client()
    batch = client.batches.retrieve(metadata["batch_id"])
    if batch.status != "completed":
        raise RuntimeError(f"Batch is not completed: {batch.status}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if batch.output_file_id:
        content = _download_text(client.files.content(batch.output_file_id))
        _write_atomic(output_dir / "raw_batch_output.jsonl", content)
    if batch.error_file_id:
        errors = _download_text(client.files.content(batch.error_file_id))
        _write_atomic(output_dir / "raw_batch_errors.jsonl", errors)
    result = {"batch_id": batch.id, "status": batch.status, "output_file_id": batch.output_file_id, "error_file_id": batch.error_file_id, "fetched_at": datetime.now(UTC).isoformat()}
    _write_json(output_dir / "fetch_metadata.json", result)
    return result


def _extract_structured(body: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if body.get("status") not in {None, "completed"}:
        return None, f"response status is {body.get('status')}"
    texts: list[str] = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "refusal":
                return None, "model refusal"
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                texts.append(content["text"])
    if not texts and isinstance(body.get("output_text"), str):
        texts.append(body["output_text"])
    if len(texts) != 1:
        return None, "expected exactly one structured output text"
    try:
        return json.loads(texts[0]), None
    except json.JSONDecodeError as exc:
        return None, f"malformed structured output: {exc}"


def validate_resolution(response: dict[str, Any], request: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if response.get("request_id") != request.get("request_id"):
        errors.append("request_id mismatch")
    requested = request["subtitle_ids"]
    resolutions = response.get("resolutions")
    if not isinstance(resolutions, list):
        return errors + ["resolutions must be an array"]
    actual = [item.get("subtitle_id") for item in resolutions if isinstance(item, dict)]
    if len(actual) != len(set(actual)):
        errors.append("duplicate subtitle resolution")
    if actual != requested:
        errors.append("subtitle resolutions must exactly match request order")
    candidate_order = {item["block_id"]: index for index, item in enumerate(request.get("dialogue_candidates", []))}
    last_order = -1
    for item in resolutions:
        if not isinstance(item, dict):
            errors.append("resolution must be an object"); continue
        decision, block_ids, confidence, basis = item.get("decision"), item.get("block_ids"), item.get("confidence"), item.get("decision_basis")
        if decision not in DECISIONS:
            errors.append(f"invalid decision for {item.get('subtitle_id')}")
        if basis not in DECISION_BASES:
            errors.append(f"invalid decision_basis for {item.get('subtitle_id')}")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
            errors.append(f"invalid confidence for {item.get('subtitle_id')}")
        if not isinstance(block_ids, list) or len(block_ids) != len(set(block_ids)):
            errors.append(f"block_ids must be a unique array for {item.get('subtitle_id')}"); continue
        if decision == "match" and not block_ids:
            errors.append(f"match requires block_ids for {item.get('subtitle_id')}")
        if decision in {"no_match", "uncertain"} and block_ids:
            errors.append(f"{decision} requires empty block_ids for {item.get('subtitle_id')}")
        if any(block_id not in candidate_order for block_id in block_ids):
            errors.append(f"block outside request candidates for {item.get('subtitle_id')}"); continue
        orders = [candidate_order[block_id] for block_id in block_ids]
        if orders != sorted(orders) or any(right != left + 1 for left, right in zip(orders, orders[1:])):
            errors.append(f"selected blocks must be ordered and adjacent for {item.get('subtitle_id')}")
        scenes = {next(candidate["scene_id"] for candidate in request["dialogue_candidates"] if candidate["block_id"] == block_id) for block_id in block_ids}
        if len(scenes) > 1:
            errors.append(f"selected blocks cross scenes for {item.get('subtitle_id')}")
        if orders and orders[0] < last_order:
            errors.append("non-monotonic block selection")
        if orders:
            last_order = orders[-1]
    return errors


def validate_responses(raw_path: Path, requests_path: Path, output_dir: Path) -> dict[str, Any]:
    requests = read_jsonl(requests_path); request_by_id = {item["request_id"]: item for item in requests}
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
        resolution_errors = validate_resolution(structured, request_by_id[custom_id])  # type: ignore[arg-type]
        if resolution_errors:
            errors.append({"line": line_number, "custom_id": custom_id, "errors": resolution_errors}); continue
        valid.append({**structured, "custom_id": custom_id, "response_id": body.get("id"), "model": body.get("model")})  # type: ignore[arg-type]
    missing = sorted(set(request_by_id) - seen)
    report = {"schema_version": "1.0", "request_count": len(requests), "valid_count": len(valid), "invalid_count": len(errors), "missing_request_ids": missing, "errors": errors, "passed": not errors and not missing}
    output_dir.mkdir(parents=True, exist_ok=True); _write_jsonl(output_dir / "validated_responses.jsonl", valid); _write_json(output_dir / "response_validation_report.json", report)
    return report


def apply_validated_responses(
    alignment_path: Path, requests_path: Path, validated_path: Path, context_path: Path,
    shots_path: Path, output_dir: Path,
) -> dict[str, Any]:
    baseline_hashes = {path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in (alignment_path, requests_path, context_path, shots_path)}
    baseline, requests, responses = read_jsonl(alignment_path), read_jsonl(requests_path), read_jsonl(validated_path)
    context = json.loads(context_path.read_text(encoding="utf-8")); shots = read_jsonl(shots_path)
    request_by_id = {item["request_id"]: item for item in requests}; row_by_subtitle = {row["subtitle_id"]: row for row in deepcopy(baseline)}
    block_lookup: dict[str, tuple[int, dict[str, Any], dict[str, Any]]] = {}; order = 0
    for scene in context["script_scenes"]:
        for block in scene["script_blocks"]:
            if block["block_type"] == "dialogue":
                block_lookup[block["block_id"]] = (order, scene, block); order += 1
    changed: set[str] = set()
    for response in responses:
        request = request_by_id.get(response["request_id"])
        if request is None or validate_resolution(response, request):
            raise ValueError(f"Validated response no longer validates: {response.get('request_id')}")
        for resolution in response["resolutions"]:
            subtitle_id = resolution["subtitle_id"]; row = row_by_subtitle[subtitle_id]; original = deepcopy(row)
            decision, block_ids = resolution["decision"], resolution["block_ids"]
            audit = {"resolver": "openai_responses_batch", "model": response.get("model"), "request_id": response["request_id"], "response_id": response.get("response_id"), "confidence": resolution["confidence"], "decision": decision, "decision_basis": resolution["decision_basis"], "original_automatic": original}
            if decision == "match":
                selected = [block_lookup[block_id] for block_id in block_ids]
                row["scene_id"] = selected[0][1]["scene_id"]
                row["script_matches"] = [{"block_id": block["block_id"], "speaker": block["speaker"], "matched_text": block["text"], "lexical_score": None, "semantic_score": None, "combined_score": resolution["confidence"], "screenplay_order": block_order} for block_order, _scene, block in selected]
                row["alignment"] = {"method": "openai_reviewed", "status": "llm_aligned", "candidate_margin": 0.0, "needs_review": False, "reliable_anchor": False, "script_order_start": selected[0][0], "script_order_end": selected[-1][0], "llm_resolution": audit}
            elif decision == "no_match":
                row["scene_id"] = None; row["script_matches"] = []
                row["alignment"] = {"method": "openai_reviewed", "status": "llm_no_match", "candidate_margin": 0.0, "needs_review": False, "reliable_anchor": False, "script_order_start": None, "script_order_end": None, "llm_resolution": audit}
            else:
                row["alignment"] = {**row["alignment"], "status": "needs_review", "needs_review": True, "reliable_anchor": False, "llm_resolution": audit}
            changed.add(subtitle_id)
    reviewed = [row_by_subtitle[row["subtitle_id"]] for row in baseline]
    reviewed_shots = map_shots(shots, reviewed, context, baseline[0]["movie_id"])
    validation = validate_data(context, reviewed, reviewed_shots, shots)
    if not validation.passed:
        raise RuntimeError("Reviewed output validation failed: " + "; ".join(validation.errors[:20]))
    alignment_output = output_dir / "subtitle_script_alignment.llm_reviewed.jsonl"; shot_output = output_dir / "shot_script_context.llm_reviewed.jsonl"
    _write_jsonl(alignment_output, reviewed); _write_jsonl(shot_output, reviewed_shots)
    diagnostics = _add_reviewed_status_counts(build_alignment_diagnostics(context, reviewed, requests), reviewed)
    openai_dir = output_dir / "review" / "openai"; _write_json(openai_dir / "reviewed_alignment_diagnostics.json", diagnostics)
    unchanged = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest for path, digest in baseline_hashes.items())
    report = {"schema_version": "1.0", "changed_subtitle_count": len(changed), "alignment_rows": len(reviewed), "shot_rows": len(reviewed_shots), "validation_passed": validation.passed, "baseline_files_unchanged": unchanged, "alignment_output": alignment_output.as_posix(), "shot_output": shot_output.as_posix()}
    if not unchanged:
        raise RuntimeError("Deterministic baseline changed during reviewed application")
    _write_json(openai_dir / "apply_report.json", report)
    return report


def evaluate_pilot(gold_path: Path, validated_path: Path, manifest_path: Path, output_path: Path) -> dict[str, Any]:
    gold, responses = read_jsonl(gold_path), read_jsonl(validated_path); manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if any(item.get("decision") is None or item.get("block_ids") is None for row in gold for item in row["resolutions"]):
        raise ValueError("Pilot gold contains null labels; accuracy cannot be computed")
    predicted = {(row["request_id"], item["subtitle_id"]): item for row in responses for item in row["resolutions"]}
    strata = {row["request_id"]: row for row in manifest["requests"]}; records = []
    for row in gold:
        for expected in row["resolutions"]:
            actual = predicted.get((row["request_id"], expected["subtitle_id"]))
            records.append({"request_id": row["request_id"], "expected": expected, "actual": actual, **strata[row["request_id"]]})
    decision_correct = sum(bool(record["actual"] and record["actual"]["decision"] == record["expected"]["decision"]) for record in records)
    block_correct = sum(bool(record["actual"] and set(record["actual"]["block_ids"]) == set(record["expected"]["block_ids"])) for record in records)
    def accuracy(field: str, value: str) -> float | None:
        subset = [record for record in records if record[field] == value]
        return None if not subset else sum(bool(record["actual"] and set(record["actual"]["block_ids"]) == set(record["expected"]["block_ids"])) for record in subset) / len(subset)
    decisions = ("match", "no_match", "uncertain")
    confusion = {expected: {actual: 0 for actual in (*decisions, "missing")} for expected in decisions}
    for record in records:
        expected = record["expected"]["decision"]
        actual = "missing" if record["actual"] is None else record["actual"]["decision"]
        confusion[expected][actual] += 1
    missing_count = sum(record["actual"] is None for record in records)
    validation_report_path = validated_path.with_name("response_validation_report.json")
    validation_report = json.loads(validation_report_path.read_text(encoding="utf-8")) if validation_report_path.is_file() else {}
    invalid_count = int(validation_report.get("invalid_count", 0))
    multi = [record for record in records if len(record["expected"]["block_ids"]) > 1]
    multi_accuracy = None if not multi else sum(bool(record["actual"] and set(record["actual"]["block_ids"]) == set(record["expected"]["block_ids"])) for record in multi) / len(multi)
    predicted_no_match = sum(bool(record["actual"] and record["actual"]["decision"] == "no_match") for record in records)
    expected_no_match = sum(record["expected"]["decision"] == "no_match" for record in records)
    true_no_match = sum(bool(record["actual"] and record["actual"]["decision"] == "no_match" and record["expected"]["decision"] == "no_match") for record in records)
    no_match_precision = None if not predicted_no_match else true_no_match / predicted_no_match
    no_match_recall = None if not expected_no_match else true_no_match / expected_no_match
    total = len(records)
    overall_accuracy = block_correct / total if total else 0.0
    easy_accuracy = accuracy("stratum", "easy")
    criteria = {
        "zero_invalid_responses": invalid_count == 0,
        "zero_missing_predictions": missing_count == 0,
        "easy_block_set_accuracy_at_least_0_95": easy_accuracy is not None and easy_accuracy >= 0.95,
        "overall_block_set_accuracy_at_least_0_90": overall_accuracy >= 0.90,
    }
    result = {
        "schema_version": "1.0", "subtitle_count": total,
        "exact_decision_accuracy": decision_correct / total if total else 0.0,
        "decision_confusion_matrix": confusion,
        "block_set_exact_match": overall_accuracy,
        "multi_block_block_set_accuracy": multi_accuracy,
        "accuracy_by_stratum": {name: accuracy("stratum", name) for name in ("easy", "fuzzy", "multi", "difficult")},
        "accuracy_by_region": {name: accuracy("timeline_region", name) for name in ("early", "middle", "late")},
        "invalid_response_count": invalid_count, "missing_prediction_count": missing_count,
        "no_match_precision": no_match_precision, "no_match_recall": no_match_recall,
        "uncertain_rate": sum(bool(record["actual"] and record["actual"]["decision"] == "uncertain") for record in records) / total if total else 0.0,
        "acceptance_criteria": {"checks": criteria, "passed": all(criteria.values())},
    }
    _write_json(output_path, result); return result
