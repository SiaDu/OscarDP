from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oscardp.shots.schema import json_dumps

from .openai_review import _submit_validated_batch, apply_validated_responses
from .openai_schema import V32_POLICY_SYSTEM_INSTRUCTIONS, alignment_response_schema_v3
from .pipeline import _write_json, _write_jsonl
from .schema import read_jsonl
from .stage23 import _unique_ids, _validate_request_subtitles, prepare_remaining_requests
from .stage253 import (
    REQUEST_PREFIX_V32_POLICY,
    batch_line_v32_policy,
    validate_batch_lines_v32_policy,
    validate_resolution_v3,
)


PRODUCTION_REVIEWER_VERSION = "v3.2-production.1"
PRODUCTION_OUTPUT_TAG = "v3_2_production_1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_reviewer_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("production_reviewer_version") != PRODUCTION_REVIEWER_VERSION:
        raise ValueError("production reviewer manifest has the wrong version")
    if manifest.get("status") != "promoted_frozen":
        raise ValueError("production reviewer is not promoted and frozen")
    prompt_hash = hashlib.sha256(V32_POLICY_SYSTEM_INSTRUCTIONS.encode("utf-8")).hexdigest()
    if manifest.get("prompt_sha256") != prompt_hash:
        raise ValueError("production reviewer prompt hash differs from code")
    if manifest.get("decision_schema_version") != "candidate_task_v3":
        raise ValueError("production reviewer decision schema is not candidate_task_v3")
    return manifest


def prepare_production_remaining_v3(
    full_requests_path: Path, pilot_requests_path: Path, output_path: Path,
    manifest_path: Path, reviewer_manifest_path: Path,
) -> dict[str, Any]:
    _load_reviewer_manifest(reviewer_manifest_path)
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite versioned production remaining artifacts")
    result = prepare_remaining_requests(
        full_requests_path, pilot_requests_path, output_path, manifest_path, "gpt-5.6-terra",
    )
    result.update({
        "lifecycle_schema_version": "v3_production_1",
        "production_reviewer_version": PRODUCTION_REVIEWER_VERSION,
        "decision_schema_version": "candidate_task_v3",
        "reviewer_manifest": reviewer_manifest_path.resolve().as_posix(),
        "reviewer_manifest_sha256": _sha(reviewer_manifest_path),
    })
    _write_json(manifest_path, result)
    return result


def prepare_production_batch_v3(
    requests_path: Path, reviewer_manifest_path: Path, output_path: Path,
) -> dict[str, Any]:
    reviewer = _load_reviewer_manifest(reviewer_manifest_path)
    if output_path.exists() or output_path.with_suffix(output_path.suffix + ".manifest.json").exists():
        raise FileExistsError("refusing to overwrite versioned production Batch artifacts")
    requests = read_jsonl(requests_path)
    ids = _unique_ids(requests, "request_id", "production requests")
    _validate_request_subtitles(requests, "production requests")
    rows = [batch_line_v32_policy(request, reviewer["model"]) for request in requests]
    errors = validate_batch_lines_v32_policy(rows)
    if errors:
        raise ValueError("Invalid production v3 Batch input: " + "; ".join(errors))
    _write_jsonl(output_path, rows)
    schema_hashes = {
        request["request_id"]: hashlib.sha256(
            json_dumps(alignment_response_schema_v3(request)).encode("utf-8")
        ).hexdigest()
        for request in requests
    }
    manifest = {
        "schema_version": "1.0", "lifecycle_schema_version": "v3_production_1",
        "production_reviewer_version": PRODUCTION_REVIEWER_VERSION,
        "decision_schema_version": "candidate_task_v3", "model": reviewer["model"],
        "request_count": len(rows), "request_ids": ids,
        "source_requests": requests_path.resolve().as_posix(), "source_requests_sha256": _sha(requests_path),
        "reviewer_manifest": reviewer_manifest_path.resolve().as_posix(),
        "reviewer_manifest_sha256": _sha(reviewer_manifest_path),
        "instructions_sha256": hashlib.sha256(V32_POLICY_SYSTEM_INSTRUCTIONS.encode("utf-8")).hexdigest(),
        "request_prefix_sha256": hashlib.sha256(REQUEST_PREFIX_V32_POLICY.encode("utf-8")).hexdigest(),
        "schema_sha256_by_request": schema_hashes,
        "schema_sha256_aggregate": hashlib.sha256(json_dumps(schema_hashes).encode("utf-8")).hexdigest(),
        "batch_input_sha256": _sha(output_path), "generated_at": datetime.now(UTC).isoformat(),
    }
    _write_json(output_path.with_suffix(output_path.suffix + ".manifest.json"), manifest)
    return manifest


def split_production_requests_v3(
    requests_path: Path, reviewer_manifest_path: Path, output_dir: Path,
    *, max_estimated_tokens: int = 300_000, max_requests: int = 100,
) -> dict[str, Any]:
    reviewer = _load_reviewer_manifest(reviewer_manifest_path)
    if max_estimated_tokens < 10_000 or max_estimated_tokens > 800_000:
        raise ValueError("max_estimated_tokens must be between 10000 and 800000")
    if max_requests < 1 or max_requests > 50_000:
        raise ValueError("max_requests must be between 1 and 50000")
    manifest_path = output_dir / "chunk_manifest.v3_2_production_1.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("refusing to overwrite production request chunks")
    requests = read_jsonl(requests_path)
    _unique_ids(requests, "request_id", "production chunk source")
    _validate_request_subtitles(requests, "production chunk source")

    def estimate(request: dict[str, Any]) -> int:
        # JSON is mostly ASCII. Dividing exact request-line bytes by three is
        # deliberately more conservative than the usual four-bytes/token rule.
        line = json_dumps(batch_line_v32_policy(request, reviewer["model"])) + "\n"
        return math.ceil(len(line.encode("utf-8")) / 3)

    chunks: list[list[tuple[dict[str, Any], int]]] = []
    current: list[tuple[dict[str, Any], int]] = []
    current_tokens = 0
    for request in requests:
        tokens = estimate(request)
        if tokens > max_estimated_tokens:
            raise ValueError(f"single request {request['request_id']} exceeds the chunk token budget")
        if current and (len(current) >= max_requests or current_tokens + tokens > max_estimated_tokens):
            chunks.append(current)
            current = []
            current_tokens = 0
        current.append((request, tokens))
        current_tokens += tokens
    if current:
        chunks.append(current)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, 1):
        path = output_dir / f"requests.chunk_{index:03d}.v3_2_production_1.jsonl"
        _write_jsonl(path, [item[0] for item in chunk])
        records.append({
            "chunk_index": index, "requests_path": path.resolve().as_posix(),
            "requests_sha256": _sha(path), "request_count": len(chunk),
            "resolution_count": sum(len(item[0]["subtitle_ids"]) for item in chunk),
            "conservative_estimated_enqueued_tokens": sum(item[1] for item in chunk),
            "first_request_id": chunk[0][0]["request_id"], "last_request_id": chunk[-1][0]["request_id"],
        })
    flattened = [item[0]["request_id"] for chunk in chunks for item in chunk]
    if flattened != [item["request_id"] for item in requests]:
        raise RuntimeError("production chunks do not preserve exact source order")
    manifest = {
        "schema_version": "1.0", "lifecycle_schema_version": "v3_production_1_chunked",
        "production_reviewer_version": PRODUCTION_REVIEWER_VERSION,
        "decision_schema_version": "candidate_task_v3", "model": reviewer["model"],
        "packing_policy": "exact_batch_line_utf8_bytes_divided_by_3_conservative_estimate",
        "max_estimated_tokens_per_chunk": max_estimated_tokens, "max_requests_per_chunk": max_requests,
        "source_requests": requests_path.resolve().as_posix(), "source_requests_sha256": _sha(requests_path),
        "reviewer_manifest": reviewer_manifest_path.resolve().as_posix(), "reviewer_manifest_sha256": _sha(reviewer_manifest_path),
        "request_count": len(requests), "resolution_count": sum(len(item["subtitle_ids"]) for item in requests),
        "chunk_count": len(records), "chunks": records, "generated_at": datetime.now(UTC).isoformat(),
    }
    _write_json(manifest_path, manifest)
    return manifest


def submit_production_batch_v3(
    batch_input_path: Path, reviewer_manifest_path: Path, job_path: Path, *, confirm_submit: bool,
) -> dict[str, Any]:
    if not confirm_submit:
        raise RuntimeError("Refusing paid production submission without --confirm-submit")
    _load_reviewer_manifest(reviewer_manifest_path)
    if job_path.exists():
        raise FileExistsError("refusing to overwrite a versioned production Batch job")
    batch_manifest_path = batch_input_path.with_suffix(batch_input_path.suffix + ".manifest.json")
    batch_manifest = json.loads(batch_manifest_path.read_text(encoding="utf-8"))
    if batch_manifest.get("production_reviewer_version") != PRODUCTION_REVIEWER_VERSION:
        raise ValueError("Batch manifest has the wrong production reviewer version")
    if batch_manifest.get("batch_input_sha256") != _sha(batch_input_path):
        raise ValueError("production Batch input hash differs from its manifest")
    if batch_manifest.get("reviewer_manifest_sha256") != _sha(reviewer_manifest_path):
        raise ValueError("production reviewer manifest hash differs from Batch manifest")
    rows = read_jsonl(batch_input_path)
    errors = validate_batch_lines_v32_policy(rows)
    if errors:
        raise ValueError("Invalid production v3 Batch input: " + "; ".join(errors))
    return _submit_validated_batch(
        batch_input_path, job_path, rows,
        metadata_extra={
            "schema_version": "1.0", "lifecycle_schema_version": "v3_production_1",
            "production_reviewer_version": PRODUCTION_REVIEWER_VERSION,
            "decision_schema_version": "candidate_task_v3",
            "review_policy_version": "annotation_policy_v1_plus_generic_v3.2_instructions",
            "batch_input_sha256": _sha(batch_input_path),
            "reviewer_manifest_sha256": _sha(reviewer_manifest_path),
        },
    )


def merge_production_responses_v3(
    full_requests_path: Path, pilot_requests_path: Path, remaining_requests_path: Path,
    pilot_responses_path: Path, remaining_responses_path: Path,
    reviewer_manifest_path: Path, output_path: Path, report_path: Path,
) -> dict[str, Any]:
    _load_reviewer_manifest(reviewer_manifest_path)
    if output_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite versioned merged production responses")
    full, pilot, remaining = (
        read_jsonl(full_requests_path), read_jsonl(pilot_requests_path), read_jsonl(remaining_requests_path)
    )
    full_ids = _unique_ids(full, "request_id", "full requests")
    pilot_ids = _unique_ids(pilot, "request_id", "pilot requests")
    remaining_ids = _unique_ids(remaining, "request_id", "remaining requests")
    if set(pilot_ids) & set(remaining_ids) or set(pilot_ids) | set(remaining_ids) != set(full_ids):
        raise ValueError("pilot and remaining requests are not an exact disjoint full-request partition")
    pilot_responses, remaining_responses = read_jsonl(pilot_responses_path), read_jsonl(remaining_responses_path)
    if set(_unique_ids(pilot_responses, "request_id", "pilot responses")) != set(pilot_ids):
        raise ValueError("pilot response request IDs differ from pilot requests")
    if set(_unique_ids(remaining_responses, "request_id", "remaining responses")) != set(remaining_ids):
        raise ValueError("remaining response request IDs differ from remaining requests")
    request_by_id = {row["request_id"]: row for row in full}
    response_by_id = {row["request_id"]: row for row in pilot_responses + remaining_responses}
    errors: list[dict[str, Any]] = []
    for request_id in full_ids:
        current, _diagnostics = validate_resolution_v3(response_by_id[request_id], request_by_id[request_id])
        if current:
            errors.append({"request_id": request_id, "errors": current})
    if errors:
        raise ValueError("production v3 response merge validation failed: " + json.dumps(errors, sort_keys=True))
    ordered = [response_by_id[request_id] for request_id in full_ids]
    _write_jsonl(output_path, ordered)
    report = {
        "schema_version": "1.0", "lifecycle_schema_version": "v3_production_1",
        "production_reviewer_version": PRODUCTION_REVIEWER_VERSION,
        "decision_schema_version": "candidate_task_v3", "request_count": len(full),
        "pilot_request_count": len(pilot), "remaining_request_count": len(remaining),
        "resolution_count": sum(len(row["resolutions"]) for row in ordered),
        "invalid_count": 0, "missing_count": 0, "foreign_count": 0,
        "source_hashes": {
            "full_requests": _sha(full_requests_path), "pilot_requests": _sha(pilot_requests_path),
            "remaining_requests": _sha(remaining_requests_path), "pilot_responses": _sha(pilot_responses_path),
            "remaining_responses": _sha(remaining_responses_path), "reviewer_manifest": _sha(reviewer_manifest_path),
        },
        "output_sha256": _sha(output_path), "passed": True,
    }
    _write_json(report_path, report)
    return report


def merge_production_response_chunks_v3(
    chunk_manifest_path: Path, response_paths: list[Path], reviewer_manifest_path: Path,
    output_path: Path, report_path: Path,
) -> dict[str, Any]:
    _load_reviewer_manifest(reviewer_manifest_path)
    if output_path.exists() or report_path.exists():
        raise FileExistsError("refusing to overwrite merged production chunk responses")
    manifest = json.loads(chunk_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("production_reviewer_version") != PRODUCTION_REVIEWER_VERSION:
        raise ValueError("chunk manifest has the wrong production reviewer version")
    if manifest.get("reviewer_manifest_sha256") != _sha(reviewer_manifest_path):
        raise ValueError("production reviewer manifest hash differs from chunk manifest")
    source_requests_path = Path(manifest["source_requests"])
    if manifest.get("source_requests_sha256") != _sha(source_requests_path):
        raise ValueError("source production requests hash differs from chunk manifest")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != len(response_paths):
        raise ValueError("response path count differs from production chunk count")
    all_requests: list[dict[str, Any]] = []
    all_responses: list[dict[str, Any]] = []
    response_hashes: list[str] = []
    for chunk, response_path in zip(chunks, response_paths):
        request_path = Path(chunk["requests_path"])
        if _sha(request_path) != chunk["requests_sha256"]:
            raise ValueError(f"chunk {chunk['chunk_index']} request hash changed")
        requests, responses = read_jsonl(request_path), read_jsonl(response_path)
        request_ids = _unique_ids(requests, "request_id", f"chunk {chunk['chunk_index']} requests")
        response_ids = _unique_ids(responses, "request_id", f"chunk {chunk['chunk_index']} responses")
        if set(request_ids) != set(response_ids):
            raise ValueError(f"chunk {chunk['chunk_index']} responses do not exactly cover requests")
        response_by_id = {row["request_id"]: row for row in responses}
        for request in requests:
            errors, _diagnostics = validate_resolution_v3(response_by_id[request["request_id"]], request)
            if errors:
                raise ValueError(f"chunk {chunk['chunk_index']} invalid v3 response {request['request_id']}: {errors}")
            all_requests.append(request)
            all_responses.append(response_by_id[request["request_id"]])
        response_hashes.append(_sha(response_path))
    if [row["request_id"] for row in all_requests] != [row["request_id"] for row in read_jsonl(source_requests_path)]:
        raise ValueError("chunk requests do not reconstruct source request order")
    if len(all_responses) != manifest["request_count"]:
        raise ValueError("merged chunk response count differs from manifest")
    _write_jsonl(output_path, all_responses)
    report = {
        "schema_version": "1.0", "lifecycle_schema_version": "v3_production_1_chunked",
        "production_reviewer_version": PRODUCTION_REVIEWER_VERSION,
        "decision_schema_version": "candidate_task_v3", "chunk_count": len(chunks),
        "request_count": len(all_responses), "resolution_count": sum(len(row["resolutions"]) for row in all_responses),
        "invalid_count": 0, "missing_count": 0, "foreign_count": 0,
        "chunk_manifest_sha256": _sha(chunk_manifest_path), "response_sha256_by_chunk": response_hashes,
        "output_sha256": _sha(output_path), "passed": True,
    }
    _write_json(report_path, report)
    return report


_BASIS_TO_HISTORICAL = {
    "exact_or_near_exact": "exact_or_near_exact", "paraphrase": "paraphrase",
    "expanded_or_contracted_turn": "paraphrase", "subtitle_fragment": "substring_or_minor_edit",
    "repeated_or_reordered_dialogue": "repeated_or_reordered_dialogue",
    "vocative_attachment": "substring_or_minor_edit", "no_supplied_candidate": "changed_or_improvised_dialogue",
    "other": "insufficient_context",
}


def apply_production_responses_v3(
    alignment_path: Path, requests_path: Path, validated_path: Path, context_path: Path,
    shots_path: Path, output_dir: Path, reviewer_manifest_path: Path,
) -> dict[str, Any]:
    _load_reviewer_manifest(reviewer_manifest_path)
    openai_dir = output_dir / "review" / "openai"
    normalized_path = openai_dir / f"validated_responses.{PRODUCTION_OUTPUT_TAG}.apply_normalized.jsonl"
    report_path = openai_dir / f"{PRODUCTION_OUTPUT_TAG}_apply_report.json"
    alignment_output = output_dir / f"subtitle_script_alignment.llm_reviewed_{PRODUCTION_OUTPUT_TAG}.jsonl"
    shot_output = output_dir / f"shot_script_context.llm_reviewed_{PRODUCTION_OUTPUT_TAG}.jsonl"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_sources = {"alignment": _sha(alignment_path), "requests": _sha(requests_path), "responses": _sha(validated_path), "context": _sha(context_path), "shots": _sha(shots_path)}
        if report.get("production_reviewer_version") != PRODUCTION_REVIEWER_VERSION or report.get("source_hashes") != expected_sources:
            raise RuntimeError("existing production apply report does not match requested immutable sources")
        if not alignment_output.is_file() or not shot_output.is_file():
            raise RuntimeError("existing production apply report has missing reviewed outputs")
        return {**report, "resumed": True}
    if any(path.exists() for path in (normalized_path, alignment_output, shot_output)):
        raise FileExistsError("refusing to overwrite incomplete versioned production apply artifacts")
    requests = read_jsonl(requests_path)
    _unique_ids(requests, "request_id", "production apply requests")
    request_by_id = {row["request_id"]: row for row in requests}
    responses = read_jsonl(validated_path)
    _unique_ids(responses, "request_id", "production apply responses")
    normalized: list[dict[str, Any]] = []
    for response in responses:
        request = request_by_id.get(response.get("request_id"))
        if request is None:
            raise ValueError(f"response outside requests: {response.get('request_id')}")
        errors, _diagnostics = validate_resolution_v3(response, request)
        if errors:
            raise ValueError(f"invalid production v3 response {response['request_id']}: {errors}")
        converted = deepcopy(response)
        for original, resolution in zip(response["resolutions"], converted["resolutions"]):
            resolution["openai_resolution"] = deepcopy(original)
            resolution["decision"] = "no_match" if original["decision"] == "no_candidate_match" else "match"
            resolution["decision_basis"] = _BASIS_TO_HISTORICAL[original["decision_basis"]]
        normalized.append(converted)
    if len(normalized) != len(requests) or set(row["request_id"] for row in normalized) != set(request_by_id):
        raise ValueError("production responses do not cover the complete request set")
    _write_jsonl(normalized_path, normalized)
    base = apply_validated_responses(
        alignment_path, requests_path, normalized_path, context_path, shots_path,
        output_dir, PRODUCTION_OUTPUT_TAG,
    )
    report = {
        **base, "lifecycle_schema_version": "v3_production_1",
        "production_reviewer_version": PRODUCTION_REVIEWER_VERSION,
        "decision_schema_version": "candidate_task_v3",
        "source_hashes": {"alignment": _sha(alignment_path), "requests": _sha(requests_path), "responses": _sha(validated_path), "context": _sha(context_path), "shots": _sha(shots_path)},
        "reviewer_manifest_sha256": _sha(reviewer_manifest_path),
        "normalized_responses": normalized_path.as_posix(), "normalized_responses_sha256": _sha(normalized_path),
        "alignment_output_sha256": _sha(alignment_output), "shot_output_sha256": _sha(shot_output),
        "resumed": False,
    }
    _write_json(report_path, report)
    return report
