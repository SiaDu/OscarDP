from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .alignment import tokenize
from .openai_review import validate_resolution
from .pilot import _region, _stratum
from .pipeline import _write_json, _write_jsonl
from .schema import read_jsonl


def _unique_ids(rows: list[dict[str, Any]], field: str, label: str) -> list[str]:
    values = [row.get(field) for row in rows]
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{label} contains an invalid {field}")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate {field} values")
    return values  # type: ignore[return-value]


def _validate_request_subtitles(requests: list[dict[str, Any]], label: str) -> list[str]:
    subtitle_ids = [subtitle_id for request in requests for subtitle_id in request.get("subtitle_ids", [])]
    if any(not isinstance(value, str) or not value for value in subtitle_ids):
        raise ValueError(f"{label} contains an invalid subtitle_id")
    if len(subtitle_ids) != len(set(subtitle_ids)):
        raise ValueError(f"{label} contains duplicate subtitle IDs")
    return subtitle_ids


def prepare_remaining_requests(
    full_requests_path: Path, pilot_requests_path: Path, output_path: Path,
    manifest_path: Path, model: str,
) -> dict[str, Any]:
    full, pilot = read_jsonl(full_requests_path), read_jsonl(pilot_requests_path)
    full_ids = _unique_ids(full, "request_id", "full requests")
    pilot_ids = _unique_ids(pilot, "request_id", "pilot requests")
    _validate_request_subtitles(full, "full requests")
    pilot_subtitles = _validate_request_subtitles(pilot, "pilot requests")
    full_set, pilot_set = set(full_ids), set(pilot_ids)
    foreign = sorted(pilot_set - full_set)
    if foreign:
        raise ValueError(f"Pilot contains requests outside full set: {foreign}")
    remaining = [request for request in full if request["request_id"] not in pilot_set]
    remaining_ids = [request["request_id"] for request in remaining]
    remaining_subtitles = _validate_request_subtitles(remaining, "remaining requests")
    if pilot_set & set(remaining_ids) or pilot_set | set(remaining_ids) != full_set:
        raise ValueError("Pilot and remaining requests are not a disjoint complete partition")
    if len(pilot_subtitles) + len(remaining_subtitles) != len(_validate_request_subtitles(full, "full requests")):
        raise ValueError("Pilot and remaining subtitle counts do not cover the full set")
    _write_jsonl(output_path, remaining)
    remaining_sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "1.0", "model": model,
        "generated_at": datetime.now(UTC).isoformat(),
        "full_requests_sha256": hashlib.sha256(full_requests_path.read_bytes()).hexdigest(),
        "pilot_requests_sha256": hashlib.sha256(pilot_requests_path.read_bytes()).hexdigest(),
        "remaining_requests_sha256": remaining_sha,
        "counts": {
            "full_requests": len(full), "pilot_requests": len(pilot), "remaining_requests": len(remaining),
            "full_subtitles": sum(len(row["subtitle_ids"]) for row in full),
            "pilot_subtitles": len(pilot_subtitles), "remaining_subtitles": len(remaining_subtitles),
        },
        "excluded_pilot_request_ids": [request_id for request_id in full_ids if request_id in pilot_set],
    }
    _write_json(manifest_path, manifest)
    return manifest


def merge_validated_responses(
    full_requests_path: Path, pilot_responses_path: Path, remaining_responses_path: Path,
    output_path: Path, report_path: Path,
) -> dict[str, Any]:
    requests = read_jsonl(full_requests_path)
    request_ids = _unique_ids(requests, "request_id", "full requests")
    _validate_request_subtitles(requests, "full requests")
    responses = read_jsonl(pilot_responses_path) + read_jsonl(remaining_responses_path)
    response_ids = _unique_ids(responses, "request_id", "validated responses")
    expected = set(request_ids)
    foreign = sorted(set(response_ids) - expected)
    missing = sorted(expected - set(response_ids))
    errors: list[dict[str, Any]] = []
    if foreign:
        errors.append({"type": "foreign_requests", "request_ids": foreign})
    if missing:
        errors.append({"type": "missing_requests", "request_ids": missing})
    response_by_id = {response["request_id"]: response for response in responses}
    request_by_id = {request["request_id"]: request for request in requests}
    for request_id in request_ids:
        if request_id not in response_by_id:
            continue
        validation_errors = validate_resolution(response_by_id[request_id], request_by_id[request_id])
        if validation_errors:
            errors.append({"type": "invalid_response", "request_id": request_id, "errors": validation_errors})
    if errors:
        raise ValueError("Full response merge validation failed: " + json.dumps(errors, sort_keys=True))
    ordered = [response_by_id[request_id] for request_id in request_ids]
    subtitle_count = sum(len(response["resolutions"]) for response in ordered)
    _write_jsonl(output_path, ordered)
    report = {
        "schema_version": "1.0", "request_count": len(requests), "valid_count": len(ordered),
        "invalid_count": 0, "subtitle_count": subtitle_count, "missing": 0, "duplicate": 0,
        "passed": True, "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    }
    _write_json(report_path, report)
    return report


def _candidate_tokens(candidate: dict[str, Any]) -> set[str]:
    return set(tokenize(candidate.get("text", "")).tokens)


def _meaningful_overlap(text_tokens: set[str], candidate: dict[str, Any]) -> bool:
    overlap = text_tokens & _candidate_tokens(candidate)
    return len(overlap) >= 2 or bool(overlap) and len(text_tokens) <= 4


def build_composite_audit(
    requests_path: Path, responses_path: Path, output_path: Path, summary_path: Path,
    total_subtitles: int = 1839,
) -> dict[str, Any]:
    requests, responses = read_jsonl(requests_path), read_jsonl(responses_path)
    response_by_id = {response["request_id"]: response for response in responses}
    audit: list[dict[str, Any]] = []
    for request in requests:
        response = response_by_id.get(request["request_id"])
        if response is None:
            raise ValueError(f"Missing response for audit request {request['request_id']}")
        subtitles = {row["subtitle_id"]: row for row in request["subtitles"]}
        automatic = {row["subtitle_id"]: row for row in request.get("automatic_candidate_mappings", [])}
        candidates = request.get("dialogue_candidates", [])
        candidate_by_id = {row["block_id"]: row for row in candidates}
        stratum = _stratum(request)
        for resolution in response["resolutions"]:
            subtitle_id = resolution["subtitle_id"]
            text = subtitles[subtitle_id]["text"]
            text_tokens = set(tokenize(text).tokens)
            auto_blocks = [match["block_id"] for match in automatic.get(subtitle_id, {}).get("matches", [])]
            selected = resolution["block_ids"]
            flags: list[str] = []
            dash_turns = len(re.findall(r"(?:^|\s)-\s*", text)) >= 2 or bool(re.search(r"\s-\s", text))
            if dash_turns:
                flags.append("multiple_dialogue_turns")
            if len(auto_blocks) > len(selected):
                flags.append("automatic_mapping_has_more_blocks")
            if stratum == "multi":
                flags.append("multi_request")
            adjacent_overlap = False
            for left, right in zip(candidates, candidates[1:]):
                if right["screenplay_order"] == left["screenplay_order"] + 1 and _meaningful_overlap(text_tokens, left) and _meaningful_overlap(text_tokens, right):
                    adjacent_overlap = True
                    break
            if len(selected) == 1 and adjacent_overlap:
                flags.append("one_block_with_two_adjacent_overlaps")
            if len(candidates) >= 2:
                first_tokens = tokenize(candidates[0].get("text", "")).tokens
                last_tokens = tokenize(candidates[-1].get("text", "")).tokens
                subtitle_tokens = tokenize(text).tokens
                if first_tokens and last_tokens and subtitle_tokens and set(subtitle_tokens[:4]) & set(first_tokens[-6:]) and set(subtitle_tokens[-4:]) & set(last_tokens[:6]):
                    flags.append("candidate_boundary_spanning_fragment")
            if resolution["decision"] == "uncertain":
                flags.append("uncertain")
            if float(resolution["confidence"]) < 0.80:
                flags.append("low_confidence")
            if dash_turns and adjacent_overlap:
                flags.append("multiple_fragments_match_adjacent_candidates")
            if not flags:
                continue
            selected_orders = [candidate_by_id[block_id]["screenplay_order"] for block_id in selected if block_id in candidate_by_id]
            audit.append({
                "request_id": request["request_id"], "subtitle_id": subtitle_id, "subtitle_text": text,
                "automatic_blocks": auto_blocks, "openai_selected_blocks": selected,
                "candidate_block_texts": [{"block_id": row["block_id"], "text": row["text"], "screenplay_order": row["screenplay_order"]} for row in candidates],
                "confidence": resolution["confidence"], "decision": resolution["decision"],
                "decision_basis": resolution["decision_basis"], "reason_flags": flags,
                "screenplay_order": selected_orders, "stratum": stratum,
                "timeline_region": _region(request, total_subtitles),
            })
    _write_jsonl(output_path, audit)
    summary = {
        "schema_version": "1.0", "total_flagged": len(audit),
        "multi_request_count": len({row["request_id"] for row in audit if "multi_request" in row["reason_flags"]}),
        "one_block_underselection_count": sum(
            len(row["openai_selected_blocks"]) == 1
            and ("one_block_with_two_adjacent_overlaps" in row["reason_flags"] or "automatic_mapping_has_more_blocks" in row["reason_flags"])
            for row in audit
        ),
        "uncertain_count": sum("uncertain" in row["reason_flags"] for row in audit),
        "low_confidence_count": sum("low_confidence" in row["reason_flags"] for row in audit),
    }
    _write_json(summary_path, summary)
    return summary


def build_human_audit_sample(
    requests_path: Path, responses_path: Path, composite_path: Path,
    output_path: Path, manifest_path: Path, total_subtitles: int = 1839,
) -> dict[str, Any]:
    requests, responses, composite = read_jsonl(requests_path), read_jsonl(responses_path), read_jsonl(composite_path)
    request_by_id = {request["request_id"]: request for request in requests}
    response_by_id = {response["request_id"]: response for response in responses}
    composite_keys = {(row["request_id"], row["subtitle_id"]) for row in composite}
    records: list[dict[str, Any]] = []
    for request in requests:
        response = response_by_id[request["request_id"]]
        subtitles = {row["subtitle_id"]: row for row in request["subtitles"]}
        for resolution in response["resolutions"]:
            key = (request["request_id"], resolution["subtitle_id"])
            reasons: list[str] = []
            if resolution["decision"] == "uncertain": reasons.append("uncertain")
            if key in composite_keys: reasons.append("composite_risk")
            if resolution["decision"] == "match" and len(resolution["block_ids"]) > 1: reasons.append("openai_multi_block")
            if resolution["decision"] == "no_match": reasons.append("openai_no_match")
            records.append({
                "request_id": request["request_id"], "subtitle_id": resolution["subtitle_id"],
                "subtitle_text": subtitles[resolution["subtitle_id"]]["text"], "resolution": resolution,
                "stratum": _stratum(request), "timeline_region": _region(request, total_subtitles),
                "inclusion_reasons": reasons,
            })
    selected = [record for record in records if record["inclusion_reasons"]]
    selected_keys = {(record["request_id"], record["subtitle_id"]) for record in selected}
    sample_counts: Counter[str] = Counter()
    for category in ("easy", "fuzzy", "early", "middle", "late"):
        matches = [record for record in records if (record["stratum"] == category or record["timeline_region"] == category) and (record["request_id"], record["subtitle_id"]) not in selected_keys]
        for record in matches[:6]:
            copied = {**record, "inclusion_reasons": [f"deterministic_sample_{category}"]}
            selected.append(copied); selected_keys.add((record["request_id"], record["subtitle_id"])); sample_counts[category] += 1
    if sum(sample_counts.values()) != 30:
        raise ValueError(f"Could not build 30 stratified additional samples: {dict(sample_counts)}")
    selected.sort(key=lambda row: (int(row["request_id"].rsplit("_", 1)[-1]), row["subtitle_id"]))
    _write_jsonl(output_path, selected)
    reason_counts = Counter(reason for row in selected for reason in row["inclusion_reasons"])
    manifest = {
        "schema_version": "1.0", "provisional": True, "human_labels_present": False,
        "record_count": len(selected), "required_record_count": len(selected) - 30,
        "additional_sample_count": 30, "additional_sample_distribution": dict(sample_counts),
        "inclusion_reason_counts": dict(sorted(reason_counts.items())),
        "source_responses_sha256": hashlib.sha256(responses_path.read_bytes()).hexdigest(),
        "source_composite_audit_sha256": hashlib.sha256(composite_path.read_bytes()).hexdigest(),
    }
    _write_json(manifest_path, manifest)
    return manifest
