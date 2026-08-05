from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from .openai_review import apply_validated_responses, validate_resolution
from .pipeline import _write_json, _write_jsonl
from .schema import read_jsonl
from .validation import validate_files


def build_non_anchor_sequence_audit(alignment_path: Path, output_path: Path, summary_path: Path) -> dict[str, Any]:
    alignments = read_jsonl(alignment_path)
    mapped = [row for row in alignments if row.get("script_matches") and isinstance(row.get("alignment", {}).get("script_order_start"), int) and isinstance(row.get("alignment", {}).get("script_order_end"), int)]
    audit: list[dict[str, Any]] = []
    for previous, current in zip(mapped, mapped[1:]):
        previous_start, previous_end = previous["alignment"]["script_order_start"], previous["alignment"]["script_order_end"]
        current_start, current_end = current["alignment"]["script_order_start"], current["alignment"]["script_order_end"]
        reasons = []
        if current_start < previous_start:
            reasons.append("start_regression")
        if current_end < previous_end:
            reasons.append("end_regression")
        if not reasons:
            continue
        previous_has_llm = bool(previous.get("alignment", {}).get("llm_resolution"))
        current_has_llm = bool(current.get("alignment", {}).get("llm_resolution"))
        subject, target_role = (current, "current") if current_has_llm or not previous_has_llm else (previous, "previous")
        def sequence_row(row: dict[str, Any], start: int, end: int) -> dict[str, Any]:
            return {
                "subtitle_id": row["subtitle_id"], "text": row["text"], "time": row["time"],
                "span": {"start": start, "end": end}, "scene_id": row.get("scene_id"),
                "block_ids": [match["block_id"] for match in row["script_matches"]],
                "status": row["alignment"]["status"], "reliable_anchor": row["alignment"].get("reliable_anchor"),
            }
        audit.append({
            "schema_version": "2.0",
            "review_target_subtitle_id": subject["subtitle_id"], "review_target_text": subject["text"],
            "review_target_time": subject["time"], "review_target_role": target_role,
            "regression_trigger_subtitle_id": current["subtitle_id"],
            "sequence_previous": sequence_row(previous, previous_start, previous_end),
            "sequence_current": sequence_row(current, current_start, current_end),
            "reason_flags": reasons,
        })
    _write_jsonl(output_path, audit)
    summary = {
        "schema_version": "2.0", "record_count": len(audit),
        "start_regression_count": sum("start_regression" in row["reason_flags"] for row in audit),
        "end_regression_count": sum("end_regression" in row["reason_flags"] for row in audit),
        "review_target_subtitle_ids": [row["review_target_subtitle_id"] for row in audit],
        "regression_trigger_subtitle_ids": [row["regression_trigger_subtitle_id"] for row in audit],
        "source_alignment_sha256": hashlib.sha256(alignment_path.read_bytes()).hexdigest(),
    }
    _write_json(summary_path, summary)
    return summary


def _nearby_context(alignments: list[dict[str, Any]], index: int, radius: int = 2) -> dict[str, list[dict[str, Any]]]:
    def compact(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "subtitle_id": row["subtitle_id"], "text": row["text"], "time": row["time"],
            "scene_id": row.get("scene_id"), "block_ids": [match["block_id"] for match in row.get("script_matches", [])],
            "alignment_status": row.get("alignment", {}).get("status"),
        }
    return {
        "before": [compact(row) for row in alignments[max(0, index - radius):index]],
        "after": [compact(row) for row in alignments[index + 1:index + 1 + radius]],
    }


def build_human_audit_v2(
    requests_path: Path, responses_path: Path, composite_path: Path, non_anchor_path: Path,
    alignment_path: Path, shot_context_path: Path, prior_audit_path: Path,
    output_path: Path, manifest_path: Path,
) -> dict[str, Any]:
    requests, responses = read_jsonl(requests_path), read_jsonl(responses_path)
    composite, regressions = read_jsonl(composite_path), read_jsonl(non_anchor_path)
    alignments, shots, prior = read_jsonl(alignment_path), read_jsonl(shot_context_path), read_jsonl(prior_audit_path)
    response_by_id = {response["request_id"]: response for response in responses}
    alignment_index = {row["subtitle_id"]: index for index, row in enumerate(alignments)}
    composite_by_key = {(row["request_id"], row["subtitle_id"]): row for row in composite}
    regression_ids = {row["review_target_subtitle_id"] for row in regressions}
    prior_sample_reasons = {
        (row["request_id"], row["subtitle_id"]): [reason for reason in row.get("inclusion_reasons", []) if reason.startswith("deterministic_sample_")]
        for row in prior
    }
    shot_refs: dict[str, list[dict[str, Any]]] = {}
    for shot in shots:
        for subtitle in shot.get("subtitles", []):
            shot_refs.setdefault(subtitle["subtitle_id"], []).append({"shot_id": shot["shot_id"], "keyframe_path": shot["keyframe"]["path"]})
    selected: list[dict[str, Any]] = []
    for request in requests:
        response = response_by_id[request["request_id"]]
        resolution_by_id = {resolution["subtitle_id"]: resolution for resolution in response["resolutions"]}
        subtitle_by_id = {subtitle["subtitle_id"]: subtitle for subtitle in request["subtitles"]}
        automatic_by_id = {mapping["subtitle_id"]: mapping for mapping in request.get("automatic_candidate_mappings", [])}
        for subtitle_id in request["subtitle_ids"]:
            resolution = resolution_by_id[subtitle_id]
            key = (request["request_id"], subtitle_id)
            composite_row = composite_by_key.get(key)
            reasons: list[str] = []
            if resolution["decision"] == "uncertain": reasons.append("uncertain")
            if resolution["decision"] == "no_match": reasons.append("openai_no_match")
            if float(resolution["confidence"]) < .80: reasons.append("low_confidence")
            if composite_row:
                if len(resolution["block_ids"]) == 1 and ("one_block_with_two_adjacent_overlaps" in composite_row["reason_flags"] or "automatic_mapping_has_more_blocks" in composite_row["reason_flags"]):
                    reasons.append("one_block_underselection")
                reasons.extend(f"composite:{flag}" for flag in composite_row["reason_flags"])
            if subtitle_id in regression_ids: reasons.append("non_anchor_sequence_regression")
            reasons.extend(prior_sample_reasons.get(key, []))
            reasons = list(dict.fromkeys(reasons))
            if not reasons:
                continue
            index = alignment_index[subtitle_id]
            selected.append({
                "request_id": request["request_id"], "subtitle_id": subtitle_id,
                "subtitle_text": subtitle_by_id[subtitle_id]["text"], "subtitle_time": subtitle_by_id[subtitle_id]["time"],
                "nearby_subtitle_context": _nearby_context(alignments, index),
                "previous_anchor": request.get("previous_anchor"), "next_anchor": request.get("next_anchor"),
                "dialogue_candidates": [{
                    "scene_id": candidate["scene_id"], "block_id": candidate["block_id"],
                    "screenplay_order": candidate["screenplay_order"], "speaker": candidate["speaker"], "text": candidate["text"],
                } for candidate in request.get("dialogue_candidates", [])],
                "automatic_mapping": automatic_by_id.get(subtitle_id),
                "openai_resolution": resolution,
                "matched_shots": shot_refs.get(subtitle_id, []), "inclusion_reasons": reasons,
                "human_decision": None, "human_block_ids": None, "reviewer_notes": None, "review_status": "pending",
            })
    selected.sort(key=lambda row: alignment_index[row["subtitle_id"]])
    _write_jsonl(output_path, selected)
    reason_counts = Counter(reason for row in selected for reason in row["inclusion_reasons"])
    required_fields = [
        "request_id", "subtitle_id", "subtitle_text", "subtitle_time", "nearby_subtitle_context",
        "previous_anchor", "next_anchor", "dialogue_candidates", "automatic_mapping", "openai_resolution",
        "matched_shots", "inclusion_reasons", "human_decision", "human_block_ids", "reviewer_notes", "review_status",
    ]
    manifest = {
        "schema_version": "3.0", "provisional": True, "human_labels_present": False,
        "record_count": len(selected), "pending_count": len(selected), "required_fields": required_fields,
        "inclusion_reason_counts": dict(sorted(reason_counts.items())),
        "existing_stratified_sample_count": sum(any(reason.startswith("deterministic_sample_") for reason in row["inclusion_reasons"]) for row in selected),
        "source_hashes": {
            "requests": hashlib.sha256(requests_path.read_bytes()).hexdigest(),
            "responses": hashlib.sha256(responses_path.read_bytes()).hexdigest(),
            "alignment": hashlib.sha256(alignment_path.read_bytes()).hexdigest(),
            "shot_context": hashlib.sha256(shot_context_path.read_bytes()).hexdigest(),
            "composite_audit": hashlib.sha256(composite_path.read_bytes()).hexdigest(),
            "non_anchor_audit": hashlib.sha256(non_anchor_path.read_bytes()).hexdigest(),
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def apply_human_corrections(
    audit_path: Path, requests_path: Path, openai_responses_path: Path,
    alignment_path: Path, context_path: Path, shots_path: Path,
    output_dir: Path, output_tag: str,
) -> dict[str, Any]:
    if output_tag in {"full", "pilot"}:
        raise ValueError("Human corrections require a new output tag")
    audit, requests, responses = read_jsonl(audit_path), read_jsonl(requests_path), read_jsonl(openai_responses_path)
    request_by_id = {request["request_id"]: request for request in requests}
    response_by_id = {response["request_id"]: deepcopy(response) for response in responses}
    resolution_lookup = {(response["request_id"], resolution["subtitle_id"]): resolution for response in response_by_id.values() for resolution in response["resolutions"]}
    corrected = 0
    for row in audit:
        status = row.get("review_status")
        if status == "pending":
            if any(row.get(field) is not None for field in ("human_decision", "human_block_ids", "reviewer_notes")):
                raise ValueError(f"Pending audit row contains human edits: {row.get('subtitle_id')}")
            continue
        if status != "completed":
            raise ValueError(f"Invalid review_status for {row.get('subtitle_id')}: {status}")
        request_id, subtitle_id = row.get("request_id"), row.get("subtitle_id")
        if request_id not in request_by_id or (request_id, subtitle_id) not in resolution_lookup:
            raise ValueError(f"Human correction references unknown request/subtitle: {request_id}/{subtitle_id}")
        decision, block_ids = row.get("human_decision"), row.get("human_block_ids")
        if decision not in {"match", "no_match", "uncertain"} or not isinstance(block_ids, list):
            raise ValueError(f"Incomplete human correction for {subtitle_id}")
        original = deepcopy(resolution_lookup[(request_id, subtitle_id)])
        corrected_resolution = {
            **original, "decision": decision, "block_ids": block_ids,
            "confidence": 1.0, "openai_resolution": original,
            "human_correction": {"review_status": status, "reviewer_notes": row.get("reviewer_notes")},
        }
        resolution_lookup[(request_id, subtitle_id)].clear()
        resolution_lookup[(request_id, subtitle_id)].update(corrected_resolution)
        corrected += 1
    if corrected == 0:
        raise ValueError("No completed human corrections found")
    for request_id, request in request_by_id.items():
        errors = validate_resolution(response_by_id[request_id], request)
        if errors:
            raise ValueError(f"Invalid human corrections for {request_id}: {errors}")
    openai_dir = output_dir / "review/openai"
    targets = [
        output_dir / f"subtitle_script_alignment.llm_reviewed_{output_tag}.jsonl",
        output_dir / f"shot_script_context.llm_reviewed_{output_tag}.jsonl",
        openai_dir / f"{output_tag}_apply_report.json", openai_dir / f"{output_tag}_reviewed_alignment_diagnostics.json",
        openai_dir / f"{output_tag}_human_correction_report.json", openai_dir / f"{output_tag}_human_validation_report.json",
    ]
    if any(path.exists() for path in targets):
        raise FileExistsError("Refusing to overwrite existing human-corrected outputs")
    ordered = [response_by_id[request["request_id"]] for request in requests]
    with tempfile.TemporaryDirectory(prefix="oscardp_human_corrections_") as temporary_dir:
        corrected_path = Path(temporary_dir) / "responses.jsonl"
        _write_jsonl(corrected_path, ordered)
        applied = apply_validated_responses(alignment_path, requests_path, corrected_path, context_path, shots_path, output_dir, output_tag)
    validation = validate_files(
        context_path, Path(applied["alignment_output"]), Path(applied["shot_output"]), shots_path,
    )
    correction_report = {
        "schema_version": "1.0", "output_tag": output_tag, "corrected_subtitle_count": corrected,
        "original_openai_responses_preserved": True, "baseline_files_unchanged": applied["baseline_files_unchanged"],
        "alignment_output": applied["alignment_output"], "shot_output": applied["shot_output"],
    }
    validation_report = {"schema_version": "1.0", "passed": validation.passed, "errors": validation.errors, "alignment_count": validation.alignment_count, "shot_count": validation.shot_count}
    _write_json(openai_dir / f"{output_tag}_human_correction_report.json", correction_report)
    _write_json(openai_dir / f"{output_tag}_human_validation_report.json", validation_report)
    if not validation.passed:
        raise RuntimeError("Human-corrected output validation failed")
    return {**correction_report, "validation_passed": True}
