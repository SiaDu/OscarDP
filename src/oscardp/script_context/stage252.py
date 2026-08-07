from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .openai_review import candidate_outcome
from .pipeline import _write_atomic, _write_json, _write_jsonl
from .schema import read_jsonl


GENERATOR_VERSION = "stage_2_5_2"
POLICY_SECTIONS = (
    "Exact / near-exact dialogue", "Paraphrased dialogue",
    "Expanded or contracted final-film dialogue", "Repeated dialogue", "Reordered dialogue",
    "Subtitle fragmentation", "Vocatives / isolated names", "Courtesy words / fillers",
    "Improvised dialogue", "Graphic / telegram / title / sign text", "Narration / voice-over",
    "Candidate-recall failure", "Multiple plausible screenplay blocks",
    "One subtitle spanning multiple dialogue blocks", "Several subtitles mapping to one screenplay block",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True,
        cwd=Path(__file__).resolve().parents[3],
    )
    return result.stdout.strip()


def _source_paths(
    gold_path: Path, validated_path: Path, requests_path: Path, manifest_path: Path,
    context_path: Path, alignment_path: Path, evaluation_path: Path, disagreements_path: Path,
) -> dict[str, Path]:
    v2_raw_path = evaluation_path.parent.parent / "pilot_result_v2" / "raw_batch_output.jsonl"
    return {
        "frozen_gold": gold_path, "pilot_requests": requests_path, "pilot_manifest": manifest_path,
        "validated_responses_stage251": validated_path, "evaluation_stage251": evaluation_path,
        "disagreements_stage251": disagreements_path, "screenplay_context": context_path,
        "deterministic_alignment": alignment_path, "v2_raw_output": v2_raw_path,
    }


def _block_indexes(context: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[int, int]]:
    lookup: dict[str, dict[str, Any]] = {}
    flat: list[dict[str, Any]] = []
    dialogue_flat_position: dict[int, int] = {}
    dialogue_order = 0
    for scene in context["script_scenes"]:
        scene_info = {
            "scene_id": scene["scene_id"], "screenplay_scene_id": scene.get("screenplay_scene_id"),
            "scene_heading": scene.get("slugline"), "location": scene.get("location"),
            "time_of_day": scene.get("time_of_day"),
        }
        for block in scene["script_blocks"]:
            item = {
                "block_id": block["block_id"], "scene_id": scene["scene_id"],
                "screenplay_order": dialogue_order if block["block_type"] == "dialogue" else None,
                "block_type": block["block_type"], "speaker": block.get("speaker"),
                "text": block.get("text", ""), "parenthetical": block.get("parenthetical"),
                "source_order": block.get("source_order"), **scene_info,
            }
            lookup[item["block_id"]] = item
            flat.append(item)
            if block["block_type"] == "dialogue":
                dialogue_flat_position[dialogue_order] = len(flat) - 1
                dialogue_order += 1
    return lookup, flat, dialogue_flat_position


def _local_screenplay_context(
    request: dict[str, Any], gold_ids: list[str], predicted_ids: list[str],
    lookup: dict[str, dict[str, Any]], flat: list[dict[str, Any]], dialogue_positions: dict[int, int],
) -> list[dict[str, Any]]:
    seeds = list(gold_ids) + list(predicted_ids)
    for anchor_name in ("previous_anchor", "next_anchor"):
        anchor = request.get(anchor_name)
        if isinstance(anchor, dict):
            seeds.extend(anchor.get("block_ids", []))
    candidates = request.get("dialogue_candidates", [])
    if candidates:
        nearest = max(candidates, key=lambda row: float(row.get("retrieval_score") or 0.0))
        seeds.append(nearest["block_id"])
    flat_indexes: set[int] = set()
    max_dialogue_order = max(dialogue_positions, default=-1)
    for block_id in dict.fromkeys(seeds):
        block = lookup.get(block_id)
        if block is None or block["screenplay_order"] is None:
            continue
        center = int(block["screenplay_order"])
        orders = [order for order in range(max(0, center - 3), min(max_dialogue_order, center + 3) + 1) if order in dialogue_positions]
        if not orders:
            continue
        start, end = dialogue_positions[orders[0]], dialogue_positions[orders[-1]]
        flat_indexes.update(range(start, end + 1))
    return [flat[index] for index in sorted(flat_indexes)]


def _subtitle_view(row: dict[str, Any], target_id: str) -> dict[str, Any]:
    time = row["time"]
    return {
        "subtitle_id": row["subtitle_id"], "text": row["text"],
        "start": time["start"], "end": time["end"],
        "start_sec": time.get("start_sec"), "end_sec": time.get("end_sec"),
        "alignment_status": row.get("alignment", {}).get("status"),
        "script_matches": row.get("script_matches", []), "is_target": row["subtitle_id"] == target_id,
    }


def _diagnostic_tags(
    subtitle_text: str, expected: dict[str, Any], actual: dict[str, Any], events: list[dict[str, Any]],
) -> list[str]:
    tags: list[str] = []
    if expected["decision"] == "no_match" and actual["decision"] == "match":
        tags.extend(["gold_no_match_predicted_match", "possible_gold_policy_issue"])
    elif expected["decision"] == "match" and actual["decision"] == "match" and set(expected["block_ids"]) != set(actual["block_ids"]):
        tags.extend(["gold_match_predicted_wrong_candidate", "possible_wrong_model_candidate"])
    elif expected["decision"] == "match" and actual["decision"] in {"no_match", "uncertain"}:
        tags.append("gold_match_predicted_no_candidate_match")
    notes = str(expected.get("reviewer_notes") or "").lower()
    if expected["decision"] == "uncertain" and "absent" in notes and "candidate" in notes:
        tags.append("possible_candidate_recall_failure")
    reasons = {event.get("reason") for event in events}
    if "repeated_block_mapping" in reasons or actual.get("decision_basis") == "repeated_or_reordered_dialogue":
        tags.append("possible_repeated_dialogue")
    if {"bounded_backward_mapping", "large_backward_mapping"} & reasons:
        tags.append("possible_reordered_dialogue")
    words = re.findall(r"[\w’'-]+", subtitle_text, flags=re.UNICODE)
    if len(words) <= 2 and subtitle_text.rstrip().endswith((",", ":")):
        tags.append("possible_vocative_fragment")
    if len(words) <= 4:
        tags.append("possible_subtitle_fragment")
    letters = [character for character in subtitle_text if character.isalpha()]
    if letters and sum(character.isupper() for character in letters) / len(letters) >= 0.8:
        tags.append("possible_graphic_or_insert_text")
    basis_tags = {
        "paraphrase": "possible_paraphrase",
        "changed_or_improvised_dialogue": "possible_improvised_dialogue",
        "split_across_blocks": "possible_expanded_film_dialogue",
    }
    if actual.get("decision_basis") in basis_tags:
        tags.append(basis_tags[actual["decision_basis"]])
    return list(dict.fromkeys(tags))


def _annotation_policy_template() -> str:
    sections = ["# Annotation policy template", "", "Status: DRAFT — no policy decisions have been finalized.", ""]
    for index, title in enumerate(POLICY_SECTIONS, 1):
        sections.extend([
            f"## {index}. {title}", "", "Policy decision:", "TODO", "",
            "Positive example:", "TODO", "", "Negative example:", "TODO", "", "Notes:", "TODO", "",
        ])
    return "\n".join(sections)


def build_gold_adjudication(
    gold_path: Path, validated_path: Path, requests_path: Path, manifest_path: Path,
    context_path: Path, alignment_path: Path, evaluation_path: Path,
    disagreements_path: Path, output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError(f"Refusing to overwrite existing adjudication package: {output_dir}")
    sources = _source_paths(
        gold_path, validated_path, requests_path, manifest_path, context_path,
        alignment_path, evaluation_path, disagreements_path,
    )
    if any(not path.is_file() for path in sources.values()):
        missing = [name for name, path in sources.items() if not path.is_file()]
        raise ValueError("Missing source artifacts: " + ", ".join(missing))
    source_hashes = {name: _sha256(path) for name, path in sources.items()}
    gold, responses, requests = read_jsonl(gold_path), read_jsonl(validated_path), read_jsonl(requests_path)
    alignment, disagreements = read_jsonl(alignment_path), read_jsonl(disagreements_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    context = json.loads(context_path.read_text(encoding="utf-8"))
    movie_id = context.get("movie", {}).get("movie_id") or alignment[0].get("movie_id")
    if not isinstance(movie_id, str) or not movie_id:
        raise ValueError("Unable to determine movie_id from screenplay context or alignment")
    selected = [row for row in disagreements if row.get("candidate_task_correct") is False]
    expected_count = evaluation["resolution_count"] - round(evaluation["candidate_task_accuracy"] * evaluation["resolution_count"])
    if len(selected) != expected_count:
        raise ValueError(f"Candidate-task disagreement count is {len(selected)}, expected {expected_count}: {[row.get('subtitle_id') for row in selected]}")
    gold_by_request = {row["request_id"]: row for row in gold}
    response_by_request = {row["request_id"]: row for row in responses}
    request_by_id = {row["request_id"]: row for row in requests}
    selection_by_id = {row["request_id"]: row for row in manifest["requests"]}
    alignment_index = {row["subtitle_id"]: index for index, row in enumerate(alignment)}
    block_lookup, flat_blocks, dialogue_positions = _block_indexes(context)
    package: list[dict[str, Any]] = []
    for disagreement in selected:
        request_id, subtitle_id = disagreement["request_id"], disagreement["subtitle_id"]
        request, response = request_by_id[request_id], response_by_request[request_id]
        expected = next(row for row in gold_by_request[request_id]["resolutions"] if row["subtitle_id"] == subtitle_id)
        actual = next(row for row in response["resolutions"] if row["subtitle_id"] == subtitle_id)
        subtitle = next(row for row in request["subtitles"] if row["subtitle_id"] == subtitle_id)
        alignment_position = alignment_index[subtitle_id]
        nearby_rows = alignment[max(0, alignment_position - 3):alignment_position + 4]
        request_index = request["subtitle_ids"].index(subtitle_id)
        sequence_ids = request["subtitle_ids"][max(0, request_index - 3):request_index + 4]
        neighbor_ids = set(sequence_ids)
        gold_resolution = {row["subtitle_id"]: row for row in gold_by_request[request_id]["resolutions"]}
        predicted_resolution = {row["subtitle_id"]: row for row in response["resolutions"]}
        all_events = response.get("validation_diagnostics", {}).get("sequence_quality", {}).get("events", [])
        touching_events = [
            event for event in all_events
            if event.get("subtitle_id") in neighbor_ids or event.get("previous_subtitle_id") in neighbor_ids
        ]
        candidates = [
            {**candidate, "is_gold_selected": candidate["block_id"] in expected["block_ids"], "is_v2_selected": candidate["block_id"] in actual["block_ids"]}
            for candidate in request["dialogue_candidates"]
        ]
        sequence_rows = [{
            "subtitle_id": value,
            "gold_decision": gold_resolution[value]["decision"], "gold_block_ids": gold_resolution[value]["block_ids"],
            "predicted_decision": predicted_resolution[value]["decision"], "predicted_block_ids": predicted_resolution[value]["block_ids"],
            "is_target": value == subtitle_id,
        } for value in sequence_ids]
        target_view = _subtitle_view(alignment[alignment_position], subtitle_id)
        tags = _diagnostic_tags(subtitle["text"], expected, actual, touching_events)
        package.append({
            "schema_version": "1.0", "movie_id": movie_id, "request_id": request_id, "subtitle_id": subtitle_id,
            "subtitle": {
                "text": subtitle["text"], "start": subtitle["time"]["start"], "end": subtitle["time"]["end"],
                "start_sec": subtitle["time"].get("start_sec"), "end_sec": subtitle["time"].get("end_sec"),
            },
            "nearby_subtitles": {
                "before": [_subtitle_view(row, subtitle_id) for row in nearby_rows if row["subtitle_id"] != subtitle_id and alignment_index[row["subtitle_id"]] < alignment_position],
                "target": target_view,
                "after": [_subtitle_view(row, subtitle_id) for row in nearby_rows if row["subtitle_id"] != subtitle_id and alignment_index[row["subtitle_id"]] > alignment_position],
                "request_local": request["subtitles"],
            },
            "request_context": {
                "stratum": selection_by_id[request_id]["stratum"], "timeline_region": selection_by_id[request_id]["timeline_region"],
                "fallback_used": selection_by_id[request_id].get("fallback_used", False),
                "candidate_limit_saturated": selection_by_id[request_id].get("candidate_limit_saturated", False),
                "candidate_count": len(candidates),
            },
            "gold": {"decision": expected["decision"], "block_ids": expected["block_ids"], "reviewer_notes": expected.get("reviewer_notes")},
            "v2_prediction": {
                "decision": actual["decision"], "block_ids": actual["block_ids"], "confidence": actual.get("confidence"),
                "decision_basis": actual.get("decision_basis"),
            },
            "candidate_task": {
                "gold_outcome": candidate_outcome(expected), "predicted_outcome": candidate_outcome(actual),
                "candidate_task_correct": False,
            },
            "selected_gold_blocks": [block_lookup[block_id] for block_id in expected["block_ids"] if block_id in block_lookup],
            "selected_prediction_blocks": [block_lookup[block_id] for block_id in actual["block_ids"] if block_id in block_lookup],
            "dialogue_candidates": candidates,
            "screenplay_local_context": _local_screenplay_context(
                request, expected["block_ids"], actual["block_ids"], block_lookup, flat_blocks, dialogue_positions,
            ),
            "sequence_context": {"mappings": sequence_rows, "quality_events": touching_events},
            "diagnostic_classification": tags,
            "human_adjudication": {
                "status": "pending", "adjudication": None, "final_decision": None,
                "final_block_ids": None, "policy_tags": [], "notes": None,
            },
        })
    output_dir.mkdir(parents=True, exist_ok=False)
    adjudication_path = output_dir / "gold_adjudication.jsonl"
    summary_path = output_dir / "gold_adjudication_summary.json"
    policy_path = output_dir / "annotation_policy_template.md"
    _write_jsonl(adjudication_path, package)
    _write_atomic(policy_path, _annotation_policy_template())
    classification_counts = Counter(tag for row in package for tag in row["diagnostic_classification"])
    reference_cases = [
        record for record in evaluation.get("candidate_recall_gold_records", [])
        if record.get("predicted_decision") == "uncertain"
    ]
    summary = {
        "schema_version": "1.0", "movie_id": movie_id, "source_commit": _source_commit(),
        "selected_disagreement_count": len(package), "selected_subtitle_ids": [row["subtitle_id"] for row in package],
        "disagreement_counts_by_type": dict(sorted(classification_counts.items())),
        "affected_request_count": len({row["request_id"] for row in package}),
        "affected_request_ids": sorted({row["request_id"] for row in package}),
        "candidate_limit_saturation_count": sum(row["request_context"]["candidate_limit_saturated"] for row in package),
        "fallback_used_count": sum(row["request_context"]["fallback_used"] for row in package),
        "sequence_warning_overlap_count": sum(bool(row["sequence_context"]["quality_events"]) for row in package),
        "pre_adjudication_metrics": {
            "resolution_count": evaluation["resolution_count"], "three_way_resolution_accuracy": evaluation["resolution_exact_match"],
            "candidate_task_accuracy": evaluation["candidate_task_accuracy"],
            "candidate_presence_decision_accuracy": evaluation["candidate_presence_decision_accuracy"],
        },
        "candidate_recall_successful_reference_cases": reference_cases,
        "potential_correction_accounting": {
            "pending_adjudication_count": len(package), "gold_correct": 0, "gold_should_change": 0, "ambiguous": 0,
            "adjusted_accuracy": None,
        },
    }
    _write_json(summary_path, summary)
    if any(_sha256(path) != digest for name, path in sources.items() for digest in [source_hashes[name]]):
        raise RuntimeError("A source artifact changed during adjudication package generation")
    generated = {path.name: {"path": path.as_posix(), "sha256": _sha256(path)} for path in (adjudication_path, summary_path, policy_path)}
    artifact_manifest = {
        "schema_version": "1.0", "generated_at": datetime.now(UTC).isoformat(),
        "source_commit": summary["source_commit"], "generator_version": GENERATOR_VERSION,
        "source_artifacts": {name: {"path": path.as_posix(), "sha256": source_hashes[name]} for name, path in sources.items()},
        "generated_artifacts": generated,
    }
    _write_json(output_dir / "artifact_manifest.json", artifact_manifest)
    return {"output_dir": output_dir.as_posix(), **summary, "validation_ready": True}


def validate_gold_adjudication(
    gold_path: Path, validated_path: Path, requests_path: Path, manifest_path: Path,
    context_path: Path, alignment_path: Path, evaluation_path: Path,
    disagreements_path: Path, output_dir: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    sources = _source_paths(
        gold_path, validated_path, requests_path, manifest_path, context_path,
        alignment_path, evaluation_path, disagreements_path,
    )
    rows = read_jsonl(output_dir / "gold_adjudication.jsonl")
    gold, responses, requests = read_jsonl(gold_path), read_jsonl(validated_path), read_jsonl(requests_path)
    disagreements = read_jsonl(disagreements_path)
    expected_ids = [row["subtitle_id"] for row in disagreements if row.get("candidate_task_correct") is False]
    actual_ids = [row.get("subtitle_id") for row in rows]
    if actual_ids != expected_ids:
        errors.append("adjudication subtitle IDs must exactly match candidate-task disagreements in source order")
    if len(actual_ids) != len(set(actual_ids)):
        errors.append("duplicate adjudication subtitle ID")
    gold_by_request = {row["request_id"]: {item["subtitle_id"]: item for item in row["resolutions"]} for row in gold}
    response_by_request = {row["request_id"]: {item["subtitle_id"]: item for item in row["resolutions"]} for row in responses}
    request_by_id = {row["request_id"]: row for row in requests}
    for row in rows:
        request_id, subtitle_id = row.get("request_id"), row.get("subtitle_id")
        request = request_by_id.get(request_id)
        expected = gold_by_request.get(request_id, {}).get(subtitle_id)
        actual = response_by_request.get(request_id, {}).get(subtitle_id)
        if request is None or subtitle_id not in request.get("subtitle_ids", []):
            errors.append(f"{subtitle_id}: target missing from request")
            continue
        if expected is None:
            errors.append(f"{subtitle_id}: target missing from gold")
        if actual is None:
            errors.append(f"{subtitle_id}: target missing from prediction")
        source_candidates = request["dialogue_candidates"]
        packaged_candidates = row.get("dialogue_candidates", [])
        if len(source_candidates) != len(packaged_candidates):
            errors.append(f"{subtitle_id}: candidate count differs from source request")
        else:
            for source, packaged in zip(source_candidates, packaged_candidates):
                if any(packaged.get(key) != value for key, value in source.items()):
                    errors.append(f"{subtitle_id}: source candidate content or order changed")
                    break
                if expected is not None and packaged.get("is_gold_selected") != (source["block_id"] in expected["block_ids"]):
                    errors.append(f"{subtitle_id}: candidate gold-selection marker differs from source gold")
                    break
                if actual is not None and packaged.get("is_v2_selected") != (source["block_id"] in actual["block_ids"]):
                    errors.append(f"{subtitle_id}: candidate prediction-selection marker differs from source prediction")
                    break
        if expected is not None and row.get("gold", {}).get("decision") != expected["decision"]:
            errors.append(f"{subtitle_id}: gold decision differs from source")
        if expected is not None and row.get("gold", {}).get("block_ids") != expected["block_ids"]:
            errors.append(f"{subtitle_id}: gold block IDs differ from source")
        if actual is not None and row.get("v2_prediction", {}).get("decision") != actual["decision"]:
            errors.append(f"{subtitle_id}: prediction decision differs from source")
        if actual is not None and row.get("v2_prediction", {}).get("block_ids") != actual["block_ids"]:
            errors.append(f"{subtitle_id}: prediction block IDs differ from source")
        if expected is not None and [block.get("block_id") for block in row.get("selected_gold_blocks", [])] != expected["block_ids"]:
            errors.append(f"{subtitle_id}: selected gold block details differ from source")
        if actual is not None and [block.get("block_id") for block in row.get("selected_prediction_blocks", [])] != actual["block_ids"]:
            errors.append(f"{subtitle_id}: selected prediction block details differ from source")
        if row.get("candidate_task", {}).get("candidate_task_correct") is not False:
            errors.append(f"{subtitle_id}: record is not marked as a candidate-task disagreement")
        human = row.get("human_adjudication", {})
        expected_human = {
            "status": "pending", "adjudication": None, "final_decision": None,
            "final_block_ids": None, "policy_tags": [], "notes": None,
        }
        if human != expected_human:
            errors.append(f"{subtitle_id}: human adjudication must remain pending with null decisions")
    artifact_manifest_path = output_dir / "artifact_manifest.json"
    if not artifact_manifest_path.is_file():
        errors.append("artifact manifest is missing")
    else:
        artifact_manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
        for name, path in sources.items():
            recorded = artifact_manifest.get("source_artifacts", {}).get(name, {}).get("sha256")
            if recorded != _sha256(path):
                errors.append(f"source artifact hash changed: {name}")
        for filename in ("gold_adjudication.jsonl", "gold_adjudication_summary.json", "annotation_policy_template.md"):
            path = output_dir / filename
            recorded = artifact_manifest.get("generated_artifacts", {}).get(filename, {}).get("sha256")
            if not path.is_file() or recorded != _sha256(path):
                errors.append(f"generated artifact hash mismatch: {filename}")
    return {
        "schema_version": "1.0", "generator_version": GENERATOR_VERSION,
        "expected_record_count": len(expected_ids), "actual_record_count": len(rows),
        "unique_subtitle_count": len(set(actual_ids)), "errors": errors, "passed": not errors,
    }
