from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from oscardp.script_context.openai_schema import (
    V321_VOCATIVE_SYSTEM_INSTRUCTIONS,
    V32_POLICY_SYSTEM_INSTRUCTIONS,
)
from oscardp.script_context.production_review import (
    PRODUCTION_REVIEWER_VERSION,
    apply_production_responses_v3,
    bind_production_request_subset_v3,
    merge_production_response_chunks_v3,
    merge_production_responses_v3,
    prepare_production_batch_v3,
    preflight_production_batch_v3,
    prepare_production_remaining_v3,
    split_production_requests_v3,
    submit_production_batch_v3,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def reviewer_manifest(path: Path) -> None:
    path.write_text(json.dumps({
        "production_reviewer_version": PRODUCTION_REVIEWER_VERSION,
        "status": "promoted_frozen", "model": "gpt-5.6-terra",
        "decision_schema_version": "candidate_task_v3",
        "prompt_sha256": hashlib.sha256(V32_POLICY_SYSTEM_INSTRUCTIONS.encode()).hexdigest(),
    }), encoding="utf-8")


def reviewer_manifest_v2(path: Path, inherited: Path) -> None:
    path.write_text(json.dumps({
        "production_reviewer_version": "v3.2-production.2",
        "parent_reviewer_version": "v3.2-production.1",
        "status": "promoted_frozen", "model": "gpt-5.6-terra",
        "decision_schema_version": "candidate_task_v3",
        "hard_validation_contract_version": "candidate_task_v3_structure_v2",
        "prompt_sha256": hashlib.sha256(V32_POLICY_SYSTEM_INSTRUCTIONS.encode()).hexdigest(),
        "inherited_production_1_evidence_manifest": inherited.resolve().as_posix(),
        "inherited_production_1_evidence_manifest_sha256": hashlib.sha256(inherited.read_bytes()).hexdigest(),
    }), encoding="utf-8")


def reviewer_manifest_v321(path: Path, evidence: Path) -> None:
    path.write_text(json.dumps({
        "schema_version": "1.0",
        "reviewer_version": "v3.2.1-production.1",
        "status": "promoted_global_production_reviewer",
        "components": {
            "prompt_version": "v3.2.1-vocative-candidate",
            "prompt_sha256": hashlib.sha256(V321_VOCATIVE_SYSTEM_INSTRUCTIONS.encode()).hexdigest(),
            "retrieval_version": "global_lexical_rescue_v2",
            "decision_schema_version": "candidate_task_v3",
            "hard_validation_contract_version": "candidate_task_v3_structure_v2",
            "model": "gpt-5.6-terra",
        },
        "independent_calibration": {
            "reference_frozen_before_reviewer_output": True,
            "new_systematic_failure_class_found": False,
            "numeric_gate": {"passed": True},
        },
        "artifact_paths": {"reference": evidence.resolve().as_posix()},
        "artifact_sha256": {"reference": hashlib.sha256(evidence.read_bytes()).hexdigest()},
    }), encoding="utf-8")


def reviewer_manifest_v321_validator_v3(
    path: Path, inherited: Path, evidence: Path, validator_evidence: Path,
) -> None:
    source = json.loads(inherited.read_text(encoding="utf-8"))
    source.update({
        "reviewer_version": "v3.2.1-production.2-validator-v3",
        "parent_reviewer_version": "v3.2.1-production.1",
        "components": {
            **source["components"],
            "hard_validation_contract_version": "candidate_task_v3_structure_v3",
        },
        "inherited_production_1_evidence_manifest": inherited.resolve().as_posix(),
        "inherited_production_1_evidence_manifest_sha256": hashlib.sha256(
            inherited.read_bytes()
        ).hexdigest(),
        "validator_independent_calibration": {
            "reference_frozen_before_reviewer_output": True,
            "new_systematic_failure_class_found": False,
            "numeric_gate": {"passed": True},
        },
        "validator_artifact_paths": {
            "evaluation": validator_evidence.resolve().as_posix(),
        },
        "validator_artifact_sha256": {
            "evaluation": hashlib.sha256(validator_evidence.read_bytes()).hexdigest(),
        },
    })
    path.write_text(json.dumps(source), encoding="utf-8")


def reviewer_manifest_v321_retrieval_v3(path: Path, parent: Path, evidence: Path) -> None:
    path.write_text(json.dumps({
        "reviewer_version": "v3.2.1-production.3-retrieval-v3-validator-v3",
        "parent_reviewer_version": "v3.2.1-production.2-validator-v3",
        "status": "promoted_global_production_reviewer", "model": "gpt-5.6-terra",
        "decision_schema_version": "candidate_task_v3",
        "hard_validation_contract_version": "candidate_task_v3_structure_v3",
        "prompt_sha256": hashlib.sha256(V321_VOCATIVE_SYSTEM_INSTRUCTIONS.encode()).hexdigest(),
        "retrieval_version": "global_lexical_rescue_v3",
        "independent_calibration": {
            "reference_frozen_before_reviewer_output": True,
            "new_systematic_failure_class_found": False,
            "numeric_gate": {"passed": True},
        },
        "artifact_paths": {"evaluation": evidence.resolve().as_posix()},
        "artifact_sha256": {"evaluation": hashlib.sha256(evidence.read_bytes()).hexdigest()},
        "parent_reviewer_manifest": parent.resolve().as_posix(),
        "parent_reviewer_manifest_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
    }), encoding="utf-8")


def retrieval_manifest(path: Path) -> None:
    path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps({
        "retrieval_version": "global_lexical_rescue_v2",
        "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }), encoding="utf-8")


def retrieval_manifest_v3(path: Path) -> None:
    path.with_suffix(path.suffix + ".manifest.json").write_text(json.dumps({
        "retrieval_version": "global_lexical_rescue_v3",
        "output_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }), encoding="utf-8")


def request(index: int, subtitle_ids: list[str] | None = None) -> dict:
    ids = subtitle_ids or [f"subtitle_{index:06d}"]
    return {
        "request_id": f"alignment_review_{index:06d}", "subtitle_ids": ids,
        "subtitles": [{"subtitle_id": sid, "text": sid, "time": {"start_sec": i, "end_sec": i + .8}} for i, sid in enumerate(ids)],
        "dialogue_candidates": [{
            "scene_id": "scene_001", "block_id": f"scene_001_dialogue_{i:03d}",
            "screenplay_order": i - 1, "speaker": "A", "text": f"dialogue {i}",
        } for i in range(1, len(ids) + 1)],
        "automatic_candidate_mappings": [], "candidate_scenes": ["scene_001"],
        "insufficient_candidates": False, "fallback_used": False,
    }


def response(req: dict, no_match_last: bool = False) -> dict:
    resolutions = []
    for i, sid in enumerate(req["subtitle_ids"]):
        no_match = no_match_last and i == len(req["subtitle_ids"]) - 1
        resolutions.append({
            "subtitle_id": sid, "decision": "no_candidate_match" if no_match else "match",
            "block_ids": [] if no_match else [req["dialogue_candidates"][i]["block_id"]],
            "confidence": .9, "decision_basis": "no_supplied_candidate" if no_match else "exact_or_near_exact",
        })
    return {"request_id": req["request_id"], "resolutions": resolutions, "model": "gpt-5.6-terra"}


def test_production_remaining_and_batch_are_versioned_and_write_once(tmp_path: Path) -> None:
    full_rows = [request(i) for i in range(1, 4)]; full = tmp_path / "full.jsonl"; pilot = tmp_path / "pilot.jsonl"
    remaining = tmp_path / "remaining.v3.jsonl"; remaining_manifest = tmp_path / "remaining.v3.manifest.json"
    reviewer = tmp_path / "reviewer.json"; reviewer_manifest(reviewer)
    write_jsonl(full, full_rows); write_jsonl(pilot, [full_rows[1]])
    result = prepare_production_remaining_v3(full, pilot, remaining, remaining_manifest, reviewer)
    assert result["production_reviewer_version"] == PRODUCTION_REVIEWER_VERSION
    assert result["counts"]["remaining_requests"] == 2
    batch = tmp_path / "batch.v3.jsonl"
    batch_manifest = prepare_production_batch_v3(remaining, reviewer, batch)
    assert batch_manifest["decision_schema_version"] == "candidate_task_v3"
    assert batch_manifest["batch_input_sha256"] == hashlib.sha256(batch.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        prepare_production_batch_v3(remaining, reviewer, batch)


def test_production_submit_revalidates_hash_and_uses_no_network_on_tamper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requests = tmp_path / "requests.jsonl"; reviewer = tmp_path / "reviewer.json"; batch = tmp_path / "batch.jsonl"
    write_jsonl(requests, [request(1)]); reviewer_manifest(reviewer); prepare_production_batch_v3(requests, reviewer, batch)
    files = SimpleNamespace(create=Mock(return_value=SimpleNamespace(id="file-prod")))
    batches = SimpleNamespace(create=Mock(return_value=SimpleNamespace(id="batch-prod", status="validating")))
    monkeypatch.setattr("oscardp.script_context.openai_review._client", lambda: SimpleNamespace(files=files, batches=batches))
    result = submit_production_batch_v3(batch, reviewer, tmp_path / "job.json", confirm_submit=True)
    assert result["production_reviewer_version"] == PRODUCTION_REVIEWER_VERSION
    files.create.assert_called_once(); batches.create.assert_called_once()
    tampered = tmp_path / "tampered.jsonl"; tampered.write_bytes(batch.read_bytes() + b"\n")
    tampered.with_suffix(".jsonl.manifest.json").write_bytes(batch.with_suffix(".jsonl.manifest.json").read_bytes())
    with pytest.raises(ValueError, match="hash differs"):
        submit_production_batch_v3(tampered, reviewer, tmp_path / "job2.json", confirm_submit=True)
    files.create.assert_called_once()


def test_production_2_preparation_is_versioned_and_bound_to_inherited_manifest(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    inherited = tmp_path / "reviewer-v1.json"
    reviewer = tmp_path / "reviewer-v2.json"
    batch = tmp_path / "batch.v3_2_production_2.jsonl"
    write_jsonl(requests, [request(1)])
    reviewer_manifest(inherited)
    reviewer_manifest_v2(reviewer, inherited)

    result = prepare_production_batch_v3(requests, reviewer, batch)

    assert result["production_reviewer_version"] == "v3.2-production.2"
    assert result["lifecycle_schema_version"] == "v3_production_2"
    assert result["reviewer_manifest_sha256"] == hashlib.sha256(reviewer.read_bytes()).hexdigest()
    inherited.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="inherited production.1 evidence hash differs"):
        prepare_production_batch_v3(requests, reviewer, tmp_path / "other.jsonl")


def test_production_321_uses_promoted_prompt_retrieval_and_calibration_bindings(tmp_path: Path) -> None:
    rows = [request(i) for i in range(1, 4)]
    full = tmp_path / "full.v321.jsonl"
    pilot = tmp_path / "pilot.v321.jsonl"
    remaining = tmp_path / "remaining.v321.jsonl"
    remaining_manifest = remaining.with_suffix(remaining.suffix + ".manifest.json")
    evidence = tmp_path / "frozen-reference.jsonl"
    reviewer = tmp_path / "reviewer-v321.json"
    evidence.write_text("frozen\n", encoding="utf-8")
    reviewer_manifest_v321(reviewer, evidence)
    write_jsonl(full, rows)
    write_jsonl(pilot, [rows[1]])
    retrieval_manifest(full)
    bind_production_request_subset_v3(
        full, pilot, reviewer, pilot.with_suffix(pilot.suffix + ".manifest.json"),
    )

    result = prepare_production_remaining_v3(
        full, pilot, remaining, remaining_manifest, reviewer,
    )
    assert result["production_reviewer_version"] == "v3.2.1-production.1"
    assert result["retrieval_version"] == "global_lexical_rescue_v2"
    batch = tmp_path / "batch.v321.jsonl"
    batch_manifest = prepare_production_batch_v3(remaining, reviewer, batch)
    batch_rows = [json.loads(line) for line in batch.read_text().splitlines()]
    assert batch_manifest["lifecycle_schema_version"] == "v3_2_1_production_1"
    assert batch_manifest["hard_validation_contract_version"] == "candidate_task_v3_structure_v2"
    assert batch_manifest["instructions_sha256"] == hashlib.sha256(
        V321_VOCATIVE_SYSTEM_INSTRUCTIONS.encode()
    ).hexdigest()
    assert all(row["body"]["instructions"] == V321_VOCATIVE_SYSTEM_INSTRUCTIONS for row in batch_rows)

    unbound = tmp_path / "unbound.jsonl"
    write_jsonl(unbound, [request(9)])
    with pytest.raises(ValueError, match="missing their retrieval manifest"):
        prepare_production_batch_v3(unbound, reviewer, tmp_path / "unbound-batch.jsonl")

    evidence.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="calibration artifact hash differs"):
        prepare_production_batch_v3(remaining, reviewer, tmp_path / "changed-evidence.jsonl")


def test_production_321_retrieval_v3_binds_parent_calibration_and_requests(tmp_path: Path) -> None:
    requests = tmp_path / "requests.v3.jsonl"
    parent = tmp_path / "parent-reviewer.json"
    evidence = tmp_path / "calibration-evaluation.json"
    reviewer = tmp_path / "reviewer-retrieval-v3.json"
    write_jsonl(requests, [request(1)])
    retrieval_manifest_v3(requests)
    parent.write_text("frozen parent\n", encoding="utf-8")
    evidence.write_text("frozen calibration\n", encoding="utf-8")
    reviewer_manifest_v321_retrieval_v3(reviewer, parent, evidence)

    output = tmp_path / "batch.jsonl"
    result = prepare_production_batch_v3(requests, reviewer, output)
    assert result["production_reviewer_version"] == "v3.2.1-production.3-retrieval-v3-validator-v3"
    assert result["retrieval_version"] == "global_lexical_rescue_v3"
    assert result["hard_validation_contract_version"] == "candidate_task_v3_structure_v3"

    parent.write_text("changed parent\n", encoding="utf-8")
    with pytest.raises(ValueError, match="parent reviewer hash differs"):
        prepare_production_batch_v3(requests, reviewer, tmp_path / "other.jsonl")


def test_production_batch_preflight_reconstructs_exact_payload_and_rejects_tampering(tmp_path: Path) -> None:
    requests = tmp_path / "requests.v3.jsonl"
    parent = tmp_path / "parent-reviewer.json"; evidence = tmp_path / "calibration-evaluation.json"
    reviewer = tmp_path / "reviewer-retrieval-v3.json"; batch = tmp_path / "batch.jsonl"
    write_jsonl(requests, [request(1), request(2)])
    retrieval_manifest_v3(requests)
    parent.write_text("frozen parent\n", encoding="utf-8"); evidence.write_text("frozen calibration\n", encoding="utf-8")
    reviewer_manifest_v321_retrieval_v3(reviewer, parent, evidence)
    prepare_production_batch_v3(requests, reviewer, batch)

    report = tmp_path / "preflight.json"
    result = preflight_production_batch_v3(batch, requests, reviewer, report)
    assert result["passed"] is True
    assert result["deterministic_payload_reconstruction_equal"] is True
    assert result["request_count"] == 2
    with pytest.raises(FileExistsError):
        preflight_production_batch_v3(batch, requests, reviewer, report)

    rows = [json.loads(line) for line in batch.read_text().splitlines()]
    rows[0]["body"]["input"] = "tampered"
    write_jsonl(batch, rows)
    tampered = preflight_production_batch_v3(batch, requests, reviewer, tmp_path / "tampered.json")
    assert tampered["passed"] is False
    assert "Batch payload differs from deterministic reconstruction" in tampered["errors"]
    assert "Batch companion manifest differs: batch_input_sha256" in tampered["errors"]


def test_production_321_validator_v3_is_versioned_and_binds_both_calibrations(
    tmp_path: Path,
) -> None:
    requests = tmp_path / "requests.jsonl"
    inherited = tmp_path / "reviewer-v321-production-1.json"
    reviewer = tmp_path / "reviewer-v321-validator-v3.json"
    prompt_evidence = tmp_path / "prompt-calibration.json"
    validator_evidence = tmp_path / "validator-calibration.json"
    prompt_evidence.write_text("prompt evidence\n", encoding="utf-8")
    validator_evidence.write_text("validator evidence\n", encoding="utf-8")
    reviewer_manifest_v321(inherited, prompt_evidence)
    reviewer_manifest_v321_validator_v3(
        reviewer, inherited, prompt_evidence, validator_evidence,
    )
    write_jsonl(requests, [request(1)])
    retrieval_manifest(requests)

    result = prepare_production_batch_v3(
        requests, reviewer, tmp_path / "batch.v321-validator-v3.jsonl",
    )

    assert result["production_reviewer_version"] == "v3.2.1-production.2-validator-v3"
    assert result["lifecycle_schema_version"] == "v3_2_1_production_2_validator_v3"
    assert result["hard_validation_contract_version"] == "candidate_task_v3_structure_v3"
    assert result["retrieval_version"] == "global_lexical_rescue_v2"
    validator_evidence.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="validator-v3 calibration artifact hash differs"):
        prepare_production_batch_v3(requests, reviewer, tmp_path / "changed.jsonl")


def test_production_321_validator_v3_can_merge_frozen_production_1_chunks(
    tmp_path: Path,
) -> None:
    rows = [request(1), request(2)]
    requests = tmp_path / "requests.jsonl"
    inherited = tmp_path / "reviewer-v321-production-1.json"
    reviewer = tmp_path / "reviewer-v321-validator-v3.json"
    prompt_evidence = tmp_path / "prompt-calibration.json"
    validator_evidence = tmp_path / "validator-calibration.json"
    prompt_evidence.write_text("prompt evidence\n", encoding="utf-8")
    validator_evidence.write_text("validator evidence\n", encoding="utf-8")
    reviewer_manifest_v321(inherited, prompt_evidence)
    reviewer_manifest_v321_validator_v3(
        reviewer, inherited, prompt_evidence, validator_evidence,
    )
    write_jsonl(requests, rows)
    retrieval_manifest(requests)
    manifest = split_production_requests_v3(
        requests, inherited, tmp_path / "chunks", max_estimated_tokens=10_000,
        max_requests=1,
    )
    response_paths = []
    for chunk, row in zip(manifest["chunks"], rows):
        response_path = tmp_path / f"responses-{chunk['chunk_index']}.jsonl"
        write_jsonl(response_path, [response(row)])
        response_paths.append(response_path)

    result = merge_production_response_chunks_v3(
        tmp_path / "chunks/chunk_manifest.v3_2_1_production_1.json",
        response_paths, reviewer, tmp_path / "merged.jsonl", tmp_path / "report.json",
    )

    assert result["production_reviewer_version"] == "v3.2.1-production.2-validator-v3"
    assert result["source_production_reviewer_version"] == "v3.2.1-production.1"
    assert result["hard_validation_contract_version"] == "candidate_task_v3_structure_v3"


def test_production_321_validator_v3_merge_accepts_and_preserves_reverse_order(
    tmp_path: Path,
) -> None:
    row = request(1, ["subtitle_000001"])
    row["dialogue_candidates"] = [
        {"scene_id": "scene_001", "block_id": "scene_001_dialogue_001", "screenplay_order": 0, "speaker": "A", "text": "First"},
        {"scene_id": "scene_001", "block_id": "scene_001_dialogue_002", "screenplay_order": 1, "speaker": "B", "text": "Second"},
    ]
    requests = tmp_path / "requests.jsonl"
    inherited = tmp_path / "reviewer-v321-production-1.json"
    reviewer = tmp_path / "reviewer-v321-validator-v3.json"
    prompt_evidence = tmp_path / "prompt-calibration.json"
    validator_evidence = tmp_path / "validator-calibration.json"
    prompt_evidence.write_text("prompt evidence\n", encoding="utf-8")
    validator_evidence.write_text("validator evidence\n", encoding="utf-8")
    reviewer_manifest_v321(inherited, prompt_evidence)
    reviewer_manifest_v321_validator_v3(
        reviewer, inherited, prompt_evidence, validator_evidence,
    )
    write_jsonl(requests, [row])
    retrieval_manifest(requests)
    manifest = split_production_requests_v3(
        requests, inherited, tmp_path / "chunks", max_estimated_tokens=10_000,
        max_requests=1,
    )
    reversed_response = response(row)
    reversed_response["resolutions"][0].update({
        "block_ids": ["scene_001_dialogue_002", "scene_001_dialogue_001"],
        "decision_basis": "repeated_or_reordered_dialogue",
    })
    responses = tmp_path / "responses.jsonl"
    write_jsonl(responses, [reversed_response])

    result = merge_production_response_chunks_v3(
        tmp_path / "chunks/chunk_manifest.v3_2_1_production_1.json",
        [responses], reviewer, tmp_path / "merged.jsonl", tmp_path / "report.json",
    )

    assert result["passed"]
    merged = json.loads((tmp_path / "merged.jsonl").read_text())
    assert merged["resolutions"][0]["block_ids"] == [
        "scene_001_dialogue_002", "scene_001_dialogue_001",
    ]


def test_production_chunk_packing_is_exact_ordered_bounded_and_write_once(tmp_path: Path) -> None:
    rows = [request(i) for i in range(1, 6)]
    requests = tmp_path / "requests.jsonl"
    reviewer = tmp_path / "reviewer.json"
    chunks_dir = tmp_path / "chunks"
    write_jsonl(requests, rows)
    reviewer_manifest(reviewer)

    result = split_production_requests_v3(
        requests, reviewer, chunks_dir, max_estimated_tokens=10_000, max_requests=2,
    )

    assert result["chunk_count"] == 3
    assert [chunk["request_count"] for chunk in result["chunks"]] == [2, 2, 1]
    assert all(chunk["conservative_estimated_enqueued_tokens"] <= 10_000 for chunk in result["chunks"])
    reconstructed = []
    for chunk in result["chunks"]:
        path = Path(chunk["requests_path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == chunk["requests_sha256"]
        reconstructed.extend(json.loads(line)["request_id"] for line in path.read_text().splitlines())
    assert reconstructed == [row["request_id"] for row in rows]
    with pytest.raises(FileExistsError):
        split_production_requests_v3(
            requests, reviewer, chunks_dir, max_estimated_tokens=10_000, max_requests=2,
        )


def test_production_chunk_response_merge_validates_coverage_and_v3_schema(tmp_path: Path) -> None:
    rows = [request(i) for i in range(1, 6)]
    requests = tmp_path / "requests.jsonl"
    reviewer = tmp_path / "reviewer.json"
    write_jsonl(requests, rows)
    reviewer_manifest(reviewer)
    manifest = split_production_requests_v3(
        requests, reviewer, tmp_path / "chunks", max_estimated_tokens=10_000, max_requests=2,
    )
    response_paths = []
    by_id = {row["request_id"]: row for row in rows}
    for chunk in manifest["chunks"]:
        chunk_requests = [json.loads(line) for line in Path(chunk["requests_path"]).read_text().splitlines()]
        path = tmp_path / f"responses-{chunk['chunk_index']}.jsonl"
        write_jsonl(path, [response(row) for row in reversed(chunk_requests)])
        response_paths.append(path)
    output = tmp_path / "merged.jsonl"
    report = tmp_path / "report.json"
    result = merge_production_response_chunks_v3(
        tmp_path / "chunks/chunk_manifest.v3_2_production_1.json",
        response_paths, reviewer, output, report,
    )
    assert result["passed"] and result["request_count"] == 5
    assert [json.loads(line)["request_id"] for line in output.read_text().splitlines()] == [
        row["request_id"] for row in rows
    ]

    reviewer_v2 = tmp_path / "reviewer-v2.json"
    reviewer_manifest_v2(reviewer_v2, reviewer)
    inherited_result = merge_production_response_chunks_v3(
        tmp_path / "chunks/chunk_manifest.v3_2_production_1.json",
        response_paths, reviewer_v2, tmp_path / "merged-v2.jsonl", tmp_path / "report-v2.json",
    )
    assert inherited_result["production_reviewer_version"] == "v3.2-production.2"
    assert inherited_result["source_production_reviewer_version"] == "v3.2-production.1"
    assert inherited_result["lifecycle_schema_version"] == "v3_production_2_chunked"

    missing = tmp_path / "missing.jsonl"
    write_jsonl(missing, [])
    with pytest.raises(ValueError, match="exactly cover"):
        merge_production_response_chunks_v3(
            tmp_path / "chunks/chunk_manifest.v3_2_production_1.json",
            [missing, *response_paths[1:]], reviewer, tmp_path / "missing-out.jsonl", tmp_path / "missing-report.json",
        )

    historical = response(by_id[manifest["chunks"][0]["first_request_id"]])
    historical["resolutions"][0]["decision"] = "no_match"
    historical["resolutions"][0]["block_ids"] = []
    invalid = tmp_path / "historical.jsonl"
    first_chunk_rows = [json.loads(line) for line in Path(manifest["chunks"][0]["requests_path"]).read_text().splitlines()]
    write_jsonl(invalid, [historical, *[response(row) for row in first_chunk_rows[1:]]])
    with pytest.raises(ValueError, match="invalid v3 response"):
        merge_production_response_chunks_v3(
            tmp_path / "chunks/chunk_manifest.v3_2_production_1.json",
            [invalid, *response_paths[1:]], reviewer, tmp_path / "invalid-out.jsonl", tmp_path / "invalid-report.json",
        )


def test_production_merge_uses_v3_validator_and_preserves_full_order(tmp_path: Path) -> None:
    rows = [request(i) for i in range(1, 4)]; full = tmp_path / "full.jsonl"; pilot = tmp_path / "pilot.jsonl"; remaining = tmp_path / "remaining.jsonl"
    pilot_responses = tmp_path / "pilot-responses.jsonl"; remaining_responses = tmp_path / "remaining-responses.jsonl"; reviewer = tmp_path / "reviewer.json"
    write_jsonl(full, rows); write_jsonl(pilot, [rows[1]]); write_jsonl(remaining, [rows[0], rows[2]])
    write_jsonl(pilot_responses, [response(rows[1])]); write_jsonl(remaining_responses, [response(rows[2]), response(rows[0])]); reviewer_manifest(reviewer)
    output = tmp_path / "merged.jsonl"; report = tmp_path / "merge-report.json"
    result = merge_production_responses_v3(full, pilot, remaining, pilot_responses, remaining_responses, reviewer, output, report)
    assert result["passed"] and result["production_reviewer_version"] == PRODUCTION_REVIEWER_VERSION
    assert [json.loads(x)["request_id"] for x in output.read_text().splitlines()] == [r["request_id"] for r in rows]
    historical = response(rows[0]); historical["resolutions"][0]["decision"] = "no_match"; historical["resolutions"][0]["block_ids"] = []
    write_jsonl(remaining_responses, [historical, response(rows[2])])
    with pytest.raises(ValueError, match="production v3 response merge"):
        merge_production_responses_v3(full, pilot, remaining, pilot_responses, remaining_responses, reviewer, tmp_path / "other.jsonl", tmp_path / "other-report.json")


def test_production_apply_translates_binary_decision_and_preserves_provenance(tmp_path: Path) -> None:
    ids = ["subtitle_000001", "subtitle_000002"]; req = request(1, ids); requests = tmp_path / "requests.jsonl"; responses = tmp_path / "responses.jsonl"
    alignment = tmp_path / "subtitle_script_alignment.jsonl"; context = tmp_path / "movie_script_context.json"; shots = tmp_path / "shots.jsonl"; reviewer = tmp_path / "reviewer.json"
    req["dialogue_candidates"].append({
        "scene_id": "scene_001", "block_id": "scene_001_dialogue_003",
        "screenplay_order": 2, "speaker": "A", "text": "dialogue 3",
    })
    production_response = response(req, no_match_last=True)
    production_response["resolutions"][0]["block_ids"] = ["scene_001_dialogue_001", "scene_001_dialogue_003"]
    write_jsonl(requests, [req]); write_jsonl(responses, [production_response]); reviewer_manifest(reviewer)
    blocks = [{"block_id": f"scene_001_dialogue_{i:03d}", "block_type": "dialogue", "source_order": i, "script_page": 1, "speaker": "A", "character_cue": "A", "parenthetical": None, "text": f"dialogue {i}"} for i in (1, 2, 3)]
    scene = {"scene_id": "scene_001", "screenplay_scene_id": "1", "slugline": "INT. ROOM", "int_ext": "INT", "location": "ROOM", "time_of_day": None, "script_pages": {"start": 1, "end": 1}, "scene_characters": ["A"], "script_blocks": blocks, "semantic_annotations": {}, "parsing": {"status": "parsed", "needs_review": False}}
    context.write_text(json.dumps({"schema_version": "1.0", "movie": {"movie_id": "tt1"}, "script_scenes": [scene]}), encoding="utf-8")
    baseline = []
    for i, sid in enumerate(ids):
        baseline.append({"movie_id": "tt1", "subtitle_id": sid, "alignment_group_id": f"g{i}", "time": {"start": f"00:00:0{i}.000", "end": f"00:00:0{i}.800", "start_sec": float(i), "end_sec": float(i)+.8}, "text": sid, "scene_id": "scene_001", "script_matches": [], "alignment": {"method": "no_match", "status": "needs_review", "candidate_margin": 0., "needs_review": True, "reliable_anchor": False, "script_order_start": None, "script_order_end": None}})
    write_jsonl(alignment, baseline)
    shot_rows = [{"shot_id": f"shot_{i+1:06d}", "start_frame": i*10, "end_frame": (i+1)*10, "frame_count": 10, "start_time": f"00:00:0{i}.000", "end_time": f"00:00:0{i+1}.000", "start_sec": float(i), "end_sec": float(i+1), "duration_sec": 1., "keyframe_frame": i*10+4, "keyframe_time_sec": i+.4, "keyframe_relpath": f"keyframes/shot_{i+1:06d}.jpg"} for i in range(2)]
    write_jsonl(shots, shot_rows); before = hashlib.sha256(alignment.read_bytes()).hexdigest()
    seed_dir = tmp_path / "interrupted-seed"
    seed = apply_production_responses_v3(alignment, requests, responses, context, shots, seed_dir, reviewer)
    partial_normalized = tmp_path / "review/openai/validated_responses.v3_2_production_1.apply_normalized.jsonl"
    partial_normalized.parent.mkdir(parents=True)
    shutil.copyfile(seed["normalized_responses"], partial_normalized)
    result = apply_production_responses_v3(alignment, requests, responses, context, shots, tmp_path, reviewer)
    assert result["baseline_files_unchanged"] and hashlib.sha256(alignment.read_bytes()).hexdigest() == before
    reviewed = [json.loads(x) for x in Path(result["alignment_output"]).read_text().splitlines()]
    assert reviewed[0]["alignment"]["status"] == "llm_aligned"
    assert [item["block_id"] for item in reviewed[0]["script_matches"]] == [
        "scene_001_dialogue_001", "scene_001_dialogue_003",
    ]
    assert reviewed[1]["alignment"]["status"] == "llm_no_match"
    assert reviewed[1]["alignment"]["llm_resolution"]["original_openai_resolution"]["decision"] == "no_candidate_match"
    resumed = apply_production_responses_v3(alignment, requests, responses, context, shots, tmp_path, reviewer)
    assert resumed["resumed"] is True
    reviewer_v2 = tmp_path / "reviewer-v2.json"
    reviewer_manifest_v2(reviewer_v2, reviewer)
    v2 = apply_production_responses_v3(alignment, requests, responses, context, shots, tmp_path, reviewer_v2)
    assert v2["production_reviewer_version"] == "v3.2-production.2"
    assert v2["alignment_output"].endswith("subtitle_script_alignment.llm_reviewed_v3_2_production_2.jsonl")
    assert v2["shot_output"].endswith("shot_script_context.llm_reviewed_v3_2_production_2.jsonl")

    evidence = tmp_path / "frozen-reference-v321.jsonl"
    reviewer_v321 = tmp_path / "reviewer-v321.json"
    evidence.write_text("frozen\n", encoding="utf-8")
    reviewer_manifest_v321(reviewer_v321, evidence)
    retrieval_manifest(requests)
    v321 = apply_production_responses_v3(
        alignment, requests, responses, context, shots, tmp_path / "v321", reviewer_v321,
    )
    assert v321["production_reviewer_version"] == "v3.2.1-production.1"
    assert v321["hard_validation_contract_version"] == "candidate_task_v3_structure_v2"
    assert v321["alignment_output"].endswith(
        "subtitle_script_alignment.llm_reviewed_v3_2_1_production_1.jsonl"
    )
    assert v321["shot_output"].endswith(
        "shot_script_context.llm_reviewed_v3_2_1_production_1.jsonl"
    )
