from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from oscardp.script_context.openai_schema import (
    V3_SYSTEM_INSTRUCTIONS,
    V321_VOCATIVE_SYSTEM_INSTRUCTIONS,
    V32_POLICY_SYSTEM_INSTRUCTIONS,
    alignment_response_schema,
    alignment_response_schema_v3,
)
from oscardp.script_context.openai_review import batch_line, submit_batch
from oscardp.script_context.stage253 import (
    batch_line_v3,
    batch_line_v321_vocative,
    batch_line_v32_policy,
    batch_line_v33_action_context,
    evaluate_pilot_v3,
    evaluate_independent_calibration_adjudicated_v3,
    evaluate_independent_calibration_v3,
    prepare_batch_v3,
    prepare_batch_v321_vocative,
    prepare_batch_v32_policy,
    prepare_batch_v33_action_context,
    prepare_review_action_context_v33,
    prepare_review_context_v31,
    submit_batch_v3,
    submit_batch_v321_vocative,
    submit_batch_v32_policy,
    submit_batch_v33_action_context,
    validate_batch_lines_v3,
    validate_batch_lines_v321_vocative,
    validate_batch_lines_v32_policy,
    validate_batch_lines_v33_action_context,
    validate_independent_calibration_reference,
    validate_resolution_v3,
)


def test_v321_vocative_candidate_changes_only_prompt_and_is_calibration_limited(tmp_path: Path, monkeypatch) -> None:
    req = request(); requests = tmp_path / "requests.jsonl"; policy = tmp_path / "policy.md"
    requests.write_text(json.dumps(req) + "\n", encoding="utf-8"); policy.write_text("frozen", encoding="utf-8")
    output = tmp_path / "batch.jsonl"
    manifest = prepare_batch_v321_vocative(requests, policy, output, "gpt-5.6-terra")
    row = json.loads(output.read_text())
    baseline = batch_line_v32_policy(req, "gpt-5.6-terra")
    assert row["body"]["instructions"] == V321_VOCATIVE_SYSTEM_INSTRUCTIONS
    assert {k: v for k, v in row["body"].items() if k != "instructions"} == {k: v for k, v in baseline["body"].items() if k != "instructions"}
    assert manifest["calibration_only"] and not validate_batch_lines_v321_vocative([row])
    tampered = json.loads(json.dumps(row)); tampered["body"]["instructions"] += " changed"
    assert validate_batch_lines_v321_vocative([tampered])
    files = SimpleNamespace(create=Mock(return_value=SimpleNamespace(id="file-1")))
    batches = SimpleNamespace(create=Mock(return_value=SimpleNamespace(id="batch-1", status="validating")))
    monkeypatch.setattr("oscardp.script_context.openai_review._client", lambda: SimpleNamespace(files=files, batches=batches))
    result = submit_batch_v321_vocative(output, tmp_path / "job.json", confirm_submit=True)
    assert result["reviewer_version"] == "v3.2.1-vocative-candidate"
    many = tmp_path / "many.jsonl"
    many.write_text("".join(json.dumps({**req, "request_id": f"r{i}"}) + "\n" for i in range(31)), encoding="utf-8")
    with pytest.raises(ValueError, match="calibration-only"):
        prepare_batch_v321_vocative(many, policy, tmp_path / "too-many.jsonl", "gpt-5.6-terra")


def request(subtitle_ids: list[str] | None = None) -> dict:
    ids = subtitle_ids or ["s1"]
    return {
        "request_id": "r1", "subtitle_ids": ids,
        "subtitles": [{"subtitle_id": value, "text": value, "time": {"start": "00:00:00.000", "end": "00:00:01.000"}} for value in ids],
        "dialogue_candidates": [
            {"block_id": block_id, "scene_id": "scene_1", "screenplay_order": index, "speaker": "A", "text": text}
            for index, (block_id, text) in enumerate((("A", "The exact turn"), ("B", "A related but different proposition"), ("C", "A repeated turn")))
        ],
    }


def resolution(subtitle_id: str, decision: str, block_ids: list[str], basis: str) -> dict:
    return {"subtitle_id": subtitle_id, "decision": decision, "block_ids": block_ids, "confidence": .9, "decision_basis": basis}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize("basis", ["exact_or_near_exact", "paraphrase", "expanded_or_contracted_turn", "subtitle_fragment"])
def test_v3_match_bases_are_structurally_valid(basis: str) -> None:
    req = request(); response = {"request_id": "r1", "resolutions": [resolution("s1", "match", ["A"], basis)]}
    assert validate_resolution_v3(response, req)[0] == []


@pytest.mark.parametrize("case", ["new proposition", "courtesy filler", "physical protest", "telegram insert"])
def test_v3_no_supplied_candidate_cases_are_valid(case: str) -> None:
    req = request(); req["subtitles"][0]["text"] = case
    response = {"request_id": "r1", "resolutions": [resolution("s1", "no_candidate_match", [], "no_supplied_candidate")]}
    assert validate_resolution_v3(response, req)[0] == []


def test_policy_instructions_cover_new_propositions_and_graphic_text_without_movie_ids() -> None:
    assert "new proposition" in V3_SYSTEM_INSTRUCTIONS
    assert "telegram" in V3_SYSTEM_INSTRUCTIONS and "visible insert text is not dialogue" in V3_SYSTEM_INSTRUCTIONS
    assert "same speaker" in V3_SYSTEM_INSTRUCTIONS and "same proposition" in V3_SYSTEM_INSTRUCTIONS
    assert "tt27847051" not in V3_SYSTEM_INSTRUCTIONS and "subtitle_" not in V3_SYSTEM_INSTRUCTIONS


def test_v32_policy_is_generic_and_covers_observed_failure_classes() -> None:
    policy = V32_POLICY_SYSTEM_INSTRUCTIONS
    for phrase in ("short replies", "expanded or contracted", "spelling variation", "multiple speakers", "Do not under-select", "Preserve negative discrimination"):
        assert phrase in policy
    assert "nearby subtitles" in policy and "are not evidence" in policy
    assert "tt27847051" not in policy and "subtitle_" not in policy


def test_reordered_dialogue_can_choose_specific_earlier_candidate() -> None:
    req = request(["s1", "s2"])
    response = {"request_id": "r1", "resolutions": [
        resolution("s1", "match", ["C"], "exact_or_near_exact"),
        resolution("s2", "match", ["A"], "repeated_or_reordered_dialogue"),
    ]}
    errors, diagnostics = validate_resolution_v3(response, req)
    assert errors == []
    assert diagnostics["sequence_quality"]["backward_mapping_count"] == 1


def test_vocative_attachment_basis_is_valid() -> None:
    req = request(); response = {"request_id": "r1", "resolutions": [resolution("s1", "match", ["A"], "vocative_attachment")]}
    assert validate_resolution_v3(response, req)[0] == []


def test_request_specific_v3_schema_and_local_validation_reject_foreign_id() -> None:
    req = request(); schema = alignment_response_schema_v3(req)
    enum = schema["properties"]["resolutions"]["items"]["properties"]["block_ids"]["items"]["enum"]
    assert enum == ["A", "B", "C"] and "D" not in enum
    response = {"request_id": "r1", "resolutions": [resolution("s1", "match", ["D"], "exact_or_near_exact")]}
    assert any("outside request" in error for error in validate_resolution_v3(response, req)[0])


def test_v3_decision_block_contract_is_enforced_locally() -> None:
    req = request()
    empty_match = {"request_id": "r1", "resolutions": [resolution("s1", "match", [], "other")]}
    selecting_non_match = {"request_id": "r1", "resolutions": [resolution("s1", "no_candidate_match", ["A"], "other")]}
    assert any("match requires" in error for error in validate_resolution_v3(empty_match, req)[0])
    assert any("requires empty" in error for error in validate_resolution_v3(selecting_non_match, req)[0])


def test_v3_non_adjacent_same_scene_selection_is_quality_risk_not_malformed() -> None:
    req = request()
    response = {"request_id": "r1", "resolutions": [
        resolution("s1", "match", ["A", "C"], "expanded_or_contracted_turn"),
    ]}
    errors, diagnostics = validate_resolution_v3(response, req)
    assert errors == []
    quality = diagnostics["sequence_quality"]
    assert quality["non_adjacent_resolution_count"] == 1
    assert quality["cross_scene_resolution_count"] == 0
    assert any(event["reason"] == "non_adjacent_blocks_within_resolution" for event in quality["events"])


def test_v3_ordered_cross_scene_selection_is_quality_risk_not_malformed() -> None:
    req = request()
    req["dialogue_candidates"][2]["scene_id"] = "scene_2"
    response = {"request_id": "r1", "resolutions": [
        resolution("s1", "match", ["A", "C"], "expanded_or_contracted_turn"),
    ]}
    errors, diagnostics = validate_resolution_v3(response, req)
    assert errors == []
    quality = diagnostics["sequence_quality"]
    assert quality["non_adjacent_resolution_count"] == 1
    assert quality["cross_scene_resolution_count"] == 1
    assert quality["high_risk_sequence_event_count"] == 2


def test_v3_multi_block_selection_must_preserve_candidate_order() -> None:
    req = request()
    response = {"request_id": "r1", "resolutions": [
        resolution("s1", "match", ["C", "A"], "repeated_or_reordered_dialogue"),
    ]}
    errors, _diagnostics = validate_resolution_v3(response, req)
    assert any("preserve request candidate order" in error for error in errors)


def test_v3_structure_v3_treats_reversed_block_order_as_quality_risk() -> None:
    req = request()
    response = {"request_id": "r1", "resolutions": [
        resolution("s1", "match", ["C", "A"], "repeated_or_reordered_dialogue"),
    ]}
    errors, diagnostics = validate_resolution_v3(
        response, req, "candidate_task_v3_structure_v3",
    )
    assert errors == []
    quality = diagnostics["sequence_quality"]
    assert quality["reversed_block_order_resolution_count"] == 1
    assert quality["non_adjacent_resolution_count"] == 1
    assert quality["high_risk_sequence_event_count"] == 2
    assert any(event["reason"] == "reversed_block_order_within_resolution" for event in quality["events"])


def test_prepare_v3_batch_preserves_payload_and_policy_manifest(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"; policy = tmp_path / "policy.md"; output = tmp_path / "batch.jsonl"
    req = request(); write_jsonl(requests, [req]); policy.write_text("frozen policy", encoding="utf-8")
    manifest = prepare_batch_v3(requests, policy, output, "test-model")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert validate_batch_lines_v3(rows) == []
    assert manifest["review_policy_version"] == "annotation_policy_v1"
    assert manifest["decision_schema_version"] == "candidate_task_v3"
    assert manifest["request_payload_data_unchanged"] is True
    assert manifest["request_context_versions"] == ["v3_no_nearby_subtitle_context"]


def test_prepare_v31_context_adds_only_non_target_nearby_subtitles(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"; alignment = tmp_path / "alignment.jsonl"; output = tmp_path / "requests-v31.jsonl"
    req = request(["s2", "s3"]); write_jsonl(requests, [req])
    write_jsonl(alignment, [
        {"subtitle_id": f"s{index}", "text": f"subtitle {index}", "time": {"start_sec": float(index), "end_sec": float(index) + .5}}
        for index in range(1, 6)
    ])
    manifest = prepare_review_context_v31(requests, alignment, output, radius=1)
    augmented = json.loads(output.read_text(encoding="utf-8"))
    assert augmented["subtitle_ids"] == req["subtitle_ids"]
    assert augmented["dialogue_candidates"] == req["dialogue_candidates"]
    assert [row["subtitle_id"] for row in augmented["review_context"]["before"]] == ["s1"]
    assert [row["subtitle_id"] for row in augmented["review_context"]["after"]] == ["s4"]
    assert set(augmented["review_context"]["before"][0]) == {"subtitle_id", "text", "time"}
    assert manifest["target_ids_unchanged"] and manifest["candidate_ids_unchanged"]
    assert manifest["gold_labels_included"] is False


def test_v31_batch_manifest_records_context_design_without_changing_policy(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"; alignment = tmp_path / "alignment.jsonl"; augmented = tmp_path / "augmented.jsonl"
    policy = tmp_path / "policy.md"; batch = tmp_path / "batch.jsonl"
    write_jsonl(requests, [request(["s2"])]); write_jsonl(alignment, [
        {"subtitle_id": f"s{index}", "text": f"subtitle {index}", "time": {"start_sec": index, "end_sec": index + .5}}
        for index in range(1, 4)
    ]); policy.write_text("frozen policy", encoding="utf-8")
    prepare_review_context_v31(requests, alignment, augmented, radius=1)
    manifest = prepare_batch_v3(augmented, policy, batch, "test-model")
    assert manifest["request_context_versions"] == ["v3.1_nearby_subtitles"]
    assert manifest["request_context_design"]["gold_labels_included"] is False
    row = json.loads(batch.read_text(encoding="utf-8"))
    assert row["body"]["instructions"] == V3_SYSTEM_INSTRUCTIONS


def test_prepare_v32_changes_only_policy_and_preserves_v3_schema(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"; policy = tmp_path / "policy.md"; output = tmp_path / "batch-v32.jsonl"
    write_jsonl(requests, [request()]); policy.write_text("frozen policy", encoding="utf-8")
    manifest = prepare_batch_v32_policy(requests, policy, output, "test-model")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert validate_batch_lines_v32_policy(rows) == []
    assert manifest["changed_layer"] == "reviewer_prompt_policy_only"
    assert manifest["request_context_version"] == "v3_no_nearby_subtitle_context"
    assert rows[0]["body"]["instructions"] == V32_POLICY_SYSTEM_INSTRUCTIONS
    assert rows[0]["body"]["text"]["format"]["schema"] == alignment_response_schema_v3(request())
    assert validate_batch_lines_v3(rows)


def screenplay_context() -> dict:
    return {"script_scenes": [{
        "scene_id": "scene_1", "script_blocks": [
            {"block_id": "act_1", "block_type": "action", "source_order": 1, "text": "She remains outside."},
            {"block_id": "A", "block_type": "dialogue", "source_order": 2, "text": "The exact turn", "speaker": "A"},
            {"block_id": "act_2", "block_type": "action", "source_order": 3, "text": "He pounds on the door."},
            {"block_id": "B", "block_type": "dialogue", "source_order": 4, "text": "A related turn", "speaker": "A"},
            {"block_id": "act_3", "block_type": "action", "source_order": 5, "text": "She opens it."},
            {"block_id": "C", "block_type": "dialogue", "source_order": 6, "text": "A repeated turn", "speaker": "A"},
        ],
    }]}


def test_v33_action_context_is_bounded_non_selectable_and_preserves_task(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"; screenplay = tmp_path / "screenplay.json"; output = tmp_path / "requests-v33.jsonl"
    req = request(); write_jsonl(requests, [req]); screenplay.write_text(json.dumps(screenplay_context()), encoding="utf-8")
    manifest = prepare_review_action_context_v33(requests, screenplay, output, radius=2, max_actions=2)
    augmented = json.loads(output.read_text(encoding="utf-8"))
    assert augmented["subtitle_ids"] == req["subtitle_ids"]
    assert augmented["dialogue_candidates"] == req["dialogue_candidates"]
    actions = augmented["review_context"]["screenplay_action_blocks"]
    assert len(actions) == 2
    assert all(row["block_type"] == "action" and row["selectable"] is False for row in actions)
    assert all(set(row) == {"block_id", "scene_id", "source_order", "text", "block_type", "selectable"} for row in actions)
    assert manifest["changed_layer"] == "request_context_only"
    assert manifest["gold_labels_included"] is False


def test_v33_batch_keeps_v32_policy_and_v3_candidate_enum(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"; screenplay = tmp_path / "screenplay.json"; augmented = tmp_path / "requests-v33.jsonl"
    policy = tmp_path / "policy.md"; batch = tmp_path / "batch-v33.jsonl"
    write_jsonl(requests, [request()]); screenplay.write_text(json.dumps(screenplay_context()), encoding="utf-8"); policy.write_text("frozen", encoding="utf-8")
    prepare_review_action_context_v33(requests, screenplay, augmented)
    manifest = prepare_batch_v33_action_context(augmented, policy, batch, "test-model")
    row = json.loads(batch.read_text(encoding="utf-8"))
    assert validate_batch_lines_v33_action_context([row]) == []
    assert row["body"]["instructions"] == V32_POLICY_SYSTEM_INSTRUCTIONS
    enum = row["body"]["text"]["format"]["schema"]["properties"]["resolutions"]["items"]["properties"]["block_ids"]["items"]["enum"]
    assert enum == ["A", "B", "C"] and "act_1" not in enum
    assert manifest["decision_schema_version"] == "candidate_task_v3"
    assert validate_batch_lines_v32_policy([row])


def test_v33_submit_mocked_lifecycle_uses_own_validator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requests = tmp_path / "requests.jsonl"; screenplay = tmp_path / "screenplay.json"; augmented = tmp_path / "requests-v33.jsonl"
    batch = tmp_path / "batch-v33.jsonl"; job = tmp_path / "job-v33.json"
    write_jsonl(requests, [request()]); screenplay.write_text(json.dumps(screenplay_context()), encoding="utf-8")
    prepare_review_action_context_v33(requests, screenplay, augmented)
    write_jsonl(batch, [batch_line_v33_action_context(json.loads(augmented.read_text()), "test-model")])
    files = SimpleNamespace(create=Mock(return_value=SimpleNamespace(id="file-v33")))
    batches = SimpleNamespace(create=Mock(return_value=SimpleNamespace(id="batch-v33", status="validating")))
    monkeypatch.setattr("oscardp.script_context.openai_review._client", lambda: SimpleNamespace(files=files, batches=batches))
    result = submit_batch_v33_action_context(batch, job, confirm_submit=True)
    assert result["reviewer_version"] == "v3.3-action-context"
    assert result["request_context_version"] == "v3.3_nearby_screenplay_actions"
    files.create.assert_called_once(); batches.create.assert_called_once()


def test_independent_calibration_reference_is_self_contained_and_hash_validated(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"; reference = tmp_path / "reference.jsonl"
    manifest = tmp_path / "reference-manifest.json"; report = tmp_path / "reference-validation.json"
    req = request(); write_jsonl(requests, [req])
    row = {
        "schema_version": "1.0", "reference_type": "codex_provisional_independent_calibration_reference",
        "human_gold": False, "movie_id": "movie", "request_id": "r1", "request": req,
        "reference_resolutions": [{
            "subtitle_id": "s1", "subtitle_text": "s1", "decision": "match", "block_ids": ["A"],
            "reference_status": "resolved", "reviewer_notes": "evidence",
        }],
    }
    write_jsonl(reference, [row])
    manifest.write_text(json.dumps({
        "human_gold": False, "frozen_before_reviewer_output": True,
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "source_requests_sha256": hashlib.sha256(requests.read_bytes()).hexdigest(),
        "request_count": 1, "subtitle_resolution_count": 1,
        "decision_counts": {"match": 1, "no_candidate_match": 0},
    }), encoding="utf-8")
    result = validate_independent_calibration_reference(reference, requests, manifest, report)
    assert result["passed"] and result["resolution_count"] == 1
    row["reference_resolutions"][0]["block_ids"] = ["foreign"]
    write_jsonl(reference, [row])
    result = validate_independent_calibration_reference(reference, requests, manifest, report)
    assert not result["passed"]
    assert any("SHA-256" in error for error in result["errors"])
    assert any("foreign candidate" in error for error in result["errors"])


def test_independent_calibration_evaluator_reports_numeric_gate_and_not_human_gold(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"; responses = tmp_path / "validated.jsonl"
    pilot_manifest = tmp_path / "pilot-manifest.json"; validation = tmp_path / "response-validation.json"; output = tmp_path / "evaluation.json"
    req = request(); write_jsonl(reference, [{
        "request_id": "r1", "reference_resolutions": [{"subtitle_id": "s1", "decision": "match", "block_ids": ["A"]}],
    }])
    write_jsonl(responses, [{"request_id": "r1", "resolutions": [resolution("s1", "match", ["A"], "exact_or_near_exact")]}])
    pilot_manifest.write_text(json.dumps({"requests": [{"request_id": "r1", "stratum": "easy", "timeline_region": "early"}]}), encoding="utf-8")
    validation.write_text(json.dumps({"valid_count": 30, "invalid_count": 0, "foreign_candidate_output_count": 0, "sequence_quality": {}}), encoding="utf-8")
    result = evaluate_independent_calibration_v3(reference, responses, pilot_manifest, validation, output)
    assert result["human_gold"] is False
    assert result["metrics"]["candidate_task_accuracy"] == 1.0
    assert result["numeric_acceptance_gate"]["passed"] is True
    assert result["promotion_requires_error_class_audit"] is True


def test_adjudicated_calibration_evaluator_preserves_frozen_reference_and_constrains_corrections(tmp_path: Path) -> None:
    reference = tmp_path / "reference.jsonl"; responses = tmp_path / "validated.jsonl"
    pilot_manifest = tmp_path / "pilot-manifest.json"; validation = tmp_path / "response-validation.json"
    adjudication = tmp_path / "adjudication.jsonl"; output = tmp_path / "evaluation.json"
    req = request()
    write_jsonl(reference, [{
        "request_id": "r1", "request": req,
        "reference_resolutions": [{"subtitle_id": "s1", "decision": "no_candidate_match", "block_ids": []}],
    }])
    frozen_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    write_jsonl(responses, [{"request_id": "r1", "resolutions": [resolution("s1", "match", ["A"], "exact_or_near_exact")]}])
    pilot_manifest.write_text(json.dumps({"requests": [{"request_id": "r1", "stratum": "easy", "timeline_region": "early"}]}), encoding="utf-8")
    validation.write_text(json.dumps({"valid_count": 30, "invalid_count": 0, "foreign_candidate_output_count": 0, "sequence_quality": {}}), encoding="utf-8")
    write_jsonl(adjudication, [{
        "request_id": "r1", "subtitle_id": "s1", "adjudication": "correct_reference",
        "original_reference": {"decision": "no_candidate_match", "block_ids": []},
        "corrected_reference": {"decision": "match", "block_ids": ["A"]},
        "attribution_layer": "gold_annotation_policy", "evidence": "The supplied line is exact source evidence.",
    }])
    result = evaluate_independent_calibration_adjudicated_v3(reference, responses, pilot_manifest, validation, adjudication, output)
    assert result["frozen_reference_sha256"] == frozen_hash
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == frozen_hash
    assert result["reference_correction_count"] == 1
    assert result["resolved_adjudicated_metrics"]["candidate_task_accuracy"] == 1.0
    assert result["resolved_accuracy_by_stratum"]["easy"] == {"resolution_count": 1, "candidate_task_accuracy": 1.0}
    assert result["numeric_acceptance_gate"]["passed"] is True

    row = json.loads(adjudication.read_text())
    row["corrected_reference"]["block_ids"] = ["FOREIGN"]
    write_jsonl(adjudication, [row])
    with pytest.raises(ValueError, match="foreign candidate"):
        evaluate_independent_calibration_adjudicated_v3(reference, responses, pilot_manifest, validation, adjudication, output)


def test_v32_submit_mocked_lifecycle_uses_its_own_validator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = tmp_path / "batch-v32.jsonl"; job = tmp_path / "job-v32.json"
    write_jsonl(batch, [batch_line_v32_policy(request(), "test-model")])
    files = SimpleNamespace(create=Mock(return_value=SimpleNamespace(id="file-v32")))
    batches = SimpleNamespace(create=Mock(return_value=SimpleNamespace(id="batch-v32", status="validating")))
    monkeypatch.setattr("oscardp.script_context.openai_review._client", lambda: SimpleNamespace(files=files, batches=batches))
    result = submit_batch_v32_policy(batch, job, confirm_submit=True)
    assert result["reviewer_version"] == "v3.2-policy"
    assert result["decision_schema_version"] == "candidate_task_v3"
    files.create.assert_called_once(); batches.create.assert_called_once()


def test_historical_submit_uses_historical_validator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = tmp_path / "batch.jsonl"; write_jsonl(batch, [batch_line(request(), "test-model")])
    validator = Mock(return_value=["historical validator marker"])
    client = Mock(side_effect=AssertionError("client must not be accessed"))
    monkeypatch.setattr("oscardp.script_context.openai_review.validate_batch_lines", validator)
    monkeypatch.setattr("oscardp.script_context.openai_review._client", client)
    with pytest.raises(ValueError, match="historical validator marker"):
        submit_batch(batch, tmp_path / "job.json", confirm_submit=True)
    validator.assert_called_once()
    client.assert_not_called()


def test_v3_submit_mocked_lifecycle_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = tmp_path / "batch-v3.jsonl"; job = tmp_path / "job-v3.json"
    write_jsonl(batch, [batch_line_v3(request(), "test-model")])
    expected_hash = hashlib.sha256(batch.read_bytes()).hexdigest()
    files = SimpleNamespace(create=Mock(return_value=SimpleNamespace(id="file-v3")))
    batches = SimpleNamespace(create=Mock(return_value=SimpleNamespace(id="batch-v3", status="validating")))
    monkeypatch.setattr("oscardp.script_context.openai_review._client", lambda: SimpleNamespace(files=files, batches=batches))

    result = submit_batch_v3(batch, job, confirm_submit=True)

    assert result == json.loads(job.read_text(encoding="utf-8"))
    assert result["decision_schema_version"] == "candidate_task_v3"
    assert result["review_policy_version"] == "annotation_policy_v1"
    assert result["request_count"] == 1 and result["model"] == "test-model"
    assert result["batch_input_sha256"] == expected_hash
    files.create.assert_called_once()
    batches.create.assert_called_once_with(input_file_id="file-v3", endpoint="/v1/responses", completion_window="24h")


def test_versioned_submit_paths_reject_the_other_schema_before_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    historical = tmp_path / "historical.jsonl"; v3 = tmp_path / "v3.jsonl"
    write_jsonl(historical, [batch_line(request(), "test-model")])
    write_jsonl(v3, [batch_line_v3(request(), "test-model")])
    client = Mock(side_effect=AssertionError("client must not be accessed"))
    monkeypatch.setattr("oscardp.script_context.openai_review._client", client)

    with pytest.raises(ValueError, match="Invalid v3 Batch input"):
        submit_batch_v3(historical, tmp_path / "v3-job.json", confirm_submit=True)
    with pytest.raises(ValueError, match="Invalid batch input"):
        submit_batch(v3, tmp_path / "historical-job.json", confirm_submit=True)
    client.assert_not_called()


def test_v3_submit_rejects_tampered_candidate_enum_before_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    row = batch_line_v3(request(), "test-model")
    enum = row["body"]["text"]["format"]["schema"]["properties"]["resolutions"]["items"]["properties"]["block_ids"]["items"]["enum"]
    enum.append("FOREIGN")
    batch = tmp_path / "tampered-v3.jsonl"; write_jsonl(batch, [row])
    client = Mock(side_effect=AssertionError("client must not be accessed"))
    monkeypatch.setattr("oscardp.script_context.openai_review._client", client)

    with pytest.raises(ValueError, match="request-specific v3 schema"):
        submit_batch_v3(batch, tmp_path / "job.json", confirm_submit=True)
    client.assert_not_called()


def test_v3_submit_requires_confirmation_before_api_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    batch = tmp_path / "batch-v3.jsonl"; write_jsonl(batch, [batch_line_v3(request(), "test-model")])
    client = Mock(side_effect=AssertionError("client must not be accessed"))
    monkeypatch.setattr("oscardp.script_context.openai_review._client", client)

    with pytest.raises(RuntimeError, match="confirm-submit"):
        submit_batch_v3(batch, tmp_path / "job.json", confirm_submit=False)
    client.assert_not_called()


def test_v3_evaluator_binary_gold_and_ambiguous_exclusion(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"; validated = tmp_path / "validated_responses.jsonl"
    manifest = tmp_path / "manifest.json"; adjudication = tmp_path / "adjudication.jsonl"; output = tmp_path / "evaluation.json"
    write_jsonl(gold, [{"request_id": "r1", "resolutions": [
        {"subtitle_id": "s1", "decision": "match", "block_ids": ["A"]},
        {"subtitle_id": "s2", "decision": "no_match", "block_ids": []},
        {"subtitle_id": "s3", "decision": "uncertain", "block_ids": []},
        {"subtitle_id": "s4", "decision": "match", "block_ids": ["B"]},
    ]}])
    write_jsonl(validated, [{"request_id": "r1", "resolutions": [
        resolution("s1", "match", ["A"], "exact_or_near_exact"),
        resolution("s2", "no_candidate_match", [], "no_supplied_candidate"),
        resolution("s3", "no_candidate_match", [], "no_supplied_candidate"),
        resolution("s4", "match", ["A"], "paraphrase"),
    ]}])
    (tmp_path / "response_validation_report.json").write_text(json.dumps({"valid_count": 30, "invalid_count": 0, "foreign_candidate_output_count": 0, "sequence_quality": {"events": []}}))
    manifest.write_text(json.dumps({"requests": [{"request_id": "r1", "stratum": "easy", "timeline_region": "early"}]}))
    write_jsonl(adjudication, [{"subtitle_id": "s4", "human_adjudication": {"adjudication": "ambiguous"}}])
    result = evaluate_pilot_v3(gold, validated, manifest, adjudication, output)
    assert result["all_records_metrics"]["resolution_count"] == 4
    assert result["all_records_metrics"]["candidate_task_accuracy"] == .75
    assert result["resolved_gold_metrics"]["resolution_count"] == 3
    assert result["resolved_gold_metrics"]["candidate_task_accuracy"] == 1.0
    assert result["excluded_ambiguous_subtitle_ids"] == ["s4"]
    assert result["all_records_metrics"]["confusion_matrix"]["no_candidate_match"]["no_candidate_match"] == 2


def test_historical_three_way_schema_remains_unchanged() -> None:
    historical = alignment_response_schema()
    decisions = historical["properties"]["resolutions"]["items"]["properties"]["decision"]["enum"]
    assert decisions == ["match", "no_match", "uncertain"]
    assert "no_candidate_match" not in decisions
