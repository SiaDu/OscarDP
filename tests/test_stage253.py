from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from oscardp.script_context.openai_schema import (
    V3_SYSTEM_INSTRUCTIONS,
    alignment_response_schema,
    alignment_response_schema_v3,
)
from oscardp.script_context.openai_review import batch_line, submit_batch
from oscardp.script_context.stage253 import (
    batch_line_v3,
    evaluate_pilot_v3,
    prepare_batch_v3,
    prepare_review_context_v31,
    submit_batch_v3,
    validate_batch_lines_v3,
    validate_resolution_v3,
)


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
