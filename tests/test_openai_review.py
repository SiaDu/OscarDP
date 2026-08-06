from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from oscardp.script_context.openai_review import (
    _add_reviewed_status_counts,
    _download_text,
    apply_validated_responses,
    batch_line,
    check_batch,
    evaluate_pilot,
    fetch_batch,
    prepare_batch,
    submit_batch,
    validate_batch_lines,
    validate_resolution,
    validate_responses,
)
from oscardp.script_context.openai_schema import alignment_response_schema
from oscardp.script_context.pilot import prepare_pilot
from oscardp.script_context.shot_mapping import map_shots
from oscardp.script_context.validation import validate_data
from oscardp.shots.schema import json_dumps


def request(ordinal: int, kind: str, subtitle_ordinal: int) -> dict:
    subtitle_ids = [f"subtitle_{subtitle_ordinal:06d}"]
    method, status, matches = "normalized_substring", "needs_review", [{"block_id": f"scene_001_dialogue_{ordinal:03d}"}]
    insufficient = False
    if kind == "fuzzy": method = "rapidfuzz_token_span"
    if kind == "multi": subtitle_ids.append(f"subtitle_{subtitle_ordinal + 1:06d}")
    if kind == "difficult": status, matches = "no_match", []; insufficient = ordinal % 8 == 1
    return {
        "request_id": f"alignment_review_{ordinal:06d}", "subtitle_ids": subtitle_ids,
        "subtitles": [{"subtitle_id": value, "text": f"text {value}", "time": {"start_sec": subtitle_ordinal, "end_sec": subtitle_ordinal + .5}} for value in subtitle_ids],
        "previous_anchor": None, "next_anchor": None, "candidate_scenes": [] if insufficient else ["scene_001"],
        "dialogue_candidates": [] if insufficient else [{"scene_id": "scene_001", "block_id": f"scene_001_dialogue_{ordinal:03d}", "screenplay_order": ordinal, "speaker": "HART", "text": "candidate text"}],
        "automatic_candidate_mappings": [{"subtitle_id": value, "scene_id": None, "matches": matches, "alignment": {"method": method, "status": status, "needs_review": True}} for value in subtitle_ids],
        "reason_for_review": ["low_confidence"], "candidate_interval_reason": "fixture", "insufficient_candidates": insufficient,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_deterministic_pilot_has_30_stratified_requests_and_null_gold(tmp_path: Path) -> None:
    rows = []
    ordinal = 1
    for kind, count in (("easy", 12), ("fuzzy", 12), ("multi", 8), ("difficult", 8)):
        for index in range(count):
            rows.append(request(ordinal, kind, 1 + (ordinal * 23) % 299)); ordinal += 1
    requests = tmp_path / "requests.jsonl"; alignment = tmp_path / "alignment.jsonl"
    write_jsonl(requests, rows); write_jsonl(alignment, [{"subtitle_id": f"subtitle_{i:06d}"} for i in range(1, 301)])
    first = prepare_pilot(requests, alignment, tmp_path / "one")
    second = prepare_pilot(requests, alignment, tmp_path / "two")
    assert first["request_count"] == 30 and sum(first["strata"].values()) == 30
    assert all(first["strata"].values())
    assert all(first["timeline"].values())
    assert first["source_pool_distribution"]["request_count"] == 40
    assert (tmp_path / "one/pilot_requests.jsonl").read_bytes() == (tmp_path / "two/pilot_requests.jsonl").read_bytes()
    gold = [json.loads(line) for line in (tmp_path / "one/pilot_gold_template.jsonl").read_text().splitlines()]
    assert all(value is None for row in gold for resolution in row["resolutions"] for value in (resolution["decision"], resolution["block_ids"], resolution["reviewer_notes"]))


def test_pilot_backfills_missing_strata_and_uses_all_when_fewer_than_requested(tmp_path: Path) -> None:
    rows = [request(index, "easy", index) for index in range(1, 13)]
    requests = tmp_path / "requests.jsonl"; alignment = tmp_path / "alignment.jsonl"
    write_jsonl(requests, rows); write_jsonl(alignment, [{"subtitle_id": f"subtitle_{i:06d}"} for i in range(1, 13)])
    result = prepare_pilot(requests, alignment, tmp_path / "pilot", 30)
    assert result["request_count"] == 12
    assert result["strata"] == {"easy": 12, "fuzzy": 0, "multi": 0, "difficult": 0}
    selected = [json.loads(line) for line in (tmp_path / "pilot/pilot_requests.jsonl").read_text().splitlines()]
    assert len({row["request_id"] for row in selected}) == 12


def test_pilot_backfills_a_partial_stratum_to_requested_count(tmp_path: Path) -> None:
    rows = [request(index, "easy", index) for index in range(1, 28)]
    rows.extend(request(index, "difficult", index) for index in range(28, 34))
    requests = tmp_path / "requests.jsonl"; alignment = tmp_path / "alignment.jsonl"
    write_jsonl(requests, rows); write_jsonl(alignment, [{"subtitle_id": f"subtitle_{i:06d}"} for i in range(1, 40)])
    result = prepare_pilot(requests, alignment, tmp_path / "pilot", 30)
    assert result["request_count"] == 30
    assert result["strata"]["difficult"] >= 5
    assert result["strata"]["easy"] + result["strata"]["difficult"] == 30
    assert result["strata"]["fuzzy"] == result["strata"]["multi"] == 0


def test_pilot_reports_and_materially_represents_candidate_limit_saturation(tmp_path: Path) -> None:
    rows = [request(index, "easy", index) for index in range(1, 41)]
    for index, row in enumerate(rows):
        row["candidate_limit"] = 4
        if index < 16:
            row["dialogue_candidates"] = [
                {"scene_id": "scene_001", "block_id": f"b_{index}_{candidate}", "screenplay_order": candidate, "speaker": "HART", "text": "candidate"}
                for candidate in range(4)
            ]
    requests = tmp_path / "requests.jsonl"; alignment = tmp_path / "alignment.jsonl"
    write_jsonl(requests, rows); write_jsonl(alignment, [{"subtitle_id": f"subtitle_{i:06d}"} for i in range(1, 50)])
    result = prepare_pilot(requests, alignment, tmp_path / "pilot", 30)
    manifest = json.loads((tmp_path / "pilot/pilot_manifest.json").read_text())
    assert result["candidate_limit_saturation"]["saturated_requests"] == 12
    assert manifest["source_pool_distribution"]["candidate_limit_saturation"]["saturated_requests"] == 16
    assert manifest["diagnostic_balanced_pilot_distribution"]["candidate_limit_saturation"]["saturated_requests"] == 12
    assert sum(row["candidate_limit_saturated"] for row in manifest["requests"]) == 12
    assert manifest["selection_design"] == "diagnostic_balanced"
    assert manifest["statistically_representative"] is False
    assert manifest["evaluation_reporting"]["source_weighted_overall_accuracy"] is True


def test_batch_uses_unique_custom_ids_and_strict_responses_schema(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"; output = tmp_path / "batch.jsonl"
    write_jsonl(requests, [request(1, "easy", 1), request(2, "easy", 2)])
    manifest = prepare_batch(requests, output, "test-model")
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert manifest["request_count"] == 2 and validate_batch_lines(rows) == []
    assert len({row["custom_id"] for row in rows}) == 2
    assert rows[0]["body"]["text"]["format"]["schema"] == alignment_response_schema()


def valid_response(req: dict) -> dict:
    candidate = req["dialogue_candidates"][0]["block_id"]
    return {"request_id": req["request_id"], "resolutions": [{"subtitle_id": value, "decision": "match", "block_ids": [candidate], "confidence": .9, "decision_basis": "substring_or_minor_edit"} for value in req["subtitle_ids"]]}


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda value: value["resolutions"].pop(), "exactly match"),
    (lambda value: value["resolutions"].append(dict(value["resolutions"][0])), "duplicate subtitle"),
    (lambda value: value["resolutions"][0].update(subtitle_id="foreign"), "exactly match"),
    (lambda value: value["resolutions"][0].update(block_ids=["foreign"]), "outside request"),
    (lambda value: value["resolutions"][0].update(decision="no_match"), "requires empty"),
    (lambda value: value["resolutions"][0].update(confidence=2), "invalid confidence"),
])
def test_invalid_structured_resolutions_are_rejected(mutate, message: str) -> None:
    req = request(1, "multi", 1); response = valid_response(req); mutate(response)
    assert any(message in error for error in validate_resolution(response, req))


def test_non_monotonic_block_selection_is_rejected() -> None:
    req = request(1, "multi", 1)
    req["dialogue_candidates"].append({"scene_id": "scene_001", "block_id": "scene_001_dialogue_002", "screenplay_order": 2, "speaker": "HART", "text": "two"})
    response = valid_response(req); response["resolutions"][0]["block_ids"] = ["scene_001_dialogue_002"]; response["resolutions"][1]["block_ids"] = ["scene_001_dialogue_001"]
    assert "non-monotonic block selection" in validate_resolution(response, req)


def interval_request(spans: list[list[int]]) -> tuple[dict, dict]:
    req = request(1, "easy", 1)
    req["subtitle_ids"] = [f"subtitle_{index + 1:06d}" for index in range(len(spans))]
    req["dialogue_candidates"] = [
        {"scene_id": "scene_001", "block_id": f"block_{index}", "screenplay_order": index, "speaker": "HART", "text": f"candidate {index}"}
        for index in range(6)
    ]
    response = {
        "request_id": req["request_id"],
        "resolutions": [{
            "subtitle_id": subtitle_id, "decision": "match",
            "block_ids": [f"block_{index}" for index in span],
            "confidence": .9, "decision_basis": "substring_or_minor_edit",
        } for subtitle_id, span in zip(req["subtitle_ids"], spans)],
    }
    return req, response


@pytest.mark.parametrize("spans", [
    [[1], [1]],
    [[1, 2], [1, 2]],
    [[0, 1], [1, 2]],
    [[0, 1], [2, 3], [2, 3]],
])
def test_interval_monotonicity_allows_reuse_and_forward_overlap(spans: list[list[int]]) -> None:
    req, response = interval_request(spans)
    assert "non-monotonic block selection" not in validate_resolution(response, req)


@pytest.mark.parametrize("spans", [
    [[2, 3], [1, 2]],
    [[0, 1, 2], [1]],
])
def test_interval_monotonicity_rejects_start_or_end_regression(spans: list[list[int]]) -> None:
    req, response = interval_request(spans)
    assert "non-monotonic block selection" in validate_resolution(response, req)


def raw_batch_row(req: dict, response: dict) -> dict:
    body = {"id": "resp_1", "model": "test-model", "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(response)}]}]}
    return {"custom_id": req["request_id"], "response": {"status_code": 200, "body": body}, "error": None}


def tiny_context() -> dict:
    blocks = [{"block_id": f"scene_001_dialogue_{i:03d}", "block_type": "dialogue", "script_page": 1, "source_order": i, "speaker": "HART", "character_cue": "HART", "parenthetical": None, "text": f"dialogue {i}"} for i in range(1, 4)]
    scene = {"scene_id": "scene_001", "screenplay_scene_id": "1", "slugline": "INT. ROOM - NIGHT", "int_ext": "INT", "location": "ROOM", "time_of_day": "NIGHT", "script_pages": {"start": 1, "end": 1}, "scene_characters": ["HART"], "script_blocks": blocks, "semantic_annotations": {"scene_summary": None, "dramatic_function": None}, "parsing": {"status": "parsed", "needs_review": False}}
    return {"schema_version": "1.0", "parser_version": "2.2", "movie": {"movie_id": "tt1", "title": "Test"}, "source_files": {}, "summary": {}, "broken_pages": [], "script_scenes": [scene]}


def baseline_row(index: int) -> dict:
    return {"movie_id": "tt1", "subtitle_id": f"subtitle_{index:06d}", "alignment_group_id": f"align_{index:06d}", "time": {"start": f"00:00:0{index}.000", "end": f"00:00:0{index}.800", "start_sec": float(index), "end_sec": float(index)+.8}, "text": f"subtitle {index}", "scene_id": "scene_001", "script_matches": [{"block_id": f"scene_001_dialogue_{index:03d}", "speaker": "HART", "matched_text": f"dialogue {index}", "combined_score": .7}], "alignment": {"method": "rapidfuzz", "status": "needs_review", "candidate_margin": .1, "needs_review": True, "reliable_anchor": False, "script_order_start": 99, "script_order_end": 99}}


def shot(index: int) -> dict:
    start, end = float(index - 1), float(index)
    return {"shot_id": f"shot_{index:06d}", "start_frame": (index-1)*10, "end_frame": index*10, "frame_count": 10, "start_time": f"00:00:{index-1:02d}.000", "end_time": f"00:00:{index:02d}.000", "start_sec": start, "end_sec": end, "duration_sec": 1., "keyframe_frame": (index-1)*10+4, "keyframe_time_sec": start+.4, "keyframe_relpath": f"keyframes/shot_{index:06d}.jpg"}


def test_mocked_validate_and_safe_apply_preserves_baselines(tmp_path: Path) -> None:
    req = request(1, "multi", 1); req["dialogue_candidates"] = [{"scene_id": "scene_001", "block_id": f"scene_001_dialogue_{i:03d}", "screenplay_order": i-1, "speaker": "HART", "text": f"dialogue {i}"} for i in range(1, 4)]
    req["subtitle_ids"] = ["subtitle_000001", "subtitle_000002", "subtitle_000003"]
    response = {"request_id": req["request_id"], "resolutions": [
        {"subtitle_id": "subtitle_000001", "decision": "match", "block_ids": ["scene_001_dialogue_001"], "confidence": .95, "decision_basis": "exact_or_near_exact"},
        {"subtitle_id": "subtitle_000002", "decision": "no_match", "block_ids": [], "confidence": .9, "decision_basis": "changed_or_improvised_dialogue"},
        {"subtitle_id": "subtitle_000003", "decision": "uncertain", "block_ids": [], "confidence": .4, "decision_basis": "insufficient_context"},
    ]}
    requests = tmp_path/"requests.jsonl"; raw=tmp_path/"raw.jsonl"; alignment=tmp_path/"alignment.jsonl"; context=tmp_path/"context.json"; shots=tmp_path/"shots.jsonl"
    write_jsonl(requests,[req]); write_jsonl(raw,[raw_batch_row(req,response)]); write_jsonl(alignment,[baseline_row(i) for i in range(1,4)]); context.write_text(json.dumps(tiny_context())); write_jsonl(shots,[shot(i) for i in range(1,1142)])
    report=validate_responses(raw,requests,tmp_path/"review/openai"); assert report["passed"]
    before=hashlib.sha256(alignment.read_bytes()).hexdigest()
    applied=apply_validated_responses(alignment,requests,tmp_path/"review/openai/validated_responses.jsonl",context,shots,tmp_path)
    assert applied["baseline_files_unchanged"] and hashlib.sha256(alignment.read_bytes()).hexdigest()==before
    reviewed=[json.loads(line) for line in (tmp_path/"subtitle_script_alignment.llm_reviewed.jsonl").read_text().splitlines()]
    assert reviewed[0]["alignment"]["script_order_start"]==0 and reviewed[0]["alignment"]["llm_resolution"]["original_automatic"]["alignment"]["script_order_start"]==99
    assert reviewed[1]["script_matches"]==[] and reviewed[1]["alignment"]["status"]=="llm_no_match"
    assert reviewed[2]["alignment"]["status"]=="needs_review" and reviewed[2]["alignment"]["needs_review"] is True
    assert len((tmp_path/"shot_script_context.llm_reviewed.jsonl").read_text().splitlines())==1141
    diagnostics = json.loads((tmp_path/"review/openai/reviewed_alignment_diagnostics.json").read_text())
    assert diagnostics["status_counts"] == {"llm_aligned": 1, "llm_no_match": 1, "needs_review": 1}
    assert diagnostics["status_total"] == 3
    tagged = apply_validated_responses(alignment, requests, tmp_path/"review/openai/validated_responses.jsonl", context, shots, tmp_path, "full")
    assert tagged["alignment_output"].endswith("subtitle_script_alignment.llm_reviewed_full.jsonl")
    assert (tmp_path/"review/openai/full_apply_report.json").is_file()
    assert (tmp_path/"review/openai/full_reviewed_alignment_diagnostics.json").is_file()


def test_submit_refuses_without_confirmation_or_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    req=request(1,"easy",1); batch=tmp_path/"batch.jsonl"; write_jsonl(batch,[batch_line(req,"test-model")])
    with pytest.raises(RuntimeError,match="confirm-submit"): submit_batch(batch,tmp_path/"job.json",confirm_submit=False)
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    with pytest.raises(RuntimeError,match="OPENAI_API_KEY"): submit_batch(batch,tmp_path/"job.json",confirm_submit=True)


def test_sdk_shaped_submit_check_and_fetch_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    req = request(1, "easy", 1); batch_input = tmp_path / "batch.jsonl"; job_file = tmp_path / "job.json"
    write_jsonl(batch_input, [batch_line(req, "test-model")])
    output_response, error_response = Mock(), SimpleNamespace(read=Mock(return_value=b'{"error":"fixture"}\n'))
    output_response.text.return_value = '{"custom_id":"alignment_review_000001"}\n'
    files = SimpleNamespace(
        create=Mock(return_value=SimpleNamespace(id="file_input")),
        content=Mock(side_effect=lambda file_id: output_response if file_id == "file_output" else error_response),
    )
    counts = Mock(); counts.model_dump.return_value = {"total": 1, "completed": 1, "failed": 0}
    submitted = SimpleNamespace(id="batch_123", status="validating")
    completed = SimpleNamespace(id="batch_123", status="completed", output_file_id="file_output", error_file_id="file_error", request_counts=counts)
    batches = SimpleNamespace(create=Mock(return_value=submitted), retrieve=Mock(return_value=completed))
    monkeypatch.setattr("oscardp.script_context.openai_review._client", lambda: SimpleNamespace(files=files, batches=batches))

    submitted_result = submit_batch(batch_input, job_file, confirm_submit=True)
    assert submitted_result["input_file_id"] == "file_input" and submitted_result["batch_id"] == "batch_123"
    files.create.assert_called_once(); batches.create.assert_called_once_with(input_file_id="file_input", endpoint="/v1/responses", completion_window="24h")
    checked = check_batch(job_file)
    assert checked["request_counts"] == {"total": 1, "completed": 1, "failed": 0}
    assert isinstance(json_dumps(checked), str); counts.model_dump.assert_called_once_with()

    replacements = []
    original_replace = Path.replace
    def tracked_replace(source: Path, target: Path) -> Path:
        replacements.append(Path(target))
        return original_replace(source, target)
    monkeypatch.setattr(Path, "replace", tracked_replace)
    fetched = fetch_batch(job_file, tmp_path / "downloads")
    assert fetched["status"] == "completed"
    assert (tmp_path / "downloads/raw_batch_output.jsonl").read_text() == '{"custom_id":"alignment_review_000001"}\n'
    assert (tmp_path / "downloads/raw_batch_errors.jsonl").read_text() == '{"error":"fixture"}\n'
    output_response.text.assert_called_once_with(); error_response.read.assert_called_once_with()
    assert not list((tmp_path / "downloads").glob(".*.tmp"))
    assert tmp_path / "downloads/raw_batch_output.jsonl" in replacements
    assert tmp_path / "downloads/raw_batch_errors.jsonl" in replacements
    assert files.content.call_args_list == [call("file_output"), call("file_error")]


def test_download_text_rejects_non_text_sdk_content() -> None:
    assert _download_text(SimpleNamespace(text=lambda: b"utf-8 bytes")) == "utf-8 bytes"
    with pytest.raises(TypeError, match="str or bytes"):
        _download_text(SimpleNamespace(text=lambda: 123))


def test_reviewed_status_counts_total_all_blue_moon_rows() -> None:
    statuses = ["auto_aligned"] * 1500 + ["needs_review"] * 200 + ["no_match"] * 39 + ["llm_aligned"] * 90 + ["llm_no_match"] * 10
    diagnostics = _add_reviewed_status_counts({}, [{"alignment": {"status": status}} for status in statuses])
    assert diagnostics["status_total"] == 1839
    assert sum(diagnostics["status_counts"].values()) == 1839
    assert diagnostics["llm_aligned"] == 90 and diagnostics["llm_no_match"] == 10


def test_global_validation_advances_only_reliable_anchors() -> None:
    context = tiny_context()
    rows = [baseline_row(index) for index in range(1, 4)]
    rows[0]["alignment"].update(status="auto_aligned", needs_review=False, reliable_anchor=True, script_order_start=1, script_order_end=1)
    rows[0]["script_matches"] = [{"block_id": "scene_001_dialogue_002", "speaker": "HART", "matched_text": "dialogue 2", "combined_score": .9}]
    rows[1]["alignment"].update(status="llm_aligned", needs_review=False, reliable_anchor=False, script_order_start=0, script_order_end=0)
    rows[1]["script_matches"] = [{"block_id": "scene_001_dialogue_001", "speaker": "HART", "matched_text": "dialogue 1", "combined_score": .9}]
    rows[2]["alignment"].update(status="auto_aligned", needs_review=False, reliable_anchor=True, script_order_start=2, script_order_end=2)
    rows[2]["script_matches"] = [{"block_id": "scene_001_dialogue_003", "speaker": "HART", "matched_text": "dialogue 3", "combined_score": .9}]
    shots = [shot(index) for index in range(1, 4)]
    mapped = map_shots(shots, rows, context, "tt1")
    result = validate_data(context, rows, mapped, shots)
    assert result.passed, result.errors


def test_evaluate_pilot_reports_complete_metrics(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"; validated = tmp_path / "validated_responses.jsonl"; manifest = tmp_path / "manifest.json"; output = tmp_path / "evaluation.json"
    gold_rows = [{"request_id": "r1", "resolutions": [
        {"subtitle_id": "s1", "decision": "match", "block_ids": ["b1", "b2"]},
        {"subtitle_id": "s2", "decision": "no_match", "block_ids": []},
        {"subtitle_id": "s3", "decision": "uncertain", "block_ids": []},
    ]}]
    predictions = [{"request_id": "r1", "resolutions": [
        {"subtitle_id": "s1", "decision": "match", "block_ids": ["b1", "b2"]},
        {"subtitle_id": "s2", "decision": "no_match", "block_ids": []},
    ]}]
    write_jsonl(gold, gold_rows); write_jsonl(validated, predictions)
    (tmp_path / "response_validation_report.json").write_text(json.dumps({"invalid_count": 2}))
    manifest.write_text(json.dumps({
        "requests": [{"request_id": "r1", "stratum": "easy", "timeline_region": "early"}],
        "source_pool_distribution": {"strata": {"easy": 10, "fuzzy": 0, "multi": 0, "difficult": 0}},
    }))
    result = evaluate_pilot(gold, validated, manifest, output)
    assert result["decision_confusion_matrix"]["uncertain"]["missing"] == 1
    assert result["invalid_response_count"] == 2 and result["missing_prediction_count"] == 1
    assert result["multi_block_block_set_accuracy"] == 1.0
    assert result["no_match_precision"] == 1.0 and result["no_match_recall"] == 1.0
    assert result["uncertain_rate"] == 0.0 and result["acceptance_criteria"]["passed"] is False
    assert result["raw_diagnostic_accuracy"] == 2 / 3
    assert result["source_weighted_overall_accuracy"] == 2 / 3


def test_evaluate_pilot_rejects_equal_empty_blocks_when_decisions_differ(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"; validated = tmp_path / "validated.jsonl"
    manifest = tmp_path / "manifest.json"; output = tmp_path / "evaluation.json"
    write_jsonl(gold, [{"request_id": "r1", "resolutions": [
        {"subtitle_id": "s1", "decision": "uncertain", "block_ids": []},
    ]}])
    write_jsonl(validated, [{"request_id": "r1", "resolutions": [
        {"subtitle_id": "s1", "decision": "no_match", "block_ids": []},
    ]}])
    manifest.write_text(json.dumps({
        "requests": [{"request_id": "r1", "stratum": "easy", "timeline_region": "early"}],
        "source_pool_distribution": {"strata": {"easy": 1}},
    }))
    result = evaluate_pilot(gold, validated, manifest, output)
    assert result["schema_version"] == "1.1"
    assert result["exact_decision_accuracy"] == 0.0
    assert result["resolution_exact_match"] == 0.0
    assert result["block_set_exact_match"] == 0.0
    assert result["raw_diagnostic_accuracy"] == 0.0
    assert result["block_ids_only_exact_match"] == 1.0
    assert result["accuracy_by_stratum"]["easy"] == 0.0
    assert result["accuracy_by_region"]["early"] == 0.0
    assert result["acceptance_criteria"]["passed"] is False


def test_evaluate_pilot_complete_resolution_cases_and_multiblock_set_order(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"; validated = tmp_path / "validated.jsonl"
    manifest = tmp_path / "manifest.json"; output = tmp_path / "evaluation.json"
    write_jsonl(gold, [{"request_id": "r1", "resolutions": [
        {"subtitle_id": "s_no_match", "decision": "no_match", "block_ids": []},
        {"subtitle_id": "s_match", "decision": "match", "block_ids": ["dialogue_001"]},
        {"subtitle_id": "s_wrong_blocks", "decision": "match", "block_ids": ["dialogue_001"]},
        {"subtitle_id": "s_missing", "decision": "uncertain", "block_ids": []},
        {"subtitle_id": "s_multi", "decision": "match", "block_ids": ["dialogue_001", "dialogue_002"]},
    ]}])
    write_jsonl(validated, [{"request_id": "r1", "resolutions": [
        {"subtitle_id": "s_no_match", "decision": "no_match", "block_ids": []},
        {"subtitle_id": "s_match", "decision": "match", "block_ids": ["dialogue_001"]},
        {"subtitle_id": "s_wrong_blocks", "decision": "match", "block_ids": ["dialogue_002"]},
        {"subtitle_id": "s_multi", "decision": "match", "block_ids": ["dialogue_002", "dialogue_001"]},
    ]}])
    manifest.write_text(json.dumps({
        "requests": [{"request_id": "r1", "stratum": "multi", "timeline_region": "middle"}],
        "source_pool_distribution": {"strata": {"multi": 1}},
    }))
    result = evaluate_pilot(gold, validated, manifest, output)
    assert result["exact_decision_accuracy"] == 4 / 5
    assert result["resolution_exact_match"] == 3 / 5
    assert result["block_set_exact_match"] == 3 / 5
    assert result["multi_block_block_set_accuracy"] == 1.0
    assert result["missing_prediction_count"] == 1


def test_source_weighted_accuracy_uses_complete_resolution_correctness(tmp_path: Path) -> None:
    gold = tmp_path / "gold.jsonl"; validated = tmp_path / "validated.jsonl"
    manifest = tmp_path / "manifest.json"; output = tmp_path / "evaluation.json"
    write_jsonl(gold, [
        {"request_id": "r_easy", "resolutions": [{"subtitle_id": "s1", "decision": "uncertain", "block_ids": []}]},
        {"request_id": "r_difficult", "resolutions": [{"subtitle_id": "s2", "decision": "no_match", "block_ids": []}]},
    ])
    write_jsonl(validated, [
        {"request_id": "r_easy", "resolutions": [{"subtitle_id": "s1", "decision": "no_match", "block_ids": []}]},
        {"request_id": "r_difficult", "resolutions": [{"subtitle_id": "s2", "decision": "no_match", "block_ids": []}]},
    ])
    manifest.write_text(json.dumps({
        "requests": [
            {"request_id": "r_easy", "stratum": "easy", "timeline_region": "early"},
            {"request_id": "r_difficult", "stratum": "difficult", "timeline_region": "late"},
        ],
        "source_pool_distribution": {"strata": {"easy": 90, "difficult": 10}},
    }))
    result = evaluate_pilot(gold, validated, manifest, output)
    assert result["block_ids_only_exact_match"] == 1.0
    assert result["raw_diagnostic_accuracy"] == 0.5
    assert result["accuracy_by_stratum"] == {"easy": 0.0, "fuzzy": None, "multi": None, "difficult": 1.0}
    assert result["source_weighted_overall_accuracy"] == 0.1
