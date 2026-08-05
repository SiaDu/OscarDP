from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oscardp.script_context.openai_review import (
    apply_validated_responses,
    batch_line,
    prepare_batch,
    submit_batch,
    validate_batch_lines,
    validate_resolution,
    validate_responses,
)
from oscardp.script_context.openai_schema import alignment_response_schema
from oscardp.script_context.pilot import prepare_pilot


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
    assert first["request_count"] == 30 and first["strata"] == {"easy": 10, "fuzzy": 10, "multi": 5, "difficult": 5}
    assert all(first["timeline"].values())
    assert (tmp_path / "one/pilot_requests.jsonl").read_bytes() == (tmp_path / "two/pilot_requests.jsonl").read_bytes()
    gold = [json.loads(line) for line in (tmp_path / "one/pilot_gold_template.jsonl").read_text().splitlines()]
    assert all(value is None for row in gold for resolution in row["resolutions"] for value in (resolution["decision"], resolution["block_ids"], resolution["reviewer_notes"]))


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


def test_submit_refuses_without_confirmation_or_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    req=request(1,"easy",1); batch=tmp_path/"batch.jsonl"; write_jsonl(batch,[batch_line(req,"test-model")])
    with pytest.raises(RuntimeError,match="confirm-submit"): submit_batch(batch,tmp_path/"job.json",confirm_submit=False)
    monkeypatch.delenv("OPENAI_API_KEY",raising=False)
    with pytest.raises(RuntimeError,match="OPENAI_API_KEY"): submit_batch(batch,tmp_path/"job.json",confirm_submit=True)
