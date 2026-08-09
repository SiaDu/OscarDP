from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from oscardp.script_context.openai_schema import V32_POLICY_SYSTEM_INSTRUCTIONS
from oscardp.script_context.production_review import (
    PRODUCTION_REVIEWER_VERSION,
    apply_production_responses_v3,
    merge_production_responses_v3,
    prepare_production_batch_v3,
    prepare_production_remaining_v3,
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
    write_jsonl(requests, [req]); write_jsonl(responses, [response(req, no_match_last=True)]); reviewer_manifest(reviewer)
    blocks = [{"block_id": f"scene_001_dialogue_{i:03d}", "block_type": "dialogue", "source_order": i, "script_page": 1, "speaker": "A", "character_cue": "A", "parenthetical": None, "text": f"dialogue {i}"} for i in (1, 2)]
    scene = {"scene_id": "scene_001", "screenplay_scene_id": "1", "slugline": "INT. ROOM", "int_ext": "INT", "location": "ROOM", "time_of_day": None, "script_pages": {"start": 1, "end": 1}, "scene_characters": ["A"], "script_blocks": blocks, "semantic_annotations": {}, "parsing": {"status": "parsed", "needs_review": False}}
    context.write_text(json.dumps({"schema_version": "1.0", "movie": {"movie_id": "tt1"}, "script_scenes": [scene]}), encoding="utf-8")
    baseline = []
    for i, sid in enumerate(ids):
        baseline.append({"movie_id": "tt1", "subtitle_id": sid, "alignment_group_id": f"g{i}", "time": {"start": f"00:00:0{i}.000", "end": f"00:00:0{i}.800", "start_sec": float(i), "end_sec": float(i)+.8}, "text": sid, "scene_id": "scene_001", "script_matches": [], "alignment": {"method": "no_match", "status": "needs_review", "candidate_margin": 0., "needs_review": True, "reliable_anchor": False, "script_order_start": None, "script_order_end": None}})
    write_jsonl(alignment, baseline)
    shot_rows = [{"shot_id": f"shot_{i+1:06d}", "start_frame": i*10, "end_frame": (i+1)*10, "frame_count": 10, "start_time": f"00:00:0{i}.000", "end_time": f"00:00:0{i+1}.000", "start_sec": float(i), "end_sec": float(i+1), "duration_sec": 1., "keyframe_frame": i*10+4, "keyframe_time_sec": i+.4, "keyframe_relpath": f"keyframes/shot_{i+1:06d}.jpg"} for i in range(2)]
    write_jsonl(shots, shot_rows); before = hashlib.sha256(alignment.read_bytes()).hexdigest()
    result = apply_production_responses_v3(alignment, requests, responses, context, shots, tmp_path, reviewer)
    assert result["baseline_files_unchanged"] and hashlib.sha256(alignment.read_bytes()).hexdigest() == before
    reviewed = [json.loads(x) for x in Path(result["alignment_output"]).read_text().splitlines()]
    assert reviewed[0]["alignment"]["status"] == "llm_aligned"
    assert reviewed[1]["alignment"]["status"] == "llm_no_match"
    assert reviewed[1]["alignment"]["llm_resolution"]["original_openai_resolution"]["decision"] == "no_candidate_match"
    resumed = apply_production_responses_v3(alignment, requests, responses, context, shots, tmp_path, reviewer)
    assert resumed["resumed"] is True
