from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from oscardp.script_context.production_qc import (
    build_production_high_risk_audit_v3,
    finalize_production_movie_v3,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request() -> dict:
    return {
        "request_id": "alignment_review_000001",
        "subtitle_ids": ["subtitle_000001", "subtitle_000002"],
        "subtitles": [
            {"subtitle_id": "subtitle_000001", "text": "- Hi.\n- Go.", "time": {"start_sec": 0.0, "end_sec": 0.8}},
            {"subtitle_id": "subtitle_000002", "text": "[SIGN: EXIT]", "time": {"start_sec": 1.0, "end_sec": 1.8}},
        ],
        "dialogue_candidates": [
            {"scene_id": "scene_001", "block_id": "scene_001_dialogue_001", "screenplay_order": 0, "speaker": "A", "text": "Sign exit", "lexical_score": .9},
            {"scene_id": "scene_001", "block_id": "scene_001_dialogue_002", "screenplay_order": 1, "speaker": "B", "text": "Go.", "lexical_score": .9},
        ],
        "automatic_candidate_mappings": [{"subtitle_id": "subtitle_000002", "matches": [{"block_id": "scene_001_dialogue_001"}]}],
        "candidate_scenes": ["scene_001"], "candidate_limit": 2,
        "fallback_used": True, "insufficient_candidates": False,
    }


def response() -> dict:
    return {
        "request_id": "alignment_review_000001", "model": "gpt-5.6-terra",
        "resolutions": [
            {"subtitle_id": "subtitle_000001", "decision": "match", "block_ids": ["scene_001_dialogue_001", "scene_001_dialogue_002"], "confidence": .7, "decision_basis": "repeated_or_reordered_dialogue"},
            {"subtitle_id": "subtitle_000002", "decision": "no_candidate_match", "block_ids": [], "confidence": .9, "decision_basis": "no_supplied_candidate"},
        ],
        "validation_diagnostics": {"sequence_quality": {"events": [{
            "previous_subtitle_id": "subtitle_000001", "subtitle_id": "subtitle_000002",
            "reason": "cross_scene_sequence_jump", "severity": "high_risk",
        }]}},
    }


def test_production_risk_audit_is_self_contained_and_covers_required_risks(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"; responses = tmp_path / "responses.jsonl"
    alignment = tmp_path / "reviewed.jsonl"; shots = tmp_path / "shots-reviewed.jsonl"; context = tmp_path / "context.json"
    write_jsonl(requests, [request()]); write_jsonl(responses, [response()])
    write_jsonl(alignment, [
        {"subtitle_id": "subtitle_000001", "text": "- Hi.\n- Go.", "time": {"start_sec": 0., "end_sec": .8}, "scene_id": "scene_001", "script_matches": [{"block_id": "scene_001_dialogue_001"}, {"block_id": "scene_001_dialogue_002"}], "alignment": {"status": "llm_aligned", "script_order_start": 0, "script_order_end": 1}},
        {"subtitle_id": "subtitle_000002", "text": "[SIGN: EXIT]", "time": {"start_sec": 1., "end_sec": 1.8}, "scene_id": None, "script_matches": [], "alignment": {"status": "llm_no_match", "script_order_start": None, "script_order_end": None}},
    ])
    write_jsonl(shots, [{"shot_id": "shot_000001", "keyframe": {"path": "keyframes/shot_000001.jpg"}, "subtitles": [{"subtitle_id": "subtitle_000001"}, {"subtitle_id": "subtitle_000002"}]}])
    context.write_text(json.dumps({"script_scenes": [{
        "scene_id": "scene_001", "parsing": {"status": "needs_review", "needs_review": True},
        "script_blocks": [
            {"block_type": "dialogue", "block_id": "scene_001_dialogue_001", "speaker": "A", "text": "Hi."},
            {"block_type": "dialogue", "block_id": "scene_001_dialogue_002", "speaker": "B", "text": "Go."},
        ],
    }]}), encoding="utf-8")
    output = tmp_path / "audit.jsonl"; summary = tmp_path / "summary.json"

    result = build_production_high_risk_audit_v3(requests, responses, alignment, shots, context, output, summary)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert result["record_count"] == 2 and result["required_fields_present"]
    first, second = rows
    assert {"low_confidence", "multi_block_selection", "multi_speaker_subtitle", "candidate_limit_saturated", "fallback_retrieval", "parser_structural_warning"} <= set(first["inclusion_reasons"])
    assert {"graphic_or_insert_like_text", "strong_lexical_overlap_no_candidate_match", "candidate_recall_risk", "cross_scene_sequence_event"} <= set(second["inclusion_reasons"])
    assert second["diagnostics"]["no_candidate_match_classification"] == "candidate_recall_risk"
    assert second["human_decision"] is None and second["review_status"] == "pending"
    assert second["dialogue_candidates"] and second["screenplay_local_context"] and second["matched_shots"]


def test_finalizer_verifies_hashes_coverage_and_freezes_manifest(tmp_path: Path, monkeypatch) -> None:
    movie = "tt1"
    source_dir = tmp_path / "sources"; source_dir.mkdir()
    video = source_dir / "movie.mkv"; subtitle = source_dir / "movie.srt"; screenplay = source_dir / "movie.pdf"
    for path, data in ((video, b"video"), (subtitle, b"subtitle"), (screenplay, b"screenplay")): path.write_bytes(data)
    context = tmp_path / "context.json"; deterministic = tmp_path / "alignment.jsonl"; deterministic_shots = tmp_path / "shot-context.jsonl"
    reviewed = tmp_path / "reviewed.jsonl"; reviewed_shots = tmp_path / "reviewed-shots.jsonl"; shots = tmp_path / "shots.jsonl"
    requests = tmp_path / "requests.jsonl"; responses = tmp_path / "responses.jsonl"; reviewer = tmp_path / "reviewer.json"
    context.write_text(json.dumps({"source_files": {"subtitle": str(subtitle)}, "script_scenes": []}), encoding="utf-8")
    write_jsonl(deterministic, []); write_jsonl(deterministic_shots, []); write_jsonl(shots, [])
    write_jsonl(reviewed, [{"subtitle_id": "subtitle_000001", "alignment": {"status": "llm_aligned"}}]); write_jsonl(reviewed_shots, [])
    req = request(); req["subtitle_ids"] = ["subtitle_000001"]; req["subtitles"] = [req["subtitles"][0]]
    res = response(); res["resolutions"] = [res["resolutions"][0]]
    write_jsonl(requests, [req]); write_jsonl(responses, [res])
    reviewer.write_text(json.dumps({"production_reviewer_version": "v3.2-production.2", "hard_validation_contract_version": "candidate_task_v3_structure_v2"}), encoding="utf-8")
    inventory = tmp_path / "inventory.json"; status = tmp_path / "status.json"
    inventory.write_text(json.dumps({"movies": [{"movie_id": movie, "video": {"path": str(video), "sha256": sha(video)}, "subtitle": {"path": str(subtitle), "sha256": sha(subtitle)}, "screenplay": {"path": str(screenplay), "sha256": sha(screenplay)}, "stage1": {"shots_sha256": sha(shots)}}]}), encoding="utf-8")
    status.write_text(json.dumps({"movies": [{"movie_id": movie, "artifact_hashes": {"video": sha(video), "subtitle": sha(subtitle), "screenplay": sha(screenplay), "shots": sha(shots)}, "deterministic_output_hashes": {"screenplay_context": sha(context), "alignment": sha(deterministic), "shot_context": sha(deterministic_shots), "review_requests": sha(requests)}}]}), encoding="utf-8")
    risk = tmp_path / "risk.jsonl"; write_jsonl(risk, [{"subtitle_id": "subtitle_000001"}])
    risk_summary = tmp_path / "risk-summary.json"; risk_summary.write_text(json.dumps({"record_count": 1, "audit_sha256": sha(risk), "inclusion_reason_counts": {"low_confidence": 1}}), encoding="utf-8")
    lifecycle = tmp_path / "merge.json"; lifecycle.write_text(json.dumps({"passed": True}), encoding="utf-8")
    monkeypatch.setattr("oscardp.script_context.production_qc.validate_files", lambda *args: SimpleNamespace(passed=True, errors=[], alignment_count=1, shot_count=0))

    result = finalize_production_movie_v3(
        movie, inventory, status, context, deterministic, deterministic_shots, reviewed, reviewed_shots,
        shots, requests, responses, reviewer, [lifecycle], risk, risk_summary,
        tmp_path / "qc.json", tmp_path / "manifest.json",
    )

    assert result["passed"] and result["unresolved_ambiguity_count"] == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETE" and manifest["counts"]["requests"] == 1
    assert all(row["unchanged"] for row in manifest["protected_hashes"].values())

    risk_summary.write_text(json.dumps({
        "record_count": 1, "audit_sha256": sha(risk), "inclusion_reason_counts": {},
        "no_candidate_match_classification_counts": {"ambiguous_needs_review": 6},
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unresolved ambiguity count 6"):
        finalize_production_movie_v3(
            movie, inventory, status, context, deterministic, deterministic_shots, reviewed, reviewed_shots,
            shots, requests, responses, reviewer, [lifecycle], risk, risk_summary,
            tmp_path / "qc-too-ambiguous.json", tmp_path / "manifest-too-ambiguous.json",
        )
    assert not (tmp_path / "manifest-too-ambiguous.json").exists()
