from __future__ import annotations

import json
from pathlib import Path

import pytest

from oscardp.script_context.stage231 import apply_human_corrections, build_human_audit_v2, build_non_anchor_sequence_audit


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def mapped(subtitle_id: str, start: int, end: int, *, llm: bool) -> dict:
    return {
        "subtitle_id": subtitle_id, "text": subtitle_id, "time": {"start_sec": float(start), "end_sec": float(start) + .5},
        "scene_id": "scene_001", "script_matches": [{"block_id": f"block_{start}", "speaker": "A", "combined_score": .9}],
        "alignment": {"status": "llm_aligned" if llm else "auto_aligned", "needs_review": False, "reliable_anchor": False, "script_order_start": start, "script_order_end": end, **({"llm_resolution": {"request_id": "r"}} if llm else {})},
    }


def test_non_anchor_audit_attributes_regressions_to_llm_rows(tmp_path: Path) -> None:
    alignment = tmp_path / "alignment.jsonl"
    write_jsonl(alignment, [
        mapped("subtitle_000862", 412, 412, llm=True), mapped("subtitle_000863", 411, 412, llm=False),
        mapped("subtitle_001714", 853, 854, llm=False), mapped("subtitle_001715", 853, 853, llm=True),
    ])
    summary = build_non_anchor_sequence_audit(alignment, tmp_path / "audit.jsonl", tmp_path / "summary.json")
    assert summary["record_count"] == 2
    assert summary["subtitle_ids"] == ["subtitle_000862", "subtitle_001715"]
    assert summary["start_regression_count"] == 1 and summary["end_regression_count"] == 1


def test_human_audit_v2_is_self_contained_and_has_null_labels(tmp_path: Path) -> None:
    request = {
        "request_id": "r1", "subtitle_ids": ["subtitle_000001"],
        "subtitles": [{"subtitle_id": "subtitle_000001", "text": "hello", "time": {"start_sec": 0., "end_sec": 1.}}],
        "previous_anchor": {"subtitle_id": "s0"}, "next_anchor": {"subtitle_id": "s2"},
        "dialogue_candidates": [{"scene_id": "scene_001", "block_id": "block_1", "screenplay_order": 1, "speaker": "A", "text": "hello"}],
        "automatic_candidate_mappings": [{"subtitle_id": "subtitle_000001", "matches": [], "alignment": {"status": "no_match"}}],
    }
    resolution = {"request_id": "r1", "resolutions": [{"subtitle_id": "subtitle_000001", "decision": "no_match", "block_ids": [], "confidence": .7, "decision_basis": "changed_or_improvised_dialogue"}]}
    alignment = mapped("subtitle_000001", 1, 1, llm=True); alignment["time"] = {"start_sec": 0., "end_sec": 1.}
    shot = {"shot_id": "shot_000001", "keyframe": {"path": "keyframes/shot_000001.jpg"}, "subtitles": [{"subtitle_id": "subtitle_000001"}]}
    files = {name: tmp_path / name for name in ("requests", "responses", "composite", "nonanchor", "alignment", "shots", "prior")}
    write_jsonl(files["requests"], [request]); write_jsonl(files["responses"], [resolution]); write_jsonl(files["composite"], [])
    write_jsonl(files["nonanchor"], []); write_jsonl(files["alignment"], [alignment]); write_jsonl(files["shots"], [shot])
    write_jsonl(files["prior"], [{"request_id": "r1", "subtitle_id": "subtitle_000001", "inclusion_reasons": ["deterministic_sample_easy"]}])
    manifest = build_human_audit_v2(files["requests"], files["responses"], files["composite"], files["nonanchor"], files["alignment"], files["shots"], files["prior"], tmp_path / "audit_v2.jsonl", tmp_path / "manifest.json")
    row = json.loads((tmp_path / "audit_v2.jsonl").read_text())
    assert set(manifest["required_fields"]) <= set(row)
    assert row["matched_shots"] == [{"shot_id": "shot_000001", "keyframe_path": "keyframes/shot_000001.jpg"}]
    assert row["human_decision"] is row["human_block_ids"] is row["reviewer_notes"] is None
    assert row["review_status"] == "pending" and manifest["human_labels_present"] is False


def test_apply_human_corrections_rejects_block_outside_original_candidates(tmp_path: Path) -> None:
    request = {"request_id": "r1", "subtitle_ids": ["s1"], "dialogue_candidates": [{"scene_id": "scene_001", "block_id": "allowed", "screenplay_order": 1}], "candidate_scenes": ["scene_001"]}
    response = {"request_id": "r1", "resolutions": [{"subtitle_id": "s1", "decision": "no_match", "block_ids": [], "confidence": .9, "decision_basis": "changed_or_improvised_dialogue"}]}
    audit = {"request_id": "r1", "subtitle_id": "s1", "review_status": "completed", "human_decision": "match", "human_block_ids": ["foreign"], "reviewer_notes": "fixture"}
    requests, responses, audit_path = tmp_path / "requests.jsonl", tmp_path / "responses.jsonl", tmp_path / "audit.jsonl"
    write_jsonl(requests, [request]); write_jsonl(responses, [response]); write_jsonl(audit_path, [audit])
    with pytest.raises(ValueError, match="outside request candidates"):
        apply_human_corrections(audit_path, requests, responses, tmp_path / "alignment", tmp_path / "context", tmp_path / "shots", tmp_path, "human_qc")


def test_apply_human_corrections_writes_new_tag_and_preserves_openai_provenance(tmp_path: Path) -> None:
    request = {
        "request_id": "r1", "subtitle_ids": ["s1"], "candidate_scenes": ["scene_001"],
        "dialogue_candidates": [{"scene_id": "scene_001", "block_id": "block_1", "screenplay_order": 0, "speaker": "A", "text": "hello"}],
        "subtitles": [{"subtitle_id": "s1", "text": "hello"}], "automatic_candidate_mappings": [], "insufficient_candidates": False,
    }
    response = {"request_id": "r1", "model": "test", "resolutions": [{"subtitle_id": "s1", "decision": "no_match", "block_ids": [], "confidence": .8, "decision_basis": "changed_or_improvised_dialogue"}]}
    audit = {"request_id": "r1", "subtitle_id": "s1", "review_status": "completed", "human_decision": "match", "human_block_ids": ["block_1"], "reviewer_notes": "confirmed"}
    context = {"script_scenes": [{"scene_id": "scene_001", "screenplay_scene_id": "1", "script_pages": {"start": 1, "end": 1}, "parsing": {"needs_review": False}, "script_blocks": [{"block_id": "block_1", "block_type": "dialogue", "source_order": 1, "speaker": "A", "text": "hello"}]}], "broken_pages": []}
    baseline = {"movie_id": "tt1", "subtitle_id": "s1", "alignment_group_id": "a1", "time": {"start_sec": 0., "end_sec": .8}, "text": "hello", "scene_id": None, "script_matches": [], "alignment": {"method": "no_match", "status": "no_match", "needs_review": True, "reliable_anchor": False, "script_order_start": None, "script_order_end": None}}
    shot = {"shot_id": "shot_000001", "start_frame": 0, "end_frame": 10, "frame_count": 10, "start_time": "00:00:00.000", "end_time": "00:00:01.000", "start_sec": 0., "end_sec": 1., "duration_sec": 1., "keyframe_frame": 4, "keyframe_time_sec": .4, "keyframe_relpath": "keyframes/shot_000001.jpg"}
    requests, responses, audit_path = tmp_path / "requests.jsonl", tmp_path / "responses.jsonl", tmp_path / "audit.jsonl"
    alignment, context_path, shots = tmp_path / "alignment.jsonl", tmp_path / "context.json", tmp_path / "shots.jsonl"
    write_jsonl(requests, [request]); write_jsonl(responses, [response]); write_jsonl(audit_path, [audit]); write_jsonl(alignment, [baseline]); write_jsonl(shots, [shot]); context_path.write_text(json.dumps(context))
    result = apply_human_corrections(audit_path, requests, responses, alignment, context_path, shots, tmp_path, "human_qc")
    reviewed = json.loads((tmp_path / "subtitle_script_alignment.llm_reviewed_human_qc.jsonl").read_text())
    provenance = reviewed["alignment"]["llm_resolution"]
    assert result["validation_passed"] and provenance["resolver"] == "human_correction"
    assert provenance["original_openai_resolution"]["decision"] == "no_match"
    assert (tmp_path / "review/openai/human_qc_human_correction_report.json").is_file()
    assert (tmp_path / "review/openai/human_qc_human_validation_report.json").is_file()
