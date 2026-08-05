from __future__ import annotations

import json
from pathlib import Path

import pytest

from oscardp.script_context.stage23 import merge_validated_responses, prepare_remaining_requests


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def request(index: int) -> dict:
    request_id = f"alignment_review_{index:06d}"
    subtitle_id = f"subtitle_{index:06d}"
    block_id = f"scene_001_dialogue_{index:03d}"
    return {
        "request_id": request_id, "subtitle_ids": [subtitle_id],
        "subtitles": [{"subtitle_id": subtitle_id, "text": f"text {index}", "time": {"start_sec": index, "end_sec": index + .5}}],
        "dialogue_candidates": [{"scene_id": "scene_001", "block_id": block_id, "screenplay_order": index, "speaker": "HART", "text": f"text {index}"}],
        "automatic_candidate_mappings": [{"subtitle_id": subtitle_id, "matches": [{"block_id": block_id}], "alignment": {"method": "normalized_exact", "status": "needs_review"}}],
        "candidate_scenes": ["scene_001"], "insufficient_candidates": False,
    }


def response(item: dict) -> dict:
    return {
        "request_id": item["request_id"],
        "resolutions": [{
            "subtitle_id": item["subtitle_ids"][0], "decision": "match",
            "block_ids": [item["dialogue_candidates"][0]["block_id"]],
            "confidence": .9, "decision_basis": "exact_or_near_exact",
        }],
    }


def test_prepare_remaining_preserves_order_and_complete_disjoint_union(tmp_path: Path) -> None:
    full_rows = [request(index) for index in range(1, 7)]
    pilot_rows = [full_rows[1], full_rows[4]]
    full, pilot = tmp_path / "full.jsonl", tmp_path / "pilot.jsonl"
    output, manifest = tmp_path / "remaining.jsonl", tmp_path / "remaining_manifest.json"
    write_jsonl(full, full_rows); write_jsonl(pilot, pilot_rows)
    result = prepare_remaining_requests(full, pilot, output, manifest, "test-model")
    remaining = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["request_id"] for row in remaining] == [full_rows[index]["request_id"] for index in (0, 2, 3, 5)]
    assert set(row["request_id"] for row in remaining).isdisjoint(row["request_id"] for row in pilot_rows)
    assert {row["request_id"] for row in remaining + pilot_rows} == {row["request_id"] for row in full_rows}
    assert result["counts"] == {"full_requests": 6, "pilot_requests": 2, "remaining_requests": 4, "full_subtitles": 6, "pilot_subtitles": 2, "remaining_subtitles": 4}
    assert result["excluded_pilot_request_ids"] == [pilot_rows[0]["request_id"], pilot_rows[1]["request_id"]]


def test_prepare_remaining_rejects_foreign_pilot_and_duplicate_subtitle(tmp_path: Path) -> None:
    full_rows = [request(1), request(2)]; full, pilot = tmp_path / "full.jsonl", tmp_path / "pilot.jsonl"
    write_jsonl(full, full_rows); write_jsonl(pilot, [request(9)])
    with pytest.raises(ValueError, match="outside full set"):
        prepare_remaining_requests(full, pilot, tmp_path / "out.jsonl", tmp_path / "manifest.json", "model")
    full_rows[1]["subtitle_ids"] = full_rows[0]["subtitle_ids"]
    write_jsonl(full, full_rows); write_jsonl(pilot, [])
    with pytest.raises(ValueError, match="duplicate subtitle"):
        prepare_remaining_requests(full, pilot, tmp_path / "out.jsonl", tmp_path / "manifest.json", "model")


def test_merge_revalidates_and_reconstructs_full_order(tmp_path: Path) -> None:
    full_rows = [request(index) for index in range(1, 5)]
    full, pilot, remaining = tmp_path / "full.jsonl", tmp_path / "pilot.jsonl", tmp_path / "remaining.jsonl"
    write_jsonl(full, full_rows)
    write_jsonl(pilot, [response(full_rows[2]), response(full_rows[0])])
    write_jsonl(remaining, [response(full_rows[3]), response(full_rows[1])])
    output, report = tmp_path / "merged.jsonl", tmp_path / "report.json"
    result = merge_validated_responses(full, pilot, remaining, output, report)
    merged = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["request_id"] for row in merged] == [row["request_id"] for row in full_rows]
    assert result["request_count"] == result["valid_count"] == result["subtitle_count"] == 4
    assert result["invalid_count"] == result["missing"] == result["duplicate"] == 0


@pytest.mark.parametrize("kind", ["duplicate", "missing", "foreign"])
def test_merge_rejects_duplicate_missing_and_foreign_requests(tmp_path: Path, kind: str) -> None:
    full_rows = [request(1), request(2)]; full, pilot, remaining = tmp_path / "full.jsonl", tmp_path / "pilot.jsonl", tmp_path / "remaining.jsonl"
    write_jsonl(full, full_rows)
    pilot_rows, remaining_rows = [response(full_rows[0])], [response(full_rows[1])]
    if kind == "duplicate": remaining_rows.append(response(full_rows[0]))
    if kind == "missing": remaining_rows = []
    if kind == "foreign": remaining_rows = [response(request(9))]
    write_jsonl(pilot, pilot_rows); write_jsonl(remaining, remaining_rows)
    with pytest.raises(ValueError):
        merge_validated_responses(full, pilot, remaining, tmp_path / "out.jsonl", tmp_path / "report.json")
