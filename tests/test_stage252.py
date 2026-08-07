from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oscardp.script_context.stage252 import build_gold_adjudication, validate_gold_adjudication


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def package(tmp_path: Path) -> dict:
    base = tmp_path / "openai"
    stage = base / "stage251"
    paths = {
        "gold": base / "gold.jsonl", "validated": stage / "validated.jsonl",
        "requests": base / "requests.jsonl", "manifest": base / "manifest.json",
        "context": tmp_path / "movie_script_context.json", "alignment": tmp_path / "alignment.jsonl",
        "evaluation": stage / "evaluation.json", "disagreements": stage / "disagreements.jsonl",
        "raw": base / "pilot_result_v2/raw_batch_output.jsonl", "output": base / "adjudication",
    }
    subtitles = [
        {"subtitle_id": subtitle_id, "text": text, "time": {"start": f"00:00:0{index}.000", "end": f"00:00:0{index}.800", "start_sec": float(index), "end_sec": index + .8}}
        for index, (subtitle_id, text) in enumerate((("s1", "Wrong candidate case"), ("s2", "No candidate"), ("s3", "Model selected a candidate")), 1)
    ]
    candidates = [
        {"block_id": block_id, "scene_id": "scene_001", "screenplay_order": index, "speaker": "A", "text": text,
         "parenthetical": None, "lexical_score": .5, "semantic_score": None, "retrieval_score": .5, "retrieval_methods": ["test"]}
        for index, (block_id, text) in enumerate((("A", "Gold line"), ("B", "Other supplied line")))
    ]
    request = {"request_id": "r1", "subtitle_ids": ["s1", "s2", "s3"], "subtitles": subtitles, "dialogue_candidates": candidates, "previous_anchor": None, "next_anchor": None}
    gold = [{"request_id": "r1", "resolutions": [
        {"subtitle_id": "s1", "decision": "match", "block_ids": ["A"], "reviewer_notes": None},
        {"subtitle_id": "s2", "decision": "no_match", "block_ids": [], "reviewer_notes": None},
        {"subtitle_id": "s3", "decision": "no_match", "block_ids": [], "reviewer_notes": None},
    ]}]
    prediction = {"request_id": "r1", "validation_diagnostics": {"sequence_quality": {"events": []}}, "resolutions": [
        {"subtitle_id": "s1", "decision": "match", "block_ids": ["B"], "confidence": .8, "decision_basis": "paraphrase"},
        {"subtitle_id": "s2", "decision": "uncertain", "block_ids": [], "confidence": .7, "decision_basis": "insufficient_context"},
        {"subtitle_id": "s3", "decision": "match", "block_ids": ["A"], "confidence": .8, "decision_basis": "paraphrase"},
    ]}
    disagreements = [
        {"request_id": "r1", "subtitle_id": "s1", "candidate_task_correct": False},
        {"request_id": "r1", "subtitle_id": "s2", "candidate_task_correct": True},
        {"request_id": "r1", "subtitle_id": "s3", "candidate_task_correct": False},
    ]
    context = {"movie": {"movie_id": "tt_test"}, "script_scenes": [{
        "scene_id": "scene_001", "screenplay_scene_id": "1", "slugline": "INT. ROOM - DAY", "location": "ROOM", "time_of_day": "DAY",
        "script_blocks": [
            {"block_id": "A", "block_type": "dialogue", "speaker": "A", "text": "Gold line", "parenthetical": None, "source_order": 0},
            {"block_id": "action_1", "block_type": "action", "speaker": None, "text": "A useful action.", "parenthetical": None, "source_order": 1},
            {"block_id": "B", "block_type": "dialogue", "speaker": "B", "text": "Other supplied line", "parenthetical": None, "source_order": 2},
            {"block_id": "C", "block_type": "dialogue", "speaker": "C", "text": "Relevant context outside candidates", "parenthetical": None, "source_order": 3},
        ],
    }]}
    alignment = []
    for index, (subtitle_id, text) in enumerate((("s0", "Before"), ("s1", "Wrong candidate case"), ("s2", "No candidate"), ("s3", "Model selected a candidate"), ("s4", "After"))):
        alignment.append({
            "movie_id": "tt_test", "subtitle_id": subtitle_id, "text": text,
            "time": {"start": f"00:00:0{index}.000", "end": f"00:00:0{index}.800", "start_sec": float(index), "end_sec": index + .8},
            "alignment": {"status": "needs_review"}, "script_matches": [],
        })
    write_jsonl(paths["gold"], gold); write_jsonl(paths["validated"], [prediction]); write_jsonl(paths["requests"], [request])
    write_json(paths["manifest"], {"requests": [{"request_id": "r1", "stratum": "difficult", "timeline_region": "early", "fallback_used": True, "candidate_limit_saturated": False}]})
    write_json(paths["context"], context); write_jsonl(paths["alignment"], alignment)
    write_json(paths["evaluation"], {"resolution_count": 3, "candidate_task_accuracy": 1 / 3, "candidate_presence_decision_accuracy": 1 / 3, "resolution_exact_match": 0.0, "candidate_recall_gold_records": []})
    write_jsonl(paths["disagreements"], disagreements); write_jsonl(paths["raw"], [{"custom_id": "r1"}])
    source_paths = [path for name, path in paths.items() if name not in {"output"}]
    before = {path: digest(path) for path in source_paths}
    build_gold_adjudication(paths["gold"], paths["validated"], paths["requests"], paths["manifest"], paths["context"], paths["alignment"], paths["evaluation"], paths["disagreements"], paths["output"])
    rows = [json.loads(line) for line in (paths["output"] / "gold_adjudication.jsonl").read_text().splitlines()]
    return {"paths": paths, "rows": rows, "before": before}


def validate(value: dict) -> dict:
    p = value["paths"]
    return validate_gold_adjudication(p["gold"], p["validated"], p["requests"], p["manifest"], p["context"], p["alignment"], p["evaluation"], p["disagreements"], p["output"])


def test_candidate_task_disagreements_are_selected(package: dict) -> None:
    assert [row["subtitle_id"] for row in package["rows"]] == ["s1", "s3"]


def test_candidate_correct_no_match_uncertain_is_excluded(package: dict) -> None:
    assert "s2" not in {row["subtitle_id"] for row in package["rows"]}


def test_wrong_supplied_candidate_and_gold_no_match_prediction_are_included(package: dict) -> None:
    by_id = {row["subtitle_id"]: row for row in package["rows"]}
    assert "gold_match_predicted_wrong_candidate" in by_id["s1"]["diagnostic_classification"]
    assert "gold_no_match_predicted_match" in by_id["s3"]["diagnostic_classification"]


def test_nearby_subtitles_and_complete_candidates_are_present(package: dict) -> None:
    row = package["rows"][0]
    assert row["nearby_subtitles"]["before"] and row["nearby_subtitles"]["after"]
    assert [candidate["block_id"] for candidate in row["dialogue_candidates"]] == ["A", "B"]


def test_local_context_can_include_dialogue_outside_candidates(package: dict) -> None:
    assert "C" in {block["block_id"] for block in package["rows"][0]["screenplay_local_context"]}


def test_human_adjudication_is_pending_and_null(package: dict) -> None:
    for row in package["rows"]:
        assert row["human_adjudication"] == {"status": "pending", "adjudication": None, "final_decision": None, "final_block_ids": None, "policy_tags": [], "notes": None}


def test_duplicate_adjudication_subtitle_fails_validation(package: dict) -> None:
    path = package["paths"]["output"] / "gold_adjudication.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(package["rows"][0]) + "\n")
    report = validate(package)
    assert not report["passed"] and any("duplicate" in error for error in report["errors"])


def test_modified_source_candidate_in_package_fails_validation(package: dict) -> None:
    path = package["paths"]["output"] / "gold_adjudication.jsonl"
    rows = package["rows"]; rows[0]["dialogue_candidates"][0]["text"] = "mutated"
    write_jsonl(path, rows)
    report = validate(package)
    assert not report["passed"] and any("candidate content" in error for error in report["errors"])


def test_generation_does_not_mutate_sources_and_package_validates(package: dict) -> None:
    assert all(digest(path) == value for path, value in package["before"].items())
    assert validate(package)["passed"] is True
