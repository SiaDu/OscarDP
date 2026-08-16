from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from oscardp.performance_candidates.__main__ import build_parser
from oscardp.performance_candidates.grouping import group_events
from oscardp.performance_candidates.io import sha256_file
from oscardp.performance_candidates.pipeline import MiningOptions, mine
from oscardp.performance_candidates.review import evaluate_review, prepare_review_sample
from oscardp.performance_candidates.semantic import mine_shot_semantics
from oscardp.performance_candidates.targeting import (
    mine_target_relevance,
    resolve_target,
)
from oscardp.performance_candidates.validation import validate_run


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def context_row(index: int, *, subtitle_id: str | None = None, subtitle_text: str = "Okay.") -> dict:
    start = float((index - 1) * 2)
    subtitles = [] if subtitle_id is None else [{"subtitle_id": subtitle_id, "text": subtitle_text, "overlap_sec": 1.0}]
    return {
        "movie_id": "tt12300742", "shot_id": f"shot_{index:06d}",
        "frame_range": {"start_frame": (index - 1) * 20, "end_frame": index * 20, "frame_count": 20},
        "time": {"start": f"00:00:0{int(start)}.000", "end": f"00:00:0{int(start + 2)}.000", "start_sec": start, "end_sec": start + 2, "duration_sec": 2.0},
        "keyframe": {"frame": (index - 1) * 20 + 9, "time_sec": start + 1, "path": f"keyframes/shot_{index:06d}.jpg"},
        "scene": {"scene_id": "scene_001", "screenplay_scene_id": "1", "method": "subtitle_script_alignment", "confidence": 1.0},
        "scene_transition": False, "scene_candidates": [], "subtitles": subtitles,
        "script_matches": [], "local_script_context": {"action_before": [], "action_during": [f"scene_001_action_{index:03d}"], "action_after": []},
        "dialogue_speakers": [], "alignment": {"status": "aligned", "needs_review": False},
    }


def source_shot(index: int) -> dict:
    row = context_row(index)
    return {
        "movie_key": "tt12300742", "shot_id": row["shot_id"],
        "start_frame": row["frame_range"]["start_frame"], "end_frame": row["frame_range"]["end_frame"],
        "frame_count": 20, "start_sec": row["time"]["start_sec"], "end_sec": row["time"]["end_sec"],
        "duration_sec": 2.0,
    }


def stage3_fixture(tmp_path: Path) -> tuple[MiningOptions, Path]:
    source = tmp_path / "source"
    reviewed = source / "reviewed.jsonl"
    screenplay = source / "screenplay.json"
    shots = source / "shots.jsonl"
    video = source / "video.mkv"
    face_model = source / "yunet.onnx"
    pending = source / "pending.jsonl"
    nominees = source / "nominees.csv"
    rows = [
        context_row(1), context_row(2, subtitle_id="subtitle_2", subtitle_text="Wait..."),
        context_row(3, subtitle_id="subtitle_3"), context_row(4, subtitle_id="pending_subtitle"),
    ]
    write_jsonl(reviewed, rows)
    actions = [
        "Michelle cries and screams.", "Michelle shouts, then waits.",
        "She flinches and stares.", "He punches the wall.",
    ]
    write_json(screenplay, {
        "script_scenes": [{"scene_id": "scene_001", "script_blocks": [
            {"block_id": f"scene_001_action_{index:03d}", "block_type": "action", "text": text}
            for index, text in enumerate(actions, 1)
        ]}],
    })
    write_jsonl(shots, [source_shot(index) for index in range(1, 5)])
    video.write_bytes(b"not decoded by the test extractor")
    face_model.write_bytes(b"test-yunet-model")
    write_jsonl(pending, [{"movie_id": "tt12300742", "subtitle_id": "pending_subtitle"}])
    nominees.write_text("Ceremony,Year,CanonicalCategory,Film,FilmId,Nominees,NomineeIds,Winner,Detail\n98,2025,ACTRESS IN A LEADING ROLE,Bugonia,tt12300742,Emma Stone,nm1297015,False,Michelle\n", encoding="utf-8")

    def artifact(path: Path) -> dict:
        return {"path": str(path), "sha256": sha256_file(path)}

    release = tmp_path / "release" / "release_manifest.json"
    write_json(release, {
        "schema_version": "1.0", "release_id": "test_release", "status": "FROZEN",
        "movies": [{
            "movie_id": "tt12300742", "status": "COMPLETE",
            "artifacts": {"reviewed_shot_context": artifact(reviewed)},
            "protected_artifacts": {
                "deterministic_screenplay_context": artifact(screenplay), "shots": artifact(shots), "video": artifact(video),
            },
        }],
        "pending_human_ambiguities": artifact(pending),
    })
    options = MiningOptions(release_manifest=release, output_root=tmp_path / "output", face_model=face_model, nominees_file=nominees)
    return options, options.output_root / "test_release" / "performance_candidates_v2" / "tt12300742" / "nm1297015_emma_stone"


class DummyDetector:
    def __init__(self, _model: Path) -> None:
        pass

    def analyze(self, path: Path) -> dict:
        face = path.parent.name == "shot_000001"
        return {
            "face_count": 1 if face else 0, "max_face_area_ratio": 0.1 if face else 0.0,
            "quality": {"mean_luma": 128.0, "blur_variance": 100.0, "too_dark": False, "too_bright": False, "too_blurry": False, "usable": True},
        }


def fake_extract(_video: Path, requests: list) -> None:
    for request in requests:
        request.output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), "gray").save(request.output_path)


def test_semantic_mining_preserves_source_text_and_exclusions(tmp_path: Path) -> None:
    options, _run_dir = stage3_fixture(tmp_path)
    release = json.loads(options.release_manifest.read_text())
    movie = release["movies"][0]
    rows = [json.loads(line) for line in Path(movie["artifacts"]["reviewed_shot_context"]["path"]).read_text().splitlines()]
    screenplay = json.loads(Path(movie["protected_artifacts"]["deterministic_screenplay_context"]["path"]).read_text())
    result = mine_shot_semantics(rows, screenplay, {"shot_000004"})
    assert result[0]["semantic_score"] == 0.55
    assert result[0]["evidence"][0]["text"] == "Michelle cries and screams."
    assert result[3]["excluded_reason"] == "pending_stage2_ambiguity"


def test_mine_writes_atomic_shots_then_groups_events(tmp_path: Path) -> None:
    options, run_dir = stage3_fixture(tmp_path)
    result = mine(options, detector_factory=DummyDetector, extractor=fake_extract)
    assert result["performance_shot_count"] == 2
    shots = [json.loads(line) for line in (run_dir / "performance_shots.jsonl").read_text().splitlines()]
    events = [json.loads(line) for line in (run_dir / "performance_events.jsonl").read_text().splitlines()]
    audit = [json.loads(line) for line in (run_dir / "screening_audit.jsonl").read_text().splitlines()]
    assert [row["source_shot_id"] for row in shots] == ["shot_000001", "shot_000002"]
    assert shots[0]["selection_basis"] == "semantic_and_cv"
    assert shots[1]["selection_basis"] == "semantic_override"
    assert len(events) == 1 and events[0]["performance_shot_ids"] == [row["performance_shot_id"] for row in shots]
    assert next(row for row in audit if row["source_shot_id"] == "shot_000003")["status"] == "excluded"
    assert next(row for row in audit if row["source_shot_id"] == "shot_000004")["reason"] == "pending_stage2_ambiguity"
    assert "person" not in (run_dir / "performance_shots.jsonl").read_text().lower()
    assert validate_run(run_dir).passed
    assert mine(options, detector_factory=DummyDetector, extractor=fake_extract)["resumed"] is True


def test_mine_rejects_release_bound_input_hash_change(tmp_path: Path) -> None:
    options, _run_dir = stage3_fixture(tmp_path)
    release = json.loads(options.release_manifest.read_text())
    reviewed = Path(release["movies"][0]["artifacts"]["reviewed_shot_context"]["path"])
    reviewed.write_bytes(reviewed.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        mine(options, detector_factory=DummyDetector, extractor=fake_extract)


def test_event_grouping_does_not_bridge_unselected_or_scene_boundary() -> None:
    def selected(index: int, scene: str = "scene_001") -> dict:
        row = context_row(index)
        row.update({
            "performance_shot_id": f"perfshot_tt12300742_nm1297015_{index:06d}", "source_shot_id": row.pop("shot_id"),
            "source_index": index - 1, "shot_score": 0.8,
            "semantic": {"categories": ["reaction"]},
        })
        row["scene"]["scene_id"] = scene
        return row
    all_rows = [context_row(index) for index in range(1, 5)]
    all_rows[3]["scene"]["scene_id"] = "scene_002"
    events = group_events([selected(1), selected(2), selected(4, "scene_002")], all_rows, "tt12300742", "release")
    assert [len(row["performance_shot_ids"]) for row in events] == [2, 1]


def test_event_grouping_splits_long_multi_shot_event() -> None:
    all_rows = [context_row(index) for index in range(1, 5)]
    selected = []
    for index, source in enumerate(all_rows):
        row = dict(source)
        row.update({
            "performance_shot_id": f"perfshot_tt12300742_nm1297015_{index + 1:06d}",
            "source_shot_id": row.pop("shot_id"), "source_index": index,
            "shot_score": 0.5 + index / 10, "semantic": {"categories": ["reaction"]},
        })
        row["time"] = {
            "start": "", "end": "", "start_sec": index * 12.0,
            "end_sec": (index + 1) * 12.0, "duration_sec": 12.0,
        }
        selected.append(row)
        all_rows[index]["time"] = row["time"]
    events = group_events(selected, all_rows, "tt12300742", "release", maximum_duration=30.0)
    assert len(events) >= 2
    assert all(row["time"]["duration_sec"] <= 30.0 for row in events)


def test_event_bridge_keeps_context_out_of_members() -> None:
    all_rows = [context_row(index) for index in range(1, 4)]
    selected = []
    for index in (0, 2):
        row = dict(all_rows[index])
        row.update({"performance_shot_id": f"perfshot_tt12300742_nm1297015_{index + 1:06d}", "source_shot_id": row.pop("shot_id"), "source_index": index, "shot_score": .8, "semantic": {"categories": ["reaction"]}})
        selected.append(row)
    events = group_events(selected, all_rows, "tt12300742", "release")
    assert events[0]["context_between_shot_ids"] == ["shot_000002"]
    assert "shot_000002" not in events[0]["source_shot_ids"]


def test_review_package_and_gate_evaluation(tmp_path: Path, monkeypatch) -> None:
    options, run_dir = stage3_fixture(tmp_path)
    mine(options, detector_factory=DummyDetector, extractor=fake_extract)

    def fake_preview(_video: Path, _event: dict, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True); output.write_bytes(b"preview")

    monkeypatch.setattr("oscardp.performance_candidates.review._preview", fake_preview)
    summary = prepare_review_sample(run_dir, shot_count=3, event_count=1)
    assert summary["shot_sample_count"] == 3 and summary["event_sample_count"] == 1
    shot_path = run_dir / "review/performance_shots.review.jsonl"
    event_path = run_dir / "review/performance_events.review.jsonl"
    shot_rows = [json.loads(line) for line in shot_path.read_text().splitlines()]
    for row in shot_rows:
        row["human_decision"] = "keep" if row["source_population"] == "selected" else "reject"
    write_jsonl(shot_path, shot_rows)
    event_rows = [json.loads(line) for line in event_path.read_text().splitlines()]
    event_rows[0]["boundary_decision"] = "accept"
    event_rows[0]["performance_decision"] = "keep"
    write_jsonl(event_path, event_rows)
    result = evaluate_review(shot_path, event_path)
    assert result["passed"]


def test_cli_exposes_stage3_commands() -> None:
    parser = build_parser()
    args = parser.parse_args(["validate", "--run-dir", "/tmp/run"])
    assert args.command == "validate"


def test_targeting_high_medium_none_and_metadata_resolution(tmp_path: Path) -> None:
    options, _run_dir = stage3_fixture(tmp_path)
    target = resolve_target(options.nominees_file, "tt12300742", "nm1297015", None)
    rows = [context_row(1), context_row(2), context_row(3)]
    rows[0]["dialogue_speakers"] = ["MICHELLE'S VOICE"]
    rows[1]["local_script_context"]["action_before"] = ["scene_001_action_001"]
    screenplay = {"script_scenes": [{"script_blocks": [{"block_id": "scene_001_action_001", "block_type": "action", "text": "Michelle leaves."}]}]}
    relevance = mine_target_relevance(rows, screenplay, target)
    assert relevance[0]["confidence"] == "high"
    assert relevance[1]["confidence"] == "medium"
    assert relevance[2] == {"performer_name": "Emma Stone", "character_names": ["Michelle"], "confidence": "none", "interpretation": "no_textual_support", "score": 0.0, "evidence": []}
