from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import fitz
import pytest

from oscardp.script_context.alignment import align_subtitles
from oscardp.script_context.llm_review import apply_alignment_responses
from oscardp.script_context.pipeline import ContextOptions, process_one
from oscardp.script_context.schema import AlignmentConfig, CleanSubtitle
from oscardp.script_context.screenplay import is_broken_page, normalize_character_cue, parse_layout_pages, stable_scene_id
from oscardp.script_context.shot_mapping import map_shots
from oscardp.script_context.subtitles import load_clean_subtitles
from oscardp.script_context.validation import validate_data, validate_files


def _context(dialogues: list[tuple[str, str]], *, second_scene: bool = False) -> dict:
    scenes = []
    split = len(dialogues) // 2 if second_scene else len(dialogues)
    for scene_number, selected in (("1", dialogues[:split]), ("4A", dialogues[split:])):
        if not selected:
            continue
        scene_id = stable_scene_id(scene_number)
        blocks = [{"block_id": f"{scene_id}_action_001", "block_type": "action", "script_page": 1, "source_order": 1, "text": "They enter."}]
        for index, (speaker, text) in enumerate(selected, 1):
            blocks.append({"block_id": f"{scene_id}_dialogue_{index:03d}", "block_type": "dialogue", "script_page": 1, "source_order": index + 1, "speaker": speaker, "character_cue": speaker, "parenthetical": None, "text": text})
        scenes.append({"scene_id": scene_id, "screenplay_scene_id": scene_number, "slugline": "INT. ROOM - NIGHT", "int_ext": "INT", "location": "ROOM", "time_of_day": "NIGHT", "script_pages": {"start": 1, "end": 1}, "scene_characters": [s for s, _ in selected], "script_blocks": blocks, "semantic_annotations": {"scene_summary": None, "dramatic_function": None}, "parsing": {"status": "parsed", "needs_review": False}})
    return {"schema_version": "1.0", "movie": {"movie_id": "tt1234567", "title": "Test"}, "source_files": {}, "summary": {"scene_count": len(scenes), "dialogue_block_count": len(dialogues), "action_block_count": len(scenes), "broken_page_count": 0}, "broken_pages": [], "script_scenes": scenes}


def _sub(index: int, text: str, start: float | None = None, end: float | None = None) -> CleanSubtitle:
    start = float(index) if start is None else start; end = start + 0.8 if end is None else end
    return CleanSubtitle(f"subtitle_{index:06d}", f"00:00:0{int(start)}.000", f"00:00:0{int(end)}.800", start, end, text, text, "en")


def test_scene_ids_character_and_layout_classification() -> None:
    assert [stable_scene_id(value) for value in ("1", "4A", "13B")] == ["scene_001", "scene_004A", "scene_013B"]
    assert normalize_character_cue("HART (CONT'D)") == "HART"
    pages = [{"page": 1, "width": 612, "lines": [
        {"text": "4A INT. SARDI'S BAR - NIGHT 4A", "x0": 50, "y0": 10},
        {"text": "People crowd the bar.", "x0": 70, "y0": 30},
        {"text": "HART (CONT'D)", "x0": 250, "y0": 50},
        {"text": "(quietly)", "x0": 220, "y0": 70},
        {"text": "This is the line.", "x0": 180, "y0": 90},
    ]}]
    result = parse_layout_pages(pages, "tt1234567", "Test", {})
    scene = result["script_scenes"][0]
    assert scene["screenplay_scene_id"] == "4A"
    assert [block["block_type"] for block in scene["script_blocks"]] == ["action", "dialogue"]
    assert scene["script_blocks"][1]["speaker"] == "HART"
    assert scene["script_blocks"][1]["parenthetical"] == "(quietly)"


def test_broken_fragment_page_detection() -> None:
    assert is_broken_page(list("ABCDEFGHIJKLMNOPQRST"))
    assert not is_broken_page(["This is a normal screenplay line", "Another complete sentence"])


def test_more_marker_is_ignored_but_dialogue_with_more_is_preserved() -> None:
    pages = [{"page": 1, "width": 612, "lines": [
        {"text": "1 INT. ROOM - NIGHT 1", "x0": 50, "y0": 10},
        {"text": "HART", "x0": 250, "y0": 30},
        {"text": "I need more time.", "x0": 180, "y0": 50},
        {"text": "(MORE)", "x0": 250, "y0": 70},
    ]}]
    result = parse_layout_pages(pages, "tt1", "Test", {})
    blocks = result["script_scenes"][0]["script_blocks"]
    assert [block["text"] for block in blocks] == ["I need more time."]


def test_part_markers_are_omitted_and_attached_continuous_is_parsed() -> None:
    pages = [{"page": 1, "width": 612, "lines": [
        {"text": "4H INT. SARDI'S - MORTY'S STATION - MAIN BAR AREA -CONTINUOUS 4H", "x0": 50, "y0": 10},
        {"text": "PT 1:", "x0": 108, "y0": 30},
        {"text": "PART 2:", "x0": 108, "y0": 50},
        {"text": "Part of the crowd moves closer.", "x0": 108, "y0": 70},
    ]}]
    scene = parse_layout_pages(pages, "tt1", "Test", {})["script_scenes"][0]
    assert scene["time_of_day"] == "CONTINUOUS"
    assert scene["slugline"] == "INT. SARDI'S - MORTY'S STATION - MAIN BAR AREA -CONTINUOUS"
    assert [block["text"] for block in scene["script_blocks"]] == ["Part of the crowd moves closer."]


def test_bilingual_subtitle_cleaning_and_no_blind_merge(tmp_path: Path) -> None:
    path = tmp_path / "test.srt"
    path.write_text("1\n00:00:01,000 --> 00:00:02,000\n<i>Hello there</i> ♪\n\n2\n00:00:01,000 --> 00:00:02,000\n你好\n\n3\n00:00:01,000 --> 00:00:02,000\nDifferent English line\n", encoding="utf-8")
    rows = load_clean_subtitles(path, "en")
    assert [row.cleaned_text for row in rows] == ["Different English line", "Hello there"]


def test_exact_fuzzy_multiblock_and_no_match_alignment() -> None:
    context = _context([("A", "Exact words here"), ("B", "first half"), ("B", "second half"), ("C", "A screenplay line deleted")])
    subtitles = [_sub(1, "Exact words here"), _sub(2, "first half second half"), _sub(3, "This was improvised and unrelated xyz")]
    rows = align_subtitles(subtitles, context, "tt1234567", AlignmentConfig(0.75, 0.55, 0.01, 20))
    assert rows[0]["alignment"]["method"] == "anchor_normalized_exact"
    assert len(rows[1]["script_matches"]) == 2
    assert rows[2]["alignment"]["status"] == "no_match"


def test_two_subtitles_to_one_block_and_monotonicity() -> None:
    context = _context([("A", "one combined sentence"), ("B", "later exact words")])
    rows = align_subtitles([_sub(1, "one combined"), _sub(2, "sentence"), _sub(3, "later exact words")], context, "tt1234567", AlignmentConfig(0.7, 0.5, 0.01, 10))
    assert rows[0]["alignment_group_id"] == rows[1]["alignment_group_id"]
    scene_order = [row["scene_id"] for row in rows if row["scene_id"]]
    assert scene_order == sorted(scene_order)


def _shot(ordinal: int, start: float, end: float) -> dict:
    return {"shot_id": f"shot_{ordinal:06d}", "start_frame": (ordinal - 1) * 10, "end_frame": ordinal * 10, "frame_count": 10, "start_time": f"00:00:0{int(start)}.000", "end_time": f"00:00:0{int(end)}.000", "start_sec": start, "end_sec": end, "duration_sec": end - start, "keyframe_frame": (ordinal - 1) * 10 + 4, "keyframe_time_sec": start + 0.4, "keyframe_relpath": f"keyframes/shot_{ordinal:06d}.jpg"}


def test_subtitle_crosses_shots_and_same_scene_interpolation() -> None:
    context = _context([("A", "hello there my friend"), ("A", "we meet once again")])
    alignments = align_subtitles([_sub(1, "hello there my friend", 0.5, 1.5), _sub(2, "we meet once again", 2.5, 3.0)], context, "tt1234567")
    rows = map_shots([_shot(1, 0, 1), _shot(2, 1, 2), _shot(3, 2, 3)], alignments, context, "tt1234567", 10)
    assert rows[0]["subtitles"][0]["subtitle_coverage"] == 0.5
    assert rows[1]["subtitles"][0]["subtitle_coverage"] == 0.5
    assert rows[2]["scene"]["scene_id"] == "scene_001"


def interpolated_fixture() -> tuple[dict, list[dict], list[dict], list[dict]]:
    context = _context([("A", "hello there my friend"), ("A", "we meet once again")])
    alignments = align_subtitles([_sub(1, "hello there my friend", .1, .5), _sub(2, "we meet once again", 2.5, 2.9)], context, "tt1234567")
    shots = [_shot(1, 0, 1), _shot(2, 1, 2), _shot(3, 2, 3)]
    return context, alignments, shots, map_shots(shots, alignments, context, "tt1234567", 10)


def test_interpolated_shot_has_no_direct_scene_evidence_and_inherits_confidence() -> None:
    context, alignments, shots, rows = interpolated_fixture()
    interpolated = rows[1]
    assert interpolated["scene"]["method"] == "same_scene_interpolation"
    assert interpolated["scene"]["scene_id"] == "scene_001"
    assert interpolated["scene"]["confidence"] == min(rows[0]["scene"]["confidence"], rows[2]["scene"]["confidence"])
    assert interpolated["scene_candidates"] == [] and interpolated["scene_transition"] is False
    assert interpolated["subtitles"] == [] and interpolated["script_matches"] == []
    assert interpolated["alignment"] == {"status": "interpolated", "needs_review": False}
    assert validate_data(context, alignments, rows, shots).passed


@pytest.mark.parametrize("mutation", ["candidate", "transition", "subtitles", "matches"])
def test_validator_rejects_interpolated_direct_evidence_or_transition(mutation: str) -> None:
    context, alignments, shots, rows = interpolated_fixture()
    invalid = deepcopy(rows)
    row = invalid[1]
    if mutation == "candidate":
        row["scene_candidates"] = [{"scene_id": "scene_001", "screenplay_scene_id": "1", "overlap_sec": 0., "confidence": 1.}]
    elif mutation == "transition":
        row["scene_transition"] = True
    elif mutation == "subtitles":
        row["subtitles"] = [{"subtitle_id": "subtitle_000001"}]
    else:
        row["script_matches"] = [{"block_id": "scene_001_dialogue_001"}]
    assert not validate_data(context, alignments, invalid, shots).passed


def test_no_cross_scene_interpolation() -> None:
    context = _context([("A", "hello"), ("B", "goodbye")], second_scene=True)
    alignments = align_subtitles([_sub(1, "hello", 0, 0.5), _sub(2, "goodbye", 2.5, 3)], context, "tt1234567")
    rows = map_shots([_shot(1, 0, 1), _shot(2, 1, 2), _shot(3, 2, 3)], alignments, context, "tt1234567", 10)
    assert rows[1]["scene"] is None
    assert rows[1]["scene_candidates"] == [] and rows[1]["scene_transition"] is False


BLUE_MOON_TRANSITION_PATTERNS = [
    ("shot_000434", .8, .5, "scene_001"),
    ("shot_000456", .4, .1, "scene_004A"),
    ("shot_000703", .8, .5, "scene_001"),
    ("shot_000796", .8, .5, "scene_001"),
    ("shot_000831", .4, .1, "scene_004A"),
    ("shot_000868", .8, .5, "scene_001"),
    ("shot_001030", .8, .5, "scene_001"),
]


@pytest.mark.parametrize(("shot_id", "left_end", "right_start", "primary_scene"), BLUE_MOON_TRANSITION_PATTERNS)
def test_blue_moon_scene_transition_patterns_use_global_order_and_primary_local_context(shot_id: str, left_end: float, right_start: float, primary_scene: str) -> None:
    context = _context([("A", "left dialogue"), ("B", "right dialogue")], second_scene=True)
    ordinal = int(shot_id.rsplit("_", 1)[1])
    shot_row = _shot(ordinal, 0, 1)
    alignments = [
        {"movie_id": "tt1", "subtitle_id": "subtitle_000001", "alignment_group_id": "a1", "time": {"start_sec": 0., "end_sec": left_end}, "text": "left", "scene_id": "scene_001", "script_matches": [{"block_id": "scene_001_dialogue_001", "speaker": "A", "combined_score": .9}], "alignment": {"status": "llm_aligned", "needs_review": False, "reliable_anchor": False, "script_order_start": 0, "script_order_end": 0}},
        {"movie_id": "tt1", "subtitle_id": "subtitle_000002", "alignment_group_id": "a2", "time": {"start_sec": right_start, "end_sec": 1.}, "text": "right", "scene_id": "scene_004A", "script_matches": [{"block_id": "scene_004A_dialogue_001", "speaker": "B", "combined_score": .9}], "alignment": {"status": "llm_aligned", "needs_review": False, "reliable_anchor": False, "script_order_start": 1, "script_order_end": 1}},
    ]
    mapped = map_shots([shot_row], alignments, context, "tt1")
    row = mapped[0]
    assert row["shot_id"] == shot_id and row["scene_transition"] is True
    assert row["scene"]["scene_id"] == primary_scene
    assert [candidate["scene_id"] for candidate in row["scene_candidates"]] == ["scene_001", "scene_004A"]
    assert abs(sum(candidate["confidence"] for candidate in row["scene_candidates"]) - 1) < 1e-6
    assert [match["block_id"] for match in row["script_matches"]] == ["scene_001_dialogue_001", "scene_004A_dialogue_001"]
    assert row["alignment"] == {"status": "scene_transition", "needs_review": True}
    local_ids = [block_id for values in row["local_script_context"].values() for block_id in values]
    assert all(block_id.startswith(primary_scene + "_") for block_id in local_ids)
    assert validate_data(context, alignments, mapped, [shot_row]).passed


def test_direct_single_scene_shot_has_one_positive_evidence_candidate() -> None:
    context = _context([("A", "hello")])
    alignments = [{"subtitle_id": "subtitle_000001", "alignment_group_id": "a1", "time": {"start_sec": 0., "end_sec": .8}, "text": "hello", "scene_id": "scene_001", "script_matches": [{"block_id": "scene_001_dialogue_001", "speaker": "A", "combined_score": .9}], "alignment": {"status": "llm_aligned", "needs_review": False, "reliable_anchor": False, "script_order_start": 0, "script_order_end": 0}}]
    row = map_shots([_shot(1, 0, 1)], alignments, context, "tt1234567")[0]
    assert row["scene"]["method"] == "subtitle_script_alignment"
    assert row["scene_transition"] is False and len(row["scene_candidates"]) == 1
    assert row["scene_candidates"][0]["overlap_sec"] > 0 and row["scene_candidates"][0]["confidence"] == 1.0


def test_shot_validation_rejects_cross_scene_action_and_bad_transition_metadata() -> None:
    context = _context([("A", "left dialogue"), ("B", "right dialogue")], second_scene=True)
    shot_row = _shot(1, 0, 1)
    alignments = [
        {"subtitle_id": "subtitle_000001", "alignment_group_id": "a1", "time": {"start_sec": 0., "end_sec": .7}, "text": "left", "scene_id": "scene_001", "script_matches": [{"block_id": "scene_001_dialogue_001", "speaker": "A", "combined_score": .9}], "alignment": {"status": "llm_aligned", "needs_review": False, "reliable_anchor": False, "script_order_start": 0, "script_order_end": 0}},
        {"subtitle_id": "subtitle_000002", "alignment_group_id": "a2", "time": {"start_sec": .4, "end_sec": 1.}, "text": "right", "scene_id": "scene_004A", "script_matches": [{"block_id": "scene_004A_dialogue_001", "speaker": "B", "combined_score": .9}], "alignment": {"status": "llm_aligned", "needs_review": False, "reliable_anchor": False, "script_order_start": 1, "script_order_end": 1}},
    ]
    mapped = map_shots([shot_row], alignments, context, "tt1234567")
    other_scene = "scene_004A" if mapped[0]["scene"]["scene_id"] == "scene_001" else "scene_001"
    mapped[0]["local_script_context"]["action_before"] = [f"{other_scene}_action_001", "missing_action", f"{mapped[0]['scene']['scene_id']}_dialogue_001"]
    mapped[0]["scene_transition"] = False
    mapped[0]["scene_candidates"] = mapped[0]["scene_candidates"][:1]
    mapped[0]["scene_candidates"][0]["overlap_sec"] = -1
    mapped[0]["scene_candidates"][0]["confidence"] = 2
    mapped[0]["keyframe"]["frame"] += 1
    errors = validate_data(context, alignments, mapped, [shot_row]).errors
    assert any("local action crosses primary scene" in error for error in errors)
    assert any("unknown local action block" in error for error in errors)
    assert any("local context references non-action block" in error for error in errors)
    assert any("scene transition flag mismatch" in error for error in errors)
    assert any("omit matched scenes" in error for error in errors)
    assert any("invalid scene candidate overlap" in error for error in errors)
    assert any("invalid scene candidate confidence" in error for error in errors)
    assert any("keyframe changed" in error for error in errors)


def test_invalid_llm_block_is_rejected() -> None:
    context = _context([("A", "hello")])
    alignments = align_subtitles([_sub(1, "hello")], context, "tt1234567")
    with pytest.raises(ValueError, match="unknown block_id"):
        apply_alignment_responses(alignments, [{"subtitle_id": "subtitle_000001", "block_ids": ["invented"]}], context)


def test_end_to_end_fixture(tmp_path: Path) -> None:
    pdf = tmp_path / "tt1234567_Test.pdf"
    document = fitz.open(); page = document.new_page()
    page.insert_text((50, 50), "1 INT. ROOM - NIGHT 1")
    page.insert_text((70, 80), "They enter.")
    page.insert_text((250, 110), "HART")
    page.insert_text((180, 140), "Hello there.")
    document.save(pdf); document.close()
    subtitle = tmp_path / "test.srt"; subtitle.write_text("1\n00:00:00,200 --> 00:00:00,800\nHello there.\n", encoding="utf-8")
    shots = tmp_path / "shots.jsonl"; shots.write_text(json.dumps(_shot(1, 0, 1)) + "\n", encoding="utf-8")
    result = process_one(ContextOptions("tt1234567", pdf, subtitle, shots, tmp_path / "out", llm_mode="export"))
    assert result["validation_passed"]
    validation = validate_files(tmp_path / "out/movie_script_context.json", tmp_path / "out/subtitle_script_alignment.jsonl", tmp_path / "out/shot_script_context.jsonl", shots)
    assert validation.passed
