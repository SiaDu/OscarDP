from __future__ import annotations

import json
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
from oscardp.script_context.validation import validate_files


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


def test_no_cross_scene_interpolation() -> None:
    context = _context([("A", "hello"), ("B", "goodbye")], second_scene=True)
    alignments = align_subtitles([_sub(1, "hello", 0, 0.5), _sub(2, "goodbye", 2.5, 3)], context, "tt1234567")
    rows = map_shots([_shot(1, 0, 1), _shot(2, 1, 2), _shot(3, 2, 3)], alignments, context, "tt1234567", 10)
    assert rows[1]["scene"] is None


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
