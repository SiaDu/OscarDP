from __future__ import annotations

from typing import Any

import pytest

from oscardp.script_context.alignment import align_subtitles, build_anchors, build_script_units, tokenize
from oscardp.script_context.llm_review import build_alignment_diagnostics, build_review_requests, validate_review_requests
from oscardp.script_context.schema import AlignmentConfig, CleanSubtitle


def context_for(scenes: list[list[str]]) -> dict[str, Any]:
    result = []
    for scene_index, lines in enumerate(scenes, 1):
        scene_id = f"scene_{scene_index:03d}"
        blocks = []
        for block_index, line in enumerate(lines, 1):
            blocks.append({"block_id": f"{scene_id}_dialogue_{block_index:03d}", "block_type": "dialogue", "script_page": scene_index, "source_order": block_index, "speaker": "HART", "character_cue": "HART", "parenthetical": None, "text": line})
        result.append({"scene_id": scene_id, "screenplay_scene_id": str(scene_index), "slugline": f"INT. ROOM {scene_index} - NIGHT", "int_ext": "INT", "location": f"ROOM {scene_index}", "time_of_day": "NIGHT", "script_pages": {"start": scene_index, "end": scene_index}, "scene_characters": ["HART"], "script_blocks": blocks, "semantic_annotations": {"scene_summary": None, "dramatic_function": None}, "parsing": {"status": "parsed", "needs_review": False}})
    return {"schema_version": "1.0", "movie": {"movie_id": "tt1", "title": "Fixture"}, "source_files": {}, "summary": {}, "broken_pages": [], "script_scenes": result}


def subs(lines: list[str]) -> list[CleanSubtitle]:
    return [CleanSubtitle(f"subtitle_{index:06d}", f"00:00:{index:02d}.000", f"00:00:{index:02d}.800", float(index), float(index) + .8, line, line, "en") for index, line in enumerate(lines, 1)]


def test_five_subtitles_share_one_long_block_with_token_spans() -> None:
    parts = ["Every morning we walk together", "past the old theater doors", "and remember all the songs", "that filled this crowded room", "before the final curtain fell"]
    rows = align_subtitles(subs(parts), context_for([[" ".join(parts)]]), "tt1")
    assert all(row["alignment"]["status"] == "auto_aligned" for row in rows)
    assert len({row["script_matches"][0]["block_id"] for row in rows}) == 1
    spans = [(row["script_matches"][0]["matched_script_token_start"], row["script_matches"][0]["matched_script_token_end"]) for row in rows]
    assert spans == sorted(spans) and all(left[1] <= right[0] for left, right in zip(spans, spans[1:]))


def test_ten_subtitles_can_reference_one_block_without_consuming_it() -> None:
    parts = [f"fragment number {index} has words" for index in range(10)]
    rows = align_subtitles(subs(parts), context_for([[" ".join(parts)]]), "tt1")
    assert len(rows) == 10
    assert len({row["script_matches"][0]["block_id"] for row in rows}) == 1


def test_one_subtitle_spans_two_adjacent_blocks() -> None:
    context = context_for([["Opening reliable anchor has enough words", "This thought begins in one place", "and finishes in another place tonight", "Closing reliable anchor has enough words"]])
    lines = ["Opening reliable anchor has enough words", "This thought begins in one place and finishes in another place tonight", "Closing reliable anchor has enough words"]
    rows = align_subtitles(subs(lines), context, "tt1", AlignmentConfig(.8, .6, .05, 20))
    assert len(rows[1]["script_matches"]) == 2
    assert rows[1]["alignment"]["method"] == "rapidfuzz_multi_block"


def test_normalized_substring_fragment_has_offsets_and_coverage() -> None:
    context = context_for([["Before we leave the city we should remember every promise that we made together"]])
    rows = align_subtitles(subs(["we should remember every promise"]), context, "tt1")
    match = rows[0]["script_matches"][0]
    assert rows[0]["alignment"]["method"] == "anchor_unique_substring"
    assert match["matched_script_token_start"] == 5
    assert match["subtitle_token_coverage"] == 1.0
    assert 0 < match["script_token_coverage"] < 1


def test_ambiguous_short_phrase_is_not_global_anchor() -> None:
    context = context_for([["Yes", "A sufficiently long unique sentence appears here"]])
    anchors = build_anchors(subs(["Yes", "A sufficiently long unique sentence appears here"]), build_script_units(context))
    assert [anchor.subtitle_index for anchor in anchors] == [1]


def test_anchors_split_regions_and_deleted_script_gap_is_allowed() -> None:
    context = context_for([["First reliable anchor has enough words", "This deleted screenplay dialogue never appears", "Last reliable anchor also has enough words"]])
    rows = align_subtitles(subs(["First reliable anchor has enough words", "Last reliable anchor also has enough words"]), context, "tt1")
    assert [row["alignment"]["script_order_start"] for row in rows] == [0, 2]
    diagnostics = build_alignment_diagnostics(context, rows, [])
    assert diagnostics["screenplay_deletion_gaps"] == 1
    assert diagnostics["monotonicity_violations"] == 0


def test_improvised_subtitle_between_anchors_remains_no_match() -> None:
    context = context_for([["First reliable anchor has enough words", "Last reliable anchor also has enough words"]])
    rows = align_subtitles(subs(["First reliable anchor has enough words", "This improvised business is nowhere in the screenplay", "Last reliable anchor also has enough words"]), context, "tt1")
    assert rows[1]["alignment"]["status"] == "no_match"


def test_no_anchor_review_is_insufficient_and_grouped() -> None:
    context = context_for([["Some screenplay sentence that does not match"]])
    rows = align_subtitles(subs(["Yes", "No", "Okay"]), context, "tt1")
    requests = build_review_requests(context, rows)["alignment_requests"]
    assert len(requests) == 1
    assert requests[0]["subtitle_ids"] == ["subtitle_000001", "subtitle_000002", "subtitle_000003"]
    assert requests[0]["insufficient_candidates"] is True
    assert requests[0]["dialogue_candidates"] == []


def test_same_scene_anchor_window_stays_in_scene() -> None:
    context = context_for([["First same scene anchor has enough words", "Candidate dialogue in the middle", "Second same scene anchor has enough words"], ["Unrelated next scene dialogue words"]])
    rows = align_subtitles(subs(["First same scene anchor has enough words", "unmatched improvised middle subtitle words", "Second same scene anchor has enough words"]), context, "tt1")
    request = build_review_requests(context, rows)["alignment_requests"][0]
    assert request["candidate_scenes"] == ["scene_001"]


def test_adjacent_scene_window_has_only_two_scenes_and_not_scene_one_fallback() -> None:
    context = context_for([["Opening anchor sentence has enough words"], ["Middle anchor before unresolved has words"], ["Closing anchor after unresolved has enough words"]])
    rows = align_subtitles(subs(["Opening anchor sentence has enough words", "Middle anchor before unresolved has words", "unmatched words from middle film segment", "Closing anchor after unresolved has enough words"]), context, "tt1")
    request = build_review_requests(context, rows)["alignment_requests"][0]
    assert request["candidate_scenes"] == ["scene_002", "scene_003"]
    assert "scene_001" not in request["candidate_scenes"]


def test_invalid_review_candidate_reference_is_rejected() -> None:
    context = context_for([["A reliable sentence with enough distinct words"]])
    request = {"request_id": "bad", "candidate_scenes": ["scene_001"], "dialogue_candidates": [{"block_id": "invented", "screenplay_order": 0}], "insufficient_candidates": False}
    assert validate_review_requests([request], context) == ["bad references invalid candidate invented"]


def test_blue_moon_style_long_fragment_fixture_has_no_substring_false_no_match() -> None:
    parts = ["All right for the last time", "put the letters on the table", "and tell me what happened", "before everybody arrived here", "at the crowded bar tonight"]
    context = context_for([[" ".join(parts)]])
    rows = align_subtitles(subs(parts), context, "tt1")
    requests = build_review_requests(context, rows)["alignment_requests"]
    diagnostics = build_alignment_diagnostics(context, rows, requests)
    assert diagnostics["unresolved_local_substring_false_no_match"] == 0
    assert diagnostics["n_subtitles_to_one_block_count"] == 1
