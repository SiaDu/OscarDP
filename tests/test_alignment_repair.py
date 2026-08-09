from __future__ import annotations

import hashlib
from typing import Any

import pytest

from oscardp.script_context.alignment import align_subtitles, build_anchors, build_script_units, tokenize
from oscardp.script_context.llm_review import augment_review_requests_global_lexical, build_alignment_diagnostics, build_review_requests, validate_review_requests
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


def test_versioned_global_lexical_rescue_adds_outside_candidate_without_overwriting(tmp_path) -> None:
    import json
    context = context_for([["Local unrelated dialogue"], ["I got gold and they did not get all of it"]])
    context_path = tmp_path / "context.json"; context_path.write_text(json.dumps(context), encoding="utf-8")
    request = {
        "request_id": "alignment_review_000001", "subtitle_ids": ["subtitle_000001"],
        "subtitles": [{"subtitle_id": "subtitle_000001", "text": "I got gold", "time": {"start_sec": 1, "end_sec": 2}}],
        "dialogue_candidates": [{"scene_id": "scene_001", "block_id": "scene_001_dialogue_001", "screenplay_order": 0, "speaker": "HART", "text": "Local unrelated dialogue", "parenthetical": None, "retrieval_methods": ["anchor_window"], "lexical_score": None, "semantic_score": None, "retrieval_score": None}],
        "candidate_scenes": ["scene_001"], "candidate_limit": 2, "estimated_screenplay_range": {"start_screenplay_order": 0, "end_screenplay_order": 0, "start_scene_id": "scene_001", "end_scene_id": "scene_001"},
    }
    requests = tmp_path / "requests.jsonl"; requests.write_text(json.dumps(request) + "\n", encoding="utf-8")
    output = tmp_path / "rescued.jsonl"
    result = augment_review_requests_global_lexical(requests, context_path, output)
    row = json.loads(output.read_text())
    assert result["rescued_target_count"] == 1
    assert [item["block_id"] for item in row["dialogue_candidates"]] == ["scene_001_dialogue_001", "scene_002_dialogue_001"]
    assert row["retrieval_version"] == "global_lexical_rescue_v2"
    retrieval_manifest = json.loads(output.with_suffix(".jsonl.manifest.json").read_text())
    assert retrieval_manifest["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert retrieval_manifest["source_requests_sha256"] == hashlib.sha256(requests.read_bytes()).hexdigest()
    assert requests.read_text() == json.dumps(request) + "\n"
    with pytest.raises(FileExistsError):
        augment_review_requests_global_lexical(requests, context_path, output)


def test_global_lexical_rescue_rejects_function_word_only_cross_scene_phrase(tmp_path) -> None:
    import json
    context = context_for([["Local scene response"], ["How are you doing today?"]])
    context_path = tmp_path / "context.json"; context_path.write_text(json.dumps(context), encoding="utf-8")
    request = {
        "request_id": "alignment_review_000001", "subtitle_ids": ["subtitle_000001"],
        "subtitles": [{"subtitle_id": "subtitle_000001", "text": "How are you", "time": {"start_sec": 1, "end_sec": 2}}],
        "dialogue_candidates": [{"scene_id": "scene_001", "block_id": "scene_001_dialogue_001", "screenplay_order": 0, "speaker": "HART", "text": "Local scene response", "parenthetical": None, "retrieval_methods": ["anchor_window"], "lexical_score": None, "semantic_score": None, "retrieval_score": None}],
        "candidate_scenes": ["scene_001"], "candidate_limit": 2,
    }
    requests = tmp_path / "requests.jsonl"; requests.write_text(json.dumps(request) + "\n", encoding="utf-8")
    output = tmp_path / "rescued.jsonl"
    result = augment_review_requests_global_lexical(requests, context_path, output)
    row = json.loads(output.read_text())
    assert result["rescued_target_count"] == 0
    assert [item["block_id"] for item in row["dialogue_candidates"]] == ["scene_001_dialogue_001"]
    assert "retrieval_version" not in row


def test_anchors_split_regions_and_deleted_script_gap_is_allowed() -> None:
    context = context_for([["First reliable anchor has enough words", "This deleted screenplay dialogue never appears", "Last reliable anchor also has enough words"]])
    rows = align_subtitles(subs(["First reliable anchor has enough words", "Last reliable anchor also has enough words"]), context, "tt1")
    assert [row["alignment"]["script_order_start"] for row in rows] == [0, 2]
    diagnostics = build_alignment_diagnostics(context, rows, [])
    assert diagnostics["screenplay_deletion_gaps"] == 1
    assert diagnostics["monotonicity_violations"] == 0


def test_backward_same_block_future_exact_does_not_starve_current_fragment() -> None:
    context = context_for([["Opening unique anchor has words target fragment has enough words target fragment has enough words"], ["Final reliable anchor has distinct words"]])
    rows = align_subtitles(subs([
        "Opening unique anchor has words",
        "target fragment has enough words",
        "Opening unique anchor has words",
        "Final reliable anchor has distinct words",
    ]), context, "tt1")
    assert rows[1]["alignment"]["status"] != "no_match"
    assert rows[1]["script_matches"][0]["block_id"] == "scene_001_dialogue_001"
    requests = build_review_requests(context, rows)["alignment_requests"]
    diagnostics = build_alignment_diagnostics(context, rows, requests)
    assert diagnostics["unresolved_local_substring_false_no_match"] == 0


def test_repeated_refrain_keeps_earlier_nonmonotonic_exact_as_review_candidate() -> None:
    context = context_for([["Opening reliable anchor has distinct words"], [
        "first refrain has enough words second refrain has enough words third refrain has enough words final refrain has enough words"
    ], ["Closing reliable anchor has distinct words"]])
    rows = align_subtitles(subs([
        "Opening reliable anchor has distinct words",
        "final refrain has enough words",
        "first refrain has enough words",
        "second refrain has enough words",
        "third refrain has enough words",
        "final refrain has enough words",
        "Closing reliable anchor has distinct words",
    ]), context, "tt1")
    first_final = rows[1]
    assert first_final["alignment"]["status"] == "needs_review"
    assert first_final["alignment"]["method"] == "normalized_substring_nonmonotonic"
    assert first_final["alignment"]["review_reason"] == "nonmonotonic_exact_fragment"
    assert first_final["script_matches"][0]["block_id"] == "scene_002_dialogue_001"
    assert rows[5]["alignment"]["status"] == "auto_aligned"


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
    assert requests[0]["fallback_used"] is True
    assert requests[0]["candidate_interval_reason"] == "fallback_timeline_estimate"


def test_short_reply_between_anchors_separated_by_empty_scene_keeps_boundary_candidates() -> None:
    context = context_for([
        ["Opening reliable anchor has enough distinct words"],
        [],
        ["Closing reliable anchor also has enough distinct words"],
    ])
    rows = align_subtitles(subs([
        "Opening reliable anchor has enough distinct words",
        "Well",
        "Closing reliable anchor also has enough distinct words",
    ]), context, "tt1")
    request = build_review_requests(context, rows)["alignment_requests"][0]
    assert request["fallback_used"] is True
    assert request["candidate_interval_reason"] == "fallback_between_reliable_anchors"
    assert request["insufficient_candidates"] is False
    assert [candidate["block_id"] for candidate in request["dialogue_candidates"]] == [
        "scene_001_dialogue_001", "scene_003_dialogue_001",
    ]
    assert all("anchor_boundary" in candidate["retrieval_methods"] for candidate in request["dialogue_candidates"])


def test_fallback_keeps_existing_low_confidence_automatic_mapping() -> None:
    context = context_for([["Opening reliable anchor has enough distinct words", "A long line ending with stop"]])
    rows = align_subtitles(subs(["Opening reliable anchor has enough distinct words", "Stop"]), context, "tt1")
    assert rows[1]["script_matches"]
    assert rows[1]["alignment"]["needs_review"] is True
    request = build_review_requests(context, rows)["alignment_requests"][0]
    automatic_id = rows[1]["script_matches"][0]["block_id"]
    candidate = next(item for item in request["dialogue_candidates"] if item["block_id"] == automatic_id)
    assert "automatic_mapping" in candidate["retrieval_methods"]
    assert request["insufficient_candidates"] is False


def test_short_tail_reply_uses_anchor_boundary_only_when_retrieval_is_empty() -> None:
    context = context_for([["Opening reliable anchor has enough distinct words", "First tail option", "Second tail option", "Third tail option", "Fourth tail option"]])
    rows = align_subtitles(subs(["Opening reliable anchor has enough distinct words", "Kisses"]), context, "tt1")
    request = build_review_requests(context, rows)["alignment_requests"][0]
    assert request["fallback_used"] is True
    assert request["candidate_interval_reason"] == "fallback_after_reliable_anchor"
    assert request["insufficient_candidates"] is False
    assert 1 <= len(request["dialogue_candidates"]) <= 4
    assert any("anchor_boundary" in candidate["retrieval_methods"] for candidate in request["dialogue_candidates"])
    assert all(
        set(candidate["retrieval_methods"]) <= {"anchor_boundary", "adjacent_dialogue"}
        for candidate in request["dialogue_candidates"]
    )


def test_wide_anchor_fallback_retrieves_vehicle_exit_dialogue_in_global_order() -> None:
    context = context_for([
        ["Opening reliable anchor has enough distinct words"],
        ["Unrelated dialogue in the second scene"],
        ["More unrelated dialogue before the relevant exchange"],
        [
            "Could you get out of the vehicle, please?",
            "Do you really need to come into my car, sir?",
            "I do. Trust me.",
        ],
        ["Unrelated dialogue after the relevant exchange"],
        ["Another unrelated scene with spoken words"],
        ["Closing reliable anchor also has enough distinct words"],
    ])
    subtitle_texts = [
        "Opening reliable anchor has enough distinct words",
        "Could you step out of the car, sir?", "Why?", "I need to get inside the car.",
        "You can trust me.", "Is it really necessary?", "Yes.", "Please step out.",
        "Closing reliable anchor also has enough distinct words",
    ]
    rows = []
    for index, text in enumerate(subtitle_texts):
        reliable = index in {0, len(subtitle_texts) - 1}
        order = 0 if index == 0 else 8 if reliable else None
        rows.append({
            "subtitle_id": f"subtitle_{index + 1:06d}", "alignment_group_id": f"align_{index + 1:06d}",
            "time": {"start_sec": float(index), "end_sec": float(index) + .8}, "text": text,
            "scene_id": "scene_001" if index == 0 else "scene_007" if reliable else None,
            "script_matches": ([{"block_id": "scene_001_dialogue_001", "combined_score": 1.0}] if index == 0 else [{"block_id": "scene_007_dialogue_001", "combined_score": 1.0}] if reliable else []),
            "alignment": {"method": "anchor_normalized_exact" if reliable else "no_match", "status": "auto_aligned" if reliable else "no_match", "needs_review": not reliable, "reliable_anchor": reliable, "script_order_start": order, "script_order_end": order, "candidate_margin": 0.0},
        })
    request = build_review_requests(context, rows)["alignment_requests"][0]
    ids = [candidate["block_id"] for candidate in request["dialogue_candidates"]]
    expected = {f"scene_004_dialogue_{index:03d}" for index in range(1, 4)}
    assert expected <= set(ids)
    assert request["fallback_used"] is True
    assert request["candidate_interval_reason"] == "fallback_between_reliable_anchors"
    assert request["insufficient_candidates"] is False
    assert ids == list(dict.fromkeys(ids))
    assert [candidate["screenplay_order"] for candidate in request["dialogue_candidates"]] == sorted(candidate["screenplay_order"] for candidate in request["dialogue_candidates"])
    assert all(candidate["retrieval_methods"] for candidate in request["dialogue_candidates"])
    assert validate_review_requests([request], context) == []


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
