import csv

import pytest

from oscardp.stage06_3.behavior_units import (
    build_units,
    carry_over,
    coverage_gaps,
    machine_summary,
    movement_groups,
    previous_by_unit,
    split_state,
    state_kind,
)
from oscardp.stage06_3.timeline import (
    COLUMNS,
    candidate_rows,
    evaluate_timeline,
    read_timeline,
    segments,
    write_timeline,
)

SHOT = {"shot_id": "shot_000215",
        "stage01_temporal": {"start": 10.0, "end": 20.0, "start_frame": 240, "end_frame": 480},
        "reported_faces": [
            {"track_id": "shot_000215_track_001", "start_sec": 10.0, "end_sec": 12.0},
            {"track_id": "shot_000215_track_002", "start_sec": 12.5, "end_sec": 12.6},
            {"track_id": "shot_000215_track_003", "start_sec": 13.0, "end_sec": 20.0},
        ]}
TRACKS = {
    "shot_000215_track_001": {"shot_id": "shot_000215", "track_id": "shot_000215_track_001",
                              "character": "MICHELLE", "measured": True},
    "shot_000215_track_002": {"shot_id": "shot_000215", "track_id": "shot_000215_track_002",
                              "character": "MICHELLE", "measured": False},
    "shot_000215_track_003": {"shot_id": "shot_000215", "track_id": "shot_000215_track_003",
                              "character": "MICHELLE", "measured": True},
}


def state(track, number, start, end, *, direction=True, qc="stable", anchor=None):
    anchor = anchor or (start, end)
    return {"state_id": f"{track}_look_{number:03d}", "shot_id": "shot_000215", "track_id": track,
            "character": "MICHELLE", "start_frame": round(start * 24), "end_frame": round(end * 24),
            "start_s": start, "end_s": end, "direction_source": "eye_head" if direction else "unobservable",
            "qc_status": qc, "look_direction": {"vector": [0, 0, -1]} if direction else None,
            "gaze_evidence": {"anchor": {"start_s": anchor[0], "end_s": anchor[1]} if direction else None}}


def event(track, number, start, end, kind="head_or_gaze_motion", assessment="return", **extra):
    return {"event_id": f"{track}_direction_{number:03d}", "track_id": track,
            "start_frame": round(start * 24), "end_frame": round(end * 24), "start_s": start, "end_s": end,
            "assessment": assessment, "direction_change_deg": None, "peak_gaze_departure_deg": None,
            "source_event": {"type": kind, "trigger_signals": ["head"], "head_change_deg": extra.get("head")}}


def prediction(state_id, preferred, ranking):
    return {state_id: {"preferred_target": preferred, "target_ranking": ranking,
                       "evidence_level": "visual_directional", "decision_status": "preferred_but_unconfirmed"}}


T1, T3 = "shot_000215_track_001", "shot_000215_track_003"
STATES = [state(T1, 1, 10.0, 12.0), state(T3, 1, 13.0, 14.0, direction=False, qc="blink_gap"),
          state(T3, 2, 14.0, 20.0, anchor=(16.0, 17.0))]
EVENTS = [event(T3, 1, 15.0, 15.3, head=20.0), event(T3, 2, 15.4, 15.6, head=-30.0),
          event(T3, 3, 17.0, 17.2, assessment="adjustment_or_same_direction"),
          event(T3, 4, 13.2, 13.6, kind="blink_candidate", assessment="blink_or_blink_overlap"),
          event(T3, 5, 18.0, 18.3, assessment="unreliable_motion"),
          event(T3, 6, 14.0, 16.5, assessment="missing_comparable_stable_flanks")]
PREDICTIONS = {m: {**prediction(f"{T1}_look_001", "TEDDY", [["TEDDY"], ["DON"]]),
                   **prediction(f"{T3}_look_001", None, []),
                   **prediction(f"{T3}_look_002", None, [["TEDDY", "DON"]])} for m in ("fused", "visual", "geometry")}


def units():
    return build_units([SHOT], TRACKS, STATES, EVENTS, PREDICTIONS)


def test_state_kinds():
    assert [state_kind(s) for s in STATES] == ["HOLD", "BLINK", "HOLD"]
    assert state_kind(state(T1, 9, 0, 1, direction=False, qc="no_sustained_direction")) == "UNMEASURED"


def test_movements_merge_short_gaps_and_skip_blinks_and_micro_adjustments():
    groups = movement_groups([e for e in EVENTS if e["track_id"] == T3])
    assert [[e["event_id"][-3:] for e in g] for g in groups] == [["001", "002"], ["005"]]


def test_movements_cut_states_into_one_non_overlapping_timeline():
    built = units()
    assert [(u["unit_id"].removeprefix("shot_000215_"), u["start_s"], u["end_s"]) for u in built] == [
        ("track_001_look_001", 10.0, 12.0), ("michelle_gap_001", 12.0, 13.0), ("track_003_look_001", 13.0, 14.0),
        ("track_003_look_002_p1", 14.0, 15.0), ("track_003_move_001", 15.0, 15.6),
        ("track_003_look_002_p2", 15.6, 18.0), ("track_003_move_002", 18.0, 18.3),
        ("track_003_look_002_p3", 18.3, 20.0)]
    pieces = [u for u in built if u.get("look_state_id") == f"{T3}_look_002"]
    assert [(u["direction_measured_here"], u["primary_piece"]) for u in pieces] == [
        (False, False), (True, True), (False, False)]
    move = next(u for u in built if u["unit_id"].endswith("move_001"))
    assert (move["head_change_deg"], move["look_states"], move["qc_flagged"]) == (30.0, [f"{T3}_look_002"], False)
    assert next(u for u in built if u["unit_id"].endswith("move_002"))["qc_flagged"]
    assert "head 30°" in machine_summary(move)
    assert "direction measured in another part" in machine_summary(pieces[0])
    # The 2.5 s drift (event 006) does not cut the state; every piece reports it.
    assert all(u["long_motion_s"] == [2.5] for u in pieces) and "long motion 2.5 s" in machine_summary(pieces[1])


def test_short_leftovers_join_the_adjacent_movement():
    unit = {**units()[0], "unit_id": "s_look_001", "look_state_id": "s_look_001",
            "start_frame": 0, "end_frame": 48, "start_s": 0.0, "end_s": 2.0, "anchor": {"start_s": .5, "end_s": 1.5}}
    first = {"start_frame": 2, "end_frame": 10, "start_s": .1, "end_s": .4}
    second = {"start_frame": 46, "end_frame": 60, "start_s": 1.9, "end_s": 2.5}
    pieces = split_state(unit, [first, second])
    assert [(p["unit_id"], p["start_s"], p["end_s"], p["primary_piece"]) for p in pieces] == [
        ("s_look_001_p1", .4, 1.9, True)]
    assert (first["start_s"], first["start_frame"]) == (0.0, 0)


def test_old_labels_go_to_the_measured_piece_only():
    built = units()
    old = {f"{T3}_look_002": {**carry_over("teddy", "100%"), "v1_3_label": "teddy", "v1_3_note": "看两次 Don"}}
    placed = previous_by_unit(built, old)
    assert placed[f"{T3}_look_002_p2"]["human_type"] == "TARGET"
    assert placed[f"{T3}_look_002_p1"] == {"v1_3_label": "teddy", "v1_3_note": "看两次 Don"}
    assert placed[f"{T3}_move_001"] == {"v1_3_label": "teddy", "v1_3_note": "看两次 Don"}


def test_a_state_swallowed_by_a_movement_hands_over_its_label():
    swallowed = [*STATES, state(T1, 2, 12.0, 12.4)]
    events = [*EVENTS, event(T1, 1, 11.9, 12.4)]
    built = build_units([SHOT], TRACKS, swallowed, events,
                        {m: {**p, **prediction(f"{T1}_look_002", None, [])} for m, p in PREDICTIONS.items()})
    assert not any(u.get("look_state_id") == f"{T1}_look_002" for u in built)
    placed = previous_by_unit(built, {f"{T1}_look_002": {**carry_over("don", "100%"), "v1_3_label": "don",
                                                          "v1_3_note": ""}})
    assert placed[f"{T1}_move_001"]["human_target"] == "DON"


def test_gaps_report_unmeasured_fragments_and_ignore_short_edges():
    gaps = coverage_gaps(SHOT, "MICHELLE", [STATES[0], STATES[2]],
                         [{"track_id": "shot_000215_track_002", "start_s": 12.5, "end_s": 12.6}])
    assert [(g["start_s"], g["end_s"], g["position"], g["unmeasured_fragments"]) for g in gaps] == [
        (12.0, 14.0, "between_tracks", ["shot_000215_track_002"])]


@pytest.mark.parametrize("label,expected", [
    ("michalle", ("TARGET", "MICHELLE", "")), ("Don", ("TARGET", "DON", "")), ("blink", ("BLINK", "", "")),
    ("down right(以teddy为坐标)", ("EXPRESSIVE", "", "DOWN")), ("look at the wall", None), ("", None)])
def test_carry_over_v13_labels(label, expected):
    found = carry_over(label, "100%")
    assert (None if found is None else (found["human_type"], found["human_target"], found["human_form"])) == expected
    assert found is None or found["human_certainty"] == "HIGH"


def row(start, gaze, source="x"):
    return {"shot_id": "shot_000215", "character": "MICHELLE", "start_s": start, "gaze": gaze, "source_unit": source}


def test_candidate_rows_keep_labels_and_edited_times_and_dissolve_none():
    built = units()
    labels = {f"{T1}_look_001": {"human_type": "TARGET", "human_target": "TEDDY", "human_certainty": "high"},
              f"{T3}_look_001": {"human_type": "BLINK"},
              f"{T3}_move_001": {"human_type": "NONE", "notes": "no move"},
              f"{T3}_look_002_p2": {"human_type": "TARGET", "human_target": "DON", "start_s": 15.65}}
    rows = candidate_rows(built, labels)
    assert [(r["start_s"], r["gaze"], r["certainty"]) for r in rows] == [
        (10.0, "TEDDY", "HIGH"), (12.0, "", ""), (13.0, "BLINK", ""), (14.0, "", ""), (15.65, "DON", ""),
        (18.0, "", ""), (18.3, "", "")]
    assert rows[3]["notes"] == "[track_003_move_001] no move"
    first = candidate_rows(built, {f"{T1}_look_001": {"human_type": "NONE"}})
    assert (first[0]["start_s"], first[0]["source_unit"]) == (10.0, "shot_000215_michelle_gap_001")


def test_timeline_sheet_round_trip_and_problems(tmp_path):
    rows = candidate_rows(units(), {})
    write_timeline(tmp_path / "timeline.csv", rows)
    assert len(read_timeline(tmp_path / "timeline.csv", [SHOT])) == len(rows)
    bad = [{**dict.fromkeys(COLUMNS, ""), "shot_id": "shot_000215", "character": "michelle", "start_s": s, "gaze": g}
           for s, g in (("11", "michalle"), ("25", "DON"), ("11", "TEDDY"))]
    with (tmp_path / "bad.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(bad)
    with pytest.raises(ValueError) as error:
        read_timeline(tmp_path / "bad.csv", [SHOT])
    message = str(error.value)
    assert "michalle" in message and "outside" in message and "two rows start at 11.0" in message


def test_segments_continue_through_blinks_merge_repeats_and_end_at_the_cut():
    rows = [row(15.0, "DON"), row(10.0, "TEDDY"), row(13.0, "BLINK"), row(14.0, "teddy".upper()), row(15.6, "")]
    assert segments(rows, [SHOT])[("shot_000215", "MICHELLE")] == [
        (10.0, 15.0, "TEDDY"), (15.0, 15.6, "DON"), (15.6, 20.0, "")]


def test_evaluate_timeline_by_time_and_shift_detection():
    built = units()
    rows = [row(10.0, "TEDDY", f"{T1}_look_001"), row(15.0, "DON", f"{T3}_move_001"),
            row(15.6, "TEDDY", ""), row(18.0, "TEDDY", f"{T3}_move_002")]
    report = evaluate_timeline(built, rows, segments(rows, [SHOT]))
    fused = report["target_accuracy_by_time"]["fused"]
    assert (fused["seconds_with_prediction"], fused["time_accuracy"], fused["time_coverage"]) == (2.0, 1.0, 0.2)
    shifts = report["gaze_shifts"]
    assert (shifts["human_shifts"], shifts["found_by_move_start"], shifts["found_by_any_machine_boundary"]) == (
        2, 0.5, 1.0)
    assert (shifts["reviewed_move_starts"], shifts["move_starts_on_a_shift"]) == (2, 1)
    assert report["rows_added_by_reviewer"] == 1 and report["machine_rows_removed"] == 5
