import csv

import pytest

from oscardp.stage06_3.gaze_labels import build_labels, evaluate, read_sheet

SHOTS = [{"shot_id": "shot_000001", "stage01_temporal": {"start": 10.0, "end": 20.0}},
         {"shot_id": "shot_000002", "stage01_temporal": {"start": 20.0, "end": 25.0}}]
FIELDS = ("unit_id", "shot_id", "character", "kind", "start_s", "end_s", "human_type", "human_target", "human_form",
          "human_certainty", "notes")


def row(shot, who, kind, start, end, kind_type="", target="", form="", certainty="HIGH", notes=""):
    return dict(zip(FIELDS, ("", shot, who, kind, str(start), str(end), kind_type, target, form, certainty, notes),
                    strict=True))


def rows(*items):
    return list(enumerate(items, 2))


GOOD = rows(
    row("shot_000001", "MICHELLE", "HOLD", 10.0, 11.0, "TARGET", "TEDDY"),
    row("shot_000001", "MICHELLE", "HOLD", 11.0, 12.0, "TARGET", "TEDDY", notes="merge me"),
    row("shot_000001", "MICHELLE", "MOVE, BLINK", 12.0, 12.2, "TARGET", "DON"),
    row("shot_000001", "MICHELLE", "HOLD", 12.9, 14.0, "TARGET", "DON"),  # gap after the MOVE is closed
    row("", "MICHELLE", "MOVE", 14.0, 14.1, "EXPRESSIVE", form="slightly down"),  # Shot found by time
    row("shot_000001", "MICHELLE", "HOLD", 14.1, 15.0, "EXPRESSIVE", form="slightly down"),
    row("shot_000001", "MICHELLE", "BLINK", 15.0, 15.3, "BLINK", form="slow blink"),
    row("shot_000001", "MICHELLE", "MOVE", 15.3, 15.5, "TARGET", "TEDDY", certainty="medium"),
    row("shot_000001", "MICHELLE", "HOLD", 15.5, 20.0, "TARGET", "TEDDY"),
    row("shot_000002", "DON", "UNMEASURED", 20.0, 25.0, "UNMEASURED", certainty="", notes="face blurred"),
)


def test_labels_are_cleaned_by_time():
    tracks, corrections = build_labels(GOOD, SHOTS)
    michelle, don = tracks
    segments = michelle["segments"]
    assert [(s["kind"], s["start_s"], s["end_s"], s["target"] or s["form"]) for s in segments] == [
        ("HOLD", 10.0, 12.0, "TEDDY"), ("MOVE", 12.0, 12.2, "DON"), ("HOLD", 12.2, 14.0, "DON"),
        ("MOVE", 14.0, 14.1, "DOWN"), ("HOLD", 14.1, 15.0, "DOWN"), ("BLINK", 15.0, 15.3, None),
        ("MOVE", 15.3, 15.5, "TEDDY"), ("HOLD", 15.5, 20.0, "TEDDY")]
    assert segments[0]["sheet_lines"] == [2, 3] and segments[0]["notes"] == "merge me"
    assert segments[1]["blink"] and segments[3]["form_text"] == "slightly down"
    assert segments[5]["form_text"] == "slow blink" and segments[6]["certainty"] == "MEDIUM"
    assert corrections == [{"track_key": "shot_000001_michelle", "sheet_line": 5, "field": "start_s", "from": 12.9,
                            "to": 12.2, "rule": "a HOLD starts when the MOVE to its target ends"}]
    assert [(s["from"], s["to"], s["landing_s"]) for s in michelle["shifts"]] == [
        ("TEDDY", "DON", 12.2), ("DON", "DOWN", 14.1), ("DOWN", "TEDDY", 15.5)]  # the BLINK is skipped
    assert don["status"] == "not_annotatable" and don["reason"] == "face blurred" and not don["segments"]


@pytest.mark.parametrize(("bad", "message"), [
    (row("shot_000001", "TEDDY", "HOLD", 10, 11, "TARGET", "NOBODY"), "target"),
    (row("shot_000001", "TEDDY", "HOLD", 10, 11, "EXPRESSIVE", form="sideways"), "FORMS"),
    (row("shot_000001", "TEDDY", "HOLD", 10, 11, "TARGET", "DON", certainty="sure"), "certainty"),
    (row("shot_000001", "TEDDY", "HOLD", 19, 21, "TARGET", "DON"), "not inside one Shot"),
    (row("shot_000001", "NOBODY", "HOLD", 10, 11, "TARGET", "DON"), "character"),
])
def test_bad_rows_are_rejected(bad, message):
    with pytest.raises(ValueError, match=message):
        build_labels(rows(bad), SHOTS)


def test_unexplained_gaps_overlaps_and_partial_tracks_are_rejected():
    hold = row("shot_000001", "TEDDY", "HOLD", 10, 11, "TARGET", "DON")
    with pytest.raises(ValueError, match="gap"):
        build_labels(rows(hold, row("shot_000001", "TEDDY", "HOLD", 12, 13, "TARGET", "MICHELLE")), SHOTS)
    with pytest.raises(ValueError, match="overlap"):
        build_labels(rows(hold, row("shot_000001", "TEDDY", "HOLD", 10.5, 13, "TARGET", "MICHELLE")), SHOTS)
    with pytest.raises(ValueError, match="no gaze"):
        build_labels(rows(hold, row("shot_000001", "TEDDY", "HOLD", 11, 13)), SHOTS)


def test_read_sheet_skips_blank_lines(tmp_path):
    path = tmp_path / "units.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(GOOD[0][1])
        writer.writerow(dict.fromkeys(FIELDS, ""))
        writer.writerow(GOOD[1][1])
    assert [n for n, _ in read_sheet(path)] == [2, 4]


def unit(kind, start, end, **extra):
    return {"kind": kind, "shot_id": "shot_000001", "track_id": "t1", "character": "MICHELLE",
            "start_s": start, "end_s": end, **extra}


def test_evaluate_against_machine_units():
    tracks, _ = build_labels(GOOD, SHOTS)
    proposals = {m: {"preferred": "TEDDY"} for m in ("fused", "visual", "geometry")}
    units = [
        unit("HOLD", 10.0, 12.1, primary_piece=True, proposals=proposals),
        unit("MOVE", 12.1, 13.0, signals=["eye", "head"]),  # on the first shift, late end
        unit("MOVE", 17.0, 18.0, signals=["eye"]),  # inside the final hold
        unit("HOLD", 13.0, 20.0, primary_piece=True, proposals=proposals),
    ]
    events = [{"shot_id": "shot_000001", "track_id": "t1", "start_s": 15.4, "end_s": 15.6}]
    report = evaluate(tracks, units, events)
    assert report["human"]["moves"] == 3
    assert report["move_detection"]["found_by_machine_move"] == 1
    assert report["move_detection"]["found_by_any_raw_event"] == 1
    assert report["machine_moves"] == {"total": 2, "on_human_move": 1, "inside_human_hold": 1}
    assert report["machine_move_timing"]["with_head"]["median_end_offset_s"] == 0.8
    fused = report["hold_target_by_time"]["fused"]
    # TEDDY is right on 10-12 and 15.5-20, wrong on DON 13-14; DOWN seconds are not target seconds
    assert fused["seconds_predicted"] == 7.5 and fused["seconds_correct"] == 6.5
