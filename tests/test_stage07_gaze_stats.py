"""Synthetic checks for Stage07 gaze statistics from human gaze labels."""
import pytest

from oscardp.stage07 import gaze_stats as gs

PEOPLE = {"ANNA", "BEN", "CARL"}


def seg(kind, start, end, target=None, form=None):
    return {"kind": kind, "start_s": start, "end_s": end, "target": target, "form": form}


def track(shot, who, segments, shifts=()):
    return {"shot_id": shot, "character": who, "status": "annotated", "segments": segments,
            "shifts": [{"shift_id": f"{who}_{n}", "start_s": a, "landing_s": b, "to": to}
                       for n, (a, b, to) in enumerate(shifts)]}


def test_roles_and_shares_follow_speech():
    cues = [(0., 4., "ANNA"), (4., 8., "BEN")]
    voiced = [[0., 8.]]
    anna = track("s1", "ANNA", [seg("HOLD", 0., 3., "BEN"), seg("MOVE", 3., 3.5, "BEN"),
                                seg("HOLD", 3.5, 6., None, "DOWN"), seg("HOLD", 6., 8., "BEN")])
    stats = gs.character_stats([anna], PEOPLE, cues, voiced)["ANNA"]
    assert stats["speaking"]["share"]["BEN"] == pytest.approx(3 / 4.2, abs=0.01)  # margin widens the own cue
    assert stats["speaking"]["shifts"] == 1
    listening = stats["listening"]
    assert listening["share"]["away"] == pytest.approx(1.8 / 3.8, abs=0.01)
    assert listening["on_current_speaker"] == pytest.approx(2 / 3.8, abs=0.01)


def test_following_mutual_and_joint_attention():
    anna = track("s1", "ANNA", [seg("HOLD", 0., 2., "BEN"), seg("MOVE", 2., 2.3, "CARL"), seg("HOLD", 2.3, 6., "CARL")],
                 shifts=[(2., 2.3, "CARL")])
    ben = track("s1", "BEN", [seg("HOLD", 0., 3., "ANNA"), seg("MOVE", 3., 3.4, "CARL"), seg("HOLD", 3.4, 6., "CARL")],
                shifts=[(3., 3.4, "CARL")])
    mutual, joint, following = gs.pair_events([anna, ben], PEOPLE)
    assert mutual[0]["seconds"] == pytest.approx(2., abs=0.02)
    assert joint[0]["target"] == "CARL" and joint[0]["seconds"] == pytest.approx(2.6, abs=0.02)
    (f,) = following
    assert (f["leader"], f["follower"], f["target"]) == ("ANNA", "BEN", "CARL") and f["lag_s"] == pytest.approx(0.7)


def test_late_turn_is_not_following():
    anna = track("s1", "ANNA", [seg("MOVE", 0., 0.3, "CARL"), seg("HOLD", 0.3, 9., "CARL")], shifts=[(0., 0.3, "CARL")])
    ben = track("s1", "BEN", [seg("HOLD", 0., 5., "ANNA"), seg("MOVE", 5., 5.3, "CARL"), seg("HOLD", 5.3, 9., "CARL")],
                shifts=[(5., 5.3, "CARL")])
    assert gs.pair_events([anna, ben], PEOPLE)[2] == []


def test_joint_spans_need_both_on_the_same_third_person():
    from oscardp.stage07.timeline_figure import joint_spans
    labels = {"tracks": [
        {"character": "MICHELLE", "status": "annotated", "segments": [seg("HOLD", 0., 2., "TEDDY"), seg("HOLD", 2.5, 6., "DON")]},
        {"character": "TEDDY", "status": "annotated", "segments": [seg("HOLD", 0., 3., "MICHELLE"), seg("HOLD", 3.4, 5., "DON")]}]}
    assert joint_spans(labels, ["MICHELLE", "TEDDY"]) == [(3.4, 5., "DON")]
