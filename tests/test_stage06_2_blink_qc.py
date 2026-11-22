from collections import Counter

import pytest

from oscardp.stage06_2.blink_qc import blink_qc
from oscardp.stage06_2.observability_replay import evidence_rows, run
from oscardp.stage06_2.stable import DIRECTION_LIMIT, MAX_TRANSITION_S, POSITION_LIMIT
from oscardp.stage06_2.evidence_replay import WEAK_X


def lids(aperture):
    return [[[0,10],[7,10-10*aperture],[13,10-10*aperture],[20,10]],
            [[0,10],[7,10+10*aperture],[13,10+10*aperture],[20,10]]]


def rows(left, right=None):
    if right is None:
        right = left
    return [{"global_frame_index":i,"movie_timestamp_sec":1500+i*.04,
             "source_next_timestamp_sec":1500+(i+1)*.04,
             "tracking_usable":True,"head_pose_valid":True,"pose_Rx":0.,"pose_Ry":0.,"pose_Rz":0.,
             "face_bbox_xywh":[0,0,100,100],
             "left_eye_visibility":"usable" if a is not None else "detection_failure",
             "right_eye_visibility":"usable" if b is not None else "detection_failure",
             "left_mp_eyelids_px":lids(a) if a is not None else None,
             "right_mp_eyelids_px":lids(b) if b is not None else None}
            for i,(a,b) in enumerate(zip(left,right,strict=True))]


def old_event(end):
    return {"event_type":"eye_closure_candidate","event_id":"prior","start_frame":4,"end_frame":end}


OPEN = [.23]*5


@pytest.mark.parametrize("reason", ["detection_failure","track_alignment_failure"])
def test_failed_landmarks_break_the_cycle_even_with_valid_neighbours(reason):
    source = rows(OPEN+[.03,.03]+OPEN)
    for row in source[5:7]:
        for side in ("left","right"):
            # Even a numerically valid contour cannot override contradictory fit QC.
            row[f"{side}_eye_visibility"] = reason
    observations, events = blink_qc(source,[old_event(8)])
    assert not events
    assert observations[5]["left"]["state"] == "eye_unobservable"
    assert reason in observations[5]["left"]["reasons"]


def test_profile_with_missing_landmarks_does_not_become_blink():
    source = rows([None]*50)
    for row in source:
        row["pose_Ry"] = 1.2
    observations, events = blink_qc(source,[old_event(45)])
    assert not events
    assert Counter(r["left"]["state"] for r in observations) == {"eye_unobservable":50}
    assert "extreme_profile" in observations[0]["right"]["reasons"]


def test_valid_continuous_bilateral_blink_keeps_event_fields_and_movie_pts():
    observations, [event] = blink_qc(rows(OPEN+[.03,.03]+OPEN),[old_event(8)])
    assert [observations[i]["left"]["state"] for i in [4,5,7]] == ["eye_open","eye_closed","eye_open"]
    assert (event["start_frame"],event["end_frame"],event["source_event_id"]) == (5,7,"prior")
    assert (event["start_s"],event["end_s"]) == (1500.20,1500.28)
    assert event["eye_support"] == ["left","right"]


def test_monocular_blink_does_not_infer_opposite_eye():
    left = OPEN+[.03,.03]+OPEN
    observations,[event] = blink_qc(rows(left,[None]*len(left)))
    assert event["eye_support"] == ["left"]
    assert observations[5]["right"]["state"] == "eye_unobservable"


def test_long_closure_is_per_frame_and_no_open_reference_is_unknown():
    observations,events = blink_qc(rows([.23]*30+[.033]*25+[.23]*30))
    assert not events
    assert observations[40]["left"]["state"] == "eye_closed"
    observations,events = blink_qc(rows([.033]*15))
    assert not events
    assert observations[0]["left"]["state"] == "eye_unknown"


def test_missing_pts_cannot_complete_a_blink():
    source = rows(OPEN+[.03,.03]+OPEN)
    for row in source[7:]:
        row["movie_timestamp_sec"] += .04
        row["source_next_timestamp_sec"] += .04
    assert not blink_qc(source)[1]


def test_head_and_independently_valid_gaze_survive_eye_failure():
    source = rows([None,None])
    for i,row in enumerate(source):
        row.update(movie_id="tt12300742",shot_id="shot_test",track_id="track_test",extraction_mode="static",
                   parquet_row=i,source_qc_reasons=[],gaze_valid=i==0,gaze_0_x=0.,gaze_0_y=0.,gaze_0_z=-1.,
                   gaze_1_x=0.,gaze_1_y=0.,gaze_1_z=-1.)
    previous = [{"head_pose_status":{"usable":True},"gaze_direction_status":{"usable":False},
                 "eye_position_status":{s:{"usable":False,"reason":"closure_candidate"} for s in ("left","right")}}]*2
    first,second = evidence_rows(source,previous,blink_qc(source)[0])
    assert first["head_facing_direction"] == second["head_facing_direction"] == [0.,0.,-1.]
    assert second["pose_Ry"] == 0.
    assert first["gaze_direction_status"]["usable"] and second["gaze_direction"] is None
    assert first["blink_status"]["left"] == "unobservable"
    assert second["eye_position_status"]["right"]["state"] == "unobservable"


def test_eye_motion_thresholds_and_output_protection(tmp_path):
    assert (DIRECTION_LIMIT, POSITION_LIMIT, MAX_TRANSITION_S, WEAK_X) == (3., .03, .55, .020)
    source,output = tmp_path/"prior",tmp_path/"fix"
    source.mkdir()
    output.mkdir()
    sentinel = output/"review.md"
    sentinel.write_text("past run")
    with pytest.raises(FileExistsError):
        run(source,output)
    assert sentinel.read_text() == "past run"
    with pytest.raises(ValueError,match="independent"):
        run(source,source/"fix")
