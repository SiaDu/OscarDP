import pytest

from oscardp.stage06_2.evidence_replay import assess, run, select_frames, weak_motion
from oscardp.stage06_2.stable import analyze


def frames():
    return [{"global_frame_index":n+40,"movie_timestamp_sec":1432.013+n*.04,
             "source_next_timestamp_sec":1432.013+(n+1)*.04,
             "left_eye_x":.4,"right_eye_x":.45,"left_eye_y":0.,"right_eye_y":0.,
             "left_eye_visibility":"usable","right_eye_visibility":"usable",
             "left_eye_landmark_jump":False,"right_eye_landmark_jump":False,
             "tracking_usable":True,"head_pose_valid":True,"gaze_valid":True,
             "pose_Rx":0.,"pose_Ry":0.,"pose_Rz":0.,
             **{f"gaze_{eye}_{axis}":(-1. if axis=="z" else 0.)
                for eye in range(2) for axis in "xyz"}} for n in range(15)]


def states(rows):
    return analyze(rows,[],[],allow_monocular=True,independent_gaze=True)[0]


def test_gaze_valid_with_missing_eye_position_and_pose():
    raw=frames()
    for r in raw:
        r.update(left_eye_visibility="low_visibility",right_eye_visibility="low_visibility",head_pose_valid=False)
    rows=assess(raw,[])
    assert all(r["left_eye_x"] is None and r["camera_gaze_usable"] and not r["head_usable"] for r in rows)
    [state]=states(rows)
    assert state["type"]=="stable_gaze_candidate" and state["head_direction"] is None


def test_eye_position_and_head_valid_with_unusable_gaze():
    raw=frames()
    for r in raw:
        r["gaze_valid"]=False
    rows=assess(raw,[])
    assert all(r["head_pose_status"]["usable"] and not r["gaze_direction_status"]["usable"]
               and r["eye_position_status"]["left"]["usable"] for r in rows)
    assert [s["type"] for s in states(rows)]==["eye_position_stable"]


@pytest.mark.parametrize(("case","expected"),[("one_eye",True),("single_spike",False),
                                              ("opposite",False),("closure",False),("failure",False)])
def test_weak_motion_is_sustained_and_cannot_mask_conflicting_evidence(case,expected):
    raw=frames()
    for i,r in enumerate(raw):
        if i>=7:
            r["left_eye_x"]+=.04
        if case=="one_eye":
            r["right_eye_visibility"]="low_visibility"
        elif case=="single_spike":
            r["left_eye_x"]=.5 if i==7 else .4
            r["left_eye_landmark_jump"]=i==7
        elif case=="opposite" and i>=7:
            r["right_eye_x"]-=.04
        elif case=="failure" and i==7:
            r["tracking_usable"]=False
    closures=([{"event_type":"eye_closure_candidate","event_id":"blink","start_frame":47,"end_frame":48}]
              if case=="closure" else [])
    rows=assess(raw,closures)
    events=weak_motion(rows,[])
    assert bool(events)==expected
    if expected:
        assert events[0]["eye_support"]==["left"]
        assert events[0]["other_eye_unavailable_statuses"]==["low_visibility"]
        assert events[0]["gaze_direction_evidence"]=="gaze_direction_unknown"
    if case=="opposite":
        _,motion,_=analyze(rows,[],[],allow_monocular=True,independent_gaze=True)
        assert not any(e["type"]=="eye_shift_candidate" for e in motion)


def test_periodic_pts_fallback_missing_bins_and_reuse():
    raw=frames()
    # Irregular source timestamps, a completely missing second bin and a
    # selected head-only frame in the third bin.
    times=[1432.013,1432.151,1432.289,1432.54,1432.9,1433.05,1433.22]
    raw=raw[:len(times)]
    for i,r in enumerate(raw):
        r.update(movie_timestamp_sec=times[i],source_next_timestamp_sec=times[i+1] if i+1<len(times) else 1433.5)
        if i>=3:
            r.update(left_eye_visibility="low_visibility",right_eye_visibility="low_visibility",gaze_valid=False)
        if i in (3,4):
            r["tracking_usable"]=False
    rows=assess(raw,[])
    existing=[{"global_frame_index":46,"reasons":["after_eye_motion"]}]
    selected,coverage=select_frames(rows,existing,[])
    assert [r["status"] for r in coverage]==["covered","missing","covered"]
    assert [r["movie_timestamp_sec"] for r in selected]==[1432.289,1433.22]
    assert selected[-1]["available_modalities"]==["head_pose"]
    assert {"motion_event","periodic_check"}<=set(selected[-1]["reasons"])


def test_refuses_existing_or_nested_pilot_output(tmp_path):
    source=tmp_path/"v13"
    source.mkdir()
    f=source/"review.md"
    f.write_text("existing")
    out=tmp_path/"v14"
    out.mkdir()
    with pytest.raises(FileExistsError):
        run(source,out)
    with pytest.raises(ValueError):
        run(source,source/"nested")
    assert f.read_text()=="existing"
