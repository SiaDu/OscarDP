from math import cos, radians, sin

from oscardp.stage06_3.look_review import choose_frame
from oscardp.stage06_3.look_states import make_states, motion_events
from oscardp.stage06_3.look_targets import event_targets, fuse

IDENTITY={"shot_id":"shot_a","track_id":"shot_a_track_001","character":"MICHELLE"}
FORWARD=[0.,0.,-1.]


def ray(deg):
    return [sin(radians(deg)),0.,-cos(radians(deg))]


def rows(directions, visible=True):
    return [{**IDENTITY,"global_frame_index":n,"movie_timestamp_sec":100.+n*.05,
             "source_next_timestamp_sec":100.+(n+1)*.05,"gaze_direction":v,
             "head_facing_direction":FORWARD,"eye_position_status":{
                 s:{"usable":visible,"landmark_jump":False} for s in ("left","right")}}
            for n,v in enumerate(directions)]


def stable(a,b,v):
    return {"type":"stable_gaze_candidate","start_frame":a,"end_frame":b,
            "start_s":100.+a*.05,"end_s":100.+b*.05,"camera_direction":v}


def test_initial_state_survives_without_any_motion_events():
    [s],e=make_states(rows([FORWARD]*30),[stable(0,30,FORWARD)],[],[],[],IDENTITY)
    assert s["state_origin"]=="initial" and s["direction_source"]=="eye_head"
    assert s["start_s"]==100. and s["end_s"]==101.5 and not e
    [s],_=make_states(rows([FORWARD]*30,False),[stable(0,30,FORWARD)],[],[],[],IDENTITY)
    assert s["direction_source"]=="gaze_only"
    [s],_=make_states(rows([None]*30,False),[],[],[],[],IDENTITY)
    assert s["look_direction"]["meaning"]=="head_facing" and s["direction_source"]=="head_only"
    assert s["gaze_evidence"]["usable_gaze_frames"]==0


def test_sustained_camera_change_can_open_a_new_state_without_fixed_spacing():
    a,b=FORWARD,ray(15.)
    s,e=make_states(rows([a]*14+[b]*16),[stable(0,14,a),stable(14,30,b)],[],[],[],IDENTITY)
    assert len(s)==2 and s[1]["start_frame"]==15 and s[1]["state_origin"]=="significant_movement"
    [direction]=e
    assert direction["assessment"]=="sustained_gaze_change"
    assert direction["pre_state_id"]==s[0]["state_id"] and direction["post_state_id"]==s[1]["state_id"]
    assert (direction["shot_id"],direction["track_id"])==(IDENTITY["shot_id"],IDENTITY["track_id"])


def test_return_blink_and_landmark_jump_do_not_trigger_sustained_change():
    r=rows([FORWARD]*12+[ray(15.)]*4+[FORWARD]*14)
    anchors=[{**stable(0,12,FORWARD),"mode":"gaze","direction":FORWARD},
             {**stable(16,30,FORWARD),"mode":"gaze","direction":FORWARD}]
    event={**stable(12,16,FORWARD),"type":"weak_eye_motion_candidate","eye_support":["left"]}
    [e],_=motion_events(r,anchors,[event],[])
    assert e["assessment"]=="return" and e["post_boundary_frame"] is None
    for row in r[12:16]:
        row["eye_position_status"]["left"]["landmark_jump"]=True
    [e],_=motion_events(r,anchors,[event],[])
    assert e["assessment"]=="unreliable_motion"
    [e],_=motion_events(r,anchors,[{**event,"type":"blink_candidate"}],[])
    assert e["assessment"]=="blink_or_blink_overlap"


def test_same_state_motion_can_hold_but_unresolved_cannot_switch():
    with_target=[{"state_id":"a","selected_target":"TEDDY"},{"state_id":"b","selected_target":"UNKNOWN"}]
    e={**IDENTITY,"event_id":"e","assessment":"return","split_supported_states":False,
       "pre_state_id":"a","post_state_id":"a","start_s":100.2,"end_s":100.4}
    assert event_targets([e],with_target)[0]["type"]=="target_hold_candidate"
    e.update(assessment="sustained_gaze_change",split_supported_states=True,post_state_id="b")
    assert event_targets([e],with_target)[0]["type"]=="target_uncertain"


def test_visual_preference_is_used_but_does_not_remove_competing_spatial_target():
    s={"direction_source":"eye_head","qc_status":"stable"}
    spatial={"geometry":{"tested_targets":[]},"reasons":[],"selected_target":"UNKNOWN",
             "target_candidates":[{"target":"TEDDY","support":"gaze_supported"},
                                  {"target":"DON","support":"gaze_supported"}]}
    visual={"target_observations":[{"target":"TEDDY","support":"visually_preferred"}]}
    r=fuse(s,spatial,visual)
    assert r["selected_target"]=="UNKNOWN"
    assert [x["target"] for x in r["possible_targets"]]==["TEDDY","DON","UNKNOWN"]
    assert "visually_preferred" in r["possible_targets"][0]["support"]


def test_review_sampling_keeps_track_box_and_stays_within_frame_window():
    rr=rows([FORWARD]*5)
    box={("shot_a",1):[{"track_id":"other"}],("shot_a",2):[{"track_id":IDENTITY["track_id"]}],
         ("shot_a",4):[{"track_id":IDENTITY["track_id"]}]}
    r=choose_frame(rr,1,4,box,near=1)
    assert r["global_frame_index"]==2
    assert r["movie_timestamp_sec"]==100.1
