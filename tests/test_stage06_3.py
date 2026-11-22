from copy import deepcopy

from oscardp.stage06_2.inputs import EYES
from oscardp.stage06_3.__main__ import coverage_gaps, link_identity_evidence, map_track
from oscardp.stage06_3.inference import direction_windows, gaze_states, match, timeline_transitions


def layout(don):
    return {"layout_id":"layout_001","people":[
        {"person":"TEDDY","position":[1.,0.],"position_quality":"clear"},
        {"person":"MICHELLE","position":[0.,0.],"position_quality":"clear"},
        {"person":"DON","position":don,"position_quality":"rough"}]}


def shot():
    return {"camera_subject":"TEDDY","camera_group":"similar",
            "camera_position":[-.2,0.],"camera_region":{"cells":[[-.25,.1],[-.25,-.1]]},
            "layout_basis":"observed","camera_clues":{"screen_order":[],"depth_order":[]}}


def test_targets_depend_on_camera_sign_and_never_include_self():
    options, selected, level, _, stats = match("TEDDY","gaze",[0.,0.,-1.],shot(),layout([1.,1.]),{})
    assert selected=="MICHELLE" and level=="gaze_supported"
    assert [c["target"] for c in options]==["MICHELLE"]
    assert set(stats["tested_targets"][0]["camera_branches"])=={"candidate_region","inherited_point"}


def test_camera_group_alone_only_provides_context():
    camera = {**shot(),"camera_position":None,"camera_region":None}
    options, selected, level, _, geometry = match(
        "TEDDY","gaze",[0.,0.,-1.],camera,layout([1.,1.]),{"MICHELLE":["speaker"],"DON":["scene"]})
    assert selected=="UNKNOWN" and level=="context_only" and geometry is None
    assert [c["target"] for c in options]==["MICHELLE","DON","UNKNOWN"]


def test_head_only_keeps_multiple_candidates_and_no_named_selection():
    options, selected, level, reasons,_ = match("TEDDY","head",[0.,0.,-1.],shot(),layout([.1,.3]),{})
    assert selected=="UNKNOWN" and level=="head_supported"
    assert {c["target"] for c in options}=={"MICHELLE","DON","UNKNOWN"}
    assert "head_only_eye_direction_unavailable" in reasons


def intervals():
    out = []
    for n in range(17):
        direction = None if n==8 else [0.,0.,-1.]
        out.append({"global_frame_index":n,"movie_timestamp_sec":1500+n*.04,
                    "source_next_timestamp_sec":1500+(n+1)*.04,
                    "gaze_direction":direction,"head_facing_direction":[1.,0.,0.],
                    **{k:v for eye in EYES for k,v in zip(eye,[0.,0.,-1.],strict=True)}})
    return out


def test_stability_keeps_original_frame_time_and_does_not_bridge_invalid_gaze():
    rows = intervals()
    states = gaze_states(rows)
    assert [(s["start_frame"],s["end_frame"]) for s in states]==[(0,8),(9,17)]
    windows = direction_windows(rows,states)
    assert [w[0] for w in windows]==["gaze","unresolved","gaze"]
    for row in rows[7:15]:
        row["gaze_direction"] = None
    assert "head" in [w[0] for w in direction_windows(rows,gaze_states(rows))]


def record(sid,t,selected,level="gaze_supported"):
    return {"interval_id":f"{sid}_{t}","shot_id":f"shot_{sid:06d}","shot_index":sid,
            "track_id":f"track_{sid}","character":"TEDDY","start_s":t,"end_s":t+1.,
            "shot_start_s":t,"shot_end_s":t+1.,"selected_target":selected,
            "evidence_level":level,"layout_id":"layout_001","mapping_strength":"supported",
            "evidence":{"plan_usable":True,"measured_frames":24}}


def test_cut_does_not_create_transition_without_adjacent_supported_intervals():
    a,b = record(212,100.,"MICHELLE"),record(213,101.,"DON")
    assert timeline_transitions([a,b])[0]["type"]=="target_switch_candidate"
    c = {**b,"evidence_level":"context_only","selected_target":"UNKNOWN"}
    assert not timeline_transitions([a,c,record(214,102.,"DON")])
    assert not timeline_transitions([a,{**b,"mapping_strength":"limited"}])
    assert not timeline_transitions([a,{**b,"start_s":101.4}])
    assert not timeline_transitions([a,{**b,"shot_index":214}])
    assert not timeline_transitions([a,{**b,"character":"MICHELLE"}])


def test_coverage_gaps_are_explicit_and_ignored_by_target_matching():
    a,b = record(212,100.,"MICHELLE"),record(214,103.,"MICHELLE")
    solo = deepcopy(b)
    solo["evidence"]["measured_frames"] = 0
    assert [(g["start_s"],g["end_s"],g["selected_target"])
            for g in coverage_gaps([a,solo],99.,105.)]==[(99.,100.,"UNKNOWN"),(101.,105.,"UNKNOWN")]
    assert not timeline_transitions([a,b])


def test_boundary_residue_retains_visual_candidate_but_does_not_map_it_into_shot():
    shot = {"shot_id":"shot_000218","visible_character_candidates":[{"character_name":"TEDDY"}],
            "stage05":{"identity":"batch_1"}}
    track = {"track_id":"shot_000218_track_001","status":"unavailable","identity_cluster_id":"identity_002"}
    review = {"shot_id":shot["shot_id"],"track_id":track["track_id"],
              "character_candidate":"MICHELLE","review_status":"boundary_residue"}
    r = map_track(shot,track,review,{"samples":[],"subjects":[]})
    assert r["character"]=="UNKNOWN" and r["character_candidates"]==["MICHELLE"]
    assert not r["human_verified"]


def test_identity_cluster_ids_are_local_to_each_batch():
    def entry(track, batch):
        return {"track_id":track,"character":"MICHELLE","sources":{"stage05_identity":{
            "status":"tentative_identity_cluster","cluster_id":"identity_002","namespace":batch}}}
    a,b,c = entry("a","batch_1"),entry("b","batch_1"),entry("c","batch_2")
    link_identity_evidence([a,b,c])
    assert a["sources"]["stage05_identity"]["consistent_same_batch_tracks"]==["b"]
    assert c["sources"]["stage05_identity"]["consistent_same_batch_tracks"]==[]
