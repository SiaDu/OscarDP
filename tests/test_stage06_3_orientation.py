import numpy as np

from oscardp.stage06_3.aligned_review import interval_samples
from oscardp.stage06_3.orientation import evaluate, heading_range, nonperson_options, orientations, projection_compatible


def layout():
    return {"layout_id":"layout_001","people":[
        {"person":"MICHELLE","position":[0.,0.],"position_quality":"clear"},
        {"person":"TEDDY","position":[1.,0.],"position_quality":"clear"},
        {"person":"DON","position":[.3,-.3],"position_quality":"rough"}]}


def camera():
    return {"shot_id":"shot_000212","layout_id":"layout_001","camera_group":"camera_001",
            "camera_subject":"TEDDY","camera_position":None,"camera_region":{"cells":[[.5,1.],[.5,-1.]]},
            "camera_clues":{"screen_order":[],"depth_order":[]},"axis_side":"unknown","axis_pair":None,
            "axis_reference_shot":None,"layout_basis":"observed"}


def observation():
    return {"reason":"co-framed","objects":[],
            "constraints":[{"constraint_id":"seen","type":"screen_order","people":["MICHELLE","TEDDY"]}]}


def test_visible_left_right_filters_mirrored_hypotheses_and_preserves_multiple_headings():
    initial,filtered,evidence=orientations(camera(),layout(),observation())
    assert len(initial)==6 and len(filtered)==3
    assert all(h[1][1]>0 for h in filtered)
    assert evidence["applied_constraint_ids"]==["seen"]
    assert evidence["status"]=="projection_constrained"
    assert evidence["usable_direction_range"]["branches"]["candidate_region"]["start_deg"] != (
        evidence["usable_direction_range"]["branches"]["candidate_region"]["end_deg"])


def test_perspective_order_uses_depth_and_camera_translation():
    view=("region",np.array([0.,0.]),np.array([0.,1.]),np.array([1.,0.]))
    positions={"A":np.array([2.,1.]),"B":np.array([3.,3.])}
    assert projection_compatible(view,positions,[{"type":"screen_order","people":["B","A"]}])
    assert projection_compatible(view,positions,[{"type":"depth_order","people":["A","B"]}])


def test_group_alone_does_not_supply_camera_heading():
    shot={**camera(),"camera_region":None}
    a,b,evidence=orientations(shot,layout(),observation())
    assert not a and not b and not evidence["applied_constraint_ids"]
    assert evidence["usable_direction_range"]["absolute_heading"] is None


def test_circular_orientation_envelope_preserves_wrapping():
    assert heading_range([[1.,-.1],[1.,.1]])["wraps_zero"]


def interval(mode="gaze"):
    return {"interval_id":"i","shot_id":"s","track_id":"t","start_frame":10,"end_frame":14,
            "start_s":101., "end_s":101.4,"direction_source":mode,
            "evidence":{"gaze_direction":[0.,0.,-1.]}}


def frame(n,t,good=True):
    return {"shot_id":"s","track_id":"t","global_frame_index":n,"movie_timestamp_sec":t,
            "gaze_direction_status":{"usable":good}}


def test_review_frame_selection_uses_valid_internal_pts_and_track():
    candidates=[frame(9,100.9),frame(10,101.,False),frame(11,101.1),frame(12,101.2),
                {**frame(13,101.3),"track_id":"other"},frame(14,101.4)]
    selected=interval_samples(interval(),candidates)
    assert [r["global_frame_index"] for r in selected]==[11,12]
    assert selected[1]["roles"]==["midpoint","near_end"]
    assert all(r["within_interval"] and r["interval_id"]=="i" for r in selected)


def test_nonperson_requires_independent_evidence_and_no_head_only_fallback():
    record=interval()
    option={"interval_id":"i","target":"OTHER","object_or_region":"folder","source_frames":[10],
            "observation":"measured gaze and eye direction align with visible folder","direction_compatible":True}
    assert not nonperson_options(record,[option],None)
    option["visual_eye_direction_supported"]=True
    assert nonperson_options(record,[option],None)[0]["target"]=="OTHER"
    assert not nonperson_options(interval("head_only"),[option],None)
    away={**option,"target":"AWAY","visible_interaction_reference":True}
    assert not nonperson_options(record,[away],None)
    disjoint={"tested_targets":[{"camera_branches":{"region":{"compatible":0}}} for _ in range(2)]}
    assert nonperson_options(record,[away],disjoint)[0]["target"]=="AWAY"


def test_even_weak_viable_rival_prevents_unique_selection(monkeypatch):
    record = {**interval(),"character":"TEDDY","selected_target":"MICHELLE",
              "evidence_level":"gaze_supported","target_candidates":[{"target":"MICHELLE"}],
              "uncertainty_reason":[]}
    record["evidence"]["context_priors"]={"MICHELLE":["speaker"],"DON":["scene"]}
    def fit(target,count,error):
        return {"target":target,"camera_branches":{"candidate_region":{
            "compatible":count,"tested":30,"median_error_deg":error}}}
    geometry = {"tested_targets":[fit("MICHELLE",29,5.),fit("DON",1,44.)]}
    monkeypatch.setattr("oscardp.stage06_3.orientation.spatial_fit",lambda *a,**k:geometry)
    r = evaluate(record,camera(),layout(),[],{"status":"projection_constrained","applied_constraint_ids":["x"]},[])
    assert r["selected_target"]=="UNKNOWN"
    assert {c["target"] for c in r["target_candidates"]}=={"DON","MICHELLE","UNKNOWN"}
