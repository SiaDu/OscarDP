"""Relative orientation constraints from reviewed frames and Stage06.1 hypotheses."""

from collections import Counter
from math import atan2, degrees

import numpy as np

from .inference import CHARACTERS, MIN_GROUP_FRACTION, camera_hypotheses, spatial_fit

FOCUS_SHOTS = tuple(f"shot_{i:06d}" for i in (212,217,219,221,222,223))
UNKNOWN = {"target":"UNKNOWN","support":"unresolved"}


def projection_compatible(hypothesis, positions, relations):
    _,camera,forward,right = hypothesis
    projected = {}
    for relation in relations:
        a,b = relation["people"]
        for name in (a,b):
            if name not in projected:
                ray = positions[name]-camera
                depth = float(ray@forward)
                if depth <= 1e-9:
                    return False
                projected[name] = (float(ray@right)/depth,depth)
        axis = 1 if relation["type"]=="depth_order" else 0
        if projected[b][axis] <= projected[a][axis]+1e-9:
            return False
    return True


def heading_range(vectors):
    """Smallest circular envelope, in degrees of the schematic plane only."""
    angles = sorted(degrees(atan2(v[1],v[0])) % 360 for v in vectors)
    cyclic = angles[1:]+[angles[0]+360]
    split = max(range(len(angles)),key=lambda i:cyclic[i]-angles[i])
    return {"start_deg":round(cyclic[split]%360,2),"end_deg":round(angles[split],2),
            "wraps_zero":cyclic[split]%360 > angles[split]}


def orientations(shot, layout, observation):
    positions = {p["person"]:np.array(p["position"],dtype=float) for p in layout["people"]
                 if p["position"] is not None}
    initial = camera_hypotheses(shot,positions)
    observed = [r for r in observation["constraints"] if r["type"] in {
                "screen_order","depth_order","bed_axis_screen_proxy"}]
    usable = [r for r in observed if set(r["people"])<=positions.keys()]
    filtered = [h for h in initial if projection_compatible(h,positions,usable)]
    before,after = (dict(Counter(h[0] for h in hs)) for hs in (initial,filtered))
    range_by_family = {family:heading_range([h[2] for h in filtered if h[0]==family]) for family in after}
    used = [r["constraint_id"] for r in usable] if initial else []
    state = ("projection_constrained" if filtered and usable else "inherited_only" if filtered else
             "projection_conflict" if initial else "relative_only")
    return initial, filtered, {"shot_id":shot["shot_id"],"status":state,"layout_id":layout["layout_id"],
                      "camera_group":shot["camera_group"],"origin":"reviewed_frames_and_stage06_1",
                      "constraints":observation["constraints"],"visible_nonperson_objects":observation["objects"],
                      "applied_constraint_ids":used,"hypotheses_before":before,"hypotheses_after":after,
                      "usable_direction_range":{"type":"schematic_camera_forward_envelopes",
                           "axes":"+x toward +y in Stage06.1 plan; conceptual, not physical yaw",
                           "branches":range_by_family} if after else
                           {"type":state,"absolute_heading":None,"group_rotation":None},
                      "inherited":{"camera_position":shot["camera_position"],"camera_region":shot["camera_region"],
                                   "axis_side":shot["axis_side"],"axis_pair":shot["axis_pair"],
                                   "axis_reference_shot":shot["axis_reference_shot"],
                                   "layout_basis":shot["layout_basis"]},
                      "reason":observation["reason"],"human_verified":False}


def nonperson_options(record, observations, geometry):
    """A mismatch alone never proposes AWAY or OTHER."""
    unavailable = record["direction_source"]!="gaze" or not record["evidence"].get("gaze_direction")
    if unavailable:
        return []
    options = []
    for item in observations:
        if (item["interval_id"]!=record["interval_id"] or not item.get("visual_eye_direction_supported")
                or not item.get("source_frames") or not item.get("observation")):
            continue
        if item["target"]=="AWAY":
            fits = (geometry or {}).get("tested_targets",[])
            if len(fits)!=len(CHARACTERS)-1 or any(
                    any(v["compatible"] for v in f["camera_branches"].values()) for f in fits):
                continue
            if not item.get("visible_interaction_reference"):
                continue
        elif item["target"]=="OTHER":
            if not item.get("object_or_region") or not item.get("direction_compatible"):
                continue
        else:
            continue
        options.append({"target":item["target"],"support":"interval_visual_direction_and_gaze","visual_evidence":item})
    return options


def evaluate(record, camera, layout, hypotheses, orientation, nonperson):
    evidence = record["evidence"]
    mode = record["direction_source"]
    actor = record["character"]
    ray = evidence.get("gaze_direction") if mode=="gaze" else evidence.get("head_facing_direction")
    geometry = (spatial_fit(actor,ray,camera,layout,"gaze" if mode=="gaze" else "head",
                            hypotheses=hypotheses,screen_veto=False)
                if actor in CHARACTERS and ray is not None else None)
    options = [c for c in record["target_candidates"] if c["target"] in CHARACTERS]
    selected = record["selected_target"]
    level = record["evidence_level"]
    reasons = list(record["uncertainty_reason"])
    if geometry is not None:
        fits = geometry["tested_targets"]
        possible = [f for f in fits if any(v["compatible"] for v in f["camera_branches"].values())]
        level = "gaze_supported" if mode=="gaze" else "head_supported"
        options = [{"target":f["target"],"support":level,"geometry":f,
                    "prior_reasons":evidence["context_priors"].get(f["target"],[])} for f in possible]
        selected = "UNKNOWN"
        reasons = ["rough_2d_layout","camera_tilt_height_uncalibrated"]
        if len(possible)==1 and mode=="gaze":
            winner = possible[0]
            branches = winner["camera_branches"]
            rivals = [f for f in fits if f["target"]!=winner["target"]]
            margin = min((f["camera_branches"][b]["median_error_deg"]-v["median_error_deg"]
                          for f in rivals for b,v in branches.items()),default=0.)
            if (min(v["compatible"]/v["tested"] for v in branches.values())>=MIN_GROUP_FRACTION
                    and margin>=15. and camera["layout_basis"] in {"assumed_static","observed"}):
                selected=winner["target"]
            else:
                reasons.append("single_candidate_but_camera_branch_support_or_margin_insufficient")
        if len(possible)>1:
            reasons.append("multiple_targets_supported_across_admissible_hypotheses")
        elif not possible:
            reasons.append("no_modeled_person_aligned_nonperson_target_unresolved")
        if mode=="head_only":
            reasons.append("head_only_eye_direction_unavailable")
        if camera["layout_basis"]=="assumed_static":
            reasons.append("uses_static_layout_working_assumption")
    elif orientation["status"]=="projection_conflict" and ray is not None:
        level,selected = "context_only","UNKNOWN"
        options = [{"target":name,"support":"context_prior","prior_reasons":why}
                   for name,why in evidence["context_priors"].items() if name!=actor]
        reasons = ["visual_projection_conflicts_with_inherited_camera_hypotheses"]
    elif orientation["status"]=="relative_only" and ray is not None:
        reasons.append("repeated_view_relative_only_no_plan_heading")

    extra = nonperson_options(record,nonperson,geometry)
    if extra:
        selected = "UNKNOWN"
        options += extra
        reasons.append("independently_reviewed_nonperson_alternative")
    if selected=="UNKNOWN":
        options.append(UNKNOWN.copy())
    return {**record,"target_candidates":options,"selected_target":selected,"evidence_level":level,
            "uncertainty_reason":reasons,"evidence":{**evidence,"scene_geometry":geometry,
                "plan_usable":geometry is not None,"relative_orientation_status":orientation["status"],
                "applied_orientation_ids":orientation["applied_constraint_ids"] if geometry else []}}
