"""Shot-local look states seeded by observable direction and supported motion."""

from collections import Counter

from oscardp.stage06_2.geometry import angular_change
from oscardp.stage06_2.stable import MAX_TRANSITION_S, MIN_STABLE_S, interval
from oscardp.stage06_2.temporal import SMALL_DEG, THRESHOLDS, valid_blocks
from .inference import direction_windows, mean_ray

SHIFT_DEG = THRESHOLDS["gaze"]["net_deg"]
EYE_EVENTS = {"eye_shift_candidate","weak_eye_motion_candidate","eye_adjustment_candidate"}
GAZE_EVENTS = {"gaze_direction_shift_candidate","continuous_gaze_transition"}


def overlaps(a, b):
    return a["start_s"] < b["end_s"] and b["start_s"] < a["end_s"]


def anchor_interval(rows, mode, idx, ray):
    return {**interval(rows,idx),"mode":mode,"direction":ray,"frame_count":len(idx)}


def find_anchor(anchors, event, before):
    boundary = event["start_s"] if before else event["end_s"]
    eligible = []
    for a in anchors:
        # An anchor that straddles the event must include enough real samples
        # on the requested side. A stable run entirely inside an event is not a flank.
        support = (min(a["end_s"],boundary)-a["start_s"] if before else
                   a["end_s"]-max(a["start_s"],boundary))
        gap = (max(0.,boundary-a["end_s"]) if before else max(0.,a["start_s"]-boundary))
        if support >= MIN_STABLE_S-1e-6 and gap <= MAX_TRANSITION_S+1e-6:
            eligible.append((gap,a))
    return min(eligible,key=lambda x:(x[0],x[1]["mode"]!="gaze",
                                      -x[1]["end_s"] if before else x[1]["start_s"]))[1] if eligible else None


def motion_events(rows, anchors, events, adjunct):
    blink = [e for e in events if e["type"]=="blink_candidate"]
    candidates = [dict(e) for e in events if e["type"] in EYE_EVENTS|GAZE_EVENTS|{"blink_candidate"}]
    candidates.extend({**e,"type":("eye_head_motion" if {"eye","head"}<=set(e["trigger_signals"]) else
                                   "eye_motion" if "eye" in e["trigger_signals"] else "head_or_gaze_motion")}
                      for e in adjunct if set(e["trigger_signals"]) & {"head","gaze","eye"})
    by_frame = {r["global_frame_index"]:i for i,r in enumerate(rows)}
    gaze = [a for a in anchors if a["mode"]=="gaze"]
    for a,b in zip(gaze,gaze[1:]):
        i,j = by_frame[a["end_frame"]-1],by_frame[b["start_frame"]]
        if (0<j-i and b["start_s"]-a["end_s"] <= MAX_TRANSITION_S+1e-6
                and angular_change(a["direction"],b["direction"]) >= SHIFT_DEG
                and all(rows[k]["gaze_direction"] is not None for k in range(i,j+1))
                and all(abs(rows[k]["source_next_timestamp_sec"]-rows[k+1]["movie_timestamp_sec"])<=1e-6
                        and rows[k+1]["global_frame_index"]==rows[k]["global_frame_index"]+1 for k in range(i,j))):
            candidates.append({"type":"continuous_gaze_transition",**interval(rows,[i,j])})

    records = []
    for e in sorted(candidates,key=lambda e:(e["start_s"],e["end_s"],e["type"])):
        a,b = find_anchor(anchors,e,True),find_anchor(anchors,e,False)
        delta = angular_change(a["direction"],b["direction"]) if a and b and a["mode"]==b["mode"] else None
        inside = [r for r in rows if overlaps(e,{"start_s":r["movie_timestamp_sec"],
                                                 "end_s":r["source_next_timestamp_sec"]})]
        weak = e["type"]=="weak_eye_motion_candidate"
        eyes = e.get("eye_support",[])
        eye_frames = [r for r in inside if any(r["eye_position_status"][s]["usable"]
                       and not r["eye_position_status"][s].get("landmark_jump") for s in eyes)]
        eye_blocks = list(valid_blocks(inside,[r in eye_frames for r in inside])) if weak else []
        reliable_weak = not weak or any(len(v)>=3 and
            inside[v[-1]]["source_next_timestamp_sec"]-inside[v[0]]["movie_timestamp_sec"]>=.16 for v in eye_blocks)
        qc_fail = {"high_median_filter_residual","binocular_motion_disagreement","binocular_motion_unusable"} & set(e.get("qc_flags",[]))
        peak = None
        if a and a["mode"]=="gaze":
            peak = max((angular_change(a["direction"],r["gaze_direction"]) for r in inside
                        if r["gaze_direction"] is not None),default=None)
        if e["type"]=="blink_candidate" or any(overlaps(e,v) for v in blink):
            assessment = "blink_or_blink_overlap"
        elif not reliable_weak or qc_fail:
            assessment = "unreliable_motion"
        elif delta is None:
            assessment = "missing_comparable_stable_flanks"
        elif delta<=SMALL_DEG:
            assessment = "return" if peak is not None and peak>=SHIFT_DEG else "adjustment_or_same_direction"
        elif delta>=SHIFT_DEG:
            assessment = "sustained_gaze_change" if a["mode"]=="gaze" else "sustained_head_change"
            if e["type"]=="eye_adjustment_candidate":
                assessment = "adjustment_with_direction_difference_pending_confirmation"
            elif weak and not any(r["gaze_direction"] is not None for r in inside):
                assessment = "uncertain_weak_motion_without_gaze_support"
        else:
            assessment = "direction_change_below_sustained_threshold"
        number = len(records)+1
        transition = max(e["end_frame"],b["start_frame"]) if assessment.startswith("sustained_") else None
        if transition is not None and transition>=b["end_frame"]:
            transition=None
            assessment="insufficient_stable_post_event_coverage"
        records.append({**{k:e[k] for k in ("start_frame","end_frame","start_s","end_s")},
                        "event_id":f"{rows[0]['track_id']}_direction_{number:03d}","source_event":e,
                        "assessment":assessment,"pre_look":a,"post_look":b,
                        "direction_change_deg":delta,"peak_gaze_departure_deg":peak,
                        "post_boundary_frame":transition})
    return records,blink


def make_states(rows, stable_gaze, eye_states, events, adjunct, identity):
    windows = direction_windows(rows,stable_gaze)
    anchors = [anchor_interval(rows,m,i,g) for m,i,g in windows if m!="unresolved"]
    motion, blink = motion_events(rows,anchors,events,adjunct)
    boundaries = {}
    for e in motion:
        n = e["post_boundary_frame"]
        if n is not None:
            # Several model events can describe the same landing. Use the earliest
            # evidenced landing in that same stable post-state.
            key = e["post_look"]["start_frame"]
            if key not in boundaries or n<boundaries[key]["post_boundary_frame"]:
                boundaries[key] = e
    split_at = {e["post_boundary_frame"]:e for e in boundaries.values()}
    available = [(r["head_facing_direction"] is not None or r["gaze_direction"] is not None)
                 and not any(e["start_frame"]<=r["global_frame_index"]<e["end_frame"] for e in blink)
                 for r in rows]
    pieces = []
    for mask in (available,[not ok for ok in available]):
        for block in valid_blocks(rows,mask):
            active = available[block[0]]
            cuts = [j for j,i in enumerate(block) if j and active and rows[i]["global_frame_index"] in split_at]
            limits = [0,*cuts,len(block)]
            for a,b in zip(limits,limits[1:]):
                pieces.append((block[a:b],active))
    pieces.sort(key=lambda x:x[0][0])

    result = []
    seen_direction = False
    for idx,active in pieces:
        span = interval(rows,idx)
        local = [rows[i] for i in idx]
        eligible = [a for a in anchors if active and
                    min(a["end_s"],span["end_s"])-max(a["start_s"],span["start_s"])>=MIN_STABLE_S-1e-6]
        gaze = [a for a in eligible if a["mode"]=="gaze"]
        head = [a for a in eligible if a["mode"]=="head"]
        # The head is a fallback for spans whose eyes supply no usable camera gaze.
        pool = gaze or (head if not any(r["gaze_direction"] is not None for r in local) else [])
        def coverage(a):
            return min(a["end_s"],span["end_s"])-max(a["start_s"],span["start_s"])
        anchor = max(pool,key=coverage,default=None)
        status = "stable"
        if anchor and any(angular_change(a["direction"],anchor["direction"])>=SHIFT_DEG for a in pool):
            # Uncorroborated drifting direction is disclosed and cannot supply
            # a unique target for the entire state.
            status = "heterogeneous_stable_support"
        if not anchor:
            status = ("no_sustained_direction" if active else
                      "blink_gap" if any(overlaps(span,e) for e in blink) else "direction_unavailable")
        support = [r for r in local if anchor and anchor["start_frame"]<=r["global_frame_index"]<anchor["end_frame"]]
        conjunct = sum(r["head_facing_direction"] is not None and r["gaze_direction"] is not None
                       and any(e["usable"] for e in r["eye_position_status"].values()) for r in support)
        source = ("unobservable" if not anchor else "head_only" if anchor["mode"]=="head" else
                  "eye_head" if conjunct>=3 and conjunct*2>=len(support) else "gaze_only")
        key = "head_facing_direction" if source=="head_only" else "gaze_direction"
        representative = mean_ray([r[key] for r in support if r[key] is not None]) if anchor else None
        trigger = split_at.get(span["start_frame"]) if active else None
        origin = ("unobservable" if not anchor else "significant_movement" if trigger else
                  "initial" if not seen_direction else "reacquired_after_gap")
        if anchor:
            seen_direction=True
        number = len(result)+1
        state = {**span,"state_id":f"{identity['track_id']}_look_{number:03d}",
                 **{k:identity[k] for k in ("shot_id","track_id","character")},
                 "state_origin":origin,"direction_source":source,
                 "look_direction":{"vector":representative,"coordinates":"same_shot_OpenFace_camera_xyz",
                                   "meaning":"head_facing" if source=="head_only" else "camera_relative_gaze"}
                                  if anchor else None,
                 "trigger_event":trigger["event_id"] if trigger else None,"qc_status":status,
                 "eye_motion_evidence":{"stable_eye_positions":[v for v in eye_states if overlaps(v,span)],
                                        "motion_event_ids":[e["event_id"] for e in motion
                                            if e["source_event"]["type"] in EYE_EVENTS and overlaps(e,span)]},
                 "head_motion_evidence":[e["event_id"] for e in motion
                    if "head" in e["source_event"].get("trigger_signals",[]) and overlaps(e,span)],
                 "gaze_evidence":{"anchor":anchor,"stable_anchor_count":len(pool),
                                  "supported_frames":len(support),"eye_head_corrob_frames":conjunct,
                                  "usable_gaze_frames":sum(r["gaze_direction"] is not None for r in local),
                                  "usable_head_frames":sum(r["head_facing_direction"] is not None for r in local),
                                  "frame_count":len(local)}}
        previous = result[-1] if result else None
        if (previous and source==previous["direction_source"]=="unobservable"
                and "blink_gap" not in (status,previous["qc_status"])
                and previous["end_frame"]==state["start_frame"]
                and abs(previous["end_s"]-state["start_s"])<=1e-6):
            previous["qc_subspans"].append({**span,"qc_status":status})
            if previous["qc_status"]!=status:
                previous["qc_status"]="no_continuous_stable_direction"
            previous.update(end_frame=state["end_frame"],end_s=state["end_s"])
            for key in ("frame_count","usable_gaze_frames","usable_head_frames"):
                previous["gaze_evidence"][key]+=state["gaze_evidence"][key]
            previous["head_motion_evidence"]=list(dict.fromkeys(
                previous["head_motion_evidence"]+state["head_motion_evidence"]))
            for key,values in state["eye_motion_evidence"].items():
                previous["eye_motion_evidence"][key].extend(v for v in values if v not in previous["eye_motion_evidence"][key])
        else:
            state["qc_subspans"]=[] if anchor else [{**span,"qc_status":status}]
            result.append(state)
    for e in motion:
        pre,post = e["pre_look"],e["post_look"]
        def linked(a, before):
            if a is None:
                return None
            n = min(a["end_frame"]-1,e["start_frame"]-1) if before else max(a["start_frame"],e["end_frame"])
            return next((s["state_id"] for s in result if s["start_frame"]<=n<s["end_frame"]),None)
        pre_id,post_id=linked(pre,True),linked(post,False)
        e.update(shot_id=identity["shot_id"],track_id=identity["track_id"],
                 pre_state_id=pre_id,post_state_id=post_id,
                 split_supported_states=bool(pre_id and post_id and pre_id!=post_id and
                     any(s["trigger_event"]==e["event_id"] for s in result)))
    return result,motion
