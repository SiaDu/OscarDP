"""Conservative, within-camera direction matching and cross-shot continuity."""

from math import atan2, cos, degrees, radians, sin

import numpy as np

from oscardp.stage06_2.geometry import angular_change, unit
from oscardp.stage06_2.inputs import EYES
from oscardp.stage06_2.stable import DIRECTION_LIMIT, direction_distance, interval, stable_runs
from oscardp.stage06_2.temporal import valid_blocks

CHARACTERS = ("MICHELLE", "TEDDY", "DON")
TARGETS = (*CHARACTERS, "AWAY", "OTHER", "UNKNOWN")
AIM_OFFSETS_DEG = (-12., 0., 12.)
PLAN_PERTURBATION = .05
GAZE_TOLERANCE_DEG = 32.
HEAD_TOLERANCE_DEG = 46.
HEAD_STABILITY_DEG = 12.
MIN_GROUP_FRACTION = .67
MAX_CONTINUITY_GAP_S = .25


def mean_ray(rays):
    return unit(np.mean(rays,axis=0)).tolist()


def gaze_states(rows):
    """Stage06.2 independent Gaze stability on the updated mask, with its original thresholds."""
    states = []
    for block in valid_blocks(rows,[r["gaze_direction"] is not None for r in rows]):
        rays = np.array([[rows[i]["gaze_direction"], *[unit([rows[i][k] for k in eye]) for eye in EYES]]
                         for i in block])
        for span, stable in stable_runs(rows,block,rays,DIRECTION_LIMIT,direction_distance):
            states.append({"type":"stable_gaze_candidate", **interval(rows,span),
                           "camera_direction":mean_ray(stable[:,0]),
                           "representative_frame":rows[span[len(span)//2]]["global_frame_index"]})
    return states



def bearing(x, z):
    return degrees(atan2(x, -z))


def difference(a, b):
    return abs((a-b+180.) % 360. - 180.)


def camera_hypotheses(shot, positions):
    # Stage06.1 uses a nominal aim and a right-handed screen-right basis in its plan.
    subject = shot["camera_subject"]
    if subject == "GROUP":
        visible = {p for kind in ("screen_order","depth_order") for pair in shot["camera_clues"][kind]
                   for p in pair.values() if p in positions}
        aim = np.mean([positions[p] for p in sorted(visible)],axis=0) if len(visible)>=2 else None
    else:
        aim = positions.get(subject)
    if aim is None:
        return []
    region = (shot.get("camera_region") or {}).get("cells",[])
    point = shot.get("camera_position")
    branches = [("candidate_region",region),("inherited_point",[point] if point is not None else [])]
    samples = []
    for branch,points in branches:
        for camera in points:
            f = np.asarray(aim)-camera
            if np.linalg.norm(f) < 1e-9:
                continue
            f = f/np.linalg.norm(f)
            for offset in AIM_OFFSETS_DEG:
                a = radians(offset)
                forward = np.array([cos(a)*f[0]-sin(a)*f[1],sin(a)*f[0]+cos(a)*f[1]])
                right = np.array([-forward[1],forward[0]])
                samples.append((branch,np.asarray(camera,dtype=float),forward,right))
    return samples


def spatial_fit(actor, ray, shot, layout, mode, *, hypotheses=None, screen_veto=True):
    """Compare rough horizontal bearings over alternative camera/plan hypotheses."""
    positions = {p["person"]:np.asarray(p["position"],dtype=float) for p in layout["people"]
                 if p["position"] is not None}
    samples = camera_hypotheses(shot,positions) if hypotheses is None else hypotheses
    if actor not in positions or not samples or np.hypot(ray[0],ray[2]) < .35:
        return None
    measured = bearing(ray[0],ray[2])
    limit = GAZE_TOLERANCE_DEG if mode=="gaze" else HEAD_TOLERANCE_DEG
    screen = shot["camera_clues"]["screen_order"]
    fits = []
    for target in CHARACTERS:
        if target == actor or target not in positions:
            continue
        delta = positions[target]-positions[actor]
        # Vary the target relative to the nominal subject. Poorly localized
        # characters receive twice the plan perturbation.
        qualities = {p["person"]:p["position_quality"] for p in layout["people"]}
        step = PLAN_PERTURBATION * (2 if "rough" in (qualities[actor],qualities[target]) else 1)
        offsets = np.array([[0,0],[step,0],[-step,0],[0,step],[0,-step]])
        deltas = delta+offsets
        errors = {}
        for family,_,forward,right in samples:
            errors.setdefault(family,[]).extend(difference(measured,bearing(d@right,d@forward))
                                                 for d in deltas if np.linalg.norm(d)>1e-9)
        # Use direct co-frame left-right only as a veto for a clear opposite sign.
        contrary = screen_veto and any((p["left"],p["right"])==(actor,target) and ray[0]<-.22
                       or (p["left"],p["right"])==(target,actor) and ray[0]>.22 for p in screen)
        by_family = {k:{"compatible":sum(e<=limit and not contrary for e in values),"tested":len(values),
                        "minimum_error_deg":round(float(min(values)),2),
                        "median_error_deg":round(float(np.median(values)),2)} for k,values in errors.items()}
        fits.append({"target":target,"camera_branches":by_family,"contradicts_screen_order":contrary})
    return {"camera_horizontal_bearing_deg":round(measured,2),"tolerance_deg":limit,
            "nominal_aim":shot["camera_subject"],"tested_targets":fits}


def match(actor, mode, ray, shot, layout, prior):
    choices = [{"target":name,"support":"context_prior","prior_reasons":reasons}
               for name,reasons in prior.items() if name!=actor]
    unknown = {"target":"UNKNOWN","support":"unresolved"}
    geometry = spatial_fit(actor,ray,shot,layout,mode) if ray is not None and actor in CHARACTERS else None
    if geometry is None:
        if actor in CHARACTERS and ray is not None and choices:
            return choices+[unknown],"UNKNOWN","context_only",["camera_orientation_unconstrained"],None
        return [unknown],"UNKNOWN","insufficient",[
            "character_mapping_unresolved" if actor=="UNKNOWN" else "no_usable_stable_direction"],None

    plausible = []
    for item in geometry["tested_targets"]:
        branches = item["camera_branches"]
        fractions = [v["compatible"]/v["tested"] for v in branches.values()]
        if max(fractions,default=0.) >= .25:
            plausible.append((item,min(fractions),max(fractions)))
    plausible.sort(key=lambda x:(-x[1],-x[2],x[0]["target"]))
    if not plausible:
        far = bool(geometry["tested_targets"]) and all(
            all(v["minimum_error_deg"] > geometry["tolerance_deg"]+10 for v in f["camera_branches"].values())
            for f in geometry["tested_targets"])
        away = [{"target":"AWAY","support":"direction_away_from_modeled_people"},
                {"target":"OTHER","support":"off_layout_alternative"}] if far else []
        return away+[unknown],"UNKNOWN","insufficient",["no_named_target_supported_across_camera_hypotheses"],geometry

    level = "gaze_supported" if mode=="gaze" else "head_supported"
    options = [{"target":f["target"],"support":level,"geometry":f,"prior_reasons":prior.get(f["target"],[])}
               for f,_,_ in plausible]
    selected = "UNKNOWN"
    reasons = ["rough_2d_layout","camera_tilt_height_uncalibrated"]
    if mode=="head":
        reasons.append("head_only_eye_direction_unavailable")
    if len(plausible)>1:
        reasons.append("multiple_spatially_compatible_people")
    else:
        winner, conservative, _ = plausible[0]
        rivals = [f for f in geometry["tested_targets"] if f["target"]!=winner["target"]]
        branches = winner["camera_branches"]
        margin = min((r["camera_branches"][b]["median_error_deg"]-v["median_error_deg"]
                      for r in rivals for b,v in branches.items()),default=0.)
        if (mode=="gaze" and conservative>=MIN_GROUP_FRACTION and margin>=15.
                and shot["layout_basis"] in {"assumed_static","observed"}):
            selected = winner["target"]
        elif mode=="gaze":
            reasons.append("support_varies_with_camera_or_insufficient_margin")
    if shot["layout_basis"]=="assumed_static":
        reasons.append("uses_static_layout_working_assumption")
    return options+([unknown] if selected=="UNKNOWN" else []),selected,level,reasons,geometry


def direction_windows(rows, states):
    """Partition measured frames into stable Gaze, head fallback, and unresolved blocks."""
    by_frame = {r["global_frame_index"]:i for i,r in enumerate(rows)}
    claimed = set()
    windows = []
    for state in states:
        a,b = by_frame[state["start_frame"]],by_frame[state["end_frame"]-1]
        indices = list(range(a,b+1))
        if (any(rows[i]["gaze_direction"] is None for i in indices)
                or len(list(valid_blocks(rows,[a<=i<=b for i in range(len(rows))])))!=1):
            raise ValueError("Gaze state crosses a missing or discontinuous frame")
        claimed.update(indices)
        windows.append(("gaze",indices,state["camera_direction"]))
    mask = [r["gaze_direction"] is None and r["head_facing_direction"] is not None for r in rows]
    for block in valid_blocks(rows,mask):
        for span, rays in stable_runs(rows,block,[rows[i]["head_facing_direction"] for i in block],
                                      HEAD_STABILITY_DEG,angular_change):
            claimed.update(span)
            windows.append(("head",span,mean_ray(rays)))
    for block in valid_blocks(rows,[i not in claimed for i in range(len(rows))]):
        windows.append(("unresolved",block,None))
    return sorted(windows,key=lambda w:w[1][0])


def timeline_transitions(records):
    """Emit explicit continuity only between neighboring, resolved Gaze intervals."""
    previous = None
    transitions = []
    for item in records:
        selected = item["selected_target"]
        known = item["evidence_level"]=="gaze_supported" and selected in CHARACTERS
        if previous is not None and known and item["character"]==previous["character"]:
            gap = item["start_s"]-previous["end_s"]
            same_track = item["track_id"]==previous["track_id"] and item["shot_id"]==previous["shot_id"]
            adjacent = item["shot_index"]==previous["shot_index"]+1
            # Named targets are compared in their shared layout, never as camera yaw/pitch.
            cross = (adjacent and item["layout_id"]==previous["layout_id"]
                     and item["mapping_strength"]==previous["mapping_strength"]=="supported"
                     and item["evidence"]["plan_usable"] and previous["evidence"]["plan_usable"]
                     and item["start_s"]-item["shot_start_s"]<=MAX_CONTINUITY_GAP_S
                     and previous["shot_end_s"]-previous["end_s"]<=MAX_CONTINUITY_GAP_S)
            if (same_track and abs(gap)<=1e-6) or (cross and -1e-6<=gap<=MAX_CONTINUITY_GAP_S):
                hold = selected==previous["selected_target"]
                transitions.append({"type":"target_hold_candidate" if hold else "target_switch_candidate",
                                    "from_interval_id":previous["interval_id"],"to_interval_id":item["interval_id"],
                                    "from_shot_id":previous["shot_id"],"to_shot_id":item["shot_id"],
                                    "from_target":previous["selected_target"],"to_target":selected,
                                    "from_stable_s":[previous["start_s"],previous["end_s"]],
                                    "to_stable_s":[item["start_s"],item["end_s"]],
                                    "start_s":previous["end_s"],"end_s":item["start_s"],"cross_shot":cross,
                                    "continuity_assumption":"same_layout_and_named_target_candidates" if cross else
                                                            "adjacent_stable_intervals_same_track"})
        previous = item if known else None
    return transitions
