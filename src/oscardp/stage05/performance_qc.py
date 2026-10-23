"""Per-modality usability gates, source-face checks and missing-interval QC."""
import math

import numpy as np


def complete(raw, names):
    return bool(names) and all(raw.get(n) is not None for n in names)


def face_agreement(raw, observation):
    if observation is None:
        return {"matched":False,"reason":"no_observed_yunet_face"}
    names = [f"{axis}_{i}" for axis in "xy" for i in range(68)]
    if not complete(raw,names):
        return {"matched":False,"reason":"missing_or_nonfinite_face_landmarks"}
    points = np.array([[raw[f"x_{i}"],raw[f"y_{i}"]] for i in range(68)])
    x,y,w,h = observation["bbox_xywh"]
    low,high = points.min(axis=0),points.max(axis=0)
    overlap = np.maximum(0,np.minimum(high,[x+w,y+h])-np.maximum(low,[x,y])).prod()
    area = np.maximum(0,high-low).prod()
    iou = float(overlap/max(1,area+w*h-overlap))
    fitted5 = np.stack([points[36:42].mean(axis=0),points[42:48].mean(axis=0),points[30],points[48],points[54]])
    error = float(np.linalg.norm(fitted5-np.asarray(observation["landmarks5"]),axis=1).mean()/max(1,math.hypot(w,h)))
    matched = iou >= .25 and error <= .18
    return {"matched":matched,"reason":"matched_spatial_evidence" if matched else "face_spatial_mismatch_review_needed",
            "landmark_bbox_iou":iou,"five_point_normalized_error":error,
            "openface_landmark_bbox_xywh":[float(low[0]),float(low[1]),float(high[0]-low[0]),float(high[1]-low[1])],
            "yunet_bbox_xywh":observation["bbox_xywh"],
            "identity_verified":False}


def assess(raw, groups, observation, min_confidence=.8):
    flags = {"openface_row_present":raw is not None,"openface_success":False,"tracking_usable":False,
             "au_valid":False,"head_pose_valid":False,"gaze_valid":False,
             "landmarks_2d_valid":False,"landmarks_3d_valid":False,
             "nonfinite_columns":[] if raw is None else raw["nonfinite_columns"],"reasons":[]}
    if raw is None:
        flags["reasons"] = ["missing_openface_frame"]
        flags["association"] = {"matched":False,"reason":"missing_openface_frame"}
        return flags
    success = raw["success"] == 1
    flags["openface_success"] = success
    agreement = face_agreement(raw,observation)
    flags["association"] = agreement
    confident = raw["confidence"] is not None and min_confidence <= raw["confidence"] <= 1
    base = bool(success and confident and agreement["matched"])
    flags["tracking_usable"] = base
    if not success: flags["reasons"].append("openface_tracking_failed")
    if not confident: flags["reasons"].append("tracking_confidence_below_gate_or_invalid")
    if not agreement["matched"]: flags["reasons"].append(agreement["reason"])
    flags["landmarks_2d_valid"] = base and len(groups["landmarks_2d"])==136 and complete(raw,groups["landmarks_2d"])
    flags["landmarks_3d_valid"] = base and len(groups["landmarks_3d"])==204 and complete(raw,groups["landmarks_3d"])
    rotations = ["pose_Rx","pose_Ry","pose_Rz"]
    pose_names = rotations+["pose_Tx","pose_Ty","pose_Tz"]
    pose_ok = complete(raw,pose_names) and all(abs(raw[n]) <= math.pi for n in rotations) and raw["pose_Tz"] > 0
    flags["head_pose_valid"] = bool(base and pose_ok)
    if not pose_ok: flags["reasons"].append("pose_nonfinite_missing_or_outside_physical_range")
    intensity,occurrence = groups["au_intensity"],groups["au_occurrence"]
    au_ok = (complete(raw,intensity) and complete(raw,occurrence)
             and all(0 <= raw[n] <= 5 for n in intensity) and all(raw[n] in {0,1} for n in occurrence))
    au_pose = complete(raw,rotations) and abs(raw["pose_Rx"]) <= .8 and abs(raw["pose_Ry"]) <= 1.
    flags["au_valid"] = bool(base and au_ok and au_pose)
    if not au_ok: flags["reasons"].append("au_missing_nonfinite_or_out_of_range")
    if not au_pose: flags["reasons"].append("au_pose_outside_pilot_gate")
    gaze_names = [f"gaze_{eye}_{axis}" for eye in range(2) for axis in "xyz"]+["gaze_angle_x","gaze_angle_y"]
    gaze_ok = complete(raw,gaze_names)
    if gaze_ok:
        norms = [math.sqrt(sum(raw[f"gaze_{eye}_{a}"]**2 for a in "xyz")) for eye in range(2)]
        gaze_ok = all(.9 <= n <= 1.1 for n in norms) and all(abs(raw[n]) <= math.pi/2 for n in ["gaze_angle_x","gaze_angle_y"])
        flags["gaze_vector_norms"] = norms
    eye_aspects = []
    if flags["landmarks_2d_valid"]:
        points = np.array([[raw[f"x_{i}"],raw[f"y_{i}"]] for i in range(68)])
        for start in [36,42]:
            eye = points[start:start+6]
            width = np.linalg.norm(eye[0]-eye[3])
            aspect = float((np.linalg.norm(eye[1]-eye[5])+np.linalg.norm(eye[2]-eye[4]))/max(1,2*width))
            eye_aspects.append(aspect if width >= 10 else 0.)
    flags["eye_aspect_ratio_proxies"] = eye_aspects
    eye_ok = (len(eye_aspects)==2 and min(eye_aspects) >= .12 and complete(raw,groups["eye_landmarks"])
              and complete(raw,rotations) and abs(raw["pose_Ry"]) <= .9)
    flags["gaze_valid"] = bool(base and gaze_ok and eye_ok)
    if not gaze_ok: flags["reasons"].append("gaze_missing_nonfinite_zero_or_out_of_range")
    if not eye_ok: flags["reasons"].append("gaze_eye_geometry_or_pose_not_usable")
    flags["usability_is_ground_truth"] = False
    return flags


def missing_intervals(rows, field):
    intervals,current = [],[]
    for row in rows:
        if not row["reliability"][field]:
            current.append(row)
        elif current:
            intervals.append(current)
            current = []
    if current: intervals.append(current)
    return [{"start_frame":r[0]["global_frame_index"],"end_frame":r[-1]["global_frame_index"]+1,
             "start_timestamp_sec":r[0]["movie_timestamp_sec"],"end_timestamp_sec":r[-1]["source_next_timestamp_sec"],
             "frame_count":len(r),"duration_sec":r[-1]["source_next_timestamp_sec"]-r[0]["movie_timestamp_sec"],
             "kind":"long_missing" if len(r)>=12 else "brief_failure"} for r in intervals]


def summarize(rows, groups):
    fields = ["openface_success","tracking_usable","au_valid","head_pose_valid","gaze_valid","landmarks_2d_valid","landmarks_3d_valid"]
    usable_iou = [r["reliability"]["association"]["landmark_bbox_iou"] for r in rows
                  if "landmark_bbox_iou" in r["reliability"]["association"]]
    jumps = []
    checks = [(groups["au_intensity"],"au_valid",2.0),(["pose_Rx","pose_Ry","pose_Rz"],"head_pose_valid",.35),
              (["gaze_angle_x","gaze_angle_y"],"gaze_valid",.5)]
    for previous,current in zip(rows,rows[1:]):
        for names,valid,limit in checks:
            if previous["reliability"][valid] and current["reliability"][valid]:
                for name in names:
                    delta = abs(current["features"][name]-previous["features"][name])
                    if delta > limit:
                        jumps.append({"global_frame_index":current["global_frame_index"],"field":name,"absolute_delta":delta,
                                      "delta_per_sec":delta/(current["movie_timestamp_sec"]-previous["movie_timestamp_sec"]),
                                      "interpretation":"QC jump flag; no performance event inferred"})
    spans = {}
    for family,valid in [("au_intensity","au_valid"),("pose","head_pose_valid"),("gaze","gaze_valid")]:
        for name in groups[family]:
            good = [r["features"][name] for r in rows if r["reliability"][valid]]
            spans[name] = {"min":min(good),"max":max(good),"median":float(np.median(good)),"range":max(good)-min(good)} if good else None
    mismatch = [r["global_frame_index"] for r in rows if r["reliability"]["openface_success"]
                and r["reliability"]["association"]["reason"]=="face_spatial_mismatch_review_needed"]
    return {"expected_frame_count":len(rows),"openface_row_count":sum(r["reliability"]["openface_row_present"] for r in rows),
            "valid_counts":{f:sum(r["reliability"][f] for r in rows) for f in fields},
            "valid_ratios":{f:sum(r["reliability"][f] for r in rows)/len(rows) for f in fields},
            "failure_intervals":{f:missing_intervals(rows,f) for f in fields},
            "bbox_iou_min":min(usable_iou) if usable_iou else None,
            "bbox_iou_median":float(np.median(usable_iou)) if usable_iou else None,
            "possible_wrong_face_frames":mismatch,"track_status":"failed_face_association" if mismatch else "measured_with_modality_qc",
            "nonfinite_value_count":sum(len(r["reliability"]["nonfinite_columns"]) for r in rows),
            "jump_flags":jumps,"valid_measurement_ranges":spans,
            "requires_human_review":True}
