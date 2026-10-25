"""Relative model orientation and camera-relative gaze changes, no semantic labels."""
import numpy as np

from .event_records import candidate, rejection, trace
from .event_series import segments, median_filter, derivative, hysteresis, rotation_distance, gaze_distance


def detect_motion(rows,kind,config):
    cfg = config[kind]
    head = kind=="head"
    fields = ["pose_Rx","pose_Ry","pose_Rz"] if head else ["gaze_angle_x","gaze_angle_y"]
    feature = "head_orientation" if head else "gaze_direction"
    event_type = "head_orientation_change" if head else "camera_relative_gaze_change"
    validity = "head_pose_valid" if head else "gaze_valid"
    raw = np.array([[r.get(n) if r.get(n) is not None else np.nan for n in fields] for r in rows])
    mask = np.array([r[validity] for r in rows]) & np.isfinite(raw).all(axis=1)
    processed = np.full(raw.shape,np.nan)
    speed_trace = np.full(len(rows),np.nan)
    events,rejected = [],[]
    distance = rotation_distance if head else gaze_distance
    for indices in segments(rows,mask,config["series"]["max_gap_sec"]):
        times = np.array([rows[i]["movie_timestamp_sec"] for i in indices])
        ends = np.array([rows[i]["source_next_timestamp_sec"] for i in indices])
        values = median_filter(times,np.unwrap(raw[indices],axis=0),cfg["median_radius_sec"])
        processed[indices] = values
        # Central angular speed from geodesic direction/orientation distance and actual PTS.
        speed = np.array([distance(values[max(0,j-1)],values[min(len(indices)-1,j+1)])/
                         (times[min(len(indices)-1,j+1)]-times[max(0,j-1)]) if len(indices)>1 else 0. for j in range(len(indices))])
        speed_trace[indices] = speed
        for a,b in hysteresis(times,ends,speed,cfg["speed_high"],cfg["speed_low"],cfg["merge_gap_sec"]):
            before,after = values[a],values[b-1]
            displacement = distance(before,after)
            reasons = []
            if b-a<cfg["min_frames"] or ends[b-1]-times[a]<cfg["min_duration_sec"]: reasons.append("minimum_motion_duration_or_samples")
            if displacement<cfg["min_displacement_rad"]: reasons.append("minimum_net_angular_displacement")
            if reasons:
                rejected.append(rejection(rows,indices,a,b,event_type,feature,reasons))
                continue
            apex = a+int(np.argmax(speed[a:b]))
            event = candidate(rows,indices,a,b,event_type,feature,validity,config,apex)
            delta = after-before
            event["measurements"] = {"angular_displacement_rad":displacement,"angular_velocity_peak_rad_per_sec":float(max(speed[a:b])),
                "angular_velocity_median_rad_per_sec":float(np.median(speed[a:b])),"before_orientation_rad":before.tolist(),
                "after_orientation_rad":after.tolist(),"per_axis_displacement_rad":dict(zip(fields,delta.tolist())),
                "duration_sec":float(ends[b-1]-times[a]),
                "measurement_stability":{"raw_minus_filtered_median_abs_rad":float(np.median(np.abs(raw[indices[a:b]]-values[a:b]))),
                                         "velocity_mad":float(np.median(np.abs(speed[a:b]-np.median(speed[a:b]))))},
                "camera_calibrated":False,"scope":"relative OpenFace camera-reference model estimates"}
            if head:
                event["measurements"]["dominant_rotation_axis"] = ["pitch","yaw","roll"][int(np.argmax(np.abs(delta)))]
                event["measurements"]["per_axis_peak_velocity_rad_per_sec"] = dict(zip(fields,np.max(np.abs(derivative(times,values)[a:b]),axis=0).tolist()))
            else:
                event["qc_flags"].append("head_motion_confound_possible")
                pose_good = all(rows[i]["head_pose_valid"] for i in indices[a:b])
                head_delta = rotation_distance([rows[indices[a]][n] for n in ["pose_Rx","pose_Ry","pose_Rz"]],
                                               [rows[indices[b-1]][n] for n in ["pose_Rx","pose_Ry","pose_Rz"]]) if pose_good else None
                event["measurements"].update(head_motion_displacement_rad=head_delta,eye_only_gaze_shift=False,
                    head_motion_contribution="not compensated; displacement is descriptive, not a decomposition")
                event["evidence"]["eye_direction_vectors_before_after"] = [
                    {n:rows[indices[j]].get(n) for n in [f"gaze_{eye}_{axis}" for eye in range(2) for axis in "xyz"]}
                    for j in [a,b-1]]
                if head_delta is None: event["qc_flags"].append("head_motion_evidence_unavailable")
                elif head_delta>=cfg["head_confound_rad"]: event["qc_flags"].append("head_gaze_simultaneous_change")
            events.append(event)
    return events,rejected,trace(rows,feature,raw,processed,mask,speed_trace)
