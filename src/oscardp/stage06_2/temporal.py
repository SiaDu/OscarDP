"""Within-shot movements, using Stage05.5 seconds-based median and hysteresis."""

import numpy as np

from oscardp.stage05.event_series import hysteresis, median_filter, segments

from .geometry import angular_change, decompose, head_change, unit
from .inputs import EYES, POSE

# Pilot heuristics, fixed before running the real shot. Gaze is supplemental.
THRESHOLDS = {
    "head": {"speed_high": 15., "speed_low": 6., "net_deg": 6., "merge_sec": .08},
    "gaze": {"speed_high": 26., "speed_low": 10., "net_deg": 6., "merge_sec": .04},
    "eye": {"speed_high": 30., "speed_low": 12., "net_deg": 6., "merge_sec": .04},
}
MEDIAN_RADIUS = .06
MAX_GAP = .25
SMALL_DEG = 3.
BINOCULAR_LIMIT_DEG = 12.
BINOCULAR_CHANGE_LIMIT_DEG = 6.
JITTER_MEDIAN_LIMIT_DEG = 3.


def prepare_frames(rows, source_events):
    closures = [e for e in source_events if e["event_type"] == "eye_closure_candidate"]
    result = []
    for row in rows:
        n = row["global_frame_index"]
        closed = [e["event_id"] for e in closures if e["start_frame"] <= n < e["end_frame"]]
        pose = [row[k] for k in POSE] if row["tracking_usable"] and row["head_pose_valid"] else None
        usable_gaze = row["tracking_usable"] and row["gaze_valid"] and not closed
        eyes = [[row[k] for k in eye] if usable_gaze else None for eye in EYES]
        values = decompose(pose, *eyes)
        left, right = map(unit, eyes)
        difference = angular_change(left, right) if left is not None and right is not None else None
        disagreement = difference is not None and difference > BINOCULAR_LIMIT_DEG
        flags = []
        if not row["tracking_usable"]:
            flags.append("tracking_unusable")
        if values["head_direction"] is None:
            flags.append("pose_unusable")
        if closed:
            flags.append("stage05_eye_closure_candidate")
        elif values["gaze_direction"] is None:
            flags.append("gaze_unusable")
        if disagreement:
            flags.append("binocular_disagreement")
        rejected = {key + "_qc_rejected": values[key].tolist() if disagreement and values[key] is not None else None
                    for key in ("gaze_direction", "eye_relative_direction")}
        if disagreement:
            values["gaze_direction"] = values["eye_relative_direction"] = None
        result.append({
            **row, **rejected, **{k: v.tolist() if v is not None else None for k, v in values.items()},
            "head_usable": values["head_direction"] is not None,
            "camera_gaze_usable": values["gaze_direction"] is not None,
            "eye_relative_usable": values["eye_relative_direction"] is not None,
            "binocular_disagreement_deg": difference, "closure_event_ids": closed, "qc_flags": flags,
        })
    return result


def valid_blocks(rows, mask):
    for block in segments(rows, mask, MAX_GAP):
        start = 0
        for j in range(1, len(block)):
            if abs(rows[block[j]]["movie_timestamp_sec"] - rows[block[j-1]]["source_next_timestamp_sec"]) > 1e-6:
                yield block[start:j]
                start = j
        yield block[start:]


def movement_type(head, eye, unreliable=False):
    if unreliable or head is None or eye is None:
        return "uncertain"
    if head >= THRESHOLDS["head"]["net_deg"] and eye >= THRESHOLDS["eye"]["net_deg"]:
        return "combined"
    if head >= THRESHOLDS["head"]["net_deg"] and eye <= SMALL_DEG:
        return "head_dominant"
    if eye >= THRESHOLDS["eye"]["net_deg"] and head <= SMALL_DEG:
        return "eye_dominant"
    return "uncertain"


def detect(rows, source_events):
    events = []
    joint = np.array([r["eye_relative_usable"] for r in rows])
    head_only = np.array([r["head_usable"] for r in rows]) & ~joint
    for mask, keys in ((joint, ("head", "gaze", "eye")), (head_only, ("head",))):
        for segment in valid_blocks(rows, mask):
            if len(segment) < 3:
                continue
            frames = [rows[i] for i in segment]
            times = np.array([r["movie_timestamp_sec"] for r in frames])
            ends = np.array([r["source_next_timestamp_sec"] for r in frames])
            raw = {"head": np.unwrap([[r[k] for k in POSE] for r in frames], axis=0)}
            if "eye" in keys:
                raw["gaze"] = [r["gaze_direction"] for r in frames]
                raw["eye"] = [r["eye_relative_direction"] for r in frames]

            filtered, intervals, residual = {}, [], {}
            for key in keys:
                distance = head_change if key == "head" else angular_change
                smooth = median_filter(times, raw[key], MEDIAN_RADIUS)
                if key != "head":
                    smooth = [unit(v) for v in smooth]
                # A degenerate vector median has no defined direction.
                if any(v is None for v in smooth):
                    continue
                filtered[key] = smooth
                residual[key] = [distance(a, b) for a, b in zip(raw[key], smooth, strict=True)]
                speeds = []
                for j in range(len(frames)):
                    a, b = max(j-1, 0), min(j+1, len(frames)-1)
                    speeds.append(distance(smooth[a], smooth[b]) / (times[b] - times[a]))
                cfg = THRESHOLDS[key]
                for a, b in hysteresis(times, ends, speeds, cfg["speed_high"], cfg["speed_low"], cfg["merge_sec"]):
                    if b-a >= 3 and ends[b-1]-times[a] >= .12 and distance(smooth[a], smooth[b-1]) >= cfg["net_deg"]:
                        intervals.append((a, b, key))
            union = []
            for a, b, key in sorted(intervals):
                if union and a <= union[-1][1]:
                    union[-1][1] = max(b, union[-1][1])
                    union[-1][2].add(key)
                else:
                    union.append([a, b, {key}])
            for a, b, triggers in union:
                changes = {key: (head_change if key == "head" else angular_change)(v[a], v[b-1])
                           for key, v in filtered.items()}
                before, after = frames[a], frames[b-1]
                flags = sorted({flag for r in frames[a:b] for flag in r["qc_flags"]})
                residuals = {key: float(np.median(value[a:b])) for key, value in residual.items()}
                if max(residuals.values()) > JITTER_MEDIAN_LIMIT_DEG:
                    flags.append("high_median_filter_residual")
                eye_deltas = None
                if "eye" in filtered:
                    endpoints = []
                    for side in ("left", "right"):
                        name = f"{side}_eye_relative_direction"
                        eye_raw = [r[name] for r in frames]
                        smoothed = median_filter(times, eye_raw, MEDIAN_RADIUS)
                        endpoints.append((unit(smoothed[a]), unit(smoothed[b-1])))
                    if all(p is not None for pair in endpoints for p in pair):
                        (l0, l1), (r0, r1) = endpoints
                        eye_deltas = [angular_change(l0, l1), angular_change(r0, r1)]
                        mismatch = float(np.degrees(2 * np.arcsin(min(1., np.linalg.norm((l1-l0)-(r1-r0))/2))))
                        if mismatch > BINOCULAR_CHANGE_LIMIT_DEG:
                            flags.append("binocular_motion_disagreement")
                    else:
                        flags.append("binocular_motion_unusable")
                unreliable = bool(flags) or not {"head", "eye"} <= changes.keys()
                start, end = before["movie_timestamp_sec"], after["source_next_timestamp_sec"]
                events.append({
                    "shot_id": before["shot_id"], "track_id": before["track_id"], "person": before["person"],
                    "start_frame": before["global_frame_index"], "end_frame": after["global_frame_index"]+1,
                    "start_s": start, "end_s": end,
                    "movement_type": movement_type(changes.get("head"), changes.get("eye"), unreliable),
                    "head_change_deg": changes.get("head"), "gaze_change_deg": changes.get("gaze"),
                    "eye_relative_change_deg": changes.get("eye"),
                    "valid_frame_ratio": sum(r["eye_relative_usable"] for r in frames[a:b]) / (b-a),
                    "status": "model_candidate", "trigger_signals": sorted(triggers), "qc_flags": flags,
                    "left_right_eye_change_deg": eye_deltas, "median_filter_residual_deg": residuals,
                    "left_censored": a == 0, "right_censored": b == len(segment),
                    "stage05_event_ids": [e["event_id"] for e in source_events
                                          if e["start_sec"] < end and e["end_sec"] > start],
                })
    events.sort(key=lambda e: (e["start_s"], e["end_s"]))
    for i, event in enumerate(events, 1):
        event["event_id"] = f"gaze_event_{i:03d}"
    return events
