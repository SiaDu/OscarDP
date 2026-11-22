"""Eye-position runs and comparisons of sustained within-shot camera directions."""

import numpy as np

from oscardp.stage05.event_series import median_filter
from .blink_qc import blink_qc
from .geometry import angular_change, head_change, unit
from .inputs import EYES, POSE
from .temporal import valid_blocks

POSITION_LIMIT = .03
DIRECTION_LIMIT = 3.
MIN_STABLE_S = .25
MAX_TRANSITION_S = .55
POSITION_FIELDS = [f"{side}_eye_{axis}" for side in ("left","right") for axis in "xy"]


def interval(rows, indices):
    a,b = rows[indices[0]],rows[indices[-1]]
    return {"start_frame": a["global_frame_index"], "end_frame": b["global_frame_index"]+1,
            "start_s": a["movie_timestamp_sec"], "end_s": b["source_next_timestamp_sec"]}


def stable_runs(rows, indices, values, limit, distance):
    start = 0
    for j in range(1, len(indices)+1):
        if j < len(indices) and all(distance(value, values[j]) <= limit for value in values[start:j]):
            continue
        span = indices[start:j]
        bounds = interval(rows, span)
        if bounds["end_s"] - bounds["start_s"] >= MIN_STABLE_S:
            yield span, values[start:j]
        start = j


def position_distance(a,b):
    return float(np.linalg.norm(np.asarray(a-b).reshape(2,2), axis=1).max())


def direction_distance(a,b):
    return max(angular_change(v,w) for v,w in zip(a,b,strict=True))


def eye_blocks(rows, support):
    for block in valid_blocks(rows, [bool(s) for s in support]):
        start = 0
        for j in range(1,len(block)):
            if support[block[j]] != support[block[start]]:
                yield block[start:j], support[block[start]]
                start = j
        yield block[start:], support[block[start]]


def analyze(rows, source_events, adjunct, *, allow_monocular=False, independent_gaze=False):
    support = [tuple(side for side in ("left","right")
                     if all(r[f"{side}_eye_{a}"] is not None for a in "xy")) for r in rows]
    if not allow_monocular:
        support = [s if len(s)==2 else () for s in support]
    mask = [bool(s) for s in support]
    states, motion, picks, accounted = [], [], {}, set()

    def select(index, reason):
        if rows[index]["camera_gaze_usable"]:
            picks.setdefault(index, set()).add(reason)

    def state(kind, span, **fields):
        record = {"type":kind, **interval(rows,span), **fields}
        states.append(record)
        if allow_monocular:
            select(span[len(span)//2], kind)
            last = rows[span[0]]["movie_timestamp_sec"]
            for i in span:
                t = rows[i]["movie_timestamp_sec"]
                if t-last >= 1.:
                    select(i, kind+"_periodic")
                    last = t
        else:
            for j in range(len(span)//2 if len(span)<48 else 12, len(span), 24 if len(span)>=48 else len(span)):
                select(span[j], kind)
        return record

    def gaze_runs(sub):
        rays = np.array([[rows[i]["gaze_direction"], *[unit([rows[i][k] for k in eye])
                         for eye in EYES]] for i in sub])
        for span, directions in stable_runs(rows, sub, rays, DIRECTION_LIMIT, direction_distance):
            r = rows[span[len(span)//2]]
            summary = unit(np.mean(directions[:,0],axis=0))
            record = state("stable_gaze_candidate", span, camera_direction=summary.tolist(),
                           representative_frame=r["global_frame_index"], head_direction=r["head_direction"],
                           eye_relative_direction=r["eye_relative_direction"],
                           binocular_absolute_flag=any("binocular_disagreement" in rows[i]["qc_flags"] for i in span),
                           **({"evidence_source":"openface_camera_gaze"} if independent_gaze else {}))
            yield span, record, np.mean(directions[:,1:], axis=0)

    def compare_gaze(gaze_states):
        for (before,a,eyes_a),(after,b,eyes_b) in zip(gaze_states,gaze_states[1:]):
            i,j = before[-1],after[0]
            if (b["start_s"]-a["end_s"] > MAX_TRANSITION_S
                    or not all(rows[k]["camera_gaze_usable" if independent_gaze else "eye_relative_usable"] for k in range(i,j+1))):
                continue
            mean = angular_change(a["camera_direction"],b["camera_direction"])
            delta = eyes_b-eyes_a
            if (mean >= 6 and min(angular_change(unit(v),unit(w)) for v,w in zip(eyes_a,eyes_b)) >= 4
                    and np.sum(delta[0]*delta[1]) > 0
                    and np.linalg.norm(delta[0]-delta[1]) <= 2*np.sin(np.radians(3))):
                motion.append({"type":"gaze_direction_shift_candidate", **interval(rows,[i,j]),
                               "gaze_change_deg":mean,
                               "before_stable_frames":[a["start_frame"],a["end_frame"]],
                               "after_stable_frames":[b["start_frame"],b["end_frame"]]})
                select(i,"before_gaze_shift")
                select(j,"after_gaze_shift")

    for block, sides in eye_blocks(rows, support):
        keys = [f"{side}_eye_{a}" for side in sides for a in "xy"]
        values = np.array([[rows[i][key] for key in keys] for i in block])
        def pos_distance(a,b):
            return float(np.linalg.norm((a-b).reshape(-1,2),axis=1).max())
        times = np.array([rows[i]["movie_timestamp_sec"] for i in block])
        smooth = median_filter(times, values, .05)
        pos_states, gaze_states = [], []
        for span, points in stable_runs(rows, block, smooth, POSITION_LIMIT, pos_distance):
            accounted.update(span)
            centre = np.median(points,axis=0)
            pos_states.append((span, centre))
            xy = dict(zip(sides,centre.reshape(-1,2).tolist(),strict=True))
            state("eye_position_stable", span, left_xy=xy.get("left"), right_xy=xy.get("right"),
                  **({"eye_support":list(sides)} if allow_monocular else {}))
            if not independent_gaze:
                allowed = [r["eye_relative_usable"] and r["head_usable"] and i in span for i,r in enumerate(rows)]
                for sub in valid_blocks(rows, allowed):
                    gaze_states.extend(gaze_runs(sub))

        for (before,a), (after,b) in zip(pos_states,pos_states[1:]):
            i,j = before[-1], after[0]
            if rows[j]["movie_timestamp_sec"]-rows[i]["source_next_timestamp_sec"] > MAX_TRANSITION_S:
                continue
            legs = (b-a).reshape(-1,2)
            lengths = np.linalg.norm(legs,axis=1)
            peak = max(pos_distance(v,a) for v in smooth[block.index(i):block.index(j)+1])
            if peak < POSITION_LIMIT and max(lengths)<POSITION_LIMIT:
                continue
            if (min(lengths)>=POSITION_LIMIT
                    and (len(sides)==1 or (np.dot(*legs)>0 and np.linalg.norm(legs[0]-legs[1]) <= .05
                         and (not allow_monocular or legs[0,0]*legs[1,0]>=0)))):
                label = "eye_shift_candidate"
            elif max(lengths)<POSITION_LIMIT:
                label = "eye_adjustment_candidate"
            else:
                label = "unknown"
            # Large concurrent yaw/pitch can move the 2D projection within the fitted aperture.
            head = (head_change([rows[i][k] for k in POSE],[rows[j][k] for k in POSE])
                    if rows[i]["head_usable"] and rows[j]["head_usable"] else None)
            if (head is None or head > 6) and label == "eye_shift_candidate":
                label = "unknown"
            motion.append({"type":label, **interval(rows,[i,j]), "eye_position_displacement":lengths.tolist(),
                           "head_change_deg":head,
                           "reason":"possible_pose_projection_or_weak_binocular_support" if label=="unknown" else "iris_fit_proxy",
                           **({"eye_support":list(sides)} if allow_monocular else {})})
            accounted.update(range(i,j+1))
            select(i,"before_eye_motion")
            select(j,"after_eye_motion")

        compare_gaze(gaze_states)

    if independent_gaze:
        for sub in valid_blocks(rows, [r["camera_gaze_usable"] for r in rows]):
            compare_gaze(list(gaze_runs(sub)))

    _, blinks = blink_qc(rows, source_events)
    motion.extend(blinks)
    for block in valid_blocks(rows, [not v for v in mask]):
        motion.append({"type":"unknown", **interval(rows,block), "reason":"eye_position_unavailable"})
    for block in valid_blocks(rows, [good and i not in accounted for i,good in enumerate(mask)]):
        motion.append({"type":"unknown", **interval(rows,block), "reason":"unstable_fit_without_supported_transition"})
    for block in valid_blocks(rows, [r["eye_relative_usable"] for r in rows]):
        if block[0]:
            select(block[0], "recovery_after_unusable_gaze_or_time_gap")
    by_frame = {r["global_frame_index"]:i for i,r in enumerate(rows)}
    for e in adjunct:
        for n in (e["start_frame"],e["end_frame"]-1):
            select(by_frame[n],"direction_motion_comparison")
    frames = [{"global_frame_index":rows[i]["global_frame_index"],"movie_timestamp_sec":rows[i]["movie_timestamp_sec"],
               "reasons":sorted(picks[i])} for i in sorted(picks)]
    return states, sorted(motion,key=lambda e:e["start_s"]), frames
