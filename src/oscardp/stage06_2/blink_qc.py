"""Per-eye observability and short closure cycles from saved landmark fits."""

import numpy as np

from oscardp.stage05.event_config import DEFAULTS
from oscardp.stage05.event_series import hysteresis, median_filter
from .temporal import valid_blocks

SIDES = ("left", "right")  # Image coordinates.
FACE_EYE_COLUMNS = [f"{axis}_{i}" for i in range(36, 48) for axis in "xy"]
EYE_RULES = DEFAULTS["eye"]
MIN_EYE_WIDTH = 10.
OPEN_APERTURE = .12
MAX_BLINK_S = .50
FAILURES = {"track_unavailable", "detection_failure", "track_alignment_failure",
            "conflicting_face_overlap", "unsupported_landmark_jump",
            "centre_outside_lids", "nonfinite_landmarks", "oblique_fit_unreliable"}


def eyelids(row, side, mp):
    if mp:
        fits = row.get(f"{side}_mp_eyelids_px")
        return [np.asarray(lid, dtype=float) for lid in fits] if fits else []
    start = 36 if side == "left" else 42
    indices = ([0, 1, 2, 3], [0, 5, 4, 3])
    return [np.array([[row.get(f"{a}_{start+i}") for a in "xy"] for i in order], dtype=float)
            for order in indices]


def lid_geometry(lids):
    if len(lids) != 2 or any(lid.ndim != 2 or lid.shape[1] != 2 or len(lid) < 4
                             or not np.isfinite(lid).all() for lid in lids):
        return None, None
    p, q = lids[0][[0, -1]]
    width = float(np.linalg.norm(q-p))
    if width < MIN_EYE_WIDTH:
        return width, None
    x = (q-p)/width
    basis = np.array([x, [-x[1], x[0]]])
    positions = []
    for lid in lids:
        u, v = ((lid-p) @ basis.T / width).T
        order = np.argsort(u)
        if u.min() > .25 or u.max() < .75:
            return width, None
        positions.append(np.interp([.35, .5, .65], u[order], v[order]))
    # Signed eyelid ordering follows the image y axis when the eye corners run left to right.
    gap = (positions[1]-positions[0]) * (1 if q[0] >= p[0] else -1)
    if np.min(gap) < -.02 or np.max(gap) > .65:
        return width, None
    return width, float(np.median(np.maximum(0., gap)))


def assess_eyes(row):
    """Return the measurable eyelid aperture, independent of iris and Gaze usability."""
    bbox = np.asarray(row.get("face_bbox_xywh"), dtype=float)
    face_size = float(min(bbox[2:])) if bbox.shape == (4,) and np.isfinite(bbox).all() else None
    yaw = row.get("pose_Ry") if row.get("head_pose_valid") else None
    yaw = abs(yaw) if yaw is not None and np.isfinite(yaw) else None
    mp = any(f"{side}_mp_eyelids_px" in row for side in SIDES)
    results = {}
    face_widths = {side: lid_geometry(eyelids(row, side, False))[0] for side in SIDES}
    for index, side in enumerate(SIDES):
        width, aperture = lid_geometry(eyelids(row, side, mp))
        other = SIDES[1-index]
        source = "mediapipe_eyelids" if mp else "openface_eyelids"
        status = row.get(f"{side}_eye_visibility") if mp else None
        reasons = []
        if status in FAILURES:
            reasons.append(status)
        if not mp and row.get("tracking_usable") is False:
            reasons.append("face_tracking_unusable")
        if not mp and row.get("landmarks_2d_valid") is False:
            reasons.append("openface_landmarks_unusable")
        if face_size is not None and face_size < 2*MIN_EYE_WIDTH:
            reasons.append("face_scale_too_small")
        if (width is not None and width < MIN_EYE_WIDTH) or (
                width is None and face_widths[side] is not None and face_widths[side] < MIN_EYE_WIDTH):
            reasons.append("eye_corner_width_below_10px")
        if yaw is not None and yaw > .9:
            reasons.append("extreme_profile")
        elif yaw is not None and yaw > .5:
            eye_width, other_width = face_widths[side], face_widths[other]
            if not mp or (eye_width is not None and other_width is not None and eye_width < .65*other_width):
                reasons.append("oblique_eye_fit_unreliable")
        if aperture is None and width is not None and width >= MIN_EYE_WIDTH:
            reasons.append("invalid_eyelid_geometry")
        if reasons:
            state = "eye_unobservable"
        elif aperture is None or face_size is None:
            state = "eye_unknown"
            reasons.append("missing_eyelid_measurement" if aperture is None else "missing_face_scale")
        else:
            state = "eye_unknown"
        results[side] = {"state":state,"reasons":reasons,"source":source,"source_status":status,
                         "eye_width_px":width,"openface_eye_width_px":face_widths[side],
                         "face_min_dimension_px":face_size,"aperture":aperture if not reasons else None}
    return results


def blink_qc(rows, source_events=()):
    """Classify each eye; require observed, contiguous open/closed/open evidence for events."""
    eyes = [assess_eyes(r) for r in rows]
    cfg = EYE_RULES
    candidates = []
    for side in SIDES:
        mask = [v[side]["aperture"] is not None for v in eyes]
        for indices in valid_blocks(rows, mask):
            times = [rows[i]["movie_timestamp_sec"] for i in indices]
            ends = [rows[i]["source_next_timestamp_sec"] for i in indices]
            aperture = median_filter(np.array(times), [eyes[i][side]["aperture"] for i in indices],
                                     cfg["median_radius_sec"])
            ref = float(np.quantile(aperture, cfg["reference_quantile"]))
            if ref < OPEN_APERTURE:
                for i in indices:
                    eyes[i][side]["reasons"].append("no_reliable_open_reference")
                continue
            opened = aperture >= max(OPEN_APERTURE, cfg["reopen_ratio"]*ref)
            closed = aperture <= cfg["complete_ratio"]*ref
            for i, is_open, is_closed in zip(indices, opened, closed, strict=True):
                if is_open or is_closed:
                    eyes[i][side]["state"] = "eye_open" if is_open else "eye_closed"
                else:
                    eyes[i][side]["reasons"].append("ambiguous_aperture")
            for a,b in hysteresis(times, ends, 1-aperture/ref, 1-cfg["partial_ratio"], 1-cfg["reopen_ratio"]):
                low_count = sum(aperture[a:b] <= cfg["partial_ratio"]*ref)
                if (a == 0 or b == len(indices) or not opened[a-1] or not opened[b]
                        or not any(closed[a:b]) or low_count < cfg["min_frames"]
                        or not cfg["min_duration_sec"] <= ends[b-1]-times[a] <= MAX_BLINK_S):
                    continue
                candidates.append((side, indices[a:b], {indices[j] for j in range(a,b) if closed[j]}))

    events = []
    paired = set()
    for i,(side, span, closure) in enumerate(candidates):
        if i in paired:
            continue
        support = {side:span}
        for j in range(i+1, len(candidates)):
            other, other_span, other_closure = candidates[j]
            union_start = min(span[0],other_span[0])
            union_end = max(span[-1],other_span[-1])
            union_duration = rows[union_end]["source_next_timestamp_sec"]-rows[union_start]["movie_timestamp_sec"]
            if j not in paired and other != side and closure & other_closure and union_duration <= MAX_BLINK_S:
                support[other] = other_span
                paired.add(j)
                break
        a = min(v[0] for v in support.values())
        b = max(v[-1] for v in support.values())
        start, end = rows[a]["global_frame_index"], rows[b]["global_frame_index"]+1
        references = sorted(((min(end,e["end_frame"])-max(start,e["start_frame"]), e["event_id"])
                             for e in source_events if e["event_type"] == "eye_closure_candidate"
                             and start < e["end_frame"] and e["start_frame"] < end), reverse=True)
        events.append({"type":"blink_candidate", "start_frame":start, "end_frame":end,
                       "start_s":rows[a]["movie_timestamp_sec"], "end_s":rows[b]["source_next_timestamp_sec"],
                       "source_event_id":references[0][1] if references else None,
                       "source_event_ids":[r[1] for r in references],
                       "eye_support":sorted(support), "cycle_evidence":"observed_open_closed_open",
                       "eye_intervals":{s:[rows[v[0]]["global_frame_index"], rows[v[-1]]["global_frame_index"]+1]
                                        for s,v in support.items()}})
    return eyes, sorted(events, key=lambda e:e["start_s"])
