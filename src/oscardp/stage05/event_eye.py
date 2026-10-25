"""Relative bilateral eye geometry cycles; AU45 is supporting, not defining evidence."""
import numpy as np

from .event_records import candidate, rejection, trace
from .event_series import eye_geometry, segments, median_filter, hysteresis


def detect_eyes(rows,dynamic,config):
    cfg = config["eye"]
    geometry = [eye_geometry(r) for r in rows]
    raw = np.array([v if v is not None else [np.nan,np.nan] for v in geometry])
    mask = np.array([r["landmarks_2d_valid"] and r["tracking_usable"] for r in rows]) & np.isfinite(raw).all(axis=1)
    processed = np.full(raw.shape,np.nan)
    events,rejected = [],[]
    reference = np.quantile(raw[mask],cfg["reference_quantile"],axis=0) if mask.any() else np.array([np.nan,np.nan])
    info = {"reference_ear":reference.tolist() if np.isfinite(reference).all() else None,"reference_quantile":cfg["reference_quantile"],
            "reference_status":"within-Track descriptive upper quantile; eye opening is not human verified"}
    if not mask.any() or not np.isfinite(reference).all() or min(reference)<cfg["min_reference_ear"]:
        info["status"] = "insufficient_evidence_for_eye_geometry_reference"
        return [],[],trace(rows,"eye_geometry",raw,processed,np.zeros(len(rows),bool)),info
    for indices in segments(rows,mask,config["series"]["max_gap_sec"]):
        times = np.array([rows[i]["movie_timestamp_sec"] for i in indices])
        ends = np.array([rows[i]["source_next_timestamp_sec"] for i in indices])
        ratios = median_filter(times,raw[indices]/reference,cfg["median_radius_sec"])
        processed[indices] = ratios
        score = 1-ratios.min(axis=1)
        for a,b in hysteresis(times,ends,score,1-cfg["partial_ratio"],1-cfg["reopen_ratio"]):
            reasons = []
            closed_count = int(np.sum(ratios[a:b].min(axis=1)<=cfg["partial_ratio"]))
            if closed_count<cfg["min_frames"] or ends[b-1]-times[a]<cfg["min_duration_sec"]:
                reasons.append("single_frame_or_insufficient_closure_duration")
            if reasons:
                rejected.append(rejection(rows,indices,a,b,"eye_closure_candidate","eye_geometry",reasons))
                continue
            apex = a+int(np.argmax(score[a:b]))
            event = candidate(rows,indices,a,b,"eye_closure_candidate","eye_geometry","landmarks_2d_valid",config,apex)
            reopened = b<len(indices) and bool(np.all(ratios[b]>=cfg["reopen_ratio"]))
            previously_open = a>0 and bool(np.all(ratios[a-1]>=cfg["reopen_ratio"]))
            bilateral_complete = bool(np.any(np.max(ratios[a:b],axis=1)<=cfg["complete_ratio"]))
            classification = "complete_closure_candidate" if bilateral_complete else "partial_closure"
            minima = ratios[a:b].min(axis=0)
            event["measurements"] = {**info,"classification":classification,"cycle_evidence":"closure_and_reopening_observed" if reopened and previously_open else "incomplete_uncertain",
                 "left_eye_min_relative_opening":float(minima[0]),"right_eye_min_relative_opening":float(minima[1]),
                 "eye_order":"image-left (36..41), image-right (42..47)","reopening_observed":reopened,
                 "preceding_opening_observed":previously_open,"duration_sec":float(ends[b-1]-times[a]),
                 "closed_support_frames":closed_count,"verified_blink":False}
            support = {}
            for name,data in [("static",rows),("dynamic",dynamic)]:
                if data is None: continue
                good = [data[i] for i in indices[a:b] if data[i]["au_valid"]]
                intensity = [r["AU45_r"] for r in good if r.get("AU45_r") is not None]
                presence = [r["AU45_c"] for r in good if r.get("AU45_c") is not None]
                support[name] = {"AU45_r_max":max(intensity) if intensity else None,
                                 "AU45_c_fraction":float(np.mean(presence)) if presence else None}
            event["evidence"]["au45_support"] = support
            primary = support.get("static",{}).get("AU45_r_max")
            if primary is None or primary<cfg["au45_support_floor"]:
                event["qc_flags"].append("eye_geometry_au45_conflict")
            if "dynamic" in support and primary is not None:
                other = support["dynamic"]["AU45_r_max"]
                if other is not None and abs(other-primary)>config["comparison"]["eye_mode_peak_delta"]:
                    event["qc_flags"].append("eye_au45_mode_disagreement")
            if not reopened or not previously_open:
                event["qc_flags"].append("incomplete_eye_cycle_or_insufficient_context")
                if not previously_open: event["onset_frame"] = None
                if not reopened: event["offset_frame"] = None
            events.append(event)
    # AU-only activation intervals are review evidence, not additional eye events.
    info["au45_without_geometry_intervals"] = []
    for name,data in [("static",rows),("dynamic",dynamic)]:
        if data is None: continue
        supported = [r["au_valid"] and r.get("AU45_r") is not None and r["AU45_r"]>=cfg["au45_support_floor"] for r in data]
        for indices in segments(rows,supported,config["series"]["max_gap_sec"]):
            if not any(e["start_frame"]<=rows[i]["global_frame_index"]<e["end_frame"] for e in events for i in indices):
                info["au45_without_geometry_intervals"].append({"mode":name,"start_frame":rows[indices[0]]["global_frame_index"],
                     "end_frame":rows[indices[-1]]["global_frame_index"]+1,"status":"insufficient_evidence_for_eye_closure",
                     "review_needed":True,"human_verified":False})
    info["status"] = "relative_geometry_processed"
    return events,rejected,trace(rows,"eye_geometry",raw,processed,mask),info
