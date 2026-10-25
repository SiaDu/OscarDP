"""Deterministic orchestration of four candidate families and mode sensitivity."""
from collections import Counter

from .event_au import detect_au
from .event_eye import detect_eyes
from .event_motion import detect_motion
from .event_records import assign_ids
from .event_series import segments


def detect(by,comparison,config,compare_dynamic):
    events,rejected,traces,sensitivity = [],[],[],[]
    eye_info = {}
    for tid in sorted({t for t,m in by}):
        rows = by[tid,"static"]
        dynamic = by[tid,"dynamic"] if compare_dynamic else None
        for feature in sorted(n for n in rows[0] if n.startswith("AU") and n.endswith("_r")):
            found,bad,series = detect_au(rows,feature,config)
            stats = comparison[tid,feature]
            disagreement = compare_dynamic and ((stats["paired_pearson"] and float(stats["paired_pearson"])<config["comparison"]["pearson_below"])
                                                or (stats["paired_rmse"] and float(stats["paired_rmse"])>config["comparison"]["rmse_above"]))
            for event in found:
                event["evidence"]["mode_comparison"] = {"scope":"whole Track/AU paired usable frames",**stats} if compare_dynamic else {"status":"not_requested"}
                if disagreement: event["qc_flags"].append("au_mode_disagreement")
            events.extend(found)
            rejected.extend(bad)
            traces.extend(series)
            if dynamic is not None:
                alternative,alt_bad,alt_series = detect_au(dynamic,feature,config)
                sensitivity.append({"track_id":tid,"feature":feature,"formal_static_count":len(found),
                    "dynamic_sensitivity_count":len(alternative),"mode_disagreement":bool(disagreement),
                    "dynamic_candidates":alternative,"dynamic_rejected_count":len(alt_bad),
                    "is_formal_event_stream":False,
                    "interval_overlap_pairs":[{"static_interval":[a["start_frame"],a["end_frame"]],
                                               "dynamic_interval":[b["start_frame"],b["end_frame"]]}
                         for a in found for b in alternative if max(a["start_frame"],b["start_frame"])<min(a["end_frame"],b["end_frame"])]})
                traces.extend(alt_series)
        found,bad,series,info = detect_eyes(rows,dynamic,config)
        events.extend(found)
        rejected.extend(bad)
        traces.extend(series)
        eye_info[tid] = info
        for kind in ["head","gaze"]:
            found,bad,series = detect_motion(rows,kind,config)
            events.extend(found)
            rejected.extend(bad)
            traces.extend(series)
    # Explicitly identify temporal overlap with independently detected head changes.
    for event in events:
        if event["event_type"]=="camera_relative_gaze_change":
            overlap = [e for e in events if e["event_type"]=="head_orientation_change" and e["track_id"]==event["track_id"]
                       and max(e["start_frame"],event["start_frame"])<min(e["end_frame"],event["end_frame"])]
            event["evidence"]["overlapping_head_intervals"] = [[e["start_frame"],e["end_frame"]] for e in overlap]
            if overlap: event["qc_flags"].append("head_gaze_simultaneous_change")
    return events,rejected,traces,sensitivity,eye_info


def validate_events(events,by):
    for event in events:
        if event["extraction_mode"]!="static" or event["human_verified"] or event["review_status"]!="algorithm_candidate":
            raise ValueError("Formal stream must contain only static unverified candidates")
        lookup = {r["global_frame_index"]:r for r in by[event["track_id"],"static"]}
        selected = [lookup[i] for i in range(event["start_frame"],event["end_frame"])]
        mask = event["evidence"]["source_measurement_qc"]["mask"]
        if not all(r[mask] for r in selected): raise ValueError("Event bridges invalid data")
        if selected[0]["movie_timestamp_sec"]!=event["start_sec"] or selected[-1]["source_next_timestamp_sec"]!=event["end_sec"]:
            raise ValueError("Event timestamp/frame reference mismatch")
        for name in ["onset_frame","apex_frame","offset_frame"]:
            if event[name] is not None and (event[name] not in lookup or not lookup[event[name]][mask]):
                raise ValueError("Phase points to invalid/nonexistent source frame")
        plateau = event["apex_plateau"]
        if plateau and not event["start_frame"]<=plateau["start_frame"]<plateau["end_frame"]<=event["end_frame"]:
            raise ValueError("Apex plateau outside event")


def review_queue(events,rejected,by,config,eye_info):
    review = []
    for event in events:
        review.append({"event_id":event["event_id"],"track_id":event["track_id"],"shot_id":event["shot_id"],
                       "event_type":event["event_type"],"feature":event["feature"],"start_frame":event["start_frame"],
                       "end_frame":event["end_frame"],"sample_kind":"ambiguous_candidate" if event["qc_flags"] else "low_flag_candidate",
                       "reasons":";".join(event["qc_flags"]) or "no_configured_qc_flags; still unverified",
                       "priority":1 if event["qc_flags"] else 3})
    for bad in rejected:
        review.append({"event_id":None,**{k:bad[k] for k in ["track_id","shot_id","event_type","feature","start_frame","end_frame"]},
                       "sample_kind":"rejected_interval","reasons":";".join(bad["reasons"]),"priority":2})
    # Algorithm-negative intervals per Track/family, explicitly not negative ground truth.
    for tid in sorted({t for t,m in by}):
        rows = by[tid,"static"]
        for typ,mask in [("au_activation","au_valid"),("eye_closure_candidate","landmarks_2d_valid"),
                         ("head_orientation_change","head_pose_valid"),("camera_relative_gaze_change","gaze_valid")]:
            good = [r[mask] and not any(e["track_id"]==tid and e["event_type"]==typ and e["start_frame"]<=r["global_frame_index"]<e["end_frame"] for e in events) for r in rows]
            spans = segments(rows,good,config["series"]["max_gap_sec"])
            if spans:
                indices = max(spans,key=lambda ids:rows[ids[-1]]["source_next_timestamp_sec"]-rows[ids[0]]["movie_timestamp_sec"])
                review.append({"event_id":None,"track_id":tid,"shot_id":rows[0]["shot_id"],"event_type":typ,"feature":None,
                               "start_frame":rows[indices[0]]["global_frame_index"],"end_frame":rows[indices[-1]]["global_frame_index"]+1,
                               "sample_kind":"algorithm_negative_interval","reasons":"No retained candidate; not negative ground truth","priority":3})
        for interval in eye_info[tid].get("au45_without_geometry_intervals",[]):
            review.append({"event_id":None,"track_id":tid,"shot_id":rows[0]["shot_id"],"event_type":"eye_closure_candidate",
                           "feature":"AU45","start_frame":interval["start_frame"],"end_frame":interval["end_frame"],
                           "sample_kind":"au45_without_geometry_support","reasons":interval["mode"]+": insufficient evidence","priority":1})
    review.sort(key=lambda r:(r["priority"],r["shot_id"],r["start_frame"],r["event_type"],r["sample_kind"],str(r["feature"])))
    for i,row in enumerate(review,1):
        row.update(review_id=f"review_{i:06d}",reviewer=None,review_status="pending_independent_review",human_verified=False)
    return review


def summary(events,rejected):
    result = {"event_count":len(events),"by_type":dict(Counter(e["event_type"] for e in events)),
              "by_track":{tid:dict(Counter(e["event_type"] for e in events if e["track_id"]==tid)) for tid in sorted({e["track_id"] for e in events})},
              "rejected_interval_count":len(rejected),"qc_flagged_event_count":sum(bool(e["qc_flags"]) for e in events),
              "review_needed_event_count":len(events),"human_verified_event_count":0}
    result["phase_observability"] = {}
    for typ in sorted({e["event_type"] for e in events}):
        selected = [e for e in events if e["event_type"]==typ]
        counts = {"onset":sum(e["onset_frame"] is not None for e in selected),
                  "apex":sum(e["apex_frame"] is not None or e["apex_plateau"] is not None for e in selected),
                  "offset":sum(e["offset_frame"] is not None for e in selected)}
        result["phase_observability"][typ] = {"denominator":len(selected),"counts":counts,
            "ratios":{k:v/len(selected) for k,v in counts.items()},"apex_semantics":"AU maximum/plateau, eye minimum opening or motion peak speed candidate; not annotated action apex"}
    return result
