"""Descriptive AU hysteresis candidates, including censored sustained observations."""
import numpy as np

from .event_series import segments, median_filter, hysteresis
from .event_records import candidate, rejection, trace


def detect_au(rows,feature,config):
    cfg = config["au"]
    raw = np.array([r[feature] if r[feature] is not None else np.nan for r in rows],float)
    mask = np.array([r["au_valid"] for r in rows]) & np.isfinite(raw)
    smoothed = np.full(len(rows),np.nan)
    events,rejected = [],[]
    for indices in segments(rows,mask,config["series"]["max_gap_sec"]):
        times = np.array([rows[i]["movie_timestamp_sec"] for i in indices])
        ends = np.array([rows[i]["source_next_timestamp_sec"] for i in indices])
        values = median_filter(times,raw[indices],cfg["median_radius_sec"])
        smoothed[indices] = values
        baseline = float(np.quantile(values,cfg["baseline_quantile"]))
        high,low = max(cfg["high_floor"],baseline+cfg["high_above_baseline"]),max(cfg["low_floor"],baseline+cfg["low_above_baseline"])
        sustained = np.median(values)>=cfg["sustained_floor"] and np.ptp(values)<cfg["min_prominence"]
        runs = [(0,len(indices))] if sustained else hysteresis(times,ends,values,high,low,cfg["merge_gap_sec"])
        for a,b in runs:
            duration = ends[b-1]-times[a]
            prominence = float(max(values[a:b])-baseline)
            reasons = []
            if b-a<cfg["min_frames"] or duration<(cfg["sustained_min_duration_sec"] if sustained else cfg["min_duration_sec"]): reasons.append("minimum_duration_or_sample_count")
            if not sustained and prominence<cfg["min_prominence"]: reasons.append("minimum_prominence")
            if reasons:
                rejected.append(rejection(rows,indices,a,b,"au_activation",feature,reasons))
                continue
            peak = a+int(np.argmax(values[a:b]))
            event = candidate(rows,indices,a,b,"au_activation",feature,"au_valid",config,apex=None if sustained else peak)
            event["measurements"] = {"phase":"sustained_observation" if sustained else "activation_onset_peak_decline_candidate",
                "baseline":baseline,"baseline_status":"segment quantile; not verified neutral expression",
                "high_threshold":high,"low_threshold":low,"prominence":prominence,"duration_sec":float(duration),
                "raw_min":float(min(raw[indices[a:b]])),"raw_max":float(max(raw[indices[a:b]])),
                "processed_min":float(min(values[a:b])),"processed_max":float(max(values[a:b])),
                "speech_sensitive_feature":feature in {"AU25_r","AU26_r"},"speech_verified":False}
            presence = feature[:-2]+"_c"
            present = [rows[i].get(presence) for i in indices[a:b] if rows[i].get(presence) is not None]
            event["evidence"]["presence"] = {"field":presence if present else None,"observed_fraction":float(np.mean(present)) if present else None,
                                               "role":"additional model evidence only"}
            if sustained:
                event["onset_frame"] = event["offset_frame"] = None
            else:
                x,y = peak,peak+1
                while x>a and values[x-1]>=values[peak]-cfg["plateau_tolerance"]: x-=1
                while y<b and values[y]>=values[peak]-cfg["plateau_tolerance"]: y+=1
                event["apex_plateau"] = {"start_frame":rows[indices[x]]["global_frame_index"],"end_frame":rows[indices[y-1]]["global_frame_index"]+1}
                if y-x>1: event["apex_frame"] = None
                event["measurements"]["declining_phase"] = {"start_frame":rows[indices[y-1]]["global_frame_index"],"end_frame":event["end_frame"]} if y<b and values[peak]-values[b-1]>=cfg["min_prominence"] else None
                event["measurements"]["multi_peak_policy"] = "Remain one event until a sufficient below-low interval; main maximum/plateau retained"
            events.append(event)
    return events,rejected,trace(rows,feature,raw,smoothed,mask)
