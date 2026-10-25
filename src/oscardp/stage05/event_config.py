"""Research pilot rules declared before running the movie; no fitted event counts."""
import copy
import hashlib
import json
import math

DEFAULTS = {
    "schema_version":"stage05.5.thresholds.v1", "algorithm_version":"temporal_candidates_v1",
    "calibration_status":"uncalibrated research parameters; no human event-annotation calibration",
    "series":{"max_gap_sec":.25,"boundary_review_sec":.12},
    "au":{"median_radius_sec":.08,"baseline_quantile":.2,"high_floor":.9,"low_floor":.45,
          "high_above_baseline":.55,"low_above_baseline":.25,"min_prominence":.45,
          "min_duration_sec":.12,"min_frames":3,"merge_gap_sec":.08,
          "sustained_floor":1.2,"sustained_min_duration_sec":.25,"plateau_tolerance":.05},
    "eye":{"median_radius_sec":.025,"reference_quantile":.8,"min_reference_ear":.05,
           "partial_ratio":.70,"complete_ratio":.35,"reopen_ratio":.85,
           "min_duration_sec":.045,"min_frames":2,"au45_support_floor":.8},
    "head":{"median_radius_sec":.06,"speed_high":.25,"speed_low":.10,
            "min_duration_sec":.12,"min_frames":3,"min_displacement_rad":.10,"merge_gap_sec":.08},
    "gaze":{"median_radius_sec":.06,"speed_high":.45,"speed_low":.18,
            "min_duration_sec":.08,"min_frames":2,"min_displacement_rad":.10,"merge_gap_sec":.04,
            "head_confound_rad":.07},
    "comparison":{"pearson_below":.5,"rmse_above":.5,"eye_mode_peak_delta":.5},
}


def config_hash(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def load_config(path=None):
    config = copy.deepcopy(DEFAULTS)
    if path:
        supplied = json.loads(path.read_text())
        for key,value in supplied.items():
            if key not in config: raise ValueError(f"Unknown event config key: {key}")
            if isinstance(config[key],dict):
                if not isinstance(value,dict) or set(value)-set(config[key]): raise ValueError(f"Unknown config parameters: {key}")
                config[key].update(value)
            elif value != config[key]: raise ValueError(f"Unsupported event algorithm/version: {key}")
    for section in [v for v in config.values() if isinstance(v,dict)]:
        if any(type(v) not in {float,int} or not math.isfinite(v) or v<0 for v in section.values()):
            raise ValueError("Event parameters must be finite, nonnegative numbers")
    for name in ["au","head","gaze","eye"]:
        if type(config[name]["min_frames"]) is not int or config[name]["min_frames"]<2 or config[name]["min_duration_sec"]<=0:
            raise ValueError("Event minimum duration/sample count invalid")
    au,eye = config["au"],config["eye"]
    if not (0<au["baseline_quantile"]<1 and au["high_floor"]>au["low_floor"]
            and au["high_above_baseline"]>au["low_above_baseline"]): raise ValueError("AU hysteresis/baseline invalid")
    if not 0<eye["complete_ratio"]<eye["partial_ratio"]<eye["reopen_ratio"]<1 or not 0<eye["reference_quantile"]<1:
        raise ValueError("Eye ratio/reference thresholds invalid")
    if config["series"]["max_gap_sec"]<=0: raise ValueError("Invalid max time gap")
    for name in ["head","gaze"]:
        if config[name]["speed_high"]<=config[name]["speed_low"]: raise ValueError("Invalid motion hysteresis")
    return config
