"""Deterministic strata with temporal diversity; never rank by signal amplitude."""
from collections import Counter

VERSION = "stratified_temporal_diversity_v1"


def select_pilot(queue, target=48):
    if target < 40 or target > 50:
        raise ValueError("Pilot size must be 40–50")
    chosen = {}

    def add(row, reason):
        chosen.setdefault(row["review_id"], {**row, "selection_reasons": []})["selection_reasons"].append(reason)

    def diversity(row):
        selected = list(chosen.values())
        same = [r for r in selected if r["shot_id"] == row["shot_id"]]
        a, b = int(row["start_frame"]), int(row["end_frame"])
        overlap = sum(max(0, min(b, int(r["end_frame"]))-max(a, int(r["start_frame"]))) /
                      max(1, b-a) for r in same)
        return (sum(r["feature"] == row["feature"] for r in selected), overlap,
                len(same), row["shot_id"], a, row["review_id"])

    def take(rows, count, reason):
        for _ in range(count):
            remaining = [r for r in rows if r["review_id"] not in chosen]
            if not remaining:
                break
            add(min(remaining, key=diversity), reason)

    for row in queue:
        if row["event_id"] and row["event_type"] in {
            "eye_closure_candidate", "head_orientation_change", "camera_relative_gaze_change"
        }:
            add(row, "complete_eye_head_gaze_stratum")
    for kind, count in [("rejected_interval", 6), ("algorithm_negative_interval", 5),
                        ("au45_without_geometry_support", 4)]:
        take([r for r in queue if r["sample_kind"] == kind], count, kind)
    for reason in ["au_mode_disagreement", "censored_event", "near_shot_boundary",
                   "low_quality_neighbor_or_invalid_segment_edge", "source_measurement_spike_review"]:
        rows = [r for r in queue if reason in r["reasons"].split(";")]
        if rows and not any(r["review_id"] in chosen for r in rows):
            take(rows, 1, "required_qc_stratum:"+reason)
    take([r for r in queue if r["event_id"] and r["event_type"] == "au_activation"],
         max(0, target-len(chosen)), "AU_feature_and_temporal_diversity")
    take(queue, max(0, target-len(chosen)), "remaining_temporal_diversity")
    if len(chosen) > target:
        raise ValueError("Required strata exceed pilot size")
    result = sorted(chosen.values(), key=lambda r: r["review_id"])
    for r in result:
        r["selection_reasons"] = ";".join(r["selection_reasons"])
        r["sampling_algorithm"] = VERSION
    return result, {"algorithm_version": VERSION, "random_seed": None, "random_process": False,
                    "target": target, "selected": len(result),
                    "policy": "All eye/head/gaze; 6 rejected, 5 negative, 4 unsupported AU45; QC coverage; AU feature/temporal diversity. No amplitude ranking.",
                    "by_sample_kind": dict(Counter(r["sample_kind"] for r in result)),
                    "by_event_type": dict(Counter(r["event_type"] for r in result)),
                    "by_shot": dict(Counter(r["shot_id"] for r in result))}
