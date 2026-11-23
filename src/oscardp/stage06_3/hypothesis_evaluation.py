"""Evaluate gaze target hypotheses after a human review has been filled in."""

import csv

from .hypotheses import PHYSICAL_TARGETS, STATE_KEYS, ranked_targets

METHODS = ("visual", "geometry", "fused")


def read_human_labels(path, states):
    known = {row["state_id"]: row for row in states}
    labels = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not set((*STATE_KEYS, "human_target")) <= set(reader.fieldnames or []):
            raise ValueError("Review CSV is missing its state identifiers or human_target")
        for row in reader:
            state_id = row["state_id"]
            if (state_id not in known or state_id in labels
                    or any(row[key] != known[state_id][key] for key in STATE_KEYS)):
                raise ValueError(f"Invalid or duplicate review state: {state_id}")
            label = (row["human_target"] or "").strip().upper()
            if label and label not in (*PHYSICAL_TARGETS, "UNCLEAR"):
                raise ValueError(f"Invalid human target {label}: {state_id}")
            labels[state_id] = label
    return labels


def accuracy_coverage(predictions, labels):
    predicted = {state: target for state, target in predictions.items() if target in PHYSICAL_TARGETS}
    correct = sum(target == labels[state] for state, target in predicted.items())
    return {
        "predicted": len(predicted),
        "correct": correct,
        "top1_accuracy": correct / len(predicted) if predicted else None,
        "prediction_coverage": len(predicted) / len(labels),
    }


def evaluate_human_labels(prediction_rows, human_labels):
    labels = {state: label for state, label in human_labels.items() if label in PHYSICAL_TARGETS}
    result = {
        "labeled_states": len(labels),
        "unclear_labels": sum(label == "UNCLEAR" for label in human_labels.values()),
        "blank_labels": sum(not label for label in human_labels.values()),
    }
    if not labels:
        return {**result, "status": "awaiting_evaluable_human_labels"}
    methods = {}
    for method in METHODS:
        rows = {row["state_id"]: row for row in prediction_rows[method] if row["state_id"] in labels}
        if rows.keys() != labels.keys():
            raise ValueError(f"Missing {method} predictions for labeled states")
        preferred = {state: row["preferred_target"] for state, row in rows.items()}
        supported = {state: row["preferred_target"] for state, row in rows.items()
                     if row["decision_status"] == "supported_hypothesis"}
        non_context = {state: target for state, target in preferred.items()
                       if rows[state]["evidence_level"] != "context_only"}
        selected = {state: row["selected_target"] for state, row in rows.items()}
        covered = sum(labels[state] in ranked_targets(row) for state, row in rows.items())
        points = {name: accuracy_coverage(choices, labels) for name, choices in (
            ("all_preferred", preferred), ("non_context_preferred", non_context),
            ("supported_hypothesis", supported), ("selected", selected)
        )}
        methods[method] = {
            **points["all_preferred"],
            "candidate_hits": covered,
            "candidate_recall": covered / len(labels),
            "accuracy_coverage_tradeoff": points,
        }
    differences = {}
    for first, second in (("geometry", "visual"), ("fused", "visual"), ("fused", "geometry")):
        measures = {}
        for metric in ("top1_accuracy", "candidate_recall", "prediction_coverage"):
            a, b = methods[first][metric], methods[second][metric]
            measures[metric] = a - b if a is not None and b is not None else None
        differences[f"{first}_minus_{second}"] = measures
    return {**result, "status": "evaluated", "methods": methods, "method_differences": differences}
