"""Publish or evaluate the Stage06.3 v1.3 Bugonia target hypothesis pilot."""

import argparse
import json
import os
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory

from oscardp.shots.transnet import sha256_file
from oscardp.stage06_2.evidence_replay import write_json
from oscardp.stage06_2.inputs import read
from .hypotheses import (
    PHYSICAL_TARGETS,
    STATE_KEYS,
    VISUAL_LEVELS,
    fused_prediction,
    geometry_prediction,
    visual_prediction,
)
from .hypothesis_evaluation import METHODS, evaluate_human_labels, read_human_labels
from .hypothesis_review import grouped_predictions, write_comparison, write_review
from .inference import CHARACTERS
from .look_pilot import MOVIE, SEQUENCE, VERSION as LOOK_STATE_VERSION
from .look_review import SOURCE
from .look_targets import spatial_inputs

VERSION = "stage06.3.target_hypotheses.v1.3"
VISUAL_VERSION = "stage06_3_visual_observations_v1_3"


def validate_visual(states, packet, observations, visual_dir):
    frames = packet["states"]
    expected = {state["state_id"] for state in states}
    if frames.keys() != expected or observations.keys() != expected or packet["source"] != SOURCE:
        raise ValueError("Visual packet and observations must cover the exact frozen Look State IDs")
    all_frames = {}
    for state in states:
        state_id = state["state_id"]
        local = frames[state_id]
        if not local or not local[0]["within_state"]:
            raise ValueError(f"No in-state frame for {state_id}")
        for frame in local:
            number = frame["global_frame_index"]
            timestamp = frame["movie_timestamp_sec"]
            within = (state["start_frame"] <= number < state["end_frame"]
                      and state["start_s"] <= timestamp < state["end_s"])
            if (any(frame[key] != state[key] for key in STATE_KEYS)
                    or frame["file"] != f"frames/frame_{number}.jpg" or frame["source"] != SOURCE
                    or frame["within_state"] != within or (not within and frame["role"] != "pre_motion")):
                raise ValueError(f"Frame identity or interval mismatch: {state_id}, {number}")
            reference = (frame["sha256"], timestamp, frame["shot_id"])
            if number in all_frames:
                if all_frames[number] != reference:
                    raise ValueError(f"Conflicting source metadata for frame {number}")
            elif sha256_file(visual_dir / frame["file"]) != frame["sha256"]:
                raise ValueError(f"Source image hash mismatch: {number}")
            all_frames[number] = reference
    if len(all_frames) != packet["unique_original_frames"]:
        raise ValueError("Unexpected original frame count")
    for state_id, observation in observations.items():
        local = frames[state_id]
        local_numbers = list(dict.fromkeys(frame["global_frame_index"] for frame in local))
        preferred = observation["preferred_target"]
        alternatives = observation["alternative_targets"]
        labels = alternatives + ([preferred] if preferred is not None else [])
        gazer = observation["gazer"]
        level = observation["evidence_level"]
        if (observation["local_frames"] != local_numbers
                or not set(observation["reference_frames"]) <= all_frames.keys()
                or len(labels) != len(set(labels)) or not set(labels) <= set(PHYSICAL_TARGETS) - {gazer}
                or gazer not in (*CHARACTERS, None) or level not in VISUAL_LEVELS
                or (level == "insufficient" and preferred is not None)
                or not observation["reason"].strip()):
            raise ValueError(f"Invalid independent visual observation: {state_id}")
        if set(labels) & {"AWAY", "OTHER"} and not observation.get("nonperson_cue", "").strip():
            raise ValueError(f"Nonperson candidates require a visible cue: {state_id}")
        if level == "visual_endpoint":
            in_state = {frame["global_frame_index"] for frame in local if frame["within_state"]}
            endpoint_frames = set(observation.get("endpoint_frames", []))
            if preferred is None or alternatives or len(endpoint_frames) < 2 or not endpoint_frames <= in_state:
                raise ValueError(f"Endpoint support requires repeated in-state evidence: {state_id}")
    return all_frames


def source(path, output, **metadata):
    return {"file": os.path.relpath(path, start=output), "sha256": sha256_file(path), **metadata}


def run(args):
    output = args.output_dir
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    states_path = args.baseline / "look_states.json"
    mapping_path = args.baseline / "character_track_mapping.json"
    visual_dir = args.baseline / "visual_evidence"
    packet_path = visual_dir / "visual_packet.json"
    frozen, mapping, packet, layout, orientation, visual = (
        read(path) for path in (states_path, mapping_path, packet_path, args.layout, args.orientation, args.observations)
    )
    if any((doc["movie_id"], doc["sequence_id"]) != (MOVIE, SEQUENCE)
           for doc in (frozen, mapping, layout, orientation, visual)):
        raise ValueError("Movie or sequence mismatch")
    if frozen["schema_version"] != LOOK_STATE_VERSION or visual["schema_version"] != VISUAL_VERSION:
        raise ValueError("Expected frozen v1.2 Look States and independent v1.3 visual observations")
    if sha256_file(args.layout) != orientation["layout_sha256"]:
        raise ValueError("Layout changed since the camera constraints were reviewed")
    frozen_hashes = frozen["input_sha256"]
    orientation_hashes = {digest for path, digest in frozen_hashes.items()
                          if Path(path).name == "camera_orientation_constraints.json"}
    if sha256_file(args.orientation) not in orientation_hashes:
        raise ValueError("Camera constraints changed since the Look States were frozen")
    if sha256_file(packet_path) != frozen_hashes["visual_evidence/visual_packet.json"]:
        raise ValueError("Original frame packet changed since the Look States were frozen")
    states = frozen["states"]
    if len(states) != len({state["state_id"] for state in states}):
        raise ValueError("Duplicate frozen Look State ID")
    tracks = {track["track_id"]: track for track in mapping["tracks"] if track["measured"]}
    for state in states:
        track = tracks[state["track_id"]]
        if (track["shot_id"], track["character"]) != (state["shot_id"], state["character"]):
            raise ValueError(f"Track mapping does not match frozen state: {state['state_id']}")
    observations = visual["observations"]
    all_frames = validate_visual(states, packet, observations, visual_dir)
    camera_inputs = spatial_inputs(layout, orientation)
    frozen_source = source(states_path, output, schema_version=frozen["schema_version"])
    geometry_sources = {
        "frozen_states": frozen_source,
        "layout": source(args.layout, output),
        "orientation": source(args.orientation, output),
        "track_mapping": source(mapping_path, output),
    }
    visual_sources = {
        "frame_packet": source(packet_path, output),
        "observations": {"file": "visual_observations.json", "sha256": sha256_file(args.observations)},
    }
    common = {"schema_version": VERSION, "movie_id": MOVIE, "sequence_id": SEQUENCE, "human_verified": False}
    documents = {
        "visual": {**common, "method": "visual_only", "sources": visual_sources,
                   "visual_assessor": visual["method"], "visual_protocol": visual["protocol"],
                   "original_frames": {str(number): {
                       "file": f"review/frames/frame_{number}.jpg", "sha256": frame_hash,
                       "movie_timestamp_sec": time, "shot_id": shot,
                   } for number, (frame_hash, time, shot) in sorted(all_frames.items())}, "predictions": []},
        "geometry": {**common, "method": "geometry_only", "sources": geometry_sources, "predictions": []},
        "fused": {**common, "method": "visual_plus_geometry", "sources": {"frozen_states": frozen_source},
                  "predictions": []},
    }
    for state in states:
        ids = {key: state[key] for key in STATE_KEYS}
        measured = {key: state[key] for key in (
            "start_frame", "end_frame", "start_s", "end_s", "direction_source", "look_direction",
            "qc_status", "gaze_evidence", "eye_motion_evidence", "head_motion_evidence"
        )}
        visual_row = visual_prediction(observations[state["state_id"]])
        geometry_row = geometry_prediction(state, tracks[state["track_id"]], camera_inputs[state["shot_id"]])
        fused_row = fused_prediction(state, visual_row, geometry_row)
        documents["visual"]["predictions"].append({**ids, **visual_row})
        documents["geometry"]["predictions"].append({**ids, **measured, **geometry_row})
        documents["fused"]["predictions"].append({**ids, **measured, **fused_row})

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as stage_dir:
        staged = Path(stage_dir) / output.name
        staged.mkdir()
        copy2(args.observations, staged / "visual_observations.json")
        for method in METHODS:
            if method == "fused":
                documents[method]["sources"].update({
                    name: source(staged / f"{name}_predictions.json", staged) for name in METHODS[:2]
                })
            write_json(staged / f"{method}_predictions.json", documents[method])
        predictions = grouped_predictions(documents)
        # The old fused candidates are read only for the retrospective comparison.
        old_path = args.baseline / "gaze_target_candidates.json"
        old = read(old_path)["look_states"]
        write_comparison(staged, states, predictions, len(all_frames), old, source(old_path, output))
        write_review(staged, states, tracks, packet, visual_dir, predictions)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        staged.rename(output)
    print(f"{len(states)} frozen Look States, {len(all_frames)} reviewed frames → {output}")


def evaluate(args):
    predictions = {method: read(args.pilot / f"{method}_predictions.json")["predictions"] for method in METHODS}
    path = args.labels if args.labels else args.pilot / "human_review.csv"
    labels = read_human_labels(path, predictions["fused"])
    print(json.dumps(evaluate_human_labels(predictions, labels), ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    pilot = commands.add_parser("run")
    pilot.add_argument("--baseline", required=True, type=Path)
    pilot.add_argument("--layout", required=True, type=Path)
    pilot.add_argument("--orientation", required=True, type=Path)
    pilot.add_argument("--observations", required=True, type=Path)
    pilot.add_argument("--output-dir", required=True, type=Path)
    pilot.set_defaults(handler=run)
    metrics = commands.add_parser("evaluate")
    metrics.add_argument("--pilot", required=True, type=Path)
    metrics.add_argument("--labels", type=Path)
    metrics.set_defaults(handler=evaluate)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
