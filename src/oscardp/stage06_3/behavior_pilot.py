"""Build or evaluate the Stage06.3 v1.4 gaze timeline for Bugonia seq_006."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from oscardp.shots.transnet import sha256_file
from oscardp.stage06_2.evidence_replay import write_json
from oscardp.stage06_2.inputs import read

from .__main__ import MOVIE, SEQUENCE
from .behavior_review import make_clips, write_guide
from .behavior_units import VERSION, build_units, carry_over, previous_by_unit
from .hypothesis_pilot import source
from .look_review import track_boxes
from .timeline import candidate_rows, evaluate_timeline, read_timeline, segments, write_timeline

METHODS = ("fused", "visual", "geometry")
UNIT_LABEL_FIELDS = ("human_type", "human_target", "human_form", "human_certainty", "notes", "v1_3_label",
                     "v1_3_note")


def inputs(args):
    paths = {
        "look_states": args.look_pilot / "look_states.json",
        "look_events": args.look_pilot / "look_direction_events.json",
        "track_mapping": args.look_pilot / "character_track_mapping.json",
        "timeline": args.full_sequence / "sequence_gaze_timeline.json",
        **{method: args.hypothesis_pilot / f"{method}_predictions.json" for method in METHODS},
    }
    docs = {name: read(path) for name, path in paths.items()}
    if any((doc["movie_id"], doc["sequence_id"]) != (MOVIE, SEQUENCE) for doc in docs.values()):
        raise ValueError("Movie or sequence mismatch")
    recorded = docs["fused"]["sources"]
    if any(recorded[name]["sha256"] != sha256_file(paths[key]) for name, key in
           (("frozen_states", "look_states"), ("visual", "visual"), ("geometry", "geometry"))):
        raise ValueError("Hypothesis pilot was not built from these Look States and predictions")
    timeline = [digest for path, digest in docs["look_states"]["input_sha256"].items()
                if path.endswith("sequence_gaze_timeline.json")]
    if timeline != [sha256_file(paths["timeline"])]:
        raise ValueError("Full-sequence timeline changed since the Look States were built")
    states = docs["look_states"]["states"]
    predictions = {m: {row["state_id"]: row for row in docs[m]["predictions"]} for m in METHODS}
    if any(rows.keys() != {s["state_id"] for s in states} for rows in predictions.values()):
        raise ValueError("Predictions must cover exactly the frozen Look States")
    return paths, docs, states, predictions


def v13_labels(path, states):
    """Free-text v1.3 blind labels keyed by Look State; raw text is kept alongside."""
    known = {s["state_id"]: s for s in states}
    carried, unmatched = {}, []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            state = known.get(row["state_id"])
            if state is None or (row["shot_id"], row["track_id"]) != (state["shot_id"], state["track_id"]):
                raise ValueError(f"Unknown v1.3 label row: {row['state_id']}")
            fields = carry_over(row["human_target"], row["human_certainty"])
            if row["human_target"].strip() and fields is None:
                unmatched.append(row["state_id"])
            carried[row["state_id"]] = {**(fields or {}), "v1_3_label": row["human_target"].strip(),
                                        "v1_3_note": row["notes"].strip()}
    return carried, unmatched


def unit_sheet_labels(path, units):
    """Labels from an earlier per-unit review sheet of the same build; edited start times are kept."""
    known = {u["unit_id"]: u for u in units}
    labels, ignored = {}, []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for n, row in enumerate(csv.DictReader(stream), 2):
            uid = (row.get("unit_id") or "").strip()
            if not uid:
                if any((v or "").strip() for v in row.values()):
                    ignored.append(n)
                continue
            if uid not in known or uid in labels:
                raise ValueError(f"Unit sheet line {n}: {uid!r} is unknown or repeated in this build")
            label = {k: (row.get(k) or "").strip() for k in UNIT_LABEL_FIELDS}
            start = float(row["start_s"])
            if abs(start - known[uid]["start_s"]) > 0.0015:  # the sheet rounds to milliseconds
                label["start_s"] = start
            labels[uid] = label
    return labels, ignored


def run(args):
    output = args.output_dir
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if any(output.resolve().is_relative_to(p.resolve()) for p in (args.look_pilot, args.hypothesis_pilot,
                                                                   args.full_sequence)):
        raise ValueError("Output must not be inside an input directory")
    paths, docs, states, predictions = inputs(args)
    tracks = {t["track_id"]: t for t in docs["track_mapping"]["tracks"]}
    shots = docs["timeline"]["shots"]
    units = build_units(shots, tracks, states, docs["look_events"]["events"], predictions)
    if args.unit_labels:
        labels, ignored = unit_sheet_labels(args.unit_labels, units)
        provenance = {"unit_labels": source(args.unit_labels, output), "ignored_unit_sheet_lines": ignored}
    else:
        labels_path = args.labels or args.hypothesis_pilot / "review" / "blind_labels.csv"
        carried, unmatched = v13_labels(labels_path, states)
        labels = previous_by_unit(units, carried)
        provenance = {"v1_3_labels": source(labels_path, output), "unmatched_v1_3_labels": unmatched}
    rows = candidate_rows(units, labels)
    video = None
    if not args.no_clips:
        record = docs["track_mapping"]["video"]
        video = Path(record["path"])
        stat = video.stat()
        if (stat.st_size, stat.st_mtime_ns) != (record["size"], record["mtime_ns"]):
            raise ValueError("Movie file does not match Stage05 provenance")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        staged = Path(temp) / output.name
        review = staged / "review"
        review.mkdir(parents=True)
        write_json(staged / "behavior_units.json", {
            "schema_version": VERSION, "movie_id": MOVIE, "sequence_id": SEQUENCE, "human_verified": False,
            "sources": {name: source(path, output) for name, path in paths.items()}, "labels": provenance,
            "shots": [{"shot_id": s["shot_id"], "stage01_temporal": s["stage01_temporal"]} for s in shots],
            "units": units,
        })
        write_timeline(review / "timeline.csv", rows)
        write_guide(review / "README.md", rows, output, clips=video is not None)
        if video:
            make_clips(review, video, shots, units, track_boxes(docs["timeline"]))
        staged.rename(output)
    labelled = sum(bool(r["gaze"]) for r in rows)
    print(f"{len(units)} machine units → {len(rows)} timeline rows, {labelled} already labelled → {output}")
    if args.unit_labels and ignored:
        print(f"Ignored unit-sheet lines without unit_id: {ignored}")


def evaluate(args):
    doc = read(args.pilot / "behavior_units.json")
    if doc["schema_version"] != VERSION:
        raise ValueError("Unexpected behavior-unit schema")
    rows = read_timeline(args.timeline or args.pilot / "review" / "timeline.csv", doc["shots"])
    report = evaluate_timeline(doc["units"], rows, segments(rows, doc["shots"]))
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    build = commands.add_parser("run")
    for key in ("look-pilot", "hypothesis-pilot", "full-sequence", "output-dir"):
        build.add_argument("--" + key, required=True, type=Path)
    build.add_argument("--labels", type=Path, help="v1.3 blind labels (default: hypothesis pilot review)")
    build.add_argument("--unit-labels", type=Path, help="reviewed units.csv from an earlier v1.4 pilot")
    build.add_argument("--no-clips", action="store_true")
    build.set_defaults(handler=run)
    check = commands.add_parser("evaluate")
    check.add_argument("--pilot", required=True, type=Path)
    check.add_argument("--timeline", type=Path)
    check.set_defaults(handler=evaluate)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
