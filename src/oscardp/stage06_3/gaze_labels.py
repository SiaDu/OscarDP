"""Freeze a reviewer's MOVE/HOLD gaze sheet as Stage06.3 human labels and score the machine units against it.

The sheet is read by time only: its unit_id column may point at reused rows. Every MOVE ends where the
gaze lands and is followed by a HOLD on the same target. Times are approximate (about one frame).
"""

import argparse
import csv
import re
import statistics
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory

from oscardp.shots.transnet import sha256_file
from oscardp.stage06_2.evidence_replay import write_json
from oscardp.stage06_2.inputs import read
from oscardp.stage06_2.stable import MIN_STABLE_S

from .__main__ import MOVIE, SEQUENCE
from .hypothesis_pilot import source
from .inference import CHARACTERS

VERSION = "stage06.3.gaze_labels.v1"
FRAME_S = 0.05  # one frame at 23.976 fps, rounded up: allowed slack on sheet times
TOLERANCE_S = MIN_STABLE_S  # a machine movement counts for a human MOVE within this distance
KINDS = ("MOVE", "HOLD", "BLINK")
# Explicit mapping of the reviewer's expressive wording; left/right are the looker's own sides.
FORMS = {
    "down": "DOWN", "slightly down": "DOWN", "up": "UP", "up and slightly left": "UP",
    "left": "LEFT", "middle left": "LEFT", "right": "RIGHT", "闭眼": "CLOSED",
    "出神，眼睛没有聚焦？": "UNFOCUSED", "眼睛往下往右往上右转了一圈": "ROLL",
}
CERTAINTY = {"HIGH", "MEDIUM", "LOW"}
METHODS = ("fused", "visual", "geometry")


def tokens(text):
    return [t for t in re.split(r"[,，\s]+", (text or "").upper()) if t]


def parse_row(row, line):
    """One sheet row → a labelled piece, or None when the reviewer left the gaze open."""
    kinds, types = tokens(row["kind"]), tokens(row["human_type"])
    kind = next((k for k in ("MOVE", "HOLD", "BLINK") if k in kinds), "HOLD")
    blink = "BLINK" in kinds or "BLINK" in types
    form_text = (row.get("human_form") or "").strip()
    certainty = (row.get("human_certainty") or "").strip().upper() or None
    if certainty and certainty not in CERTAINTY:
        raise ValueError(f"Sheet line {line}: certainty {row['human_certainty']!r}")

    def piece(kind, target=None, form=None, text=None):
        return {"kind": kind, "start_s": float(row["start_s"]), "end_s": float(row["end_s"]), "target": target,
                "form": form, "form_text": text, "blink": blink, "certainty": certainty,
                "notes": (row.get("notes") or "").strip(), "sheet_lines": [line]}
    if "TARGET" in types:
        target = (row.get("human_target") or "").strip().upper()
        if target not in CHARACTERS:
            raise ValueError(f"Sheet line {line}: target {row['human_target']!r}")
        return piece(kind, target=target)
    if "EXPRESSIVE" in types:
        form = FORMS.get(form_text.lower())
        if form is None:
            raise ValueError(f"Sheet line {line}: add {form_text!r} to FORMS")
        return piece(kind, form=form, text=form_text)
    if types == ["BLINK"]:
        return piece("BLINK", text=form_text or None)
    if types and types != ["UNMEASURED"]:
        raise ValueError(f"Sheet line {line}: type {row['human_type']!r}")
    return None


def shot_of(start, end, bounds, line):
    found = [s for s, b in bounds.items() if b["start"] - FRAME_S <= start and end <= b["end"] + FRAME_S]
    if len(found) != 1:
        raise ValueError(f"Sheet line {line}: {start}-{end} is not inside one Shot")
    return found[0]


def same_gaze(a, b):
    return (a["target"], a["form"]) == (b["target"], b["form"])


def clean(pieces, where):
    """Sort, close MOVE→HOLD gaps, merge repeated HOLDs; returns segments and the corrections made."""
    pieces = sorted(pieces, key=lambda p: p["start_s"])
    corrections, merged = [], []
    for a, b in pairwise(pieces):
        gap = b["start_s"] - a["end_s"]
        if gap < -FRAME_S:
            raise ValueError(f"{where}: sheet lines {a['sheet_lines']} and {b['sheet_lines']} overlap")
        if gap > FRAME_S:
            if not (a["kind"] == "MOVE" and b["kind"] == "HOLD" and same_gaze(a, b)):
                raise ValueError(f"{where}: {gap:.3f} s gap before sheet line {b['sheet_lines'][0]}")
            corrections.append({"sheet_line": b["sheet_lines"][0], "field": "start_s", "from": b["start_s"],
                                "to": a["end_s"], "rule": "a HOLD starts when the MOVE to its target ends"})
            b["start_s"] = a["end_s"]
    for p in pieces:
        last = merged[-1] if merged else None
        if last and last["kind"] == p["kind"] == "HOLD" and same_gaze(last, p) and not (last["blink"] or p["blink"]):
            last["end_s"] = p["end_s"]
            last["sheet_lines"] += p["sheet_lines"]
            last["notes"] = " ".join(n for n in (last["notes"], p["notes"]) if n)
            last["certainty"] = last["certainty"] or p["certainty"]
            continue
        merged.append(p)
    return merged, corrections


def label(segment):
    return segment["target"] or segment["form"] or segment["kind"]


def build_labels(rows, shots):
    """Rows are (line, sheet row) pairs; returns per-(Shot, character) tracks and the corrections."""
    bounds = {s["shot_id"]: s["stage01_temporal"] for s in shots}
    grouped, open_rows = defaultdict(list), defaultdict(list)
    for line, row in rows:
        start, end = float(row["start_s"]), float(row["end_s"])
        if end <= start:
            raise ValueError(f"Sheet line {line}: end_s must follow start_s")
        shot = (row.get("shot_id") or "").strip() or shot_of(start, end, bounds, line)
        if shot not in bounds or shot_of(start, end, {shot: bounds[shot]}, line) != shot:
            raise ValueError(f"Sheet line {line}: unknown Shot {shot!r}")
        person = row["character"].strip().upper()
        if person not in CHARACTERS:
            raise ValueError(f"Sheet line {line}: character {row['character']!r}")
        piece = parse_row(row, line)
        (grouped if piece else open_rows)[shot, person].append(piece or {"line": line, "notes": row["notes"].strip()})
    order = {s["shot_id"]: n for n, s in enumerate(shots)}
    tracks, corrections = [], []
    for shot, person in sorted(grouped.keys() | open_rows.keys(), key=lambda k: (order[k[0]], k[1])):
        key = f"{shot}_{person.lower()}"
        if (shot, person) in grouped and (shot, person) in open_rows:
            lines = [r["line"] for r in open_rows[shot, person]]
            raise ValueError(f"{key}: sheet lines {lines} have no gaze while the rest of the track does")
        if (shot, person) in open_rows:
            notes = [r["notes"] for r in open_rows[shot, person] if r["notes"]]
            tracks.append({"track_key": key, "shot_id": shot, "character": person, "status": "not_annotatable",
                           "reason": notes[0] if notes else None, "segments": [], "shifts": []})
            continue
        segments, fixed = clean(grouped[shot, person], key)
        corrections += [{"track_key": key, **c} for c in fixed]
        for n, s in enumerate(segments, 1):
            s["segment_id"] = f"{key}_g{n:03d}"
        shifts, previous = [], None
        for s in segments:
            if s["kind"] == "MOVE":
                shifts.append({"shift_id": f"{key}_shift_{len(shifts) + 1:03d}", "segment_id": s["segment_id"],
                               "start_s": s["start_s"], "landing_s": s["end_s"],
                               "from": label(previous) if previous else None, "to": label(s)})
            if s["kind"] != "BLINK":
                previous = s
        tracks.append({"track_key": key, "shot_id": shot, "character": person, "status": "annotated",
                       "reason": None, "segments": [{"segment_id": s.pop("segment_id"), **s} for s in segments],
                       "shifts": shifts})
    return tracks, corrections


def read_sheet(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return [(n, row) for n, row in enumerate(csv.DictReader(stream), 2)
                if any((v or "").strip() for v in row.values())]


def near(a0, a1, b0, b1, tolerance):
    return min(a1, b1) + tolerance > max(a0, b0)


def overlap(a0, a1, b0, b1):
    return max(0., min(a1, b1) - max(a0, b0))


def evaluate(tracks, units, events):
    """Machine Stage06.3 units and raw look-direction events against the human segments."""
    by_key = {(t["shot_id"], t["character"]): t for t in tracks}
    character = {u["track_id"]: u["character"] for u in units if u.get("track_id")}
    moves = [u for u in units if u["kind"] == "MOVE"]
    raw = [{**e, "character": character.get(e["track_id"])} for e in events]
    human = [(t, s) for t in tracks for s in t["segments"] if s["kind"] == "MOVE"]

    def hits(t, s, candidates):
        return [c for c in candidates if (c["shot_id"], c["character"]) == (t["shot_id"], t["character"])
                and near(s["start_s"], s["end_s"], c["start_s"], c["end_s"], TOLERANCE_S)]

    def ratio(a, b):
        return round(a / b, 3) if b else None
    found_units = sum(bool(hits(t, s, moves)) for t, s in human)
    found_raw = sum(bool(hits(t, s, raw)) for t, s in human)

    def place(c):
        track = by_key.get((c["shot_id"], c["character"]))
        if not track or track["status"] != "annotated":
            return "not_annotated"
        if any(s["kind"] == "MOVE" and near(s["start_s"], s["end_s"], c["start_s"], c["end_s"], TOLERANCE_S)
               for s in track["segments"]):
            return "on_human_move"
        inside = {s["kind"] for s in track["segments"] if overlap(s["start_s"], s["end_s"], c["start_s"], c["end_s"])}
        return "inside_human_blink" if inside == {"BLINK"} else "inside_human_hold" if inside else "outside_labels"
    offsets = defaultdict(list)
    for t, s in human:
        for u in hits(t, s, moves):
            signal = "eye_only" if u["signals"] == ["eye"] else "with_head"
            offsets[signal].append((u["start_s"] - s["start_s"], u["end_s"] - s["end_s"], u["end_s"] - u["start_s"]))
    timing = {signal: {"matched_pairs": len(v), "median_start_offset_s": round(statistics.median(a for a, _, _ in v), 2),
                       "median_end_offset_s": round(statistics.median(b for _, b, _ in v), 2),
                       "median_machine_move_s": round(statistics.median(d for _, _, d in v), 2)}
              for signal, v in sorted(offsets.items())}
    holds = [(t, s) for t in tracks for s in t["segments"] if s["kind"] == "HOLD" and s["target"]]
    looked = sum(s["end_s"] - s["start_s"] for _, s in holds)
    targets = {}
    for method in METHODS:
        predicted = correct = 0.
        for u in units:
            preferred = u.get("primary_piece") and u["proposals"][method]["preferred"]
            if preferred not in CHARACTERS:
                continue
            for t, s in holds:
                if (t["shot_id"], t["character"]) == (u["shot_id"], u["character"]):
                    time = overlap(u["start_s"], u["end_s"], s["start_s"], s["end_s"])
                    predicted += time
                    correct += time * (preferred == s["target"])
        targets[method] = {"seconds_predicted": round(predicted, 2), "seconds_correct": round(correct, 2),
                           "accuracy": ratio(correct, predicted), "coverage": ratio(predicted, looked)}
    durations = [s["end_s"] - s["start_s"] for _, s in human]
    return {
        "tolerance_s": TOLERANCE_S,
        "human": {"moves": len(human), "median_move_s": round(statistics.median(durations), 2) if durations else None,
                  "moves_up_to_2_frames": sum(d <= 2 * FRAME_S for d in durations),
                  "seconds_on_a_target": round(looked, 2)},
        "move_detection": {"found_by_machine_move": found_units, "found_by_any_raw_event": found_raw,
                           "recall_machine_move": ratio(found_units, len(human)),
                           "recall_any_raw_event": ratio(found_raw, len(human))},
        "machine_moves": {"total": len(moves), **Counter(place(u) for u in moves)},
        "raw_events": {"total": len(raw), **Counter(place(e) for e in raw)},
        "machine_move_timing": timing,
        "hold_target_by_time": targets,
    }


def summary(tracks, report):
    lines = ["# Bugonia seq_006 · gaze labels", "",
             "Human MOVE/HOLD labels. Times are movie seconds, start-inclusive, end-exclusive, about ±1 frame.", "",
             "| Shot | Character | Status | Shifts | Seconds per gaze |", "| --- | --- | --- | --- | --- |"]
    for t in tracks:
        seconds = Counter()
        for s in t["segments"]:
            seconds[label(s)] += s["end_s"] - s["start_s"]
        text = ", ".join(f"{k} {v:.1f}" for k, v in seconds.most_common()) or t["reason"] or ""
        lines.append(f"| {t['shot_id']} | {t['character']} | {t['status']} | {len(t['shifts'])} | {text} |")
    d, m = report["move_detection"], report["machine_moves"]
    lines += ["", "## Machine units against these labels", "",
              f"- Human moves: {report['human']['moves']}; found by a machine MOVE within ±{TOLERANCE_S} s: "
              + f"{d['found_by_machine_move']}, by any raw event: {d['found_by_any_raw_event']}.",
              f"- Machine MOVEs: {m['total']}; on a human move {m.get('on_human_move', 0)}, "
              + f"inside a human HOLD {m.get('inside_human_hold', 0)}."]
    for k, v in report["machine_move_timing"].items():
        lines.append(f"- Timing ({k}): start {v['median_start_offset_s']:+.2f} s, end {v['median_end_offset_s']:+.2f} s,"
                     f" machine move {v['median_machine_move_s']:.2f} s (medians).")
    lines += ["", "Full numbers: `evaluation.json`.", ""]
    return "\n".join(lines)


def run(args):
    output = args.output_dir
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    timeline_path = args.full_sequence / "sequence_gaze_timeline.json"
    units_path = args.units / "behavior_units.json"
    timeline, units_doc = read(timeline_path), read(units_path)
    if any((d["movie_id"], d["sequence_id"]) != (MOVIE, SEQUENCE) for d in (timeline, units_doc)):
        raise ValueError("Movie or sequence mismatch")
    recorded = units_doc["sources"]["look_events"]
    events_path = (args.units / recorded["file"]).resolve()
    if sha256_file(events_path) != recorded["sha256"]:
        raise ValueError("Look-direction events changed since the machine units were built")
    shots = [{"shot_id": s["shot_id"], "stage01_temporal": s["stage01_temporal"]} for s in timeline["shots"]]
    tracks, corrections = build_labels(read_sheet(args.sheet), shots)
    report = evaluate(tracks, units_doc["units"], read(events_path)["events"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temp:
        staged = Path(temp) / output.name
        staged.mkdir()
        write_json(staged / "gaze_labels.json", {
            "schema_version": VERSION, "movie_id": MOVIE, "sequence_id": SEQUENCE,
            "human_verified": True, "label_source": "reviewer sheet, read by time; unit_id column ignored",
            "time_basis": "movie seconds, start-inclusive, end-exclusive; sheet times about ±1 frame",
            "direction_frame": "left/right/up/down are the looker's own sides",
            "sources": {"sheet": source(args.sheet, output), "timeline": source(timeline_path, output)},
            "corrections": [{**c, "approval": args.approval} for c in corrections],
            "shots": shots, "tracks": tracks,
        })
        write_json(staged / "evaluation.json", {
            "schema_version": VERSION, "movie_id": MOVIE, "sequence_id": SEQUENCE, "human_verified": False,
            "sources": {"labels": "gaze_labels.json", "units": source(units_path, output),
                        "look_events": source(events_path, output)},
            **report,
        })
        (staged / "README.md").write_text(summary(tracks, report), encoding="utf-8")
        staged.rename(output)
    print(f"{sum(len(t['segments']) for t in tracks)} segments, {sum(len(t['shifts']) for t in tracks)} shifts, "
          f"{len(corrections)} corrections → {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheet", required=True, type=Path, help="reviewed MOVE/HOLD units.csv")
    parser.add_argument("--full-sequence", required=True, type=Path, help="Stage06.2 full-sequence run (Shot bounds)")
    parser.add_argument("--units", required=True, type=Path, help="Stage06.3 behaviour-unit pilot to evaluate")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--approval", required=True, help="who approved the gap corrections, and when")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
