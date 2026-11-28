"""Stage06.3 v1.4 gaze timeline: one row per gaze segment; a row lasts until the next row starts.

People annotate only where the eyes are and when that changes. Head motion, shift duration and
blinks are left to the measurements; machine units only propose candidate boundaries.
"""

import csv
from collections import Counter, defaultdict
from itertools import pairwise

from oscardp.stage06_2.stable import MIN_STABLE_S

from .behavior_units import machine_summary
from .inference import CHARACTERS

LOOK_AT = (*CHARACTERS, "OTHER")
EXPRESSIVE = ("DOWN", "UP", "SIDE", "AVERT", "CLOSED")
GAZE = (*LOOK_AT, *EXPRESSIVE, "BLINK", "UNCLEAR")
CERTAINTY = ("HIGH", "MEDIUM", "LOW")
# A detected boundary counts for a human gaze shift within the shortest Look State duration.
SHIFT_TOLERANCE_S = MIN_STABLE_S
EDITABLE = ("start_s", "gaze", "certainty", "notes")
HINTS = ("machine_kind", "machine_end_s", "machine_hint", "source_unit", "v1_3_label")
COLUMNS = ("shot_id", "character", *EDITABLE, *HINTS)
FORM_TO_GAZE = {"DOWN": "DOWN", "UP": "UP", "SIDE": "SIDE", "AVERT": "AVERT", "BRIEF_AWAY": "AVERT"}


def gaze_from_unit_label(label):
    """Map a per-unit label (type/target/form) to one gaze value; None means 'no behaviour here'."""
    kind = (label.get("human_type") or "").strip().upper()
    if kind == "TARGET":
        return (label.get("human_target") or "").strip().upper()
    if kind == "EXPRESSIVE":
        return FORM_TO_GAZE.get((label.get("human_form") or "").strip().upper(), "UNCLEAR")
    if kind in ("BLINK", "UNCLEAR"):
        return kind
    return None if kind == "NONE" else ""


def candidate_rows(units, labels):
    """One candidate row per machine unit, with any earlier label; 'NONE' rows dissolve into a neighbour."""
    groups = defaultdict(list)
    for u in units:
        groups[u["shot_id"], u["character"]].append(u)
    rows = []
    for (shot, person), group in groups.items():
        local, carried_start = [], None
        for u in group:
            label = labels.get(u["unit_id"], {})
            gaze = gaze_from_unit_label(label)
            short = u["unit_id"].removeprefix(shot + "_")
            note = (label.get("notes") or "").strip()
            if gaze is None:  # reviewed as no behaviour: its boundary disappears
                if local:
                    local[-1]["notes"] = " ".join(x for x in (local[-1]["notes"], f"[{short}] {note}") if x.strip())
                else:
                    carried_start = label.get("start_s", u["start_s"])
                continue
            local.append({
                "shot_id": shot, "character": person,
                "start_s": carried_start if carried_start is not None else label.get("start_s", u["start_s"]),
                "gaze": gaze, "certainty": (label.get("human_certainty") or "").strip().upper(), "notes": note,
                "machine_kind": u["kind"], "machine_end_s": u["end_s"], "machine_hint": machine_summary(u),
                "source_unit": u["unit_id"],
                "v1_3_label": " — ".join(x for x in (label.get("v1_3_label"), label.get("v1_3_note")) if x),
            })
            carried_start = None
        rows += sorted(local, key=lambda r: r["start_s"])
    return rows


def write_timeline(path, rows):
    with path.open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "start_s": f"{r['start_s']:.3f}", "machine_end_s": f"{r['machine_end_s']:.3f}"})


def read_timeline(path, shots):
    """Validated rows; every problem is reported at once so the sheet can be fixed in one pass."""
    bounds = {s["shot_id"]: s["stage01_temporal"] for s in shots}
    rows, errors, seen = [], [], set()
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for n, row in enumerate(csv.DictReader(stream), 2):
            if not any((v or "").strip() for k, v in row.items() if k in ("shot_id", "start_s", "gaze")):
                continue
            where = f"timeline.csv line {n}"
            shot, person = (row.get("shot_id") or "").strip(), (row.get("character") or "").strip().upper()
            gaze, certainty = ((row.get(k) or "").strip().upper() for k in ("gaze", "certainty"))
            try:
                start = float(row.get("start_s") or "")
            except ValueError:
                errors.append(f"{where}: start_s must be movie seconds")
                continue
            if shot not in bounds or person not in CHARACTERS:
                errors.append(f"{where}: unknown shot_id {shot!r} or character {person!r}")
                continue
            if not bounds[shot]["start"] - 0.05 <= start < bounds[shot]["end"]:
                errors.append(f"{where}: {start} is outside {shot}")
            if gaze and gaze not in GAZE:
                errors.append(f"{where}: gaze={row['gaze']!r} not in {'/'.join(GAZE)}")
            if certainty and certainty not in CERTAINTY:
                errors.append(f"{where}: certainty={row['certainty']!r} not in {'/'.join(CERTAINTY)}")
            key = (shot, person, round(start, 3))
            if key in seen:
                errors.append(f"{where}: two rows start at {start} for {person} in {shot}")
            seen.add(key)
            rows.append({"shot_id": shot, "character": person, "start_s": start, "gaze": gaze,
                         "source_unit": (row.get("source_unit") or "").strip()})
    if errors:
        raise ValueError("Timeline problems:\n" + "\n".join(errors))
    return rows


def segments(rows, shots):
    """Merged (start, end, gaze) per Shot and character. BLINK continues the gaze around it; '' is unreviewed."""
    ends = {s["shot_id"]: s["stage01_temporal"]["end"] for s in shots}
    groups = defaultdict(list)
    for r in rows:
        groups[r["shot_id"], r["character"]].append(r)
    result = {}
    for key, local in groups.items():
        local.sort(key=lambda r: r["start_s"])
        stops = [r["start_s"] for r in local[1:]] + [ends[key[0]]]
        spans = [[r["start_s"], stop, r["gaze"]] for r, stop in zip(local, stops, strict=True)]
        for i, span in enumerate(spans):
            if span[2] == "BLINK":
                before = spans[i - 1][2] if i else ""
                after = next((s[2] for s in spans[i + 1:] if s[2] != "BLINK"), "")
                span[2] = before or after
        merged = []
        for span in spans:
            if merged and merged[-1][2] == span[2]:
                merged[-1][1] = span[1]
            else:
                merged.append(span)
        result[key] = [tuple(s) for s in merged]
    return result


def overlap(a0, a1, b0, b1):
    return max(0., min(a1, b1) - max(a0, b0))


def evaluate_timeline(units, rows, segs):
    seconds = Counter()
    for spans in segs.values():
        for a, b, gaze in spans:
            seconds[gaze or "unreviewed"] += b - a
    looked_at = sum(v for k, v in seconds.items() if k in LOOK_AT)
    primary = [u for u in units if u.get("primary_piece")]
    methods = {}
    for method in ("fused", "visual", "geometry"):
        predicted = correct = listed = expressive = 0.
        for u in primary:
            proposal = u["proposals"][method]
            ranked = {t for tier in proposal["ranking"] for t in tier}
            for a, b, gaze in segs.get((u["shot_id"], u["character"]), []):
                t = overlap(u["start_s"], u["end_s"], a, b)
                if gaze in LOOK_AT:
                    listed += t * (gaze in ranked)
                    if proposal["preferred"] in LOOK_AT:
                        predicted += t
                        correct += t * (proposal["preferred"] == gaze)
                elif gaze in EXPRESSIVE and proposal["preferred"] in LOOK_AT:
                    expressive += t
        methods[method] = {
            "seconds_with_prediction": round(predicted, 2), "seconds_correct": round(correct, 2),
            "time_accuracy": round(correct / predicted, 3) if predicted else None,
            "time_coverage": round(predicted / looked_at, 3) if looked_at else None,
            "candidate_recall": round(listed / looked_at, 3) if looked_at else None,
            "target_predicted_but_expressive_s": round(expressive, 2),
        }
    shifts = [(key, b[0]) for key, spans in segs.items() for a, b in pairwise(spans)
              if a[2] and b[2] and "UNCLEAR" not in (a[2], b[2])]
    by_key = defaultdict(list)
    for u in units:
        by_key[u["shot_id"], u["character"]].append(u)
    boundaries = {k: sorted(u["start_s"] for u in v)[1:] for k, v in by_key.items()}
    move_starts = {k: [u["start_s"] for u in v if u["kind"] == "MOVE"] for k, v in by_key.items()}

    def found(table):
        hits = sum(any(abs(t - m) <= SHIFT_TOLERANCE_S for m in table.get(k, [])) for k, t in shifts)
        return round(hits / len(shifts), 3) if shifts else None

    def reviewed_at(key, t):
        return any(a <= t < b and g and g != "UNCLEAR" for a, b, g in segs.get(key, []))
    moves = [(k, t) for k, starts in move_starts.items() for t in starts if reviewed_at(k, t)]
    shift_times = defaultdict(list)
    for k, t in shifts:
        shift_times[k].append(t)
    on_shift = sum(any(abs(t - s) <= SHIFT_TOLERANCE_S for s in shift_times[k]) for k, t in moves)
    kept = {r["source_unit"] for r in rows if r["source_unit"]}
    return {
        "seconds_by_gaze": {k: round(v, 2) for k, v in sorted(seconds.items())},
        "target_accuracy_by_time": methods,
        "gaze_shifts": {
            "human_shifts": len(shifts), "tolerance_s": SHIFT_TOLERANCE_S,
            "found_by_any_machine_boundary": found(boundaries), "found_by_move_start": found(move_starts),
            "reviewed_move_starts": len(moves), "move_starts_on_a_shift": on_shift,
        },
        "rows_added_by_reviewer": sum(not r["source_unit"] for r in rows),
        "machine_rows_removed": sum(u["unit_id"] not in kept for u in units),
    }
