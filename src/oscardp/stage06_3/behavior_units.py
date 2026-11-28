"""Stage06.3 v1.4 review units: one non-overlapping timeline per character and Shot.

Movement episodes cut the frozen Look States, so a glance or a head turn inside a long
state becomes its own row and the parts before and after it can carry different targets.
Tracking gaps become rows too. Every moment belongs to exactly one unit.
"""

from collections import defaultdict
from itertools import pairwise

from oscardp.stage06_2.stable import MIN_STABLE_S

VERSION = "stage06.3.behavior_units.v1.4"
# Micro-adjustments return to the same direction (<=3 deg, no >=6 deg departure).
MICRO_ASSESSMENTS = {"adjustment_or_same_direction"}
# Two movements closer than the shortest possible Look State had no hold between them.
MERGE_GAP_S = MIN_STABLE_S
MIN_GAP_S = MIN_STABLE_S
# Review heuristic: single head/gaze shifts finish within ~1-1.5 s. Longer events in seq_006 (2.3-8.8 s)
# are drift or noisy small-face pose, so they annotate the state instead of cutting it.
MAX_MOVE_S = 2.0

# Explicit mapping of the free-text v1.3 blind labels, including the observed typo.
V13_PEOPLE = {"michelle": "MICHELLE", "michalle": "MICHELLE", "teddy": "TEDDY", "don": "DON"}


def overlaps(a, b):
    return a["start_s"] < b["end_s"] and b["start_s"] < a["end_s"]


def state_kind(state):
    if state["look_direction"]:
        return "HOLD"
    return "BLINK" if state["qc_status"] == "blink_gap" else "UNMEASURED"


def candidate_motion(event):
    return event["source_event"]["type"] != "blink_candidate" and event["assessment"] not in MICRO_ASSESSMENTS


def is_long(event):
    return event["end_s"] - event["start_s"] > MAX_MOVE_S


def movement_groups(events):
    """Merge one Track's motion events into episodes; blinks, micro-adjustments and long drifts are excluded."""
    picked = sorted((e for e in events if candidate_motion(e) and not is_long(e)),
                    key=lambda e: (e["start_s"], e["end_s"], e["event_id"]))
    groups = []
    for event in picked:
        if groups and event["start_s"] - max(e["end_s"] for e in groups[-1]) < MERGE_GAP_S:
            groups[-1].append(event)
        else:
            groups.append([event])
    return groups


def peak(values):
    found = [abs(v) for v in values if v is not None]
    return max(found, default=None)


def movement_unit(track, number, group, states):
    sources = [e["source_event"] for e in group]
    span = {"start_frame": min(e["start_frame"] for e in group), "end_frame": max(e["end_frame"] for e in group),
            "start_s": min(e["start_s"] for e in group), "end_s": max(e["end_s"] for e in group)}
    signals = set()
    for s in sources:
        signals |= set(s.get("trigger_signals") or (["eye"] if "eye" in s["type"] else []))
    assessments = list(dict.fromkeys(e["assessment"] for e in group))
    return {
        "unit_id": f"{track['track_id']}_move_{number:03d}", "kind": "MOVE",
        "shot_id": track["shot_id"], "track_id": track["track_id"], "character": track["character"], **span,
        "event_ids": [e["event_id"] for e in group],
        "source_types": sorted({s["type"] for s in sources}),
        "signals": sorted(signals),
        "assessments": assessments,
        "head_change_deg": peak(s.get("head_change_deg") for s in sources),
        "gaze_change_deg": peak(s.get("gaze_change_deg") for s in sources),
        "eye_relative_change_deg": peak(s.get("eye_relative_change_deg") for s in sources),
        "direction_change_deg": peak(e["direction_change_deg"] for e in group),
        "peak_gaze_departure_deg": peak(e["peak_gaze_departure_deg"] for e in group),
        "qc_flagged": all(a == "unreliable_motion" for a in assessments),
        "qc_flags": sorted({flag for s in sources for flag in s.get("qc_flags", [])}),
        "look_states": [s["state_id"] for s in states if overlaps(s, span)],
        "overlaps_blink": any(state_kind(s) == "BLINK" and overlaps(s, span) for s in states),
        "event_start_s": span["start_s"], "event_end_s": span["end_s"],
    }


def state_unit(state, predictions):
    proposals = {}
    for method, rows in predictions.items():
        row = rows[state["state_id"]]
        proposals[method] = {"preferred": row["preferred_target"], "ranking": row["target_ranking"],
                             "evidence_level": row["evidence_level"], "decision_status": row["decision_status"]}
    anchor = (state.get("gaze_evidence") or {}).get("anchor")
    kind = state_kind(state)
    return {
        "unit_id": state["state_id"], "kind": kind,
        **{k: state[k] for k in ("shot_id", "track_id", "character", "start_frame", "end_frame", "start_s", "end_s",
                                 "direction_source", "qc_status")},
        "look_state_id": state["state_id"], "part": 1, "parts": 1,
        # The piece holding the measured stable direction receives the whole-state proposal and old label.
        "anchor": {k: anchor[k] for k in ("start_s", "end_s")} if anchor and kind == "HOLD" else None,
        "direction_measured_here": kind == "HOLD", "primary_piece": True, "proposals": proposals,
    }


def span_of(unit):
    return {k: unit[k] for k in ("start_frame", "end_frame", "start_s", "end_s")}


def split_state(unit, moves):
    """Cut a state by movement episodes; leftovers shorter than a Look State join the adjacent movement."""
    inside = [m for m in moves if overlaps(unit, m)]
    if not inside:
        return [unit]
    segments, cursor, previous = [], span_of(unit), None
    for move in inside:
        if move["start_s"] > cursor["start_s"]:
            segments.append(({**cursor, "end_frame": move["start_frame"], "end_s": move["start_s"]}, previous, move))
        if move["end_s"] > cursor["start_s"]:
            cursor = {**cursor, "start_frame": move["end_frame"], "start_s": move["end_s"]}
        previous = move
    if cursor["start_s"] < unit["end_s"]:
        segments.append((cursor, previous, None))
    kept = []
    for span, before, after in segments:
        if span["end_s"] - span["start_s"] < MIN_STABLE_S and unit["kind"] != "BLINK" and (before or after):
            if after:  # the leftover ends where the next movement starts
                after.update(start_frame=span["start_frame"], start_s=span["start_s"])
            else:
                before.update(end_frame=span["end_frame"], end_s=span["end_s"])
        else:
            kept.append(span)
    pieces = [{**unit, **span, "unit_id": f"{unit['unit_id']}_p{n}", "part": n, "parts": len(kept)}
              for n, span in enumerate(kept, 1)]
    anchor = unit["anchor"]
    shared = [(min(p["end_s"], anchor["end_s"]) - max(p["start_s"], anchor["start_s"]) if anchor else 0, p)
              for p in pieces]
    measured = max(shared, key=lambda x: x[0], default=(0, None))
    primary = measured[1] if measured[0] > 0 else max(pieces, key=lambda p: p["end_s"] - p["start_s"], default=None)
    for p in pieces:
        p["direction_measured_here"] = p is measured[1] and measured[0] > 0
        p["primary_piece"] = p is primary
    return pieces


def coverage_gaps(shot, character, covered, fragments):
    """Spans of the Shot where this character has no measured Look State."""
    bounds = shot["stage01_temporal"]
    spans = sorted(covered, key=lambda s: s["start_s"])
    edges = ([{"end_s": bounds["start"], "end_frame": bounds["start_frame"], "edge": "shot_start"}] + spans
             + [{"start_s": bounds["end"], "start_frame": bounds["end_frame"], "edge": "shot_end"}])
    gaps = []
    for before, after in pairwise(edges):
        if after["start_s"] - before["end_s"] < MIN_GAP_S:
            continue
        span = {"start_frame": before["end_frame"], "end_frame": after["start_frame"],
                "start_s": before["end_s"], "end_s": after["start_s"]}
        gaps.append({
            "unit_id": f"{shot['shot_id']}_{character.lower()}_gap_{len(gaps) + 1:03d}", "kind": "GAP",
            "shot_id": shot["shot_id"], "track_id": None, "character": character, **span,
            "position": before.get("edge") or after.get("edge") or "between_tracks",
            "unmeasured_fragments": [f["track_id"] for f in fragments if overlaps(f, span)],
        })
    return gaps


def build_units(shots, tracks, states, events, predictions):
    by_track, events_by_track = defaultdict(list), defaultdict(list)
    for state in states:
        by_track[state["track_id"]].append(state)
    for event in events:
        events_by_track[event["track_id"]].append(event)
    units = []
    for shot in shots:
        local = [tracks[f["track_id"]] for f in shot["reported_faces"] if f["track_id"] in tracks]
        faces = {f["track_id"]: {**f, "start_s": f["start_sec"], "end_s": f["end_sec"]} for f in shot["reported_faces"]}
        people = list(dict.fromkeys(t["character"] for t in local if t["measured"]))
        for person in people:
            group = []
            measured = [t for t in local if t["character"] == person and t["measured"]]
            for track in measured:
                own = sorted(by_track[track["track_id"]], key=lambda s: s["start_frame"])
                track_events = events_by_track[track["track_id"]]
                moves = [movement_unit(track, n, g, own) for n, g in enumerate(movement_groups(track_events), 1)]
                long = [e for e in track_events if candidate_motion(e) and is_long(e)]
                for state in own:
                    unit = state_unit(state, predictions)
                    unit["long_motion_s"] = [round(e["end_s"] - e["start_s"], 2) for e in long if overlaps(e, state)]
                    group += split_state(unit, moves)
                group += moves
            fragments = [faces[t["track_id"]] for t in local if t["character"] == person and not t["measured"]]
            covered = [s for t in measured for s in by_track[t["track_id"]]]
            group += coverage_gaps(shot, person, covered, fragments)
            group.sort(key=lambda u: (u["start_s"], u["unit_id"]))
            for a, b in pairwise(group):
                if a["track_id"] == b["track_id"] and a["end_s"] > b["start_s"] + 1e-6:
                    raise ValueError(f"Overlapping review units: {a['unit_id']} / {b['unit_id']}")
            units += group
    return units


def previous_by_unit(units, carried):
    """Old labels go to the primary piece of their Look State; other pieces show the text for reference.

    A state swallowed whole by one movement hands its label to that movement.
    """
    result = {}
    remaining = {u.get("look_state_id") for u in units}
    for u in units:
        if u["kind"] == "MOVE":
            olds = [carried[s] for s in u["look_states"] if carried.get(s, {}).get("v1_3_label")]
            if olds:
                result[u["unit_id"]] = {k: " | ".join(o[k] for o in olds if o[k]) for k in ("v1_3_label", "v1_3_note")}
            swallowed = [carried[s] for s in u["look_states"]
                         if s not in remaining and carried.get(s, {}).get("human_type")]
            if len(swallowed) == 1:
                result[u["unit_id"]] = {**swallowed[0], **result[u["unit_id"]]}
            continue
        old = carried.get(u.get("look_state_id"))
        if old:
            reference = {k: old.get(k, "") for k in ("v1_3_label", "v1_3_note")}
            result[u["unit_id"]] = old if u["primary_piece"] else reference
    return result


def carry_over(label, certainty):
    """Translate one free-text v1.3 blind label; None means it needs a fresh look."""
    text = (label or "").strip().lower()
    if not text:
        return None
    fields = {"human_type": "", "human_target": "", "human_form": "",
              "human_certainty": "HIGH" if (certainty or "").strip() == "100%" else ""}
    if text in V13_PEOPLE:
        return {**fields, "human_type": "TARGET", "human_target": V13_PEOPLE[text]}
    if text == "blink":
        return {**fields, "human_type": "BLINK"}
    if text.startswith("down"):
        return {**fields, "human_type": "EXPRESSIVE", "human_form": "DOWN"}
    return None


def ranking_text(ranking):
    return " > ".join(" / ".join(tier) for tier in ranking) or "—"


def machine_summary(unit):
    kind = unit["kind"]
    if kind == "MOVE":
        parts = [f"{name} {unit[key]:.0f}°" for name, key in (
            ("head", "head_change_deg"), ("gaze", "gaze_change_deg"), ("eye", "eye_relative_change_deg"))
            if unit[key] is not None]
        parts.append("+".join(unit["signals"]) or "eye")
        if unit["qc_flagged"]:
            parts.append("QC flagged")
        if unit["overlaps_blink"]:
            parts.append("overlaps blink")
        return ", ".join(parts)
    if kind == "GAP":
        extra = f"; unmeasured fragment {', '.join(unit['unmeasured_fragments'])}" if unit["unmeasured_fragments"] else ""
        return f"no measured face ({unit['position']}){extra}"
    fused = unit["proposals"]["fused"]
    part = f"part {unit['part']}/{unit['parts']}; " if unit["parts"] > 1 or unit["unit_id"].endswith("_p1") else ""
    long = unit.get("long_motion_s", [])
    drift = f"; long motion {', '.join(f'{s:.1f}' for s in long)} s" if long else ""
    if kind == "HOLD":
        where = unit["direction_source"] if unit["direction_measured_here"] else "direction measured in another part"
        return f"{part}{where}; fused {ranking_text(fused['ranking'])} ({fused['evidence_level']}){drift}"
    return part + unit["qc_status"] + drift
