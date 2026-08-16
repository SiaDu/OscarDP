from __future__ import annotations

from typing import Any

from .schema import SCHEMA_VERSION
from .targeting import Target


def _scene_id(row: dict[str, Any]) -> str | None:
    scene = row.get("scene") or {}
    return scene.get("scene_id")


def _split_duration(group: list[dict[str, Any]], maximum: float) -> list[list[dict[str, Any]]]:
    if len(group) <= 1 or float(group[-1]["time"]["end_sec"]) - float(group[0]["time"]["start_sec"]) <= maximum:
        return [group]
    midpoint = (float(group[0]["time"]["start_sec"]) + float(group[-1]["time"]["end_sec"])) / 2
    candidates = range(1, len(group))
    split = min(
        candidates,
        key=lambda index: (
            min(float(group[index - 1]["shot_score"]), float(group[index]["shot_score"])),
            abs(float(group[index]["time"]["start_sec"]) - midpoint), index,
        ),
    )
    return _split_duration(group[:split], maximum) + _split_duration(group[split:], maximum)


def group_events(
    selected: list[dict[str, Any]], all_shots: list[dict[str, Any]], movie_id: str,
    release_id: str, maximum_duration: float = 30.0, target: Target | None = None,
    blocked_source_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    by_source_index = {int(row["source_index"]): row for row in selected}
    blocked_source_indices = blocked_source_indices or set()
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda item: int(item["source_index"])):
        bridge_ok = False
        if current:
            previous = current[-1]
            gap = all_shots[int(previous["source_index"]) + 1:int(row["source_index"])]
            bridge_ok = (
                _scene_id(row) == _scene_id(previous) and len(gap) <= 1
                and all(_scene_id(context) == _scene_id(row) and not context.get("scene_transition") for context in gap)
                and not any(int(previous["source_index"]) < value < int(row["source_index"]) for value in blocked_source_indices)
                and sum(float(context["time"]["duration_sec"]) for context in gap) <= 5.0
            )
        if current and not bridge_ok:
            groups.extend(_split_duration(current, maximum_duration))
            current = []
        current.append(row)
    if current:
        groups.extend(_split_duration(current, maximum_duration))

    events: list[dict[str, Any]] = []
    for ordinal, members in enumerate(groups, 1):
        first, last = members[0], members[-1]
        duration = float(last["time"]["end_sec"]) - float(first["time"]["start_sec"])
        total_member_duration = sum(float(row["time"]["duration_sec"]) for row in members)
        weighted = sum(float(row["shot_score"]) * float(row["time"]["duration_sec"]) for row in members)
        base_score = weighted / total_member_duration if total_member_duration > 0 else 0.0
        category_sets = [set(row.get("semantic", {}).get("categories", [])) for row in members]
        union = set().union(*category_sets) if category_sets else set()
        consistency = 0.0
        if len(members) > 1 and union:
            consistency = sum(len(categories) for categories in category_sets) / (len(members) * len(union))
        event_score = min(1.0, base_score + 0.05 * consistency)
        first_index, last_index = int(first["source_index"]), int(last["source_index"])
        before = all_shots[first_index - 1] if first_index > 0 else None
        after = all_shots[last_index + 1] if last_index + 1 < len(all_shots) else None
        before_ids = []
        after_ids = []
        if before and first_index - 1 not in by_source_index and _scene_id(before) == _scene_id(first):
            before_ids = [before["shot_id"]]
        if after and last_index + 1 not in by_source_index and _scene_id(after) == _scene_id(last):
            after_ids = [after["shot_id"]]
        events.append({
            "schema_version": SCHEMA_VERSION,
            "release_id": release_id,
            "movie_id": movie_id,
            "event_id": f"perfevent_{movie_id}_{target.performer_id if target and target.performer_id else 'target'}_{ordinal:06d}",
            "target": target.output() if target else None,
            "scene": first.get("scene"),
            "time": {
                "start": first["time"]["start"], "end": last["time"]["end"],
                "start_sec": first["time"]["start_sec"], "end_sec": last["time"]["end_sec"],
                "duration_sec": round(duration, 6),
            },
            "performance_shot_ids": [row["performance_shot_id"] for row in members],
            "source_shot_ids": [row["source_shot_id"] for row in members],
            "source_indices": [row["source_index"] for row in members],
            "categories": sorted(union),
            "event_score": round(event_score, 6),
            "semantic_consistency": round(consistency, 6),
            "context_before_shot_ids": before_ids,
            "context_between_shot_ids": [all_shots[index]["shot_id"] for left, right in zip([int(row["source_index"]) for row in members], [int(row["source_index"]) for row in members][1:]) for index in range(left + 1, right)],
            "context_after_shot_ids": after_ids,
            "duration_limit_exception": "single_source_shot" if len(members) == 1 and duration > maximum_duration else None,
        })
    ranked = sorted(events, key=lambda row: (-float(row["event_score"]), float(row["time"]["start_sec"]), row["event_id"]))
    for rank, row in enumerate(ranked, 1):
        row["target_rank"] = rank
    return sorted(events, key=lambda row: float(row["time"]["start_sec"]))
