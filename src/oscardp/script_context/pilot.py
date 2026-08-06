from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .pipeline import _write_json, _write_jsonl
from .schema import read_jsonl


TARGETS = {"easy": 10, "fuzzy": 10, "multi": 5, "difficult": 5}
WINDOW_TYPES = ("normal_candidate_window", "fallback_retrieval", "insufficient_candidates")


def _region(request: dict[str, Any], total_subtitles: int) -> str:
    ordinals = [int(value.rsplit("_", 1)[-1]) for value in request["subtitle_ids"]]
    position = sum(ordinals) / len(ordinals) / max(1, total_subtitles)
    return "early" if position < 1 / 3 else "middle" if position < 2 / 3 else "late"


def _stratum(request: dict[str, Any]) -> str:
    mappings = request.get("automatic_candidate_mappings", [])
    if request.get("insufficient_candidates") or any(item.get("alignment", {}).get("status") == "no_match" for item in mappings):
        return "difficult"
    if len(request.get("subtitle_ids", [])) > 1 or any(len(item.get("matches", [])) > 1 for item in mappings):
        return "multi"
    if any("rapidfuzz" in item.get("alignment", {}).get("method", "") for item in mappings):
        return "fuzzy"
    return "easy"


def _stratified_take(candidates: list[dict[str, Any]], count: int, total: int) -> list[dict[str, Any]]:
    buckets = {region: [] for region in ("early", "middle", "late")}
    for request in candidates:
        buckets[_region(request, total)].append(request)
    for values in buckets.values():
        values.sort(key=lambda request: request["request_id"])
    selected: list[dict[str, Any]] = []
    while len(selected) < count and any(buckets.values()):
        for region in ("early", "middle", "late"):
            if buckets[region] and len(selected) < count:
                selected.append(buckets[region].pop(0))
    return selected


def _window_type(request: dict[str, Any]) -> str:
    if request.get("insufficient_candidates"):
        return "insufficient_candidates"
    if request.get("fallback_used"):
        return "fallback_retrieval"
    return "normal_candidate_window"


def _candidate_limit_saturated(request: dict[str, Any]) -> bool:
    limit = request.get("candidate_limit")
    return isinstance(limit, int) and limit > 0 and len(request.get("dialogue_candidates", [])) == limit


def _selection_cell(request: dict[str, Any]) -> str:
    saturation = "saturated" if _candidate_limit_saturated(request) else "not_saturated"
    return f"{_window_type(request)}:{saturation}"


def _balanced_take(candidates: list[dict[str, Any]], count: int, total: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for request in candidates:
        buckets.setdefault((_region(request, total), _stratum(request)), []).append(request)
    for values in buckets.values():
        values.sort(key=lambda request: request["request_id"])
    keys = sorted(buckets, key=lambda key: (("early", "middle", "late").index(key[0]), key[1]))
    selected: list[dict[str, Any]] = []
    while len(selected) < count and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(selected) < count:
                selected.append(buckets[key].pop(0))
    return selected


def _proportional_quotas(counts: dict[str, int], target: int) -> dict[str, int]:
    total = sum(counts.values())
    if total == 0:
        return {key: 0 for key in counts}
    ideals = {key: target * value / total for key, value in counts.items()}
    quotas = {key: min(value, int(ideals[key])) for key, value in counts.items()}
    nonempty = [key for key, value in counts.items() if value]
    if target >= len(nonempty):
        for key in nonempty:
            quotas[key] = max(1, quotas[key])
    while sum(quotas.values()) < target:
        available = [key for key in counts if quotas[key] < counts[key]]
        if not available:
            break
        key = max(available, key=lambda value: (ideals[value] - quotas[value], counts[value], -list(counts).index(value)))
        quotas[key] += 1
    while sum(quotas.values()) > target:
        available = [key for key in counts if quotas[key] > (1 if counts[key] and target >= len(nonempty) else 0)]
        key = min(available, key=lambda value: (ideals[value] - quotas[value], counts[value]))
        quotas[key] -= 1
    return quotas


def _distribution(requests: list[dict[str, Any]], total_subtitles: int) -> dict[str, Any]:
    limits = sorted({request.get("candidate_limit") for request in requests if isinstance(request.get("candidate_limit"), int)})
    saturated = sum(_candidate_limit_saturated(request) for request in requests)
    return {
        "request_count": len(requests),
        "subtitle_resolution_count": sum(len(request["subtitle_ids"]) for request in requests),
        "candidate_windows": {name: sum(_window_type(request) == name for request in requests) for name in WINDOW_TYPES},
        "strata": {name: sum(_stratum(request) == name for request in requests) for name in TARGETS},
        "timeline": {name: sum(_region(request, total_subtitles) == name for request in requests) for name in ("early", "middle", "late")},
        "fuzzy_requests": sum(_stratum(request) == "fuzzy" for request in requests),
        "multi_or_composite_requests": sum(
            len(request.get("subtitle_ids", [])) > 1
            or any(len(item.get("matches", [])) > 1 for item in request.get("automatic_candidate_mappings", []))
            for request in requests
        ),
        "difficult_requests": sum(_stratum(request) == "difficult" for request in requests),
        "candidate_limit_saturation": {
            "configured_limits": limits, "saturated_requests": saturated,
            "not_saturated_requests": len(requests) - saturated,
            "saturation_rate": saturated / len(requests) if requests else 0.0,
        },
    }


def prepare_pilot(requests_path: Path, alignment_path: Path, output_dir: Path, count: int = 30) -> dict[str, Any]:
    requests, alignments = read_jsonl(requests_path), read_jsonl(alignment_path)
    if count < 1:
        raise ValueError("Pilot count must be positive")
    target_count = min(count, len(requests))
    cells = list(dict.fromkeys(_selection_cell(request) for request in requests))
    by_cell = {name: [request for request in requests if _selection_cell(request) == name] for name in cells}
    quotas = _proportional_quotas({name: len(values) for name, values in by_cell.items()}, target_count)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for name in cells:
        choices = _balanced_take(by_cell[name], quotas[name], len(alignments))
        selected.extend(choices); used.update(item["request_id"] for item in choices)

    if len(selected) < target_count:
        remaining = [item for item in requests if item["request_id"] not in used]
        choices = _balanced_take(remaining, target_count - len(selected), len(alignments))
        selected.extend(choices); used.update(item["request_id"] for item in choices)
    if len(selected) != target_count or len(used) != target_count:
        raise ValueError("Pilot selection is not unique and complete")
    selected.sort(key=lambda request: request["request_id"])
    manifest_rows = [{
        "request_id": request["request_id"], "stratum": _stratum(request),
        "timeline_region": _region(request, len(alignments)), "request_size": len(request["subtitle_ids"]),
        "candidate_count": len(request.get("dialogue_candidates", [])),
        "candidate_window": _window_type(request),
        "fallback_used": bool(request.get("fallback_used")),
        "insufficient_candidates": bool(request.get("insufficient_candidates")),
        "candidate_limit_saturated": _candidate_limit_saturated(request),
        "selection_reason": f"diagnostic_balanced_{_selection_cell(request)}_{_stratum(request)}",
    } for request in selected]
    source_distribution = _distribution(requests, len(alignments))
    pilot_distribution = _distribution(selected, len(alignments))
    warnings = []
    for name in WINDOW_TYPES:
        source_rate = source_distribution["candidate_windows"][name] / max(1, source_distribution["request_count"])
        pilot_rate = pilot_distribution["candidate_windows"][name] / max(1, pilot_distribution["request_count"])
        if abs(source_rate - pilot_rate) > 0.15:
            warnings.append(f"material candidate-window representation difference for {name}: source={source_rate:.3f}, pilot={pilot_rate:.3f}")
    source_saturation = source_distribution["candidate_limit_saturation"]["saturation_rate"]
    pilot_saturation = pilot_distribution["candidate_limit_saturation"]["saturation_rate"]
    if abs(source_saturation - pilot_saturation) > 0.10:
        warnings.append(
            f"material candidate-limit saturation representation difference: source={source_saturation:.3f}, pilot={pilot_saturation:.3f}"
        )
    warnings.append(
        "This is a diagnostic-balanced pilot with deliberately oversampled strata; raw accuracy is not a statistically representative overall estimate."
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "pilot_requests.jsonl", selected)
    _write_json(output_dir / "pilot_manifest.json", {
        "schema_version": "1.1", "source_requests": requests_path.resolve().as_posix(),
        "source_requests_sha256": hashlib.sha256(requests_path.read_bytes()).hexdigest(),
        "request_count": len(selected), "subtitle_count": sum(len(item["subtitle_ids"]) for item in selected),
        "source_pool_distribution": source_distribution,
        "diagnostic_balanced_pilot_distribution": pilot_distribution,
        "pilot_distribution": pilot_distribution,
        "selection_design": "diagnostic_balanced",
        "statistically_representative": False,
        "evaluation_reporting": {
            "raw_diagnostic_accuracy": True, "per_stratum_accuracy": True,
            "source_weighted_overall_accuracy": True,
            "source_weighting_basis": "source_pool_request_count_by_stratum",
        },
        "distribution": {"strata": pilot_distribution["strata"], "timeline": pilot_distribution["timeline"]},
        "representativeness_warnings": warnings, "requests": manifest_rows,
    })
    gold = [{"request_id": request["request_id"], "resolutions": [{"subtitle_id": subtitle_id, "decision": None, "block_ids": None, "reviewer_notes": None} for subtitle_id in request["subtitle_ids"]]} for request in selected]
    _write_jsonl(output_dir / "pilot_gold_template.jsonl", gold)
    return {
        "request_count": len(selected), "subtitle_count": sum(len(item["subtitle_ids"]) for item in selected),
        "strata": pilot_distribution["strata"], "timeline": pilot_distribution["timeline"],
        "candidate_windows": pilot_distribution["candidate_windows"],
        "candidate_limit_saturation": pilot_distribution["candidate_limit_saturation"],
        "source_pool_distribution": source_distribution,
        "representativeness_warnings": warnings,
    }
