from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .pipeline import _write_json, _write_jsonl
from .schema import read_jsonl


TARGETS = {"easy": 10, "fuzzy": 10, "multi": 5, "difficult": 5}


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


def prepare_pilot(requests_path: Path, alignment_path: Path, output_dir: Path, count: int = 30) -> dict[str, Any]:
    requests, alignments = read_jsonl(requests_path), read_jsonl(alignment_path)
    if count != 30:
        raise ValueError("Stage 2.2 pilot count must be exactly 30")
    by_stratum = {name: [] for name in TARGETS}
    for request in requests:
        by_stratum[_stratum(request)].append(request)
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for name in ("difficult", "multi", "fuzzy", "easy"):
        forced = []
        if name == "difficult":
            forced = sorted((item for item in by_stratum[name] if item.get("insufficient_candidates")), key=lambda item: item["request_id"])[:1]
        remaining = [item for item in by_stratum[name] if item not in forced]
        choices = forced + _stratified_take(remaining, TARGETS[name] - len(forced), len(alignments))
        if len(choices) < TARGETS[name]:
            raise ValueError(f"Not enough {name} requests for pilot: {len(choices)}")
        selected.extend(choices); used.update(item["request_id"] for item in choices)
    if len(selected) != count or len(used) != count:
        raise ValueError("Pilot selection is not unique and complete")
    selected.sort(key=lambda request: request["request_id"])
    if not any(request.get("insufficient_candidates") for request in selected):
        raise ValueError("Pilot must include the insufficient-candidate request")
    if sum(len(request["subtitle_ids"]) > 1 for request in selected) < 5:
        raise ValueError("Pilot must include at least five grouped-subtitle requests")
    manifest_rows = [{
        "request_id": request["request_id"], "stratum": _stratum(request),
        "timeline_region": _region(request, len(alignments)), "request_size": len(request["subtitle_ids"]),
        "candidate_count": len(request.get("dialogue_candidates", [])),
        "selection_reason": f"deterministic_{_stratum(request)}_stratum",
    } for request in selected]
    distribution = {
        "strata": {name: sum(row["stratum"] == name for row in manifest_rows) for name in TARGETS},
        "timeline": {name: sum(row["timeline_region"] == name for row in manifest_rows) for name in ("early", "middle", "late")},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "pilot_requests.jsonl", selected)
    _write_json(output_dir / "pilot_manifest.json", {
        "schema_version": "1.0", "source_requests": requests_path.resolve().as_posix(),
        "source_requests_sha256": hashlib.sha256(requests_path.read_bytes()).hexdigest(),
        "request_count": len(selected), "subtitle_count": sum(len(item["subtitle_ids"]) for item in selected),
        "distribution": distribution, "requests": manifest_rows,
    })
    gold = [{"request_id": request["request_id"], "resolutions": [{"subtitle_id": subtitle_id, "decision": None, "block_ids": None, "reviewer_notes": None} for subtitle_id in request["subtitle_ids"]]} for request in selected]
    _write_jsonl(output_dir / "pilot_gold_template.jsonl", gold)
    return {"request_count": len(selected), "subtitle_count": sum(len(item["subtitle_ids"]) for item in selected), **distribution}
