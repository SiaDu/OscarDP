from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

TARGET_RULESET_VERSION = "performance_target_rules_v1"


def _normal(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).replace("’", "'")
    return re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _normal(value).lower()).strip("_")


def _cue_base(value: str) -> str:
    value = re.sub(r"\([^)]*\)", "", value or "")
    value = re.sub(r"(?:['’]S\s+VOICE|\s+VOICE)$", "", value, flags=re.IGNORECASE)
    return _normal(value)


def _characters(detail: str) -> list[str]:
    return list(dict.fromkeys(part.strip() for part in re.split(r"[|/]", detail) if part.strip()))


@dataclass(frozen=True)
class Target:
    movie_id: str
    film: str
    performer_name: str
    performer_id: str
    category: str
    character_names: list[str]
    winner: bool
    year: str
    raw_row: dict[str, str]
    resolution_method: str

    @property
    def performance_key(self) -> str:
        return f"acting_{_slug(self.year)}_{_slug(self.category)}_{self.movie_id}_{self.performer_id or _slug(self.performer_name)}"

    @property
    def performer_slug(self) -> str:
        if self.performer_id:
            return f"{self.performer_id}_{_slug(self.performer_name)}"
        token = hashlib.sha256(self.performance_key.encode()).hexdigest()[:8]
        return f"name_{_slug(self.performer_name)}_{token}"

    def output(self) -> dict[str, Any]:
        return {
            "performance_key": self.performance_key, "movie_id": self.movie_id, "film": self.film,
            "performer_name": self.performer_name, "performer_id": self.performer_id,
            "character_names": self.character_names, "category": self.category, "winner": self.winner,
            "year": self.year,
        }


def resolve_target(path: Path, movie_id: str, performer_id: str | None, performer_name: str | None) -> Target:
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    required = {"FilmId", "Film", "Nominees", "NomineeIds", "CanonicalCategory", "Detail", "Winner", "Year"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"Nominees CSV lacks required columns: {sorted(required)}")
    candidates = [row for row in rows if row["FilmId"] == movie_id]
    if not candidates:
        raise ValueError(f"No acting nomination for movie: {movie_id}")
    method = "single_movie_nomination"
    if performer_id:
        candidates = [row for row in candidates if row["NomineeIds"] == performer_id]
        method = "performer_id"
    elif performer_name:
        name = _normal(performer_name)
        candidates = [row for row in candidates if _normal(row["Nominees"]) == name]
        method = "performer_name"
    if len(candidates) != 1:
        choices = [f"{row['NomineeIds']} ({row['Nominees']}; {row['CanonicalCategory']})" for row in candidates] if candidates else []
        if not performer_id and not performer_name:
            choices = [f"{row['NomineeIds']} ({row['Nominees']}; {row['CanonicalCategory']})" for row in rows if row["FilmId"] == movie_id]
        raise ValueError(f"Could not uniquely resolve nomination for {movie_id}; choices: {choices}")
    row = candidates[0]
    return Target(movie_id=row["FilmId"], film=row["Film"], performer_name=row["Nominees"], performer_id=row["NomineeIds"],
                  category=row["CanonicalCategory"], character_names=_characters(row["Detail"]),
                  winner=row["Winner"].strip().casefold() == "true", year=row["Year"], raw_row=row, resolution_method=method)


def _mentions(text: str, aliases: list[str]) -> bool:
    normalized = _normal(text)
    return any(re.search(rf"(?<![A-Z0-9]){re.escape(alias)}(?![A-Z0-9])", normalized) for alias in aliases)


def _target_speaker(value: str, aliases: list[str]) -> bool:
    return _cue_base(value) in aliases


def mine_target_relevance(rows: list[dict[str, Any]], screenplay: dict[str, Any], target: Target) -> list[dict[str, Any]]:
    blocks = {block["block_id"]: block for scene in screenplay.get("script_scenes", []) for block in scene.get("script_blocks", [])}
    aliases = [_normal(name) for name in target.character_names]
    result: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        evidence: list[dict[str, Any]] = []
        def add(rule_id: str, relation: str, source_type: str, source_id: str, text: str, score: float, items: list[dict[str, Any]] = evidence) -> None:
            items.append({"rule_id": rule_id, "relation": relation, "source_type": source_type, "source_id": source_id, "text": text, "weight": score})
        speakers = [str(value) for value in row.get("dialogue_speakers", [])]
        for speaker in speakers:
            if _target_speaker(speaker, aliases):
                add("target_current_dialogue_speaker_v1", "within_shot", "dialogue_speaker", row["shot_id"], speaker, 1.0)
        for match in row.get("script_matches", []):
            block = blocks.get(match.get("block_id"), {})
            speaker = str(match.get("speaker") or block.get("speaker") or "")
            if _target_speaker(speaker, aliases):
                add("target_script_match_speaker_v1", "direct_dialogue_match", "script_dialogue", str(match.get("block_id")), speaker, 1.0)
                parenthetical = block.get("parenthetical")
                if isinstance(parenthetical, str) and parenthetical.strip():
                    add("target_parenthetical_v1", "direct_dialogue_match", "script_parenthetical", str(match.get("block_id")), parenthetical, .95)
        local = row.get("local_script_context") or {}
        for relation in ("action_during", "action_before", "action_after"):
            for block_id in local.get(relation, []):
                block = blocks.get(block_id, {})
                text = block.get("text")
                if isinstance(text, str) and _mentions(text, aliases):
                    score = .90 if relation == "action_during" else .55
                    add(f"target_{relation}_mention_v1", relation, "script_action", str(block_id), text, score)
        if len(set(speakers)) > 1 and any(_target_speaker(speaker, aliases) for speaker in speakers):
            add("target_current_interaction_v1", "within_shot", "shot_structure", row["shot_id"], " | ".join(speakers), .85)
        scene_id = (row.get("scene") or {}).get("scene_id")
        for adjacent in (index - 1, index + 1):
            if 0 <= adjacent < len(rows) and scene_id and (rows[adjacent].get("scene") or {}).get("scene_id") == scene_id:
                for speaker in rows[adjacent].get("dialogue_speakers", []):
                    if _target_speaker(str(speaker), aliases):
                        add("target_adjacent_dialogue_v1", "adjacent_same_scene", "dialogue_speaker", rows[adjacent]["shot_id"], str(speaker), .45)
                        break
        score = max((float(item["weight"]) for item in evidence), default=0.0)
        confidence = "high" if score >= .80 else "medium" if score >= .45 else "none"
        relevance: dict[str, Any] = {"performer_name": target.performer_name, "character_names": target.character_names,
                                     "confidence": confidence, "score": round(score, 6), "evidence": evidence}
        if confidence == "none": relevance["interpretation"] = "no_textual_support"
        result.append(relevance)
    return result


def target_context_risk(row: dict[str, Any], relevance: dict[str, Any], target: Target) -> dict[str, Any]:
    """Flag weak textual target context without inferring visual subject identity."""
    evidence = relevance.get("evidence") or []
    weak_rules = {
        "target_action_before_mention_v1", "target_action_after_mention_v1", "target_adjacent_dialogue_v1",
    }
    direct_rules = {
        "target_current_dialogue_speaker_v1", "target_script_match_speaker_v1",
        "target_parenthetical_v1", "target_action_during_mention_v1", "target_current_interaction_v1",
    }
    aliases = [_normal(name) for name in target.character_names]
    speakers = [str(value) for value in row.get("dialogue_speakers", []) if value]
    match_speakers = [str(value) for value in row.get("script_matches", []) if value.get("speaker")]
    current_focus = speakers or match_speakers
    only_weak = bool(evidence) and all(item.get("rule_id") in weak_rules for item in evidence)
    non_target_focus = bool(current_focus) and all(not _target_speaker(value, aliases) for value in current_focus)
    flagged = relevance.get("confidence") == "medium" and only_weak and non_target_focus and not any(
        item.get("rule_id") in direct_rules for item in evidence
    )
    risk_evidence = []
    if flagged:
        risk_evidence = [
            {"source_type": "target_relevance", "rule_ids": [item["rule_id"] for item in evidence], "relation": "weak_target_context"},
            {"source_type": "current_dialogue_focus", "source_id": row["shot_id"], "text": " | ".join(current_focus), "relation": "non_target_current_focus"},
        ]
    return {"weak_context_conflict": flagged, "evidence": risk_evidence}
