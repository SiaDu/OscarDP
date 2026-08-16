from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .schema import RULESET_VERSION


@dataclass(frozen=True)
class SemanticRule:
    rule_id: str
    category: str
    pattern: re.Pattern[str]


def _rule(rule_id: str, category: str, expression: str) -> SemanticRule:
    return SemanticRule(rule_id, category, re.compile(expression, re.IGNORECASE))


RULES = (
    _rule("emotion_tears_v1", "emotion", r"\b(?:cr(?:y|ies|ied|ying)|sob(?:s|bed|bing)?|weep(?:s|ing)?|tearful|in tears)\b"),
    _rule("emotion_laughter_v1", "emotion", r"\b(?:laugh(?:s|ed|ing)?|chuckl(?:e|es|ed|ing)|giggl(?:e|es|ed|ing)|smil(?:e|es|ed|ing)|grin(?:s|ned|ning)?)\b"),
    _rule("emotion_distress_v1", "emotion", r"\b(?:angry|furious|upset|terrified|frightened|afraid|panicked|devastated|nervous)\b"),
    _rule("reaction_v1", "reaction", r"\b(?:react(?:s|ed|ing)?|flinch(?:es|ed|ing)?|freeze(?:s|ing)?|realiz(?:e|es|ed|ing)|notic(?:e|es|ed|ing)|stare(?:s|d|ing)?|gasp(?:s|ed|ing)?)\b"),
    _rule("silence_pause_v1", "silence", r"\b(?:silence|silent|pause(?:s|d)?|a beat|hesitat(?:e|es|ed|ing)|wait(?:s|ed|ing)?)\b"),
    _rule("physical_contact_v1", "physical_action", r"\b(?:grab(?:s|bed|bing)?|slap(?:s|ped|ping)?|punch(?:es|ed|ing)?|hit(?:s|ting)?|kiss(?:es|ed|ing)?|hug(?:s|ged|ging)?)\b"),
    _rule("physical_expression_v1", "physical_action", r"\b(?:trembl(?:e|es|ed|ing)|shak(?:e|es|ing)|nod(?:s|ded|ding)?|pace(?:s|d|ing)|kneel(?:s|ed|ing)?)\b"),
    _rule("conflict_v1", "conflict", r"\b(?:argu(?:e|es|ed|ing)|shout(?:s|ed|ing)?|yell(?:s|ed|ing)?|scream(?:s|ed|ing)?|threaten(?:s|ed|ing)?|fight(?:s|ing)?)\b"),
)


def _matching_evidence(
    text: str, source_type: str, source_id: str, relation: str, weight: float,
) -> list[dict[str, Any]]:
    return [
        {
            "rule_id": rule.rule_id,
            "category": rule.category,
            "source_type": source_type,
            "source_id": source_id,
            "relation": relation,
            "text": text,
            "weight": weight,
        }
        for rule in RULES if rule.pattern.search(text)
    ]


def _scene_id(row: dict[str, Any]) -> str | None:
    scene = row.get("scene")
    return scene.get("scene_id") if isinstance(scene, dict) else None


def mine_shot_semantics(
    rows: list[dict[str, Any]], screenplay: dict[str, Any], excluded_shot_ids: set[str],
) -> list[dict[str, Any]]:
    blocks = {
        block["block_id"]: block
        for scene in screenplay.get("script_scenes", [])
        for block in scene.get("script_blocks", [])
    }
    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        shot_id = row["shot_id"]
        excluded_reason = "pending_stage2_ambiguity" if shot_id in excluded_shot_ids else None
        if row.get("scene_transition"):
            excluded_reason = excluded_reason or "scene_transition"
        if not _scene_id(row):
            excluded_reason = excluded_reason or "unresolved_scene"
        evidence: list[dict[str, Any]] = []
        action_strength = 0.0
        local = row.get("local_script_context") or {}
        for relation in ("action_before", "action_during", "action_after"):
            weight = 0.45 if relation == "action_during" else 0.30
            for block_id in local.get(relation, []):
                block = blocks.get(block_id)
                if block and isinstance(block.get("text"), str):
                    found = _matching_evidence(block["text"], "script_action", block_id, relation, weight)
                    evidence.extend(found)
                    if found:
                        action_strength = max(action_strength, weight)
        for match in row.get("script_matches", []):
            block_id = match.get("block_id")
            block = blocks.get(block_id)
            parenthetical = block.get("parenthetical") if block else None
            if isinstance(parenthetical, str) and parenthetical.strip():
                found = _matching_evidence(parenthetical, "script_parenthetical", block_id, "direct_dialogue_match", 0.45)
                evidence.extend(found)
                if found:
                    action_strength = max(action_strength, 0.45)
        subtitle_strength = 0.0
        for subtitle in row.get("subtitles", []):
            text = subtitle.get("text")
            subtitle_id = subtitle.get("subtitle_id")
            if not isinstance(text, str) or not subtitle_id:
                continue
            found = _matching_evidence(text, "subtitle", subtitle_id, "overlaps_shot", 0.25)
            evidence.extend(found)
            if found:
                subtitle_strength = 0.25
            if re.search(r"(?:\.\.\.|—|--)$", text.strip()):
                evidence.append({
                    "rule_id": "subtitle_hesitation_or_interruption_v1", "category": "hesitation_or_interruption",
                    "source_type": "subtitle", "source_id": subtitle_id, "relation": "overlaps_shot",
                    "text": text, "weight": 0.25,
                })
                subtitle_strength = 0.25
        speakers = [speaker for speaker in row.get("dialogue_speakers", []) if speaker]
        interaction_strength = 0.0
        if len(set(speakers)) >= 2:
            evidence.append({
                "rule_id": "multi_speaker_interaction_v1", "category": "interaction",
                "source_type": "shot_structure", "source_id": shot_id, "relation": "within_shot",
                "text": " | ".join(dict.fromkeys(speakers)), "weight": 0.20,
            })
            interaction_strength = 0.20
        if not row.get("subtitles") and 0 < index < len(rows) - 1:
            previous, following = rows[index - 1], rows[index + 1]
            if (
                _scene_id(previous) == _scene_id(row) == _scene_id(following)
                and previous.get("subtitles") and following.get("subtitles")
            ):
                evidence.append({
                    "rule_id": "silent_between_dialogue_shots_v1", "category": "reaction_or_silence",
                    "source_type": "shot_structure", "source_id": shot_id,
                    "relation": "between_dialogue_shots", "text": "", "weight": 0.25,
                })
                subtitle_strength = max(subtitle_strength, 0.25)
        scene = row.get("scene") or {}
        context_confidence = max(0.0, min(1.0, float(scene.get("confidence") or 0.0)))
        semantic_score = min(1.0, action_strength + subtitle_strength + interaction_strength + 0.10 * context_confidence)
        categories = sorted({item["category"] for item in evidence})
        results.append({
            "source_index": index,
            "shot": row,
            "ruleset_version": RULESET_VERSION,
            "semantic_score": round(semantic_score, 6),
            "context_confidence": round(context_confidence, 6),
            "categories": categories,
            "evidence": evidence,
            "excluded_reason": excluded_reason,
        })
    return results
