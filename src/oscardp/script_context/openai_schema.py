from __future__ import annotations

from typing import Any


DECISIONS = ("match", "no_match", "uncertain")
DECISION_BASES = (
    "exact_or_near_exact", "substring_or_minor_edit", "paraphrase",
    "split_across_blocks", "changed_or_improvised_dialogue", "song_or_non_dialogue",
    "ambiguous_short_utterance", "insufficient_context",
)


def alignment_response_schema() -> dict[str, Any]:
    return {
        "type": "object", "additionalProperties": False,
        "required": ["request_id", "resolutions"],
        "properties": {
            "request_id": {"type": "string"},
            "resolutions": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["subtitle_id", "decision", "block_ids", "confidence", "decision_basis"],
                    "properties": {
                        "subtitle_id": {"type": "string"},
                        "decision": {"type": "string", "enum": list(DECISIONS)},
                        "block_ids": {"type": "array", "items": {"type": "string"}},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "decision_basis": {"type": "string", "enum": list(DECISION_BASES)},
                    },
                },
            },
        },
    }


SYSTEM_INSTRUCTIONS = """Resolve only the supplied subtitle records against only the supplied screenplay dialogue candidates.
Return one resolution for every supplied subtitle, in subtitle order. Preserve screenplay order.
Never invent or rewrite IDs, dialogue, speakers, scenes, or subtitle text. Select only supplied block IDs.
Use uncertain rather than guessing. Distinguish changed or improvised final-film dialogue from screenplay dialogue.
For match, select one or more adjacent candidate blocks from one scene. For no_match or uncertain, return an empty block_ids list.
Do not provide chain-of-thought or prose outside the required structured result."""
