from __future__ import annotations

from typing import Any


DECISIONS = ("match", "no_match", "uncertain")
DECISION_BASES = (
    "exact_or_near_exact", "substring_or_minor_edit", "paraphrase",
    "split_across_blocks", "changed_or_improvised_dialogue", "song_or_non_dialogue",
    "ambiguous_short_utterance", "insufficient_context", "repeated_or_reordered_dialogue",
)


def alignment_response_schema(request: dict[str, Any] | None = None) -> dict[str, Any]:
    request_ids = None if request is None else [request["request_id"]]
    subtitle_ids = None if request is None else list(request["subtitle_ids"])
    candidate_ids = None if request is None else [item["block_id"] for item in request.get("dialogue_candidates", [])]
    return {
        "type": "object", "additionalProperties": False,
        "required": ["request_id", "resolutions"],
        "properties": {
            "request_id": {"type": "string", **({"enum": request_ids} if request_ids is not None else {})},
            "resolutions": {
                "type": "array",
                **({"minItems": len(subtitle_ids), "maxItems": len(subtitle_ids)} if subtitle_ids is not None else {}),
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["subtitle_id", "decision", "block_ids", "confidence", "decision_basis"],
                    "properties": {
                        "subtitle_id": {"type": "string", **({"enum": subtitle_ids} if subtitle_ids is not None else {})},
                        "decision": {"type": "string", "enum": list(DECISIONS)},
                        "block_ids": {"type": "array", "items": {"type": "string", **({"enum": candidate_ids} if candidate_ids is not None else {})}},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "decision_basis": {"type": "string", "enum": list(DECISION_BASES)},
                    },
                },
            },
        },
    }


SYSTEM_INSTRUCTIONS = """Resolve every supplied subtitle against only the supplied screenplay dialogue candidates, at the identifiable screenplay-turn level.
Return one resolution for every supplied subtitle in subtitle order. Normally preserve screenplay order.
Never invent or rewrite IDs, dialogue, speakers, scenes, or subtitle text. Select only supplied block IDs.
Minor expansion, contraction, translation variation, changed wording, repeated delivery, and subtitle fragmentation can still be a match when the final-film line clearly realizes the same conversational turn or dramatic intent.
For example, “Is it really necessary?” may match “Do you really need...”; “Please step out” may match “Could you get out...”; and a vocative or short fragment may attach to its surrounding screenplay turn.
A clearly identifiable turn may be reused when the final cut repeats it. A small local backward move may represent nearby reordered turns, such as a repeated “No, I am not...” returning to an earlier negative reply. Use repeated_or_reordered_dialogue for such matches. Never use it for a distant or cross-scene jump.
Use no_match only when no supplied screenplay turn can reasonably account for the subtitle. Use uncertain when a likely corresponding screenplay line is missing from the supplied candidates or the candidates are genuinely ambiguous. Candidate-recall failure must produce uncertain, never an invented block ID.
For match, select one or more adjacent candidate blocks from one scene. For no_match or uncertain, return an empty block_ids list.
Do not provide chain-of-thought or prose outside the required structured result."""
