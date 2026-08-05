from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .schema import AlignmentConfig, CleanSubtitle


TOKEN_RE = re.compile(r"[\w]+(?:['’][\w]+)?", re.UNICODE)
GENERIC_SHORT = {"yes", "no", "what", "okay", "ok", "right", "hey", "hello", "thanks", "thank you"}


@dataclass(frozen=True)
class TokenText:
    tokens: tuple[str, ...]
    offsets: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ScriptUnit:
    block: dict[str, Any]
    block_index: int
    scene_index: int
    token_text: TokenText


@dataclass(frozen=True)
class Anchor:
    subtitle_index: int
    block_index: int
    token_start: int
    token_end: int
    method: str
    score: float


@dataclass(frozen=True)
class SpanMatch:
    block_indices: tuple[int, ...]
    spans: tuple[tuple[int, int], ...]
    method: str
    lexical_score: float
    combined_score: float
    subtitle_coverage: float
    margin: float = 0.0


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text).tokens)


def tokenize(text: str) -> TokenText:
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    normalized_source = unicodedata.normalize("NFKC", text).replace("’", "'")
    for match in TOKEN_RE.finditer(normalized_source):
        token = match.group(0).casefold()
        tokens.append(token)
        offsets.append((match.start(), match.end()))
    return TokenText(tuple(tokens), tuple(offsets))


def dialogue_blocks(context: dict[str, Any]) -> list[dict[str, Any]]:
    return [unit.block for unit in build_script_units(context)]


def build_script_units(context: dict[str, Any]) -> list[ScriptUnit]:
    units: list[ScriptUnit] = []
    for scene_index, scene in enumerate(context["script_scenes"]):
        for block in scene["script_blocks"]:
            if block["block_type"] == "dialogue":
                public = {**block, "scene_id": scene["scene_id"], "scene_index": scene_index}
                units.append(ScriptUnit(public, len(units), scene_index, tokenize(block["text"])))
    return units


def _subsequence_positions(needle: tuple[str, ...], haystack: tuple[str, ...], start: int = 0, end: int | None = None) -> list[int]:
    if not needle:
        return []
    stop = len(haystack) if end is None else min(end, len(haystack))
    width = len(needle)
    return [index for index in range(start, stop - width + 1) if haystack[index:index + width] == needle]


def _informative(tokens: tuple[str, ...]) -> bool:
    joined = " ".join(tokens)
    return len(tokens) >= 4 or (len(tokens) >= 2 and len(joined) >= 15 and joined not in GENERIC_SHORT)


def build_anchors(subtitles: list[CleanSubtitle], units: list[ScriptUnit]) -> list[Anchor]:
    candidates: list[Anchor] = []
    for subtitle_index, subtitle in enumerate(subtitles):
        sub_tokens = tokenize(subtitle.cleaned_text).tokens
        if not _informative(sub_tokens):
            continue
        occurrences: list[tuple[int, int]] = []
        for unit in units:
            occurrences.extend((unit.block_index, start) for start in _subsequence_positions(sub_tokens, unit.token_text.tokens))
        if len(occurrences) != 1:
            continue
        block_index, start = occurrences[0]
        method = "anchor_normalized_exact" if len(sub_tokens) == len(units[block_index].token_text.tokens) else "anchor_unique_substring"
        candidates.append(Anchor(subtitle_index, block_index, start, start + len(sub_tokens), method, 1.0))
    if not candidates:
        return []

    # Maximum-cardinality monotonic chain. Same-block anchors are permitted
    # when their token spans progress, which is essential for long dialogue.
    lengths = [1] * len(candidates)
    previous = [-1] * len(candidates)
    for right, candidate in enumerate(candidates):
        for left in range(right):
            earlier = candidates[left]
            ordered = earlier.block_index < candidate.block_index or (
                earlier.block_index == candidate.block_index and earlier.token_end <= candidate.token_start
            )
            if ordered and lengths[left] + 1 > lengths[right]:
                lengths[right] = lengths[left] + 1
                previous[right] = left
    cursor = max(range(len(candidates)), key=lambda index: (lengths[index], -candidates[index].subtitle_index))
    chain: list[Anchor] = []
    while cursor >= 0:
        chain.append(candidates[cursor])
        cursor = previous[cursor]
    return list(reversed(chain))


def _fuzzy(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    if not left or not right:
        return 0.0
    from rapidfuzz.fuzz import ratio, token_set_ratio
    left_text, right_text = " ".join(left), " ".join(right)
    coverage = min(len(left), len(right)) / max(len(left), len(right))
    return max(ratio(left_text, right_text) / 100.0, token_set_ratio(left_text, right_text) / 100.0 * coverage)


def _single_block_candidate(sub_tokens: tuple[str, ...], unit: ScriptUnit, token_start: int, token_end: int) -> SpanMatch | None:
    available = unit.token_text.tokens[token_start:token_end]
    positions = _subsequence_positions(sub_tokens, unit.token_text.tokens, token_start, token_end)
    if positions:
        start = positions[0]
        method = "normalized_exact" if len(sub_tokens) == len(unit.token_text.tokens) else "normalized_substring"
        script_coverage = len(sub_tokens) / max(1, len(unit.token_text.tokens))
        combined = 1.0 if method == "normalized_exact" else min(0.99, 0.9 + 0.1 * script_coverage)
        return SpanMatch((unit.block_index,), ((start, start + len(sub_tokens)),), method, 1.0, combined, 1.0)
    reverse = _subsequence_positions(available, sub_tokens)
    if reverse and available:
        return SpanMatch((unit.block_index,), ((token_start, token_end),), "script_substring", 1.0, 0.9, len(available) / len(sub_tokens))
    if not available:
        return None
    target = len(sub_tokens)
    best: tuple[float, int, int] | None = None
    for width in range(max(1, target - 3), min(len(available), target + 5) + 1):
        for local_start in range(0, len(available) - width + 1):
            score = _fuzzy(sub_tokens, available[local_start:local_start + width])
            candidate = (score, token_start + local_start, token_start + local_start + width)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return None
    score, start, end = best
    return SpanMatch((unit.block_index,), ((start, end),), "rapidfuzz_token_span", score, score, min(1.0, (end - start) / max(1, len(sub_tokens))))


def _two_block_candidate(sub_tokens: tuple[str, ...], left: ScriptUnit, right: ScriptUnit, left_start: int, right_end: int) -> SpanMatch | None:
    left_tokens = left.token_text.tokens[left_start:]
    right_tokens = right.token_text.tokens[:right_end]
    best: tuple[float, int, int] | None = None
    for left_count in range(1, min(len(left_tokens), len(sub_tokens) - 1) + 1):
        for right_count in range(1, min(len(right_tokens), len(sub_tokens) - left_count + 3) + 1):
            combined = left_tokens[-left_count:] + right_tokens[:right_count]
            score = _fuzzy(sub_tokens, combined)
            candidate = (score, left_count, right_count)
            if best is None or candidate > best:
                best = candidate
    if best is None:
        return None
    score, left_count, right_count = best
    return SpanMatch(
        (left.block_index, right.block_index),
        ((len(left.token_text.tokens) - left_count, len(left.token_text.tokens)), (0, right_count)),
        "rapidfuzz_multi_block", score, score, min(1.0, (left_count + right_count) / max(1, len(sub_tokens))),
    )


def _candidate_bounds(anchor_before: Anchor | None, anchor_after: Anchor | None, unit_count: int, window: int) -> tuple[int, int, int, int] | None:
    if anchor_before is None and anchor_after is None:
        return None
    if anchor_before is None:
        low, high = max(0, anchor_after.block_index - window), anchor_after.block_index
        return low, high, 0, anchor_after.token_start
    if anchor_after is None:
        low, high = anchor_before.block_index, min(unit_count - 1, anchor_before.block_index + window)
        return low, high, anchor_before.token_end, 10**9
    return anchor_before.block_index, anchor_after.block_index, anchor_before.token_end, anchor_after.token_start


def _best_local_match(
    sub_tokens: tuple[str, ...], units: list[ScriptUnit], low: int, high: int,
    cursor_block: int, cursor_token: int, final_token: int,
) -> SpanMatch | None:
    candidates: list[SpanMatch] = []
    for block_index in range(max(low, cursor_block), high + 1):
        unit = units[block_index]
        start = cursor_token if block_index == cursor_block else 0
        end = final_token if block_index == high else len(unit.token_text.tokens)
        end = min(end, len(unit.token_text.tokens))
        # Exact fragments may legitimately overlap another subtitle span in
        # the same long screenplay block. Check containment across the full
        # locally valid block before applying the progressive fuzzy cursor.
        overlap_exact = _single_block_candidate(sub_tokens, unit, 0, end)
        if overlap_exact and overlap_exact.method in {"normalized_exact", "normalized_substring"}:
            gap = block_index - cursor_block
            adjusted = max(0.0, overlap_exact.combined_score - min(0.12, gap * 0.004))
            candidates.append(SpanMatch(overlap_exact.block_indices, overlap_exact.spans, overlap_exact.method, overlap_exact.lexical_score, adjusted, overlap_exact.subtitle_coverage))
        candidate = _single_block_candidate(sub_tokens, unit, start, end)
        if candidate:
            gap = block_index - cursor_block
            adjusted = max(0.0, candidate.combined_score - min(0.12, gap * 0.004))
            candidates.append(SpanMatch(candidate.block_indices, candidate.spans, candidate.method, candidate.lexical_score, adjusted, candidate.subtitle_coverage))
        if block_index < high:
            multi = _two_block_candidate(sub_tokens, unit, units[block_index + 1], start, final_token if block_index + 1 == high else len(units[block_index + 1].token_text.tokens))
            if multi:
                gap = block_index - cursor_block
                adjusted = max(0.0, multi.combined_score - min(0.12, gap * 0.004) - 0.015)
                candidates.append(SpanMatch(multi.block_indices, multi.spans, multi.method, multi.lexical_score, adjusted, multi.subtitle_coverage))
    if not candidates:
        return None
    candidates.sort(key=lambda value: (-value.combined_score, value.block_indices, value.spans))
    best = candidates[0]
    second_score = next((item.combined_score for item in candidates[1:] if (item.block_indices, item.spans) != (best.block_indices, best.spans)), 0.0)
    return SpanMatch(best.block_indices, best.spans, best.method, best.lexical_score, best.combined_score, best.subtitle_coverage, max(0.0, best.combined_score - second_score))


def _match_payload(match: SpanMatch, units: list[ScriptUnit], subtitle_tokens: int) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for block_index, (start, end) in zip(match.block_indices, match.spans):
        unit = units[block_index]
        offsets = unit.token_text.offsets
        char_start = offsets[start][0] if start < len(offsets) else len(unit.block["text"])
        char_end = offsets[end - 1][1] if end > start and end <= len(offsets) else char_start
        payload.append({
            "block_id": unit.block["block_id"], "speaker": unit.block["speaker"], "matched_text": unit.block["text"],
            "matched_script_token_start": start, "matched_script_token_end": end,
            "matched_script_char_start": char_start, "matched_script_char_end": char_end,
            "subtitle_token_coverage": round(match.subtitle_coverage, 6),
            "script_token_coverage": round((end - start) / max(1, len(unit.token_text.tokens)), 6),
            "lexical_score": round(match.lexical_score, 6), "semantic_score": None,
            "combined_score": round(match.combined_score, 6), "screenplay_order": block_index,
        })
    return payload


def align_subtitles(subtitles: list[CleanSubtitle], context: dict[str, Any], movie_key: str, config: AlignmentConfig = AlignmentConfig()) -> list[dict[str, Any]]:
    units = build_script_units(context)
    anchors = build_anchors(subtitles, units)
    anchor_by_subtitle = {anchor.subtitle_index: anchor for anchor in anchors}
    results: list[dict[str, Any] | None] = [None] * len(subtitles)

    def make_row(index: int, match: SpanMatch | None, status: str, method: str, review: bool, reliable: bool, reason: str | None = None) -> dict[str, Any]:
        subtitle = subtitles[index]
        matches = _match_payload(match, units, len(tokenize(subtitle.cleaned_text).tokens)) if match else []
        scene_id = units[match.block_indices[0]].block["scene_id"] if match else None
        alignment = {
            "method": method, "status": status, "candidate_margin": round(match.margin if match else 0.0, 6),
            "needs_review": review, "reliable_anchor": reliable,
            "script_order_start": match.block_indices[0] if match else None,
            "script_order_end": match.block_indices[-1] if match else None,
        }
        if reason:
            alignment["review_reason"] = reason
        return {
            "movie_id": movie_key, "subtitle_id": subtitle.subtitle_id, "alignment_group_id": "",
            "time": {"start": subtitle.start, "end": subtitle.end, "start_sec": subtitle.start_sec, "end_sec": subtitle.end_sec},
            "text": subtitle.cleaned_text, "scene_id": scene_id, "script_matches": matches, "alignment": alignment,
        }

    for anchor in anchors:
        match = SpanMatch((anchor.block_index,), ((anchor.token_start, anchor.token_end),), anchor.method, anchor.score, anchor.score, 1.0, 1.0)
        results[anchor.subtitle_index] = make_row(anchor.subtitle_index, match, "auto_aligned", anchor.method, False, True)

    boundaries: list[tuple[int, Anchor | None, Anchor | None]] = []
    previous_sub = -1
    previous_anchor: Anchor | None = None
    for anchor in anchors:
        boundaries.append((previous_sub + 1, previous_anchor, anchor))
        previous_sub, previous_anchor = anchor.subtitle_index, anchor
    boundaries.append((previous_sub + 1, previous_anchor, None))

    for start_sub, before, after in boundaries:
        end_sub = after.subtitle_index if after else len(subtitles)
        if start_sub >= end_sub:
            continue
        bounds = _candidate_bounds(before, after, len(units), config.dialogue_window)
        if bounds is None:
            for index in range(start_sub, end_sub):
                results[index] = make_row(index, None, "no_match", "no_match", True, False, "insufficient_candidates")
            continue
        low, high, initial_token, final_token = bounds
        cursor_block, cursor_token = low, initial_token
        local_exact: dict[int, tuple[int, int, int]] = {}
        for probe_index in range(start_sub, end_sub):
            probe_tokens = tokenize(subtitles[probe_index].cleaned_text).tokens
            if len(probe_tokens) < 4:
                continue
            occurrences: list[tuple[int, int, int]] = []
            for block_index in range(low, high + 1):
                block_end = final_token if block_index == high else len(units[block_index].token_text.tokens)
                for token_start in _subsequence_positions(probe_tokens, units[block_index].token_text.tokens, 0, block_end):
                    occurrences.append((block_index, token_start, token_start + len(probe_tokens)))
            if len(occurrences) == 1:
                local_exact[probe_index] = occurrences[0]
        for index in range(start_sub, end_sub):
            sub_tokens = tokenize(subtitles[index].cleaned_text).tokens
            exact = local_exact.get(index)
            if exact is not None and exact[0] >= cursor_block:
                block_index, token_start, token_end = exact
                method = "normalized_exact" if len(sub_tokens) == len(units[block_index].token_text.tokens) else "normalized_substring"
                candidate = SpanMatch((block_index,), ((token_start, token_end),), method, 1.0, 1.0 if method == "normalized_exact" else 0.99, 1.0, 1.0)
            else:
                future = next((local_exact[future_index] for future_index in range(index + 1, end_sub) if future_index in local_exact and local_exact[future_index][0] >= cursor_block), None)
                effective_high = min(high, future[0]) if future else high
                effective_final = future[1] if future and future[0] == effective_high else final_token if effective_high == high else len(units[effective_high].token_text.tokens)
                candidate = _best_local_match(sub_tokens, units, low, effective_high, cursor_block, cursor_token, effective_final)
            informative = _informative(sub_tokens)
            if candidate and candidate.method == "normalized_substring" and len(sub_tokens) >= 4:
                status, review = "auto_aligned", False
            elif candidate and informative and candidate.combined_score >= config.alignment_threshold and candidate.margin >= config.candidate_margin:
                status, review = "auto_aligned", False
            elif candidate and candidate.combined_score >= config.review_threshold:
                status, review = "needs_review", True
            else:
                results[index] = make_row(index, None, "no_match", "no_match", True, False, "below_review_threshold")
                continue
            results[index] = make_row(index, candidate, status, candidate.method, review, exact is not None and not review, "low_confidence" if review else None)
            if status == "auto_aligned":
                cursor_block = candidate.block_indices[-1]
                cursor_token = candidate.spans[-1][1]
                if cursor_token >= len(units[cursor_block].token_text.tokens) and cursor_block < high:
                    cursor_block += 1
                    cursor_token = 0

    final = [row for row in results if row is not None]
    group = 0
    previous_key: tuple[int | None, int | None] | None = None
    for row in final:
        key = (row["alignment"]["script_order_start"], row["alignment"]["script_order_end"])
        if key != previous_key or key == (None, None):
            group += 1
        row["alignment_group_id"] = f"align_{group:06d}"
        previous_key = key
    return final
