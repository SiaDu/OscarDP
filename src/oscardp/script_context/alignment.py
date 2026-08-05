from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any, Callable

from .schema import AlignmentConfig, CleanSubtitle


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold().replace("’", "'")
    value = re.sub(r"[^\w']+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def dialogue_blocks(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scene_index, scene in enumerate(context["script_scenes"]):
        for block in scene["script_blocks"]:
            if block["block_type"] == "dialogue":
                rows.append({**block, "scene_id": scene["scene_id"], "scene_index": scene_index})
    return rows


def _semantic_scorer(model_name: str | None, texts: list[str]) -> Callable[[str, int], float] | None:
    if not model_name:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is not installed") from exc
    model = SentenceTransformer(model_name, local_files_only=True)
    embeddings = model.encode(texts, normalize_embeddings=True)

    def score(query: str, index: int) -> float:
        query_embedding = model.encode([query], normalize_embeddings=True)[0]
        return max(0.0, min(1.0, float(query_embedding @ embeddings[index])))
    return score


def align_subtitles(
    subtitles: list[CleanSubtitle], context: dict[str, Any], movie_key: str,
    config: AlignmentConfig = AlignmentConfig(),
) -> list[dict[str, Any]]:
    try:
        from rapidfuzz.fuzz import ratio, token_set_ratio
    except ImportError as exc:
        raise RuntimeError('rapidfuzz is required; install with pip install -e ".[context]"') from exc
    blocks = dialogue_blocks(context)
    block_norm = [normalize_text(block["text"]) for block in blocks]
    exact_positions: dict[str, list[int]] = defaultdict(list)
    for index, text in enumerate(block_norm):
        if text:
            exact_positions[text].append(index)
    semantic = _semantic_scorer(config.semantic_model, [block["text"] for block in blocks])

    def lexical(left: str, right: str) -> float:
        if not left or not right:
            return 0.0
        left_tokens, right_tokens = left.split(), right.split()
        coverage = min(len(left_tokens), len(right_tokens)) / max(len(left_tokens), len(right_tokens))
        # token_set_ratio is deliberately coverage-weighted: without this,
        # a short fragment receives 1.0 against any longer line containing it
        # and defeats the explicit 1:2 / 2:1 sequence candidates.
        return max(ratio(left, right) / 100.0, token_set_ratio(left, right) / 100.0 * coverage)

    def candidate_score(sub_text: str, indices: tuple[int, ...]) -> tuple[float, float | None]:
        joined = " ".join(block_norm[index] for index in indices)
        lex = lexical(sub_text, joined)
        sem = None
        if semantic is not None and len(indices) == 1:
            sem = semantic(sub_text, indices[0])
        return (0.75 * lex + 0.25 * sem if sem is not None else lex), sem

    results: list[dict[str, Any]] = []
    cursor = 0
    subtitle_index = 0
    group_index = 0
    while subtitle_index < len(subtitles):
        subtitle = subtitles[subtitle_index]
        normalized = normalize_text(subtitle.cleaned_text)
        options: list[tuple[float, tuple[int, ...], str, float | None, int]] = []
        upper = min(len(blocks), cursor + config.dialogue_window)
        all_positions = exact_positions.get(normalized, [])
        # An exact phrase is an anchor only when it is globally unique and
        # inside the current sequence window. A repeated line whose earlier
        # occurrence is behind the cursor must not suddenly jump to a much
        # later reprise (common with songs and callbacks).
        if len(all_positions) == 1 and cursor <= all_positions[0] < upper:
            options.append((1.0, (all_positions[0],), "normalized_exact", None, 1))
        for block_index in range(cursor, upper):
            score, sem = candidate_score(normalized, (block_index,))
            options.append((score, (block_index,), "semantic_fuzzy" if sem is not None else "rapidfuzz", sem, 1))
            if block_index + 1 < upper:
                score2, sem2 = candidate_score(normalized, (block_index, block_index + 1))
                options.append((score2, (block_index, block_index + 1), "rapidfuzz_sequence_1_to_2", sem2, 1))
        if subtitle_index + 1 < len(subtitles):
            combined_sub = normalize_text(subtitle.cleaned_text + " " + subtitles[subtitle_index + 1].cleaned_text)
            for block_index in range(cursor, upper):
                if min(
                    lexical(normalized, block_norm[block_index]),
                    lexical(normalize_text(subtitles[subtitle_index + 1].cleaned_text), block_norm[block_index]),
                ) < 0.2:
                    continue
                score, sem = candidate_score(combined_sub, (block_index,))
                options.append((score, (block_index,), "rapidfuzz_sequence_2_to_1", sem, 2))
        options.sort(key=lambda item: (-item[0], item[1], item[4]))
        best = options[0] if options else (0.0, (), "no_match", None, 1)
        runner_up = next((item for item in options[1:] if item[1] != best[1]), None)
        margin = best[0] - (runner_up[0] if runner_up else 0.0)
        consume = best[4]
        if best[0] < config.review_threshold:
            selected: tuple[int, ...] = ()
            status, method, needs_review, consume = "no_match", "no_match", True, 1
        elif best[0] >= config.alignment_threshold and (best[2] == "normalized_exact" or margin >= config.candidate_margin):
            selected = best[1]
            status, method, needs_review = "auto_aligned", best[2], False
        else:
            selected = best[1]
            status, method, needs_review = "needs_review", best[2], True
            # A provisional 2:1 suggestion must not consume the next subtitle;
            # only confirmed sequence matches are allowed to become anchors.
            consume = 1
        group_index += 1
        group_id = f"align_{group_index:06d}"
        for offset in range(consume):
            current_sub = subtitles[subtitle_index + offset]
            matches = []
            for index in selected:
                block = blocks[index]
                lex = lexical(normalize_text(current_sub.cleaned_text), block_norm[index])
                sem_score = semantic(normalize_text(current_sub.cleaned_text), index) if semantic is not None else None
                combined = 0.75 * lex + 0.25 * sem_score if sem_score is not None else lex
                matches.append({
                    "block_id": block["block_id"], "speaker": block["speaker"],
                    "matched_text": block["text"], "lexical_score": round(lex, 6),
                    "semantic_score": None if sem_score is None else round(sem_score, 6),
                    "combined_score": round(combined, 6),
                })
            scene_id = blocks[selected[0]]["scene_id"] if selected else None
            results.append({
                "movie_id": movie_key, "subtitle_id": current_sub.subtitle_id,
                "alignment_group_id": group_id,
                "time": {"start": current_sub.start, "end": current_sub.end, "start_sec": current_sub.start_sec, "end_sec": current_sub.end_sec},
                "text": current_sub.cleaned_text, "scene_id": scene_id, "script_matches": matches,
                "alignment": {"method": method, "status": status, "candidate_margin": round(max(0.0, margin), 6), "needs_review": needs_review},
            })
        if selected and status == "auto_aligned":
            cursor = max(cursor, selected[-1] + 1)
        subtitle_index += consume
    return results
