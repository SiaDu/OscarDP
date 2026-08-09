from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from oscardp.shots.schema import json_dumps

from .alignment import ALIGNMENT_VERSION, align_subtitles
from .llm_review import (
    apply_alignment_responses,
    build_alignment_diagnostics,
    build_review_requests,
    validate_review_requests,
)
from .schema import AlignmentConfig, ContextOptions, read_jsonl
from .screenplay import PARSER_VERSION, audit_screenplay_structure, parse_screenplay
from .shot_mapping import map_shots
from .subtitles import load_clean_subtitles
from .validation import validate_data, validate_files


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _write_atomic(path, json_dumps(value, pretty=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_atomic(path, "".join(json_dumps(row) + "\n" for row in rows))


def _input_fingerprint(options: ContextOptions) -> dict[str, Any]:
    def describe(path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {"path": path.resolve().as_posix(), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {
        "schema_version": "1.0", "movie_key": options.movie_key,
        "alignment_version": ALIGNMENT_VERSION, "parser_version": PARSER_VERSION,
        "screenplay": describe(options.screenplay), "subtitle": describe(options.subtitle), "shots": describe(options.shots),
        "subtitle_language": options.subtitle_language, "alignment_threshold": options.alignment_threshold,
        "review_threshold": options.review_threshold,
        "semantic_model": None if options.disable_semantic else options.semantic_model,
        "llm_mode": options.llm_mode, "llm_responses": None if options.llm_responses is None else describe(options.llm_responses),
        "scene_interpolation_max_gap": options.scene_interpolation_max_gap,
        "review_local_window": options.review_local_window,
        "review_fallback_window": options.review_fallback_window,
        "review_candidate_limit": options.review_candidate_limit,
    }


def _logger(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"oscardp.script_context.{os.getpid()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def _title_from_path(path: Path) -> str:
    import re
    stem = re.sub(r"^tt\d{7,8}(?:[_\.\- ]+)?", "", path.stem, flags=re.IGNORECASE)
    separated = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)
    return re.sub(r"[_\.]+", " ", separated).strip()


def process_one(options: ContextOptions) -> dict[str, Any]:
    for path, label in ((options.screenplay, "screenplay"), (options.subtitle, "subtitle"), (options.shots, "shots")):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not 0 <= options.review_threshold <= options.alignment_threshold <= 1:
        raise ValueError("Require 0 <= review-threshold <= alignment-threshold <= 1")
    if options.llm_mode not in {"none", "export", "apply"}:
        raise ValueError("llm-mode must be none, export, or apply")
    if options.llm_mode == "apply" and options.llm_responses is None:
        raise ValueError("--llm-responses is required for --llm-mode apply")
    fingerprint = _input_fingerprint(options)
    outputs = {
        "context": options.output_dir / "movie_script_context.json",
        "alignment": options.output_dir / "subtitle_script_alignment.jsonl",
        "shots": options.output_dir / "shot_script_context.jsonl",
    }
    state_path = options.output_dir / "logs" / "script_context_state.json"
    if options.dry_run:
        return {"movie_key": options.movie_key, "status": "dry_run", "fingerprint": fingerprint, "outputs": {key: value.as_posix() for key, value in outputs.items()}}
    if options.resume and not options.overwrite and state_path.is_file() and all(path.is_file() for path in outputs.values()):
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("fingerprint") == fingerprint and state.get("phase") == "complete":
            validation = validate_files(outputs["context"], outputs["alignment"], outputs["shots"], options.shots)
            if validation.passed:
                return {"movie_key": options.movie_key, "status": "completed", "resumed": True, **state["statistics"], "validation_passed": True}
    if not options.overwrite and any(path.exists() for path in outputs.values()):
        raise RuntimeError("Stage 2 outputs exist but cannot be resumed; use --overwrite")
    options.output_dir.mkdir(parents=True, exist_ok=True)
    logger = _logger(options.output_dir / "logs" / "script_context.log")
    try:
        source_files = {"screenplay": options.screenplay.resolve().as_posix(), "subtitle": options.subtitle.resolve().as_posix(), "shots": options.shots.resolve().as_posix()}
        context = None
        if not options.overwrite and outputs["context"].is_file():
            existing_context = json.loads(outputs["context"].read_text(encoding="utf-8"))
            if existing_context.get("parser_version") == PARSER_VERSION and existing_context.get("movie", {}).get("movie_id") == options.movie_key and existing_context.get("source_files", {}).get("screenplay") == source_files["screenplay"]:
                context = existing_context
                logger.info("Reusing existing movie_script_context.json")
        if context is None:
            logger.info("Parsing screenplay %s", options.screenplay)
            context = parse_screenplay(options.screenplay, options.movie_key, _title_from_path(options.screenplay), source_files)
            _write_json(outputs["context"], context)
        parsing_audit = audit_screenplay_structure(context)
        context["parsing_audit"] = parsing_audit
        if parsing_audit["confirmed_structural_error_count"]:
            raise RuntimeError(
                "Pre-OpenAI screenplay structure gate failed: "
                f"{parsing_audit['confirmed_structural_error_count']} confirmed errors"
            )
        subtitles = load_clean_subtitles(options.subtitle, options.subtitle_language)
        logger.info("Cleaned subtitles=%d", len(subtitles))
        config = AlignmentConfig(
            alignment_threshold=options.alignment_threshold, review_threshold=options.review_threshold,
            semantic_model=None if options.disable_semantic else options.semantic_model,
        )
        alignments = align_subtitles(subtitles, context, options.movie_key, config)
        if options.llm_mode == "apply":
            alignments = apply_alignment_responses(alignments, read_jsonl(options.llm_responses), context)  # type: ignore[arg-type]
        _write_jsonl(outputs["alignment"], alignments)
        requests = build_review_requests(
            context, alignments, local_window=options.review_local_window,
            fallback_window=options.review_fallback_window,
            candidate_limit=options.review_candidate_limit,
            semantic_model=None if options.disable_semantic else options.semantic_model,
        )
        diagnostics = build_alignment_diagnostics(context, alignments, requests["alignment_requests"])
        request_errors = validate_review_requests(requests["alignment_requests"], context)
        if request_errors:
            raise RuntimeError("Review candidate validation failed: " + "; ".join(request_errors[:20]))
        if options.llm_mode == "export":
            for name, rows in requests.items():
                _write_jsonl(options.output_dir / "review" / f"{name}.jsonl", rows)
            _write_json(options.output_dir / "review" / "alignment_diagnostics.json", diagnostics)
        shots = read_jsonl(options.shots)
        shot_rows = map_shots(shots, alignments, context, options.movie_key, options.scene_interpolation_max_gap)
        _write_jsonl(outputs["shots"], shot_rows)
        validation = validate_data(context, alignments, shot_rows, shots)
        if not validation.passed:
            raise RuntimeError("Stage 2 validation failed: " + "; ".join(validation.errors[:20]))
        status_counts: dict[str, int] = {}
        for row in alignments:
            status = row["alignment"]["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        statistics = {
            **context["summary"], "cleaned_subtitle_count": len(subtitles),
            "alignment_status_counts": status_counts,
            "needs_review_count": sum(row["alignment"]["needs_review"] for row in alignments),
            "no_match_count": sum(row["alignment"]["status"] == "no_match" for row in alignments),
            "shot_context_count": len(shot_rows),
            "review_request_counts": {name: len(rows) for name, rows in requests.items()},
            "alignment_diagnostics": diagnostics,
        }
        _write_json(state_path, {"fingerprint": fingerprint, "phase": "complete", "statistics": statistics})
        logger.info("Validation passed statistics=%s", statistics)
        return {"movie_key": options.movie_key, "status": "completed", "resumed": False, **statistics, "validation_passed": True}
    finally:
        _close_logger(logger)


__all__ = ["ContextOptions", "process_one"]
