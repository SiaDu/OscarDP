"""Stage 0A: read-first source preflight and safe cleanup planning."""

from __future__ import annotations

import csv
import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .media import SUBTITLE_EXTENSIONS, VIDEO_EXTENSIONS, MediaInfo, is_english, probe, probe_payload, stream_language

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
RELEASE_EXTENSIONS = {".nfo", ".txt", ".url", ".sfv", ".md5", ".sha1", ".torrent"}
ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z"}
EXECUTABLE_EXTENSIONS = {".exe", ".bat", ".cmd", ".scr"}
ID_PATTERN = re.compile(r"^(tt\d{7,10})(?:[_ .-]|$)", re.IGNORECASE)
AUXILIARY_NAME = re.compile(r"(?:^|[._ -])(sample|trailer|teaser|extras?|featurette|behind[._ -]?the[._ -]?scenes|bts|interview)(?:[._ -]|$)", re.IGNORECASE)
ENGLISH_NAME = re.compile(r"(?:^|[._ -])(en|eng|english)(?:[._ -]|$)", re.IGNORECASE)
FULL_MOVIE_SEC = 40 * 60
SHORT_VIDEO_SEC = 20 * 60

PREFLIGHT_FIELDS = [
    "movie_id", "movie_key", "movie_folder", "primary_video_path", "primary_video_original_name",
    "primary_video_canonical_name", "primary_video_confidence", "video_candidate_count", "auxiliary_video_count",
    "multiple_full_movie_candidates", "external_subtitle_count", "english_external_subtitle_count",
    "embedded_subtitle_count", "has_english_subtitle", "image_asset_count", "release_metadata_count",
    "archive_count", "unknown_file_count", "rename_required", "rename_status", "cleanup_candidate_count",
    "cleanup_status", "manual_review_required", "manual_review_reasons",
]
PLAN_FIELDS = ["movie_key", "source_relpath", "action", "category", "destination_relpath", "status", "reason"]


@dataclass(frozen=True)
class PreflightOptions:
    input_root: Path
    report_dir: Path
    apply_renames: bool = False
    quarantine: bool = False
    quarantine_root: Path | None = None
    include_unknown_quarantine: bool = False
    limit: int | None = None


def _files(folder: Path) -> list[Path]:
    return [path for path in sorted(folder.rglob("*"), key=lambda item: item.as_posix().lower()) if path.is_file() and not any(part.startswith(".") for part in path.relative_to(folder).parts)]


def _subtitle_language(path: Path) -> str:
    if ENGLISH_NAME.search(path.stem):
        return "en"
    try:
        payload = probe_payload(path)
    except Exception:
        return "und"
    streams = payload.get("streams") if isinstance(payload.get("streams"), list) else []
    for stream in streams:
        if isinstance(stream, dict) and stream.get("codec_type") == "subtitle":
            language = stream_language(stream)
            if language != "und":
                return language
    return "und"


def _video(path: Path) -> tuple[MediaInfo | None, str | None]:
    try:
        return probe(path), None
    except Exception as exc:
        return None, str(exc)


def _candidate(path: Path, info: MediaInfo | None, error: str | None) -> dict[str, Any]:
    duration = info.duration_sec if info else 0.0
    marked_auxiliary = bool(AUXILIARY_NAME.search(path.stem)) or (info is not None and duration < SHORT_VIDEO_SEC)
    return {
        "path": path, "duration_sec": duration, "info": info, "ffprobe_error": error,
        "auxiliary": marked_auxiliary, "full_length": bool(info and duration >= FULL_MOVIE_SEC and not marked_auxiliary),
    }


def _select_primary(candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str, list[str], bool]:
    reasons: list[str] = []
    viable = [item for item in candidates if item["info"] is not None and not item["auxiliary"]]
    full = sorted((item for item in viable if item["full_length"]), key=lambda item: item["duration_sec"], reverse=True)
    if len(full) >= 2 and full[1]["duration_sec"] >= full[0]["duration_sec"] * 0.90:
        reasons.extend(["MULTIPLE_FULL_MOVIE_CANDIDATES", "MANUAL_REVIEW"])
        return None, "ambiguous", reasons, True
    if full:
        return full[0], "high", reasons, False
    if len(viable) == 1:
        reasons.extend(["UNCERTAIN_PRIMARY_VIDEO", "MANUAL_REVIEW"])
        return viable[0], "low", reasons, True
    if viable:
        ordered = sorted(viable, key=lambda item: item["duration_sec"], reverse=True)
        reasons.extend(["UNCERTAIN_PRIMARY_VIDEO", "MANUAL_REVIEW"])
        return ordered[0], "low", reasons, True
    reasons.extend(["NO_USABLE_PRIMARY_VIDEO", "MANUAL_REVIEW"])
    return None, "none", reasons, True


def _category(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS: return "IMAGE_ASSET"
    if suffix in RELEASE_EXTENSIONS: return "RELEASE_METADATA"
    if suffix in ARCHIVE_EXTENSIONS: return "ARCHIVE"
    if suffix in EXECUTABLE_EXTENSIONS: return "EXECUTABLE_OR_SCRIPT"
    return "UNKNOWN"


def _move_safely(source: Path, destination: Path) -> str:
    if destination.exists():
        return "SKIPPED_DESTINATION_EXISTS"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return "MOVED"


def _rename_safely(source: Path, destination: Path) -> str:
    if source == destination: return "NOT_REQUIRED"
    if destination.exists(): return "SKIPPED_DESTINATION_EXISTS"
    source.rename(destination)
    return "RENAMED"


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def run(options: PreflightOptions) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not options.input_root.is_dir(): raise FileNotFoundError(f"Input root does not exist: {options.input_root}")
    folders = [path for path in sorted(options.input_root.iterdir(), key=lambda item: item.name.lower()) if path.is_dir() and not path.name.startswith(".")]
    if options.limit is not None: folders = folders[:options.limit]
    quarantine_root = options.quarantine_root or options.input_root.with_name(options.input_root.name + "_cleanup_quarantine")
    rows: list[dict[str, Any]] = []; plan: list[dict[str, Any]] = []
    for folder in folders:
        match = ID_PATTERN.match(folder.name)
        movie_id = match.group(1).lower() if match else ""
        reasons = [] if movie_id else ["MISSING_MOVIE_ID"]
        files = _files(folder)
        videos = [_candidate(path, *_video(path)) for path in files if path.suffix.lower() in VIDEO_EXTENSIONS]
        primary, confidence, selection_reasons, selection_review = _select_primary(videos)
        reasons.extend(selection_reasons)
        subtitles = [{"path": path, "language": _subtitle_language(path)} for path in files if path.suffix.lower() in SUBTITLE_EXTENSIONS]
        english_subtitles = [item for item in subtitles if is_english(item["language"])]
        if len(english_subtitles) > 1: reasons.append("MULTIPLE_ENGLISH_SUBTITLES")
        categories = Counter(_category(path) for path in files if path.suffix.lower() not in VIDEO_EXTENSIONS | SUBTITLE_EXTENSIONS)
        embedded = primary["info"].subtitle_streams if primary else ()
        embedded_languages = [stream_language(stream) for stream in embedded]
        canonical_name = ""; rename_required = False; rename_status = "NOT_REQUIRED"
        if primary:
            source = primary["path"]; canonical_name = f"{folder.name}{source.suffix.lower()}"; target = source.with_name(canonical_name)
            rename_required = source.name != canonical_name
            if rename_required:
                status = _rename_safely(source, target) if options.apply_renames else "PLANNED"
                rename_status = status
                plan.append({"movie_key": folder.name, "source_relpath": source.relative_to(options.input_root).as_posix(), "action": "RENAME", "category": "PRIMARY_VIDEO", "destination_relpath": target.relative_to(options.input_root).as_posix(), "status": status, "reason": "canonical primary filename"})
                if status == "SKIPPED_DESTINATION_EXISTS": reasons.extend(["CANONICAL_FILENAME_CONFLICT", "MANUAL_REVIEW"])
        for index, item in enumerate(english_subtitles, 1):
            source = item["path"]; suffix = source.suffix.lower(); suffix_part = ".en" if len(english_subtitles) == 1 else f".en.{index}"
            target = source.with_name(f"{folder.name}{suffix_part}{suffix}")
            if source.name != target.name:
                status = _rename_safely(source, target) if options.apply_renames else "PLANNED"
                plan.append({"movie_key": folder.name, "source_relpath": source.relative_to(options.input_root).as_posix(), "action": "RENAME", "category": "ENGLISH_SUBTITLE", "destination_relpath": target.relative_to(options.input_root).as_posix(), "status": status, "reason": "canonical English subtitle filename"})
                if status == "SKIPPED_DESTINATION_EXISTS": reasons.extend(["CANONICAL_FILENAME_CONFLICT", "MANUAL_REVIEW"])
        for item in videos:
            if item["auxiliary"]:
                source = item["path"]; destination = quarantine_root / source.relative_to(options.input_root)
                status = _move_safely(source, destination) if options.quarantine else "PLANNED"
                plan.append({"movie_key": folder.name, "source_relpath": source.relative_to(options.input_root).as_posix(), "action": "QUARANTINE", "category": "AUXILIARY_VIDEO", "destination_relpath": destination.relative_to(quarantine_root).as_posix(), "status": status, "reason": "auxiliary name or short duration"})
        for path in files:
            if path.suffix.lower() in VIDEO_EXTENSIONS | SUBTITLE_EXTENSIONS: continue
            category = _category(path)
            eligible = category in {"IMAGE_ASSET", "RELEASE_METADATA", "EXECUTABLE_OR_SCRIPT"} or (category == "UNKNOWN" and options.include_unknown_quarantine)
            if eligible:
                destination = quarantine_root / path.relative_to(options.input_root)
                status = _move_safely(path, destination) if options.quarantine else "PLANNED"
                plan.append({"movie_key": folder.name, "source_relpath": path.relative_to(options.input_root).as_posix(), "action": "QUARANTINE", "category": category, "destination_relpath": destination.relative_to(quarantine_root).as_posix(), "status": status, "reason": "safe cleanup category"})
        manual = selection_review or any(reason in {"MISSING_MOVIE_ID", "MULTIPLE_ENGLISH_SUBTITLES", "CANONICAL_FILENAME_CONFLICT"} for reason in reasons)
        has_embedded_english = any(is_english(language) for language in embedded_languages)
        row = {"movie_id": movie_id, "movie_key": folder.name, "movie_folder": folder.relative_to(options.input_root).as_posix(), "primary_video_path": primary["path"].relative_to(options.input_root).as_posix() if primary else "", "primary_video_original_name": primary["path"].name if primary else "", "primary_video_canonical_name": canonical_name, "primary_video_confidence": confidence, "video_candidate_count": len(videos), "auxiliary_video_count": sum(item["auxiliary"] for item in videos), "multiple_full_movie_candidates": "MULTIPLE_FULL_MOVIE_CANDIDATES" in reasons, "external_subtitle_count": len(subtitles), "english_external_subtitle_count": len(english_subtitles), "embedded_subtitle_count": len(embedded), "has_embedded_english_subtitle": has_embedded_english, "has_english_subtitle": bool(english_subtitles or has_embedded_english), "image_asset_count": categories["IMAGE_ASSET"], "release_metadata_count": categories["RELEASE_METADATA"], "archive_count": categories["ARCHIVE"], "unknown_file_count": categories["UNKNOWN"], "rename_required": rename_required, "rename_status": rename_status, "cleanup_candidate_count": sum(item["movie_key"] == folder.name and item["action"] == "QUARANTINE" for item in plan), "cleanup_status": "PLANNED" if any(item["movie_key"] == folder.name and item["action"] == "QUARANTINE" for item in plan) else "NONE", "manual_review_required": manual, "manual_review_reasons": ";".join(dict.fromkeys(reasons))}
        rows.append(row)
    options.report_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(options.report_dir / "stage0_source_preflight.csv", PREFLIGHT_FIELDS, rows)
    with (options.report_dir / "stage0_source_preflight.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_csv(options.report_dir / "stage0_cleanup_plan.csv", PLAN_FIELDS, plan)
    summary = {"movie_directory_count": len(rows), "primary_movies_confidently_identified": sum(row["primary_video_confidence"] == "high" for row in rows), "manual_review_count": sum(bool(row["manual_review_required"]) for row in rows), "multiple_video_candidate_count": sum(int(row["video_candidate_count"]) > 1 for row in rows), "external_english_subtitle_movie_count": sum(int(row["english_external_subtitle_count"]) > 0 for row in rows), "embedded_english_subtitle_movie_count": sum(bool(row["has_embedded_english_subtitle"]) for row in rows), "no_detected_english_subtitle_count": sum(not bool(row["has_english_subtitle"]) for row in rows), "image_asset_count": sum(int(row["image_asset_count"]) for row in rows), "release_metadata_count": sum(int(row["release_metadata_count"]) for row in rows), "archive_count": sum(int(row["archive_count"]) for row in rows), "auxiliary_video_count": sum(int(row["auxiliary_video_count"]) for row in rows), "unknown_file_count": sum(int(row["unknown_file_count"]) for row in rows), "planned_rename_count": sum(item["action"] == "RENAME" and item["status"] == "PLANNED" for item in plan), "quarantine_eligible_count": sum(item["action"] == "QUARANTINE" for item in plan), "file_categories": dict(Counter(item["category"] for item in plan if item["action"] == "QUARANTINE")), "manual_review_cases": [row["movie_key"] for row in rows if row["manual_review_required"]], "representative_planned_renames": [item for item in plan if item["action"] == "RENAME"][:10]}
    (options.report_dir / "stage0_cleanup_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows, plan, summary
