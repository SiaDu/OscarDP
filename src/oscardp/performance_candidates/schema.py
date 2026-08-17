from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = "1.0"
PIPELINE_VERSION = "performance_candidates_v2_1"
RULESET_VERSION = "performance_semantic_rules_v1"


@dataclass(frozen=True)
class MiningOptions:
    release_manifest: Path
    output_root: Path
    face_model: Path
    nominees_file: Path
    movie_key: str = "tt12300742"
    performer_id: str | None = None
    performer_name: str | None = None
    face_model_sha256: str | None = None
    semantic_threshold: float = 0.35
    semantic_override_threshold: float = 0.75
    max_event_duration_sec: float = 30.0
    resume: bool = True
    overwrite: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    errors: list[str]
    performance_shot_count: int
    performance_event_count: int
    screening_audit_count: int
