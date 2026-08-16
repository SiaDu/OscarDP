from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = "1.0"
PIPELINE_VERSION = "performance_candidates_v1"
RULESET_VERSION = "performance_semantic_rules_v1"


@dataclass(frozen=True)
class MiningOptions:
    release_manifest: Path
    output_root: Path
    face_model: Path
    movie_key: str = "tt12300742"
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
