from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oscardp.artifact_paths import parse_path_maps
from oscardp.shots.schema import json_dumps

from .alignment import align_subtitles
from .pipeline import ContextOptions, _write_json, _write_jsonl, process_one
from .schema import AlignmentConfig, read_jsonl
from .screenplay import parse_screenplay
from .shot_mapping import map_shots
from .subtitles import load_clean_subtitles
from .validation import validate_files


def _common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--movie-key", required=True)
    parser.add_argument("--screenplay", type=Path, required=True)
    parser.add_argument("--subtitle", type=Path, required=True)
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m oscardp.script_context")
    commands = parser.add_subparsers(dest="command", required=True)
    parse = commands.add_parser("parse-script")
    parse.add_argument("--movie-key", required=True); parse.add_argument("--screenplay", type=Path, required=True)
    parse.add_argument("--subtitle", type=Path); parse.add_argument("--shots", type=Path); parse.add_argument("--output", type=Path, required=True)
    align = commands.add_parser("align-subtitles")
    align.add_argument("--movie-key", required=True); align.add_argument("--screenplay-context", type=Path, required=True)
    align.add_argument("--subtitle", type=Path, required=True); align.add_argument("--output", type=Path, required=True)
    align.add_argument("--subtitle-language", default="en"); align.add_argument("--alignment-threshold", type=float, default=0.82)
    align.add_argument("--review-threshold", type=float, default=0.65); align.add_argument("--semantic-model"); align.add_argument("--disable-semantic", action="store_true")
    mapping = commands.add_parser("map-shots")
    mapping.add_argument("--movie-key", required=True); mapping.add_argument("--screenplay-context", type=Path, required=True)
    mapping.add_argument("--alignment", type=Path, required=True); mapping.add_argument("--shots", type=Path, required=True); mapping.add_argument("--output", type=Path, required=True)
    mapping.add_argument("--scene-interpolation-max-gap", type=float, default=10.0)
    process = commands.add_parser("process-one"); _common_paths(process)
    process.add_argument("--subtitle-language", default="en"); process.add_argument("--alignment-threshold", type=float, default=0.82)
    process.add_argument("--review-threshold", type=float, default=0.65); process.add_argument("--semantic-model"); process.add_argument("--disable-semantic", action="store_true")
    process.add_argument("--llm-mode", choices=("none", "export", "apply"), default="none"); process.add_argument("--llm-responses", type=Path)
    process.add_argument("--scene-interpolation-max-gap", type=float, default=10.0)
    process.add_argument("--review-local-window", type=int, default=40)
    process.add_argument("--review-fallback-window", type=int, default=240)
    process.add_argument("--review-candidate-limit", type=int, default=36)
    behavior = process.add_mutually_exclusive_group(); behavior.add_argument("--resume", action="store_true", default=True); behavior.add_argument("--overwrite", action="store_true")
    process.add_argument("--dry-run", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("--movie-key", required=True); validate.add_argument("--screenplay-context", type=Path, required=True)
    validate.add_argument("--alignment", type=Path, required=True); validate.add_argument("--shot-context", type=Path, required=True); validate.add_argument("--shots", type=Path, required=True)
    pilot = commands.add_parser("prepare-openai-pilot")
    pilot.add_argument("--requests", type=Path, required=True); pilot.add_argument("--alignment", type=Path, required=True)
    pilot.add_argument("--output-dir", type=Path, required=True); pilot.add_argument("--count", type=int, default=30)
    prepare_batch = commands.add_parser("prepare-openai-batch")
    prepare_batch.add_argument("--requests", type=Path, required=True); prepare_batch.add_argument("--output", type=Path, required=True); prepare_batch.add_argument("--model")
    prepare_batch_v3 = commands.add_parser("prepare-openai-batch-v3")
    prepare_batch_v3.add_argument("--requests", type=Path, required=True); prepare_batch_v3.add_argument("--annotation-policy", type=Path, required=True)
    prepare_batch_v3.add_argument("--output", type=Path, required=True); prepare_batch_v3.add_argument("--model", required=True)
    prepare_batch_v32 = commands.add_parser("prepare-openai-batch-v3-2-policy")
    prepare_batch_v32.add_argument("--requests", type=Path, required=True); prepare_batch_v32.add_argument("--annotation-policy", type=Path, required=True)
    prepare_batch_v32.add_argument("--output", type=Path, required=True); prepare_batch_v32.add_argument("--model", required=True)
    prepare_batch_v321 = commands.add_parser("prepare-openai-batch-v3-2-1-vocative")
    prepare_batch_v321.add_argument("--requests", type=Path, required=True); prepare_batch_v321.add_argument("--annotation-policy", type=Path, required=True)
    prepare_batch_v321.add_argument("--output", type=Path, required=True); prepare_batch_v321.add_argument("--model", required=True)
    prepare_batch_v33 = commands.add_parser("prepare-openai-batch-v3-3-action-context")
    prepare_batch_v33.add_argument("--requests", type=Path, required=True); prepare_batch_v33.add_argument("--annotation-policy", type=Path, required=True)
    prepare_batch_v33.add_argument("--output", type=Path, required=True); prepare_batch_v33.add_argument("--model", required=True)
    context_v31 = commands.add_parser("prepare-review-context-v3-1")
    context_v31.add_argument("--requests", type=Path, required=True); context_v31.add_argument("--alignment", type=Path, required=True)
    context_v31.add_argument("--output", type=Path, required=True); context_v31.add_argument("--radius", type=int, default=2)
    context_v33 = commands.add_parser("prepare-review-action-context-v3-3")
    context_v33.add_argument("--requests", type=Path, required=True); context_v33.add_argument("--screenplay-context", type=Path, required=True)
    context_v33.add_argument("--output", type=Path, required=True); context_v33.add_argument("--radius", type=int, default=8)
    context_v33.add_argument("--max-actions", type=int, default=24)
    remaining = commands.add_parser("prepare-openai-remaining")
    remaining.add_argument("--full-requests", type=Path, required=True); remaining.add_argument("--pilot-requests", type=Path, required=True)
    remaining.add_argument("--output", type=Path, required=True); remaining.add_argument("--manifest", type=Path, required=True); remaining.add_argument("--model", required=True)
    submit = commands.add_parser("submit-openai-batch")
    submit.add_argument("--batch-input", type=Path, required=True); submit.add_argument("--job-file", type=Path, required=True); submit.add_argument("--confirm-submit", action="store_true")
    submit_v3 = commands.add_parser("submit-openai-batch-v3")
    submit_v3.add_argument("--batch-input", type=Path, required=True); submit_v3.add_argument("--job-file", type=Path, required=True); submit_v3.add_argument("--confirm-submit", action="store_true")
    submit_v32 = commands.add_parser("submit-openai-batch-v3-2-policy")
    submit_v32.add_argument("--batch-input", type=Path, required=True); submit_v32.add_argument("--job-file", type=Path, required=True); submit_v32.add_argument("--confirm-submit", action="store_true")
    submit_v321 = commands.add_parser("submit-openai-batch-v3-2-1-vocative")
    submit_v321.add_argument("--batch-input", type=Path, required=True); submit_v321.add_argument("--job-file", type=Path, required=True); submit_v321.add_argument("--confirm-submit", action="store_true")
    submit_v33 = commands.add_parser("submit-openai-batch-v3-3-action-context")
    submit_v33.add_argument("--batch-input", type=Path, required=True); submit_v33.add_argument("--job-file", type=Path, required=True); submit_v33.add_argument("--confirm-submit", action="store_true")
    production_remaining = commands.add_parser("prepare-openai-production-remaining-v3")
    production_remaining.add_argument("--full-requests", type=Path, required=True); production_remaining.add_argument("--pilot-requests", type=Path, required=True)
    production_remaining.add_argument("--output", type=Path, required=True); production_remaining.add_argument("--manifest", type=Path, required=True)
    production_remaining.add_argument("--reviewer-manifest", type=Path, required=True)
    production_bind_subset = commands.add_parser("bind-openai-production-request-subset-v3")
    production_bind_subset.add_argument("--source-requests", type=Path, required=True)
    production_bind_subset.add_argument("--subset-requests", type=Path, required=True)
    production_bind_subset.add_argument("--reviewer-manifest", type=Path, required=True)
    production_bind_subset.add_argument("--manifest", type=Path, required=True)
    production_batch = commands.add_parser("prepare-openai-production-batch-v3")
    production_batch.add_argument("--requests", type=Path, required=True); production_batch.add_argument("--reviewer-manifest", type=Path, required=True); production_batch.add_argument("--output", type=Path, required=True)
    production_preflight = commands.add_parser("preflight-openai-production-batch-v3")
    production_preflight.add_argument("--batch-input", type=Path, required=True); production_preflight.add_argument("--requests", type=Path, required=True)
    production_preflight.add_argument("--reviewer-manifest", type=Path, required=True); production_preflight.add_argument("--output", type=Path, required=True)
    production_split = commands.add_parser("split-openai-production-requests-v3")
    production_split.add_argument("--requests", type=Path, required=True); production_split.add_argument("--reviewer-manifest", type=Path, required=True)
    production_split.add_argument("--output-dir", type=Path, required=True); production_split.add_argument("--max-estimated-tokens", type=int, default=300_000)
    production_split.add_argument("--max-requests", type=int, default=100)
    production_submit = commands.add_parser("submit-openai-production-batch-v3")
    production_submit.add_argument("--batch-input", type=Path, required=True); production_submit.add_argument("--reviewer-manifest", type=Path, required=True)
    production_submit.add_argument("--job-file", type=Path, required=True); production_submit.add_argument("--confirm-submit", action="store_true")
    check = commands.add_parser("check-openai-batch"); check.add_argument("--job-file", type=Path, required=True)
    fetch = commands.add_parser("fetch-openai-batch"); fetch.add_argument("--job-file", type=Path, required=True); fetch.add_argument("--output-dir", type=Path, required=True)
    validate_openai = commands.add_parser("validate-openai-responses")
    validate_openai.add_argument("--raw-output", type=Path, required=True); validate_openai.add_argument("--requests", type=Path, required=True); validate_openai.add_argument("--output-dir", type=Path, required=True)
    validate_openai_v3 = commands.add_parser("validate-openai-responses-v3")
    validate_openai_v3.add_argument("--raw-output", type=Path, required=True); validate_openai_v3.add_argument("--requests", type=Path, required=True); validate_openai_v3.add_argument("--output-dir", type=Path, required=True)
    validate_openai_v3.add_argument("--hard-validation-contract", choices=("candidate_task_v3_structure_v2", "candidate_task_v3_structure_v3"), default="candidate_task_v3_structure_v2")
    validate_gold = commands.add_parser("validate-openai-pilot-gold")
    validate_gold.add_argument("--gold", type=Path, required=True); validate_gold.add_argument("--requests", type=Path, required=True)
    validate_gold.add_argument("--output", type=Path, required=True); validate_gold.add_argument("--max-backward-distance", type=int, default=3)
    apply_openai = commands.add_parser("apply-openai-responses")
    apply_openai.add_argument("--alignment", type=Path, required=True); apply_openai.add_argument("--requests", type=Path, required=True)
    apply_openai.add_argument("--validated-responses", type=Path, required=True); apply_openai.add_argument("--screenplay-context", type=Path, required=True)
    apply_openai.add_argument("--shots", type=Path, required=True); apply_openai.add_argument("--output-dir", type=Path, required=True); apply_openai.add_argument("--output-tag")
    merge = commands.add_parser("merge-openai-validated-responses")
    merge.add_argument("--full-requests", type=Path, required=True); merge.add_argument("--pilot-responses", type=Path, required=True)
    merge.add_argument("--remaining-responses", type=Path, required=True); merge.add_argument("--output", type=Path, required=True); merge.add_argument("--report", type=Path, required=True)
    production_merge = commands.add_parser("merge-openai-production-responses-v3")
    production_merge.add_argument("--full-requests", type=Path, required=True); production_merge.add_argument("--pilot-requests", type=Path, required=True)
    production_merge.add_argument("--remaining-requests", type=Path, required=True); production_merge.add_argument("--pilot-responses", type=Path, required=True)
    production_merge.add_argument("--remaining-responses", type=Path, required=True); production_merge.add_argument("--reviewer-manifest", type=Path, required=True)
    production_merge.add_argument("--output", type=Path, required=True); production_merge.add_argument("--report", type=Path, required=True)
    production_chunk_merge = commands.add_parser("merge-openai-production-response-chunks-v3")
    production_chunk_merge.add_argument("--chunk-manifest", type=Path, required=True)
    production_chunk_merge.add_argument("--chunk-response", type=Path, action="append", required=True)
    production_chunk_merge.add_argument("--reviewer-manifest", type=Path, required=True)
    production_chunk_merge.add_argument("--output", type=Path, required=True); production_chunk_merge.add_argument("--report", type=Path, required=True)
    production_apply = commands.add_parser("apply-openai-production-responses-v3")
    production_apply.add_argument("--alignment", type=Path, required=True); production_apply.add_argument("--requests", type=Path, required=True)
    production_apply.add_argument("--validated-responses", type=Path, required=True); production_apply.add_argument("--screenplay-context", type=Path, required=True)
    production_apply.add_argument("--shots", type=Path, required=True); production_apply.add_argument("--output-dir", type=Path, required=True)
    production_apply.add_argument("--reviewer-manifest", type=Path, required=True)
    production_risk = commands.add_parser("build-openai-production-risk-audit-v3")
    production_risk.add_argument("--requests", type=Path, required=True); production_risk.add_argument("--validated-responses", type=Path, required=True)
    production_risk.add_argument("--reviewed-alignment", type=Path, required=True); production_risk.add_argument("--reviewed-shot-context", type=Path, required=True)
    production_risk.add_argument("--screenplay-context", type=Path, required=True); production_risk.add_argument("--output", type=Path, required=True)
    production_risk.add_argument("--summary", type=Path, required=True); production_risk.add_argument("--low-confidence-threshold", type=float, default=.8)
    production_risk.add_argument(
        "--hard-validation-contract",
        choices=("candidate_task_v3_structure_v2", "candidate_task_v3_structure_v3"),
        default="candidate_task_v3_structure_v2",
    )
    evidence_corrections = commands.add_parser("apply-production-evidence-corrections-v3")
    evidence_corrections.add_argument("--correction-plan", type=Path, required=True)
    evidence_corrections.add_argument("--diagnosis", type=Path, required=True)
    evidence_corrections.add_argument("--risk-audit", type=Path, required=True)
    evidence_corrections.add_argument("--requests", type=Path, required=True)
    evidence_corrections.add_argument("--normalized-responses", type=Path, required=True)
    evidence_corrections.add_argument("--deterministic-alignment", type=Path, required=True)
    evidence_corrections.add_argument("--screenplay-context", type=Path, required=True)
    evidence_corrections.add_argument("--shots", type=Path, required=True)
    evidence_corrections.add_argument("--output-dir", type=Path, required=True)
    evidence_corrections.add_argument("--output-tag", required=True)
    evidence_corrections.add_argument("--adjudicated-audit", type=Path, required=True)
    evidence_corrections.add_argument("--adjudicated-summary", type=Path, required=True)
    lexical_rescue = commands.add_parser("augment-review-requests-global-lexical")
    lexical_rescue.add_argument("--requests", type=Path, required=True); lexical_rescue.add_argument("--screenplay-context", type=Path, required=True)
    lexical_rescue.add_argument("--output", type=Path, required=True); lexical_rescue.add_argument("--max-rescue-candidates", type=int, default=12)
    lexical_rescue.add_argument(
        "--retrieval-version",
        choices=("global_lexical_rescue_v2", "global_lexical_rescue_v3", "global_lexical_rescue_v4"),
        default="global_lexical_rescue_v2",
    )
    lexical_rescue.add_argument("--alignment", type=Path)
    lexical_rescue.add_argument("--context-radius", type=int, default=2)
    production_finalize = commands.add_parser("finalize-openai-production-movie-v3")
    production_finalize.add_argument("--movie-key", required=True); production_finalize.add_argument("--inventory", type=Path, required=True)
    production_finalize.add_argument("--status", type=Path, required=True); production_finalize.add_argument("--screenplay-context", type=Path, required=True)
    production_finalize.add_argument("--deterministic-alignment", type=Path, required=True); production_finalize.add_argument("--deterministic-shot-context", type=Path, required=True)
    production_finalize.add_argument("--reviewed-alignment", type=Path, required=True); production_finalize.add_argument("--reviewed-shot-context", type=Path, required=True)
    production_finalize.add_argument("--shots", type=Path, required=True); production_finalize.add_argument("--requests", type=Path, required=True)
    production_finalize.add_argument("--deterministic-requests", type=Path)
    production_finalize.add_argument("--validated-responses", type=Path, required=True); production_finalize.add_argument("--reviewer-manifest", type=Path, required=True)
    production_finalize.add_argument("--lifecycle-report", type=Path, action="append", required=True)
    production_finalize.add_argument("--risk-audit", type=Path, required=True); production_finalize.add_argument("--risk-summary", type=Path, required=True)
    production_finalize.add_argument("--qc-report", type=Path, required=True); production_finalize.add_argument("--manifest", type=Path, required=True)
    production_finalize.add_argument("--max-unresolved-ambiguities", type=int, default=5)
    production_finalize.add_argument("--max-unresolved-candidate-recall-risks", type=int, default=0)
    production_finalize.add_argument("--max-unresolved-reviewer-selection-risks", type=int, default=0)
    production_release = commands.add_parser("freeze-production-release-v3")
    production_release.add_argument("--inventory", type=Path, required=True)
    production_release.add_argument("--status", type=Path, required=True)
    production_release.add_argument("--experiments", type=Path, required=True)
    production_release.add_argument("--output-root", type=Path, required=True)
    production_release.add_argument("--release-dir", type=Path, required=True)
    production_release.add_argument("--code-commit", required=True)
    production_release.add_argument("--release-id", default="v3_2_1_production_3_final_seven")
    production_release.add_argument(
        "--path-map", action="append", default=[], metavar="OLD=NEW",
        help="explicitly resolve immutable manifest paths after a dataset mount relocation",
    )
    composite = commands.add_parser("build-openai-composite-audit")
    composite.add_argument("--requests", type=Path, required=True); composite.add_argument("--validated-responses", type=Path, required=True)
    composite.add_argument("--output", type=Path, required=True); composite.add_argument("--summary", type=Path, required=True)
    human = commands.add_parser("build-openai-human-audit")
    human.add_argument("--requests", type=Path, required=True); human.add_argument("--validated-responses", type=Path, required=True)
    human.add_argument("--composite-audit", type=Path, required=True); human.add_argument("--output", type=Path, required=True); human.add_argument("--manifest", type=Path, required=True)
    sequence_audit = commands.add_parser("build-openai-non-anchor-audit")
    sequence_audit.add_argument("--alignment", type=Path, required=True); sequence_audit.add_argument("--output", type=Path, required=True); sequence_audit.add_argument("--summary", type=Path, required=True)
    human_v2 = commands.add_parser("build-openai-human-audit-v2")
    human_v2.add_argument("--requests", type=Path, required=True); human_v2.add_argument("--validated-responses", type=Path, required=True)
    human_v2.add_argument("--composite-audit", type=Path, required=True); human_v2.add_argument("--non-anchor-audit", type=Path, required=True)
    human_v2.add_argument("--alignment", type=Path, required=True); human_v2.add_argument("--shot-context", type=Path, required=True)
    human_v2.add_argument("--prior-audit", type=Path, required=True); human_v2.add_argument("--output", type=Path, required=True); human_v2.add_argument("--manifest", type=Path, required=True)
    corrections = commands.add_parser("apply-human-corrections")
    corrections.add_argument("--audit", type=Path, required=True); corrections.add_argument("--requests", type=Path, required=True)
    corrections.add_argument("--openai-responses", type=Path, required=True); corrections.add_argument("--alignment", type=Path, required=True)
    corrections.add_argument("--screenplay-context", type=Path, required=True); corrections.add_argument("--shots", type=Path, required=True)
    corrections.add_argument("--output-dir", type=Path, required=True); corrections.add_argument("--output-tag", required=True)
    evaluate = commands.add_parser("evaluate-openai-pilot")
    evaluate.add_argument("--gold", type=Path, required=True); evaluate.add_argument("--validated-responses", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True); evaluate.add_argument("--output", type=Path, required=True)
    disagreements = commands.add_parser("build-openai-pilot-disagreements")
    disagreements.add_argument("--gold", type=Path, required=True); disagreements.add_argument("--validated-responses", type=Path, required=True)
    disagreements.add_argument("--requests", type=Path, required=True); disagreements.add_argument("--manifest", type=Path, required=True)
    disagreements.add_argument("--output", type=Path, required=True)
    evaluate_v3 = commands.add_parser("evaluate-openai-pilot-v3")
    evaluate_v3.add_argument("--gold", type=Path, required=True); evaluate_v3.add_argument("--validated-responses", type=Path, required=True)
    evaluate_v3.add_argument("--manifest", type=Path, required=True); evaluate_v3.add_argument("--adjudication", type=Path, required=True)
    evaluate_v3.add_argument("--output", type=Path, required=True)
    validate_calibration = commands.add_parser("validate-independent-calibration-reference")
    validate_calibration.add_argument("--reference", type=Path, required=True); validate_calibration.add_argument("--requests", type=Path, required=True)
    validate_calibration.add_argument("--reference-manifest", type=Path, required=True); validate_calibration.add_argument("--output", type=Path, required=True)
    evaluate_calibration = commands.add_parser("evaluate-independent-calibration-v3")
    evaluate_calibration.add_argument("--reference", type=Path, required=True); evaluate_calibration.add_argument("--validated-responses", type=Path, required=True)
    evaluate_calibration.add_argument("--pilot-manifest", type=Path, required=True); evaluate_calibration.add_argument("--response-validation", type=Path, required=True)
    evaluate_calibration.add_argument("--output", type=Path, required=True)
    evaluate_spot_check = commands.add_parser("evaluate-production-spot-check-v3")
    evaluate_spot_check.add_argument("--reference", type=Path, required=True); evaluate_spot_check.add_argument("--validated-responses", type=Path, required=True)
    evaluate_spot_check.add_argument("--pilot-manifest", type=Path, required=True); evaluate_spot_check.add_argument("--response-validation", type=Path, required=True)
    evaluate_spot_check.add_argument("--output", type=Path, required=True)
    evaluate_calibration_adjudicated = commands.add_parser("evaluate-independent-calibration-adjudicated-v3")
    evaluate_calibration_adjudicated.add_argument("--reference", type=Path, required=True); evaluate_calibration_adjudicated.add_argument("--validated-responses", type=Path, required=True)
    evaluate_calibration_adjudicated.add_argument("--pilot-manifest", type=Path, required=True); evaluate_calibration_adjudicated.add_argument("--response-validation", type=Path, required=True)
    evaluate_calibration_adjudicated.add_argument("--adjudication", type=Path, required=True); evaluate_calibration_adjudicated.add_argument("--output", type=Path, required=True)
    adjudication = commands.add_parser("build-openai-gold-adjudication")
    adjudication_validate = commands.add_parser("validate-openai-gold-adjudication")
    for command in (adjudication, adjudication_validate):
        command.add_argument("--gold", type=Path, required=True); command.add_argument("--validated-responses", type=Path, required=True)
        command.add_argument("--requests", type=Path, required=True); command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--screenplay-context", type=Path, required=True); command.add_argument("--alignment", type=Path, required=True)
        command.add_argument("--evaluation", type=Path, required=True); command.add_argument("--disagreements", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "process-one":
            values = {key: value for key, value in vars(args).items() if key != "command"}
            values["resume"] = args.resume and not args.overwrite
            options = ContextOptions(**values)
            print(json_dumps(process_one(options), pretty=True)); return 0
        if args.command == "validate":
            result = validate_files(args.screenplay_context, args.alignment, args.shot_context, args.shots)
            print(json_dumps({"movie_key": args.movie_key, **vars(result)}, pretty=True)); return 0 if result.passed else 1
        if args.command == "parse-script":
            sources = {"screenplay": args.screenplay.resolve().as_posix(), "subtitle": None if args.subtitle is None else args.subtitle.resolve().as_posix(), "shots": None if args.shots is None else args.shots.resolve().as_posix()}
            _write_json(args.output, parse_screenplay(args.screenplay, args.movie_key, args.screenplay.stem, sources)); return 0
        if args.command == "align-subtitles":
            import json
            context = json.loads(args.screenplay_context.read_text(encoding="utf-8")); subtitles = load_clean_subtitles(args.subtitle, args.subtitle_language)
            config = AlignmentConfig(args.alignment_threshold, args.review_threshold, semantic_model=None if args.disable_semantic else args.semantic_model)
            _write_jsonl(args.output, align_subtitles(subtitles, context, args.movie_key, config)); return 0
        if args.command == "map-shots":
            import json
            context = json.loads(args.screenplay_context.read_text(encoding="utf-8"))
            _write_jsonl(args.output, map_shots(read_jsonl(args.shots), read_jsonl(args.alignment), context, args.movie_key, args.scene_interpolation_max_gap)); return 0
        if args.command == "prepare-openai-pilot":
            from .pilot import prepare_pilot
            print(json_dumps(prepare_pilot(args.requests, args.alignment, args.output_dir, args.count), pretty=True)); return 0
        if args.command == "prepare-openai-batch":
            from .openai_review import prepare_batch
            print(json_dumps(prepare_batch(args.requests, args.output, args.model), pretty=True)); return 0
        if args.command == "prepare-openai-batch-v3":
            from .stage253 import prepare_batch_v3
            print(json_dumps(prepare_batch_v3(args.requests, args.annotation_policy, args.output, args.model), pretty=True)); return 0
        if args.command == "prepare-openai-batch-v3-2-policy":
            from .stage253 import prepare_batch_v32_policy
            print(json_dumps(prepare_batch_v32_policy(args.requests, args.annotation_policy, args.output, args.model), pretty=True)); return 0
        if args.command == "prepare-openai-batch-v3-2-1-vocative":
            from .stage253 import prepare_batch_v321_vocative
            print(json_dumps(prepare_batch_v321_vocative(args.requests, args.annotation_policy, args.output, args.model), pretty=True)); return 0
        if args.command == "prepare-openai-batch-v3-3-action-context":
            from .stage253 import prepare_batch_v33_action_context
            print(json_dumps(prepare_batch_v33_action_context(args.requests, args.annotation_policy, args.output, args.model), pretty=True)); return 0
        if args.command == "prepare-review-context-v3-1":
            from .stage253 import prepare_review_context_v31
            print(json_dumps(prepare_review_context_v31(args.requests, args.alignment, args.output, args.radius), pretty=True)); return 0
        if args.command == "prepare-review-action-context-v3-3":
            from .stage253 import prepare_review_action_context_v33
            print(json_dumps(prepare_review_action_context_v33(args.requests, args.screenplay_context, args.output, args.radius, args.max_actions), pretty=True)); return 0
        if args.command == "prepare-openai-remaining":
            from .stage23 import prepare_remaining_requests
            print(json_dumps(prepare_remaining_requests(args.full_requests, args.pilot_requests, args.output, args.manifest, args.model), pretty=True)); return 0
        if args.command == "submit-openai-batch":
            from .openai_review import submit_batch
            print(json_dumps(submit_batch(args.batch_input, args.job_file, confirm_submit=args.confirm_submit), pretty=True)); return 0
        if args.command == "submit-openai-batch-v3":
            from .stage253 import submit_batch_v3
            print(json_dumps(submit_batch_v3(args.batch_input, args.job_file, confirm_submit=args.confirm_submit), pretty=True)); return 0
        if args.command == "submit-openai-batch-v3-2-policy":
            from .stage253 import submit_batch_v32_policy
            print(json_dumps(submit_batch_v32_policy(args.batch_input, args.job_file, confirm_submit=args.confirm_submit), pretty=True)); return 0
        if args.command == "submit-openai-batch-v3-2-1-vocative":
            from .stage253 import submit_batch_v321_vocative
            print(json_dumps(submit_batch_v321_vocative(args.batch_input, args.job_file, confirm_submit=args.confirm_submit), pretty=True)); return 0
        if args.command == "submit-openai-batch-v3-3-action-context":
            from .stage253 import submit_batch_v33_action_context
            print(json_dumps(submit_batch_v33_action_context(args.batch_input, args.job_file, confirm_submit=args.confirm_submit), pretty=True)); return 0
        if args.command == "prepare-openai-production-remaining-v3":
            from .production_review import prepare_production_remaining_v3
            print(json_dumps(prepare_production_remaining_v3(args.full_requests, args.pilot_requests, args.output, args.manifest, args.reviewer_manifest), pretty=True)); return 0
        if args.command == "bind-openai-production-request-subset-v3":
            from .production_review import bind_production_request_subset_v3
            print(json_dumps(bind_production_request_subset_v3(args.source_requests, args.subset_requests, args.reviewer_manifest, args.manifest), pretty=True)); return 0
        if args.command == "prepare-openai-production-batch-v3":
            from .production_review import prepare_production_batch_v3
            print(json_dumps(prepare_production_batch_v3(args.requests, args.reviewer_manifest, args.output), pretty=True)); return 0
        if args.command == "preflight-openai-production-batch-v3":
            from .production_review import preflight_production_batch_v3
            result = preflight_production_batch_v3(
                args.batch_input, args.requests, args.reviewer_manifest, args.output,
            )
            print(json_dumps(result, pretty=True)); return 0 if result["passed"] else 1
        if args.command == "split-openai-production-requests-v3":
            from .production_review import split_production_requests_v3
            result = split_production_requests_v3(
                args.requests, args.reviewer_manifest, args.output_dir,
                max_estimated_tokens=args.max_estimated_tokens, max_requests=args.max_requests,
            )
            print(json_dumps(result, pretty=True)); return 0
        if args.command == "submit-openai-production-batch-v3":
            from .production_review import submit_production_batch_v3
            print(json_dumps(submit_production_batch_v3(args.batch_input, args.reviewer_manifest, args.job_file, confirm_submit=args.confirm_submit), pretty=True)); return 0
        if args.command == "check-openai-batch":
            from .openai_review import check_batch
            print(json_dumps(check_batch(args.job_file), pretty=True)); return 0
        if args.command == "fetch-openai-batch":
            from .openai_review import fetch_batch
            print(json_dumps(fetch_batch(args.job_file, args.output_dir), pretty=True)); return 0
        if args.command == "validate-openai-responses":
            from .openai_review import validate_responses
            report = validate_responses(args.raw_output, args.requests, args.output_dir)
            print(json_dumps(report, pretty=True)); return 0 if report["passed"] else 1
        if args.command == "validate-openai-responses-v3":
            from .stage253 import validate_responses_v3
            report = validate_responses_v3(args.raw_output, args.requests, args.output_dir, args.hard_validation_contract)
            print(json_dumps(report, pretty=True)); return 0 if report["passed"] else 1
        if args.command == "validate-openai-pilot-gold":
            from .openai_review import validate_pilot_gold
            report = validate_pilot_gold(args.gold, args.requests, args.output, args.max_backward_distance)
            print(json_dumps(report, pretty=True)); return 0 if report["passed"] else 1
        if args.command == "apply-openai-responses":
            from .openai_review import apply_validated_responses
            print(json_dumps(apply_validated_responses(args.alignment, args.requests, args.validated_responses, args.screenplay_context, args.shots, args.output_dir, args.output_tag), pretty=True)); return 0
        if args.command == "merge-openai-validated-responses":
            from .stage23 import merge_validated_responses
            print(json_dumps(merge_validated_responses(args.full_requests, args.pilot_responses, args.remaining_responses, args.output, args.report), pretty=True)); return 0
        if args.command == "merge-openai-production-responses-v3":
            from .production_review import merge_production_responses_v3
            print(json_dumps(merge_production_responses_v3(args.full_requests, args.pilot_requests, args.remaining_requests, args.pilot_responses, args.remaining_responses, args.reviewer_manifest, args.output, args.report), pretty=True)); return 0
        if args.command == "merge-openai-production-response-chunks-v3":
            from .production_review import merge_production_response_chunks_v3
            result = merge_production_response_chunks_v3(
                args.chunk_manifest, args.chunk_response, args.reviewer_manifest, args.output, args.report,
            )
            print(json_dumps(result, pretty=True)); return 0
        if args.command == "apply-openai-production-responses-v3":
            from .production_review import apply_production_responses_v3
            print(json_dumps(apply_production_responses_v3(args.alignment, args.requests, args.validated_responses, args.screenplay_context, args.shots, args.output_dir, args.reviewer_manifest), pretty=True)); return 0
        if args.command == "build-openai-production-risk-audit-v3":
            from .production_qc import build_production_high_risk_audit_v3
            result = build_production_high_risk_audit_v3(
                args.requests, args.validated_responses, args.reviewed_alignment, args.reviewed_shot_context,
                args.screenplay_context, args.output, args.summary, low_confidence_threshold=args.low_confidence_threshold,
                hard_validation_contract_version=args.hard_validation_contract,
            )
            print(json_dumps(result, pretty=True)); return 0
        if args.command == "apply-production-evidence-corrections-v3":
            from .production_qc import apply_production_evidence_corrections_v3
            result = apply_production_evidence_corrections_v3(
                args.correction_plan, args.diagnosis, args.risk_audit, args.requests,
                args.normalized_responses, args.deterministic_alignment,
                args.screenplay_context, args.shots, args.output_dir, args.output_tag,
                args.adjudicated_audit, args.adjudicated_summary,
            )
            print(json_dumps(result, pretty=True)); return 0
        if args.command == "augment-review-requests-global-lexical":
            from .llm_review import augment_review_requests_global_lexical
            print(json_dumps(augment_review_requests_global_lexical(
                args.requests, args.screenplay_context, args.output,
                max_rescue_candidates=args.max_rescue_candidates,
                retrieval_version=args.retrieval_version,
                alignment_path=args.alignment, context_radius=args.context_radius,
            ), pretty=True)); return 0
        if args.command == "finalize-openai-production-movie-v3":
            from .production_qc import finalize_production_movie_v3
            result = finalize_production_movie_v3(
                args.movie_key, args.inventory, args.status, args.screenplay_context,
                args.deterministic_alignment, args.deterministic_shot_context, args.reviewed_alignment,
                args.reviewed_shot_context, args.shots, args.requests, args.validated_responses,
                args.reviewer_manifest, args.lifecycle_report, args.risk_audit, args.risk_summary,
                args.qc_report, args.manifest, max_unresolved_ambiguities=args.max_unresolved_ambiguities,
                max_unresolved_candidate_recall_risks=args.max_unresolved_candidate_recall_risks,
                max_unresolved_reviewer_selection_risks=args.max_unresolved_reviewer_selection_risks,
                deterministic_requests_path=args.deterministic_requests,
            )
            print(json_dumps(result, pretty=True)); return 0
        if args.command == "freeze-production-release-v3":
            from .release import freeze_stage2_release
            result = freeze_stage2_release(
                args.inventory, args.status, args.experiments, args.output_root,
                args.release_dir, args.code_commit, release_id=args.release_id,
                path_maps=parse_path_maps(args.path_map),
            )
            print(json_dumps(result, pretty=True)); return 0
        if args.command == "build-openai-composite-audit":
            from .stage23 import build_composite_audit
            print(json_dumps(build_composite_audit(args.requests, args.validated_responses, args.output, args.summary), pretty=True)); return 0
        if args.command == "build-openai-human-audit":
            from .stage23 import build_human_audit_sample
            print(json_dumps(build_human_audit_sample(args.requests, args.validated_responses, args.composite_audit, args.output, args.manifest), pretty=True)); return 0
        if args.command == "build-openai-non-anchor-audit":
            from .stage231 import build_non_anchor_sequence_audit
            print(json_dumps(build_non_anchor_sequence_audit(args.alignment, args.output, args.summary), pretty=True)); return 0
        if args.command == "build-openai-human-audit-v2":
            from .stage231 import build_human_audit_v2
            print(json_dumps(build_human_audit_v2(args.requests, args.validated_responses, args.composite_audit, args.non_anchor_audit, args.alignment, args.shot_context, args.prior_audit, args.output, args.manifest), pretty=True)); return 0
        if args.command == "apply-human-corrections":
            from .stage231 import apply_human_corrections
            print(json_dumps(apply_human_corrections(args.audit, args.requests, args.openai_responses, args.alignment, args.screenplay_context, args.shots, args.output_dir, args.output_tag), pretty=True)); return 0
        if args.command == "evaluate-openai-pilot":
            from .openai_review import evaluate_pilot
            print(json_dumps(evaluate_pilot(args.gold, args.validated_responses, args.manifest, args.output), pretty=True)); return 0
        if args.command == "build-openai-pilot-disagreements":
            from .openai_review import build_pilot_disagreements
            print(json_dumps(build_pilot_disagreements(args.gold, args.validated_responses, args.requests, args.manifest, args.output), pretty=True)); return 0
        if args.command == "evaluate-openai-pilot-v3":
            from .stage253 import evaluate_pilot_v3
            print(json_dumps(evaluate_pilot_v3(args.gold, args.validated_responses, args.manifest, args.adjudication, args.output), pretty=True)); return 0
        if args.command == "validate-independent-calibration-reference":
            from .stage253 import validate_independent_calibration_reference
            report = validate_independent_calibration_reference(args.reference, args.requests, args.reference_manifest, args.output)
            print(json_dumps(report, pretty=True)); return 0 if report["passed"] else 1
        if args.command == "evaluate-independent-calibration-v3":
            from .stage253 import evaluate_independent_calibration_v3
            print(json_dumps(evaluate_independent_calibration_v3(args.reference, args.validated_responses, args.pilot_manifest, args.response_validation, args.output), pretty=True)); return 0
        if args.command == "evaluate-production-spot-check-v3":
            from .stage253 import evaluate_production_spot_check_v3
            print(json_dumps(evaluate_production_spot_check_v3(args.reference, args.validated_responses, args.pilot_manifest, args.response_validation, args.output), pretty=True)); return 0
        if args.command == "evaluate-independent-calibration-adjudicated-v3":
            from .stage253 import evaluate_independent_calibration_adjudicated_v3
            result = evaluate_independent_calibration_adjudicated_v3(
                args.reference, args.validated_responses, args.pilot_manifest,
                args.response_validation, args.adjudication, args.output,
            )
            print(json_dumps(result, pretty=True)); return 0
        if args.command == "build-openai-gold-adjudication":
            from .stage252 import build_gold_adjudication
            result = build_gold_adjudication(args.gold, args.validated_responses, args.requests, args.manifest, args.screenplay_context, args.alignment, args.evaluation, args.disagreements, args.output_dir)
            print(json_dumps(result, pretty=True)); return 0
        if args.command == "validate-openai-gold-adjudication":
            from .stage252 import validate_gold_adjudication
            result = validate_gold_adjudication(args.gold, args.validated_responses, args.requests, args.manifest, args.screenplay_context, args.alignment, args.evaluation, args.disagreements, args.output_dir)
            print(json_dumps(result, pretty=True)); return 0 if result["passed"] else 1
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr); return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
