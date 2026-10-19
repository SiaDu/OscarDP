# Canonical Workspace and versioned releases

The mutable Canonical Workspace is
`/media/sdu/SiyaoDu/datasets/oscar_movie_analysis`. A movie directory contains the
current `Stage01_<id>.json`, `Stage02_<id>.json`, future `Stage03_<id>.json` and `Stage04_<id>.json`, `<id>.srt`, `shot_img/`, and optional
`stage02_curated/` history. Stage02 JSON is the annotation, not a Validator-generated
`Stage02Val` file. Stage01's layout check accepts these explicitly named coexisting entries while
retaining its shot, image, frame and timing checks.

Fixed research snapshots live under
`/media/sdu/SiyaoDu/datasets/oscardp_releases`. The workspace can change; a Frozen
Release cannot. Stage03 interactive GPT visual annotation accepts Stage01/Stage02 JSONs and JPG samples from an explicitly supplied folder without user-supplied Frozen Manifest or hashes. Later local dataset release packaging and Stage04 publication can independently verify frozen release IDs and hashes. No new stage code is implemented by this reorganisation.

## Frozen-selected migration

`tools/reorganise_dataset.py` defaults to a data-read-only dry-run, writing its
inventory and immutable plan under a separate report directory:

```bash
uv run python tools/reorganise_dataset.py \
  --frozen-manifest /path/to/release/frozen_manifest.json \
  --expected-manifest-sha256 VERIFIED_SHA256 \
  --workspace /path/to/oscar_movie_analysis \
  --history-root /path/to/oscar_movie_analysis_curated \
  --history-root /path/to/oscar_movie_curated \
  --movie-root /path/to/oscar_movie \
  --report-root reports/dataset_reorganisation_v1 --dry-run
```

Review `pre_migration_inventory.json`, `migration_plan.json` and
`migration_plan.md` before using the same arguments with `--apply`. Obtain
authorization if this session has not already authorized the planned replacements.
Use `--verify` afterwards. Reuse the same report root to resume an interrupted run;
a different release requires a new report directory and plan.

Different JSON versions go under the owning film's
`stage02_curated/history/stage01/` or `history/stage02/`, named by full SHA-256,
without inventing LLM authorship. Historical subtitles go inside ZIP files in
`subtitle_sources/`, so recursive discovery cannot select them. Shared old bundles
are preserved under `.dataset_history/legacy_archives/`. Duplicate content may be
deduplicated, but every source path/hash is retained. Legacy Curated directories
are not deleted.

The tool checks path boundaries, Frozen checksums, disk space and write access;
locks the workspace against concurrent tool runs; verifies old-byte backups
before replacing current files; publishes each file atomically; and journals each
operation. Recovery is per file, not an all-movies transaction. After interruption,
resume `--apply` with the saved plan, then `--verify`. Do not edit files concurrently.
Unexpected changes block migration. Videos and approval files are never modified.

## Validate working versions

```bash
uv run python -m oscardp.stage02 validate \
  --analysis-root /media/sdu/SiyaoDu/datasets/oscar_movie_analysis \
  --srt-root /media/sdu/SiyaoDu/datasets/oscar_movie_analysis \
  --srt-map /media/sdu/SiyaoDu/datasets/oscar_movie_analysis/srt_map.json \
  --manual-semantic-decisions /path/to/release/evidence/manual_semantic_decisions.json \
  --expect-movies 18 --report-root reports/stage02_canonical_validation_v1
```

Use only applicable source-bound manual decisions. Existing explicit-root/map
options and old CLI defaults remain compatible. If more films exist, do not delete
them to satisfy the count: use existing `--movie-id` selection with fresh reports,
or a separately verified input selection. Nested historical JSON never substitutes
for a complete movie-root file; ZIP-contained SRTs are not discovered.

## New films and revisions

Film 19 or 100 uses the existing Stage01 video input and Canonical output root.
Place the confirmed SRT at the movie root; pass explicit Stage01/SRT/Screenplay
paths to Source Preflight v1.0. Only after PASS or precise approved nonblocking
warnings use manual Stage02 Prompt v1.2, then Validator v1.1.1. Schema v1.0,
Prompt semantics and source-check rules are unchanged.

Prepare repairs in an independent candidate directory. After review and validation,
preserve current bytes and source/hash records before promotion. Do not overwrite
a valid annotation with an unvalidated repair. New films or corrected frozen films
require a new Release ID, manifest and checksums; never append to or modify an
existing Frozen Release.

Migration preserves bytes and evidence; it does not authenticate LLM authorship,
edition/audio identity or full semantic truth. Preflight issue IDs can include
source paths: retain original approvals and scopes, never silently rewrite or
reuse them for a newly generated issue after relocation. Historical reports remain
traceable through migration path/hash mappings.

## Stage01 image upgrades after reorganisation

Canonical movie directories contain downstream data. Use the dedicated
[Stage01 frames-upgrade command](stage01_frames_upgrade.md) after reorganisation,
with the same explicitly verified Frozen Manifest. It edits only known `shot_img/`
entries and writes internal sampling provenance; it never republishes a whole movie
directory. Stage01 `process-one --overwrite` refuses directories containing Stage02,
SRT, curation history or unknown entries. Reorganisation treats images as opaque
files and preserves both legacy and three-sample filenames; no Stage02 schema or
input selection changes are needed. Preserve `.internal/` and upgrade reports when
copying an upgraded workspace, since they bind source identity and Middle hashes.
