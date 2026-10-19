# OscarDP

OscarDP develops the Oscar Acting Dataset: a movie-first research dataset linking narrative context and character intent to observable acting, facial and gaze behaviour, and explainable performance annotations for controllable 3D character animation.

**Stage 01: Shot Detection** has an executable processing pipeline. **Stage 02: Subtitle & Screenplay Alignment** has manual annotations for 18 films using **ChatGPT GPT-6 (High thinking effort)** and an implemented deterministic batch validator. Stage02 Annotation Generation remains manual; Deterministic Validation checks structure and original-source consistency. See [the pipeline overview](docs/pipeline_overview.md), [Stage 01 contract](docs/stage01_spec.md), and [Stage 02 contract](docs/stage02_spec_v1.md).

## Layout

`src/oscardp/shots/` provides discovery, FFmpeg timing and extraction, TransNetV2 inference, checkpointing, QC, validation and the CLI. `src/oscardp/stage02/` provides the read-only Source Preflight v1.0, Stage02 Validator v1.1.1, shared SRT parsing and local reports. `src/oscardp/vendor/` contains the licensed PyTorch TransNetV2 model. `tests/` contains Stage01 and Stage02 tests. `models/` is for local weights; pass their actual path with `--weights`.

## Environment

Use Python 3.12+, `uv`, FFmpeg and FFprobe. Install the project with `uv sync --extra test`. For GPU processing, verify a CUDA-enabled PyTorch build and `torch.cuda.is_available()`. The `--device cuda` option fails if CUDA is unavailable. Supply compatible TransNetV2 PyTorch weights; the repository does not include them.

## Dataset and output

The default input is `/media/sdu/SiyaoDu/datasets/oscar_movie`; the default output is `/media/sdu/SiyaoDu/datasets/oscar_movie_analysis`. Override either path on the CLI. Supported videos are MP4, MKV, MOV, M4V, AVI, WebM and RMVB. Every source movie needs an IMDb ID in its filename or parent folder. Duplicate IDs are reported as errors. `data/metadata/movie_titles.json` is the **only Stage 01 movie-selection configuration file**: `--titles` supplies verified titles and limits batch processing to its IMDb IDs. Batch `process` requires it. `process-one` may omit it, in which case `movie_name` is `null`.

```text
oscar_movie_analysis/
  tt1234567/
    Stage01_tt1234567.json
    shot_img/shot_000001_early.jpg
    shot_img/shot_000001_middle.jpg
    shot_img/shot_000001_late.jpg
  .internal/tt1234567/video_metadata.json
```

The official JSON has one movie-level `movie_id`, one `movie_name`, and an ordered `shots` array. Each shot has only `shot_id` and `temporal` (`start`, `end`, `start_frame`, `end_frame`). Frame starts are inclusive and ends exclusive. TransNetV2 detects boundaries; FFprobe provides CFR/VFR timing; FFmpeg extracts Early/Middle/Late samples in one sequential pass (short shots use aliases). Optional boundary QC is stored under `.internal/`.

The Canonical Workspace is `/media/sdu/SiyaoDu/datasets/oscar_movie_analysis`. A completed Stage02 movie folder also holds its current `Stage02_<id>.json` and `<id>.srt`, with optional history under `stage02_curated/`. Historical subtitles stay in ZIP archives so recursive discovery sees only the current SRT. Stage01's image and JSON checks remain active when Stage02 files coexist.

Frozen Releases under `/media/sdu/SiyaoDu/datasets/oscardp_releases` are independent versioned snapshots; the mutable workspace is not a frozen research release. See [safe migration, validation and future-film workflow](docs/dataset_reorganisation.md). CLI defaults remain compatible; explicitly set both `--analysis-root` and `--srt-root` to the workspace for Canonical Stage02 validation and retain applicable `--manual-semantic-decisions`.

## Usage

```bash
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
uv run python -m oscardp.shots discover --input-root /path/to/movies --titles data/metadata/movie_titles.json
uv run python -m oscardp.shots process-one --video /path/to/movies/tt1234567.mkv --input-root /path/to/movies --output-root /path/to/oscar_movie_analysis --weights /path/to/transnetv2.pth --device cuda
uv run python -m oscardp.shots validate --output-root /path/to/oscar_movie_analysis --movie-id tt1234567
```

`process-one` resumes valid completed output and matching interrupted inference by default. Use `--qc` for sampled boundary images; `--save-all-boundary-frames` is for small tests. After one movie passes inspection, a batch may be run with:

```bash
uv run python -m oscardp.shots process --input-root /path/to/movies --output-root /path/to/oscar_movie_analysis --titles data/metadata/movie_titles.json --weights /path/to/transnetv2.pth --device cuda
```

The batch continues after individual failures and skips valid outputs. Do not run the full dataset before inspecting a test movie. Stages 03–06 remain planned and are not executable in this repository. Stage03 is now **manual GPT Visual Annotation**, not a YOLO/SCRFD pipeline; Stage05 will own video-based identity and face analysis.

## Stage 02 — Subtitle & Screenplay Alignment

Stage 02 v1.0 is a **Shot-Centric, text-derived semantic annotation**, currently completed outside this repository for **18 movies** using **ChatGPT GPT-6 with High thinking effort** (interactive LLM analysis, not an automated Stage 02 API/CLI). The original annotation instructions are preserved without modification:

- [Stage 02 v1.0 specification](docs/stage02_spec_v1.md) — fixed JSON schema and annotation rules.
- [Stage 02 annotation prompt v1.0](docs/prompts/stage02_prompt_v1.txt) — preserved historical prompt used for the existing 18-film annotations.
- [Stage 02 annotation prompt v1.1](docs/prompts/stage02_prompt_v1_1.txt) — preserved previous workflow; retained for reproducibility.
- [Stage 02 annotation prompt v1.2](docs/prompts/stage02_prompt_v1_2.txt) — **recommended for new movies**; mandatory executable source preflight before LLM, mixed-cue/SDH examples, audiovisual ambiguity QC, and post-annotation validation. The fixed Stage02 Schema remains v1.0.

**Inputs:** `Stage01_<movie_id>.json`, the movie's confirmed `<movie_id>.srt` (original or a separately approved Curated source version), and its screenplay PDF, together with the fixed specification. The selected SRT is the sole source of actual dialogue text; the screenplay informs semantic correspondence, characters, locations, events and tentative performance cues; Stage 01 is authoritative for shot IDs and timestamps. Canonical placement alone does not authenticate a version.

### Mandatory Stage02 Source Preflight — BEFORE LLM annotation

**A deterministic Stage02 Source Preflight MUST run and be reviewed before any LLM-based semantic annotation for a new movie.** The required order is **source preflight → resolve/approve blocking input issues → LLM annotation → Stage02 Validator → explicit release/activation**. An LLM assertion that it has inspected the input does **not** count as a programmatically completed preflight.

The source preflight must generate a saved, independently reviewable `source_preflight.json`, outside the Stage02 Schema, with paths and SHA-256 hashes of Stage01/SRT/PDF, Stage01 shot integrity, the actual subtitle format and encoding (including WebVTT masquerading as SRT), unique original cue IDs, text and time validity, empty/zero-duration cues, duplicate IDs or subtitle advertising, time-range/version concerns, and a per-issue disposition. The gate is **PASS** or explicitly approved, evidenced nonblocking **WARNING**. Any unresolved source-index mapping, nonempty zero-duration cue without an authoritative time, unreliable subtitle content, or incomplete mandatory check is **BLOCKED**. No invented index mapping, subtitle timing, or silent source edits.

**Implemented: Source Preflight CLI v1.0.** It shares the Validator's Stage01 contract, strict JSON reader, SRT parser and source-comparison rules; adds subtitle timeline diagnostics and deterministic PDF text-layer checks with `pypdf`; and never repairs input data. The existing `tools/dataset_preflight.py` remains an inventory/Excel audit. `validate` remains the independent post-annotation validator.

```bash
uv run python -m oscardp.stage02 preflight \
  --stage01 /path/to/Stage01_tt1234567.json \
  --srt /path/to/tt1234567.srt \
  --screenplay /path/to/tt1234567.pdf \
  --report-root reports/stage02_preflight/tt1234567
```

Optional `--stage02 PATH` checks existing annotation compatibility and reports stale cue references; it is not required for a new film. `--approved-exceptions PATH` accepts only precise, issue-specific nonblocking approvals bound to **every selected input SHA-256**. `--baseline-manifest PATH` explicitly compares a prior Preflight, per-movie Validator report or release manifest without modifying it. A changed source produces `SOURCE_VERSION_CHANGED` and needs review; a historical hash is never silently replaced.

Reports are `PREFLIGHT.md`, `source_preflight.json` and `issues.csv`, outside the input directories in a fresh/empty report directory. The report separates `preflight_status` (`PASS`, `WARNING`, `BLOCKED`) from `llm_annotation_allowed`. A WARNING remains visible after approval. Exit `0` permits annotation; `1` means blocked or awaiting necessary review; `2` means CLI/configuration/execution error. Structural/source blockers cannot be waived. Actual edition/audio synchronization and screenplay/movie semantics remain `UNVERIFIED`. See [Preflight rules, limits and approval format](docs/stage02_preflight.md).

```text
Stage01 + SRT + Screenplay
          |
          v
Deterministic Source Preflight
          |
     PASS / Approved WARNING
          |
          v
GPT-6 High Semantic Annotation
          |
          v
Stage02 Validator v1.1.1
          |
          v
Curated Dataset Release (separate approval; no automatic activation)
```

The original SRT/Stage01/PDF remain immutable. Approved curated sources require separate paths, source-to-curated cue provenance, SHA-256 and explicit approval. Source preflight does not certify semantic accuracy or the audiovisual correctness of a screenplay.

**Output:** one complete `Stage02_<movie_id>.json` containing all Stage 01 shots with `shot_id`, `temporal`, `dialogue`, `narrative` and `tags`, plus optional `performance_cues`, `sound_cues` and `song_lyrics`. Incomplete work must be explicitly marked `_partNNN`, never presented as a full-movie result. The 18 generated JSON files are **not committed to this repository**.

**Reproduction:** in ChatGPT, choose GPT-6 with **High** thinking effort; upload the Stage 01 JSON, original SRT, screenplay PDF and fixed spec; use the historical v1.0/v1.1 prompts to reproduce earlier workflows, or **v1.2 for new films** only **after a saved and reviewed executable Source Preflight**. Run the independent Stage02 Validator after annotation; it is not the preflight. Semantic alignment requires the model to read the screenplay and subtitle context rather than relying only on Python string similarity. `performance_cues` derived from the screenplay or LLM inference are **not** video-verified AU, gaze, emotion or facial observations.

**Status:** Annotation Generation still uses manual GPT-6 High interaction. Validator **v1.1.1** performs **Structural and Source Consistency Validation** against the fixed [JSON Schema](docs/schemas/stage02_v1.schema.json), Stage01 shot IDs/times and original SRT text/indices. It separates **Source Integrity** (subtitle quality) from **Annotation Consistency** (Stage02 structure and source references), and lists incomplete checks. It does not evaluate screenplay semantics or movie visuals. Downstream scene/beat aggregation and visual analysis remain future work.

```bash
# Single movie, then all discovered IMDb-ID directories.
uv run python -m oscardp.stage02 validate --movie-id tt0025316 --report-root reports/stage02_validation_single
uv run python -m oscardp.stage02 validate --expect-movies 18
```

Defaults use the dataset paths above. Override `--analysis-root`, `--srt-root` and `--report-root` as needed. SRT discovery builds one file index, using IMDb IDs in filenames/parent folders, or an identified video in the same directory. Missing or ambiguous sources fail; no fuzzy title matching is used. An optional `--srt-map` JSON object maps IMDb IDs to confirmed subtitle paths to resolve ambiguity.

Reports go to `reports/stage02_validation/`: `summary.csv`, `summary.json`, `issues.csv`, and detailed `movies/<movie_id>.json` with source hashes, shot/index evidence, text differences and incomplete checks. **PASS** means every required check completed without errors or warnings; **WARNING** means no errors but manual review is needed (including formatting-only text normalization); **FAIL** means errors or incomplete required checks. Missing/ambiguous SRT cannot pass. Reports are saved before a failure exit code. Choose a fresh report directory for reruns; existing reports are never overwritten.

The validator never modifies Stage01, Stage02 or subtitles and requires no GPU, video processing or LLM API. Passing structural/source consistency does not establish semantic correctness. See [validation checks and limitations](docs/stage02_validation.md) for CLI options, normalization rules and report details.

Validator v1.1.1 accepts complete, explicitly bounded dialogue/lyric/sound components in mixed cues; arbitrary substring matches remain invalid. It checks explicit field contamination and aggregates repeated cross-shot occurrences into root problems while preserving every affected shot. Known style tags and ASS alignment tags are formatting differences; source words and punctuation are preserved. Invalid original SRT can fail Source Integrity even when the existing annotation references are consistent. An unreadable or unnumbered source blocks validation instead of generating thousands of dependent index errors.

To preserve the saved v1.0 baseline, write the new run and audit to fresh directories:

```bash
uv run python -m oscardp.stage02 validate --expect-movies 18 --report-root reports/stage02_validation_v1_1
uv run python -m oscardp.stage02 review --baseline-root reports/stage02_validation --validation-root reports/stage02_validation_v1_1 --report-root reports/stage02_validation_v1_1/review --historical-baseline reports/stage02_validation_initial
```

The optional historical argument also audits the earlier first run. `review/REVIEW.md` provides the audit findings; the JSON/CSV artifacts map every old ERROR/WARNING to a new disposition and retain original evidence/hashes. Source curation produces **unapproved derivative files**, original-index maps and separate candidate validation only. Proposed Stage02 JSON Patches include exact before/after values and test guards and are **never applied**. Neither a candidate-source PASS nor a formatting acceptance promotes an input or silently changes the original-source validation result. Approval and unresolved actions remain explicit in the review.


## Dataset preflight

`tools/dataset_preflight.py` compares the reference Excel workbook with movie, SRT and screenplay files. It writes review reports under `reports/dataset_inventory/` and a Stage 01 title mapping at `data/metadata/movie_titles.json`. The reports are local and ignored by Git. The audit and plan commands do not rename dataset files.

```bash
uv run python tools/dataset_preflight.py audit --workbook /path/to/oscar_movie_exist.xlsx
uv run python tools/dataset_preflight.py plan-rename --workbook /path/to/oscar_movie_exist.xlsx
```

Review `rename_plan.csv`, set `approval_status` to `APPROVED` only for selected `SAFE` rows, and save that reviewed CSV as a separate file. `apply-rename --approved-plan /path/to/reviewed.csv --confirm` performs in-place, journaled renames; `verify --journal /path/to/rename_journal.jsonl` checks them, and `verify --journal /path/to/rename_journal.jsonl --rollback --confirm` reverses them if original names remain free. Renaming may invalidate Stage 01 source-path fingerprints or downstream references; inspect existing outputs before applying a plan.

Movie files use `<imdb_id>.<original_extension>` and external subtitles use `<imdb_id>.srt`. Movie directories and screenplay names stay as they are. Approve renames only after inventory, destination and RMVB decode checks pass; the journal is retained for verification and recovery.


Stage02 Validator **v1.1.1** fixes the six reviewed turn-delimiter/SDH support gaps
and supports optional source-bound user audiovisual decisions with
`--manual-semantic-decisions PATH`. These decisions do not waive schema, shot,
text or time checks and do not classify matching words globally. Source Integrity
and Annotation Consistency remain separate results. Local Phase A proposals in `reports/stage02_patch_review_v1_3/` remain historical, unmodified proposals. Separately, four locally curated Stage02 v1.1.1 copies were produced and validated in Phase B under a separate dataset root as reported by the dataset maintainer; they have **not been activated as default Pipeline inputs** and are **not stored in this public repository**. Later stages remain unimplemented.

Existing frozen-selected movies can be upgraded without inference using
`uv run python -m oscardp.shots frames-upgrade`. Default mode is a dry-run; image
changes require `--apply`. Stage01/Stage02 JSON and SRT bytes remain unchanged,
legacy Middle JPEGs retain their original hashes, and journals support recovery.
See [exact single-film and 17-film commands](docs/stage01_frames_upgrade.md).
For downstream readiness, use `validate --frame-profile three_sample`; historical
outputs remain valid under `--frame-profile legacy` or the default `auto` mode.

## Stage03 / Stage04 — Planned GPT workflow (contracts only)

**New conversation:** send `做 OscarDP Stage03，按 SiaDu/OscarDP main 的 Stage03 Prompt。Google Drive：<folder link>`. The Drive folder only needs Stage01 JSON, Stage02 JSON and actual Early/Middle/Late images. The [standing prompt](docs/prompts/stage03_gpt_visual_prompt_v1.md) and [schema](docs/schemas/stage03_gpt_visual_v1.schema.json) no longer require the user to supply frozen manifest, manual hashes, batch metadata or frame sample metadata. Formal release provenance is checked separately later; always verify real image access.


- **Stage03 GPT Visual Annotation**: manually inspect the Early/Middle/Late images in ChatGPT, optionally delivered in small batches from Google Drive; label visible shot scale, camera evidence, people, salient props and visible interactions. No mandatory local YOLO, SCRFD, bounding boxes, identity verification, AU or gaze. [Stage03 contract](docs/stage03_gpt_visual_spec_v1.md) · [Schema](docs/schemas/stage03_gpt_visual_v1.schema.json) · [Annotation prompt](docs/prompts/stage03_gpt_visual_prompt_v1.md).
- **Stage04 Scene & Character Context**: combine Stage02 textual narrative with Stage03 visual observations to assign finalized scene ranges and **unverified** character candidates. [Stage04 contract](docs/stage04_scene_context_spec_v1.md) · [Schema](docs/schemas/stage04_scene_context_v1.schema.json).
- **Stage05**: verified Actor/Character identity, continuous face detection/tracking, facial AUs, head pose and gaze on selected source-video intervals. **Stage06**: evidence-linked explainable acting annotations.

These schemas are **design contracts**, not implemented annotation/validation CLIs. A pilot is not a full-film publication. The frozen Stage02 release and its image-independent hashes are never modified; source JPEGs are verified through Stage01 sample metadata and kept separate from frozen inputs.
