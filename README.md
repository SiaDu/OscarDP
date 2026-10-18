# OscarDP

OscarDP develops the Oscar Acting Dataset: a movie-first research dataset linking narrative context and character intent to observable acting, facial and gaze behaviour, and explainable performance annotations for controllable 3D character animation.

**Stage 01: Shot Detection** has an executable processing pipeline. **Stage 02: Subtitle & Screenplay Alignment** has manual annotations for 18 films using **ChatGPT GPT-6 (High thinking effort)** and an implemented deterministic batch validator. Stage02 Annotation Generation remains manual; Deterministic Validation checks structure and original-source consistency. See [the pipeline overview](docs/pipeline_overview.md), [Stage 01 contract](docs/stage01_spec.md), and [Stage 02 contract](docs/stage02_spec_v1.md).

## Layout

`src/oscardp/shots/` provides discovery, FFmpeg timing and extraction, TransNetV2 inference, checkpointing, QC, validation and the CLI. `src/oscardp/stage02/` provides read-only Stage02 validation, SRT parsing and reports. `src/oscardp/vendor/` contains the licensed PyTorch TransNetV2 model. `tests/` contains Stage01 and Stage02 tests. `models/` is for local weights; pass their actual path with `--weights`.

## Environment

Use Python 3.12+, `uv`, FFmpeg and FFprobe. Install the project with `uv sync --extra test`. For GPU processing, verify a CUDA-enabled PyTorch build and `torch.cuda.is_available()`. The `--device cuda` option fails if CUDA is unavailable. Supply compatible TransNetV2 PyTorch weights; the repository does not include them.

## Dataset and output

The default input is `/media/sdu/SiyaoDu/datasets/oscar_movie`; the default output is `/media/sdu/SiyaoDu/datasets/oscar_movie_analysis`. Override either path on the CLI. Supported videos are MP4, MKV, MOV, M4V, AVI, WebM and RMVB. Every source movie needs an IMDb ID in its filename or parent folder. Duplicate IDs are reported as errors. `data/metadata/movie_titles.json` is the **only Stage 01 movie-selection configuration file**: `--titles` supplies verified titles and limits batch processing to its IMDb IDs. Batch `process` requires it. `process-one` may omit it, in which case `movie_name` is `null`.

```text
oscar_movie_analysis/
  tt1234567/
    Stage01_tt1234567.json
    shot_img/shot_000001_00-00-02-500.jpg
  .internal/tt1234567/video_metadata.json
```

The official JSON has one movie-level `movie_id`, one `movie_name`, and an ordered `shots` array. Each shot has only `shot_id` and `temporal` (`start`, `end`, `start_frame`, `end_frame`). Frame starts are inclusive and ends exclusive. TransNetV2 detects boundaries; FFprobe provides CFR/VFR timing; FFmpeg extracts one midpoint image per shot in a sequential pass. Optional boundary QC is stored under `.internal/`.

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

The batch continues after individual failures and skips valid outputs. Do not run the full dataset before inspecting a test movie. Stages 03–06 remain planned and are not executable in this repository.

## Stage 02 — Subtitle & Screenplay Alignment

Stage 02 v1.0 is a **Shot-Centric, text-derived semantic annotation**, currently completed outside this repository for **18 movies** using **ChatGPT GPT-6 with High thinking effort** (interactive LLM analysis, not an automated Stage 02 API/CLI). The original annotation instructions are preserved without modification:

- [Stage 02 v1.0 specification](docs/stage02_spec_v1.md) — fixed JSON schema and annotation rules.
- [Stage 02 annotation prompt v1.0](docs/prompts/stage02_prompt_v1.txt) — preserved historical prompt used for the existing 18-film annotations.
- [Stage 02 annotation prompt v1.1](docs/prompts/stage02_prompt_v1_1.txt) — recommended for future movies; adds source preflight, mixed-cue classification, and independent validation requirements without changing Schema v1.0.

**Inputs:** `Stage01_<movie_id>.json`, the movie's original `<movie_id>.srt`, and its screenplay PDF, together with the fixed specification. The SRT is the sole source of actual dialogue text; the screenplay informs semantic correspondence, characters, locations, events and tentative performance cues; Stage 01 is authoritative for shot IDs and timestamps.

**Output:** one complete `Stage02_<movie_id>.json` containing all Stage 01 shots with `shot_id`, `temporal`, `dialogue`, `narrative` and `tags`, plus optional `performance_cues`, `sound_cues` and `song_lyrics`. Incomplete work must be explicitly marked `_partNNN`, never presented as a full-movie result. The 18 generated JSON files are **not committed to this repository**.

**Reproduction:** in ChatGPT, choose GPT-6 with **High** thinking effort; upload the Stage 01 JSON, original SRT, screenplay PDF and fixed spec; use the historical v1.0 prompt to reproduce the original 18-film workflow, or **v1.1 for new films** (requires source preflight and an independent Stage02 Validator run when available). Semantic alignment requires the model to read the screenplay and subtitle context rather than relying only on Python string similarity. `performance_cues` derived from the screenplay or LLM inference are **not** video-verified AU, gaze, emotion or facial observations.

**Status:** Annotation Generation still uses manual GPT-6 High interaction. The implemented validator performs **Structural and Source Consistency Validation** against the fixed [JSON Schema](docs/schemas/stage02_v1.schema.json), Stage01 shot IDs/times and original SRT text/indices. It does not evaluate screenplay semantics or movie visuals. Downstream scene/beat aggregation and visual analysis remain future work.

```bash
# Single movie, then all discovered IMDb-ID directories.
uv run python -m oscardp.stage02 validate --movie-id tt0025316 --report-root reports/stage02_validation_single
uv run python -m oscardp.stage02 validate --expect-movies 18
```

Defaults use the dataset paths above. Override `--analysis-root`, `--srt-root` and `--report-root` as needed. SRT discovery builds one file index, using IMDb IDs in filenames/parent folders, or an identified video in the same directory. Missing or ambiguous sources fail; no fuzzy title matching is used. An optional `--srt-map` JSON object maps IMDb IDs to confirmed subtitle paths to resolve ambiguity.

Reports go to `reports/stage02_validation/`: `summary.csv`, `summary.json`, `issues.csv`, and detailed `movies/<movie_id>.json` with source hashes, shot/index evidence, text differences and incomplete checks. **PASS** means every required check completed without errors or warnings; **WARNING** means no errors but manual review is needed (including formatting-only text normalization); **FAIL** means errors or incomplete required checks. Missing/ambiguous SRT cannot pass. Reports are saved before a failure exit code. Choose a fresh report directory for reruns; existing reports are never overwritten.

The validator never modifies Stage01, Stage02 or subtitles and requires no GPU, video processing or LLM API. Passing structural/source consistency does not establish semantic correctness. See [validation checks and limitations](docs/stage02_validation.md) for CLI options, normalization rules and report details.


## Dataset preflight

`tools/dataset_preflight.py` compares the reference Excel workbook with movie, SRT and screenplay files. It writes review reports under `reports/dataset_inventory/` and a Stage 01 title mapping at `data/metadata/movie_titles.json`. The reports are local and ignored by Git. The audit and plan commands do not rename dataset files.

```bash
uv run python tools/dataset_preflight.py audit --workbook /path/to/oscar_movie_exist.xlsx
uv run python tools/dataset_preflight.py plan-rename --workbook /path/to/oscar_movie_exist.xlsx
```

Review `rename_plan.csv`, set `approval_status` to `APPROVED` only for selected `SAFE` rows, and save that reviewed CSV as a separate file. `apply-rename --approved-plan /path/to/reviewed.csv --confirm` performs in-place, journaled renames; `verify --journal /path/to/rename_journal.jsonl` checks them, and `verify --journal /path/to/rename_journal.jsonl --rollback --confirm` reverses them if original names remain free. Renaming may invalidate Stage 01 source-path fingerprints or downstream references; inspect existing outputs before applying a plan.

Movie files use `<imdb_id>.<original_extension>` and external subtitles use `<imdb_id>.srt`. Movie directories and screenplay names stay as they are. Approve renames only after inventory, destination and RMVB decode checks pass; the journal is retained for verification and recovery.
