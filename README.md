# OscarDP

OscarDP develops the Oscar Acting Dataset: a movie-first research dataset linking narrative context and character intent to observable acting, facial and gaze behaviour, and explainable performance annotations for controllable 3D character animation.

Only **Stage 01: Shot Detection** is executable. The planned six-stage architecture is in [docs/pipeline_overview.md](docs/pipeline_overview.md). The Stage 01 data contract is in [docs/stage01_spec.md](docs/stage01_spec.md).

## Layout

`src/oscardp/shots/` provides discovery, FFmpeg timing and extraction, TransNetV2 inference, checkpointing, QC, validation and the CLI. `src/oscardp/vendor/` contains the licensed PyTorch TransNetV2 model. `tests/` contains Stage 01 tests. `models/` is for local weights; pass their actual path with `--weights`.

## Environment

Use Python 3.12+, `uv`, FFmpeg and FFprobe. Install the project with `uv sync --extra test`. For GPU processing, verify a CUDA-enabled PyTorch build and `torch.cuda.is_available()`. The `--device cuda` option fails if CUDA is unavailable. Supply compatible TransNetV2 PyTorch weights; the repository does not include them.

## Dataset and output

The default input is `/media/sdu/SiyaoDu/datasets/oscar_movie`; the default output is `/media/sdu/SiyaoDu/datasets/oscar_movie_analysis`. Override either path on the CLI. Every source movie needs an IMDb ID in its filename or parent folder. Duplicate IDs are reported as errors. Reliable titles may be supplied using `--titles /path/to/titles.json`; otherwise `movie_name` is `null`.

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
uv run python -m oscardp.shots discover --input-root /path/to/movies
uv run python -m oscardp.shots process-one --video /path/to/movies/tt1234567.mkv --input-root /path/to/movies --output-root /path/to/oscar_movie_analysis --weights /path/to/transnetv2.pth --device cuda
uv run python -m oscardp.shots validate --output-root /path/to/oscar_movie_analysis --movie-id tt1234567
```

`process-one` resumes valid completed output and matching interrupted inference by default. Use `--qc` for sampled boundary images; `--save-all-boundary-frames` is for small tests. After one movie passes inspection, a batch may be run with:

```bash
uv run python -m oscardp.shots process --input-root /path/to/movies --output-root /path/to/oscar_movie_analysis --weights /path/to/transnetv2.pth --device cuda
```

The batch continues after individual failures and skips valid outputs. Do not run the full dataset before inspecting a test movie. Future Stages 02–06 will write separate JSON files and link records by stable IDs; they are not implemented.
