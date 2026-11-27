# OscarDP — Oscar Acting Dataset

OscarDP is a research pipeline for building an evidence-linked dataset of film acting: how actors
express a character's intentions through face, head and gaze, to support explainable and
controllable 3D character animation. Target performances come from Oscar nominations and awards
(selection criteria still in progress); other characters are kept as interaction context.

## Status

| Stage | What it does | Status |
| --- | --- | --- |
| 01 Shots | TransNetV2 shot boundaries, timestamps, Early/Middle/Late frames | Implemented; 18 films |
| 02 Dialogue & script | Align subtitles, screenplay context and acting cues to shots | Preflight and validator implemented; 18 films annotated with GPT and manually reviewed |
| 03 Visual shots | Framing, visible people and props from sampled frames | GPT annotation workflow; 1 film (Bugonia) |
| 04 Narrative context | Scene/character context, character profiles (04.1), scene objectives (04.2), sequence candidates (04.3) | Assistant-authored prototypes; Bugonia |
| 05 Measurements | Face tracking, identity, OpenFace AU / head pose / gaze, temporal events | Pilot on selected Bugonia shots |
| 06 Gaze targets | Scene layout (06.1), in-shot head/eye direction (06.2), cross-shot gaze target candidates (06.3) | Pilot on one Bugonia sequence (13 shots); awaiting human evaluation |
| 07 Acting interpretation | Audio, visual and narrative affect states and acting interpretations | Legacy prototype on a 115 s Bugonia excerpt |

Assistant-generated annotations (Stages 03–07) are candidates, not validated labels.
Details: [pipeline overview](docs/pipeline_overview.md) and the stage specs in [docs/](docs/).

## Getting started

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), FFmpeg/FFprobe, and the model weights for the stage you run.

```bash
uv sync --extra test
uv run python -m pytest
uv run python -m oscardp.shots --help
```

Process one movie into shots and sampled frames:

```bash
uv run python -m oscardp.shots process-one \
  --video /path/to/tt1234567.mkv \
  --input-root /path/to/movies \
  --output-root /path/to/analysis \
  --weights /path/to/transnetv2.pth \
  --device cuda
```

Optional dependency groups: `stage05`, `stage05-identity`, `stage05-performance`, `stage06-iris`, `stage07`.

## Repository layout

- `src/oscardp/shots/` — Stage01 shot detection and frame sampling
- `src/oscardp/stage02/` — Stage02 preflight and validation
- `src/oscardp/stage05/`, `stage05_review/` — Stage05 measurements, events and review viewer
- `src/oscardp/stage06_1/`, `stage06_2/`, `stage06_3/` — Stage06 gaze pilots
- `src/oscardp/stage07/` — Stage07 acting prototype ([docs](docs/stage07_prototype.md))
- `tools/` — dataset preflight and reorganisation
- `docs/` — stage specs and prompts
- `tests/`

Movies, subtitles, screenplays, model weights, sampled frames and generated reports are not included.
