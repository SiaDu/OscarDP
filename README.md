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
| 06 Gaze targets | Scene layout (06.1), in-shot head/eye direction (06.2), cross-shot gaze targets (06.3) | Pilot on Bugonia `seq_006` (13 shots); human gaze labels frozen as the 06.3 output |
| 07 Acting interpretation | Speech-gated AU and head-movement evidence; acting interpretation grounded in gaze and Stage04 context | `seq_006` run end to end; first human review done |

Assistant-generated annotations (Stages 03–07) are candidates, not validated labels.
Details: [pipeline overview](docs/pipeline_overview.md) and the stage specs in [docs/](docs/).

## Pilot: Bugonia `seq_006`

One 115 s sequence (13 shots, 3 characters) runs through every stage. Results so far:

| Layer | Method | Result |
| --- | --- | --- |
| Gaze (06.3) | Human MOVE/HOLD labels per shot and character | 45 gaze shifts; machine MOVEs find 30 within ±0.25 s, 37 with any raw event |
| Eye–head | Head motion around each labelled shift | Eyes lead; for nearby targets the head barely follows (median turn 3.9°) |
| Head (07) | OpenFace pose inside gaze holds, split by own speech | Michelle 11.3 °/s speaking vs 4.1 °/s silent; 14 nod/shake/tilt candidates |
| AU (07) | OpenFace AU with speech gating (jaw ignored, mouth AUs stricter while speaking) | Misses subtle and sustained expressions; blinks read as lid tightening |
| Interpretation (07) | LLM prompt over Stage04 context, gaze, AU and head evidence | 15 beats reviewed: 4 agree, 9 partly, 2 disagree |

The review shows that gaze-grounded readings hold and AU-grounded readings do not.
AU levels for the same actor shift by about 2 points between shots (pose, light, framing), so changing the
normalisation does not fix it. Details: [Stage06.3](docs/stage06_3_gaze_target.md),
[Stage07](docs/stage07_prototype.md), [interpretation prompt](docs/prompts/stage07_acting_interpretation_prompt_v1.md).

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
- `docs/` — stage specs; `docs/prompts/` — LLM prompts
- `tests/`

Movies, subtitles, screenplays, model weights, sampled frames and generated reports are not included.
