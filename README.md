# OscarDP — Oscar Acting Dataset

OscarDP is a research pipeline for studying **how actors express character intentions through facial, head and gaze behaviour** in films. The goal is an evidence-linked dataset that can support explainable and controllable 3D character animation.

The intended research targets are actors/performances selected using Oscar nomination or award records. Other characters matter as **interaction context**, but are not automatically dataset targets. Award-based target selection is still being developed.

## Pipeline

| Stage | What it does | Current status |
| --- | --- | --- |
| **01 · Shots** | Detect shot boundaries, timestamps and sample frames | Working pipeline |
| **02 · Script & subtitles** | Align dialogue, screenplay context and possible acting cues to shots | 18 films annotated; annotation is assisted/manual |
| **03 · Visual shots** | Describe framing, visible people and objects from sampled frames | ChatGPT annotation workflow; Bugonia output available locally |
| **04 · Narrative & sequence selection** | Scene/character context (04), film-level profiles (04.1), scene objectives (04.2), LLM + external-reference sequence shortlisting (04.3) | Bugonia full-film prototype; 04.3 review decisions are tracked separately from video QC |
| **05 · Performance measurements** | On selected shots: face tracking, AU, head pose and **camera-relative gaze candidates** | Selected-shot pilot; no verified gaze target |
| **06 · Cross-shot gaze targets** | Infer film-scene gaze targets through shot/reverse-shot continuity and visual evidence | **Research design only; not implemented** |
| **07 · Hierarchical explainable acting** | Audio, visual, narrative, affect-state and acting-event interpretations | **Existing 115-second prototype still runs under legacy `oscardp.stage06`; logic likely to change** |

**Current refinements:** Stage04.1 profiles, Stage04.2 objectives and Stage04.3 sequence candidates exist as assistant-authored Bugonia JSON prototypes, not validated labels or a scalable CLI. **Next:** review candidate videos, then develop Stage06 cross-shot gaze targets. The old Stage06 acting prototype is conceptually renumbered Stage07; its code has not yet been migrated.

See [pipeline overview](docs/pipeline_overview.md), [Stage02 specification](docs/stage02_spec_v1.md), [Stage04 scene design](docs/stage04_scene_context_spec_v1.md), [Stage05 event extraction](docs/stage05_5_temporal_events.md), and [legacy Stage06 acting pilot (now conceptually Stage07)](docs/stage06_prototype.md).

## Existing acting pilot (Stage07 concept, legacy Stage06 code)

The existing acting pilot (formerly called Stage06, conceptually Stage07) uses a 115.2-second excerpt from `tt12300742` (movie time **1432.014–1547.212 s**). It combines:

- **Audio:** local Silero VAD and SUPERB Wav2Vec2 speech-emotion predictions.
- **Visual:** sampled-video observations authored in an interactive assistant session.
- **Narrative:** dialogue and scene context from Stages 02 and 04.
- **Events:** existing Stage05 AU, head and gaze candidates linked by time.

It produced six proposed affect states and six acting interpretations, with 81 links to Stage05 events. **This is a research prototype, not validated ground truth.** Character identities and event interpretations remain unreviewed; the current pipeline does **not** infer verified gaze targets. Re-running the existing legacy Stage06 module can repeat local audio processing and reuse saved assistant annotations, but does not automatically repeat the assistant's visual/narrative reasoning. No external inference API was used for this pilot.

## Getting started

Requires **Python 3.12+**, [uv](https://docs.astral.sh/uv/), **FFmpeg/FFprobe**, and model weights for the stage you want to run.

```bash
uv sync --extra test
uv run python -m oscardp.shots --help
uv run python -m oscardp.stage02 --help
uv run python -m oscardp.stage06 --help
```

Example: process one movie into shot boundaries and sampled images:

```bash
uv run python -m oscardp.shots process-one \
  --video /path/to/tt1234567.mkv \
  --input-root /path/to/movies \
  --output-root /path/to/analysis \
  --weights /path/to/transnetv2.pth \
  --device cuda
```

Example: inspect the **legacy Stage06 module** (Stage07 source/CLI migration not yet done):

```bash
uv run python -m oscardp.stage06 inspect
```

For the existing pilot commands, see [legacy Stage06 instructions](docs/stage06_prototype.md). Do not rename `oscardp.stage06` in usage yet: this is a documentation-level stage reclassification.

## Repository layout

- `src/oscardp/shots/` — video discovery, shot detection and frame sampling.
- `src/oscardp/stage02/` — subtitle/script input checks and annotation validation.
- `src/oscardp/stage05/` and `stage05_review/` — measurements, events and review.
- `src/oscardp/stage06/` — **legacy implementation of the acting prototype** (new roadmap stage 07; not yet renamed or refactored).
- `docs/` — stage specifications and workflow notes.
- `tests/` — tests.

Movie files, screenplay PDFs, local models and pilot reports are not part of this public repository. See the relevant stage documentation for their expected inputs.
