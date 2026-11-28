# Stage07 Hierarchical Affect & Explainable Acting — Legacy Prototype

Stage07 processes one continuous Bugonia excerpt. Original Stage01–05 outputs
and source movies/subtitles remain read-only. No detection, training, release,
hashing framework, annotation website or external inference API is introduced.
The user explicitly requested that the current assistant inspect the movie,
rather than using Gemini. There is no Gemini/OpenAI API client in this module.

The actual pilot combines local pretrained waveform SER with interactive
assistant visual/text analysis. This is a research representation prototype,
not a dataset release, verified identity/event annotation, or blind experiment.

## Legacy Compatibility and reserved Stage06

This is a module/metadata renumbering, not a new affect algorithm. Stage04.3 remains LLM Sequence Candidate Selection; Stage05 retains camera-relative measurements; Stage06 cross-shot gaze targets are **Planned / Not Implemented**. The old Stage06 acting CLI is not kept as an alias, so its namespace does not conflict with future gaze code.

New outputs use `reports/stage07/`, `stage07_input.json`, `stage07_prototype.json`, `stage07.research_input.v1` and `stage07.research_prototype.v1`. State/Event/Shot IDs, timestamps, scores and interpretation semantics are unchanged. **Legacy Compatibility:** the reader still consumes the immutable Stage05.6 `stage06_input_pilot.json` packet and can read an old `stage06_input.json` preparation without editing it. Old `reports/stage06/` runs are never completed or overwritten by Stage07. Saved legacy assistant annotations can be supplied as read-only `--analysis-dir` inputs to a fresh Stage07 run.

The dependency extra is now `stage07` (`uv sync --extra stage07`). Default model location is `models/stage07`; pass `--model-dir models/stage06` to reuse the existing cache. Both model directories are ignored by Git. No model download is necessary for inspection or tests.

## Environment and real audio models

Use the existing Python/uv environment, FFmpeg/FFprobe and one optional runtime:

```sh
uv --no-cache pip install --python .venv/bin/python 'transformers>=4.51,<5'
```

The pilot used Transformers 4.57.6 and existing PyTorch on CPU. Checkpoints are
already cached under git-ignored `models/stage06/`; the CLI examples below reuse
that cache explicitly. No download is needed for this migration. For a new
installation, use `models/stage07` below and omit the legacy `--model-dir` override:

```python
from pathlib import Path
from urllib.request import urlretrieve
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

root = Path("models/stage07")
root.mkdir(parents=True, exist_ok=True)
vad = root / "silero_vad.jit"
if not vad.exists():
    urlretrieve("https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.jit", vad)
for cls in (Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification):
    cls.from_pretrained("superb/wav2vec2-base-superb-er", cache_dir=str(root / "huggingface"))
```

Downloading checkpoints sends no project audio/video/text. Inference uses
`local_files_only=True`, with no network lookup. Silero JIT processes real 16kHz
waveform in 32ms frames with a 0.5 speech threshold. Original SRT cue timing
bounds utterance proposals; VAD support excludes silence. This is approximate
localisation, not forced alignment, source separation or verified diarization.
Stage02 speaker hints remain unverified; mixed speakers stay Unknown.

The [SUPERB checkpoint](https://huggingface.co/superb/wav2vec2-base-superb-er)
only supports Neutral/Happiness/Anger/Sadness. All actual labels and softmax
scores are preserved; scores are uncalibrated and film-domain accuracy is not
established. There is no valence/arousal or pitch/tremor output. RMS is only a
descriptive waveform measurement. No subtitle text is passed into SER.

## CLI

```sh
uv --no-cache run --no-sync python -m oscardp.stage07 inspect
uv --no-cache run --no-sync python -m oscardp.stage07 prepare \
  --output-dir reports/stage07/tt12300742/new_preparation \
  --model-dir models/stage06
```

`prepare` extracts continuous original-sound review video, 16kHz WAV, silent
video and timestamped 2 FPS contact sheets with burned-in captions masked in
the bottom 22 percent. The assistant must actually inspect these sheets and the
source text to author new proposals. No mock or automatic stand-in is used.
The pilot inspected all 230 regular samples and an additional tail frame.

For an existing local preparation, assemble and check the real saved analysis:

```sh
uv --no-cache run --no-sync python -m oscardp.stage07 run --prepared \
  --analysis-dir reports/stage06/tt12300742/pilot_001/assistant_analysis \
  --output-dir reports/stage07/tt12300742/new_preparation
```

Use --prepared only for a new, unfinished Stage07 preparation with matching saved analysis; historical Stage06 reports are read-only and completed results are refused.
To reproduce local media/audio inference and replay the saved assistant analysis:

```sh
uv --no-cache run --no-sync python -m oscardp.stage07 run \
  --analysis-dir reports/stage06/tt12300742/pilot_001/assistant_analysis \
  --output-dir reports/stage07/tt12300742/pilot_001 \
  --model-dir models/stage06
```

This reruns genuine local VAD/SER and replays saved interactive judgments.
It **does not** automatically rerun this conversation's LLM/VLM cognition.
No API credentials are needed. An exact backend API model ID is not asserted.
For new interpretations, supply newly inspected, genuinely authored analysis.
Optional arguments include `--pilot-input`, `--start`, `--end`, `--video-fps`,
`--model-dir`, and a compatible Wav2Vec2 `--ser-model`. Alternate time ranges
need correspondingly reauthored proposals; out-of-range times/IDs are refused.

The analysis folder contains visual_affect.json, narrative_affect.json,
visual_identity_map.json, ablation_A.json, ablation_B.json, affect_states.json,
relation_assessments.json, acting_annotations.json and comparison_abc.json.
`overlap_candidates.json` records the numeric Python matching used before the
assistant's event-by-event semantic judgments. It is not model prediction.

## Evidence restrictions and limits

Visual analysis uses only sampled visible behaviour for its affect proposals,
anonymous person IDs, explicit visible intervals and separate inferred affect.
A later appearance-only mapping to Stage04 character candidates is unverified.
Audio classification uses real waveform only. Narrative analysis uses Stage02
text cues/dialogue and Stage04 context, without claiming observed acting.

A uses only narrative evidence; B only audio/video and identity metadata;
C fuses the three branches before seeing Stage05 event links. A/B results were
saved before C. Python checks nonoverlapping per-character state times,
evidence IDs, candidate compatibility, modality restrictions and positive
half-open temporal overlap. The assistant assessed each of 81 candidate links.
Unknown identities force unresolved relations. Earlier numeric arrays and
algorithm_candidate/human_verified authority are never edited.

This is source-separated reasoning **within one interactive conversation**.
The assistant had prior source-context/SER-label exposure and initially saw
burned-in captions in three preview sheets before masking and re-inspecting
all sheets. It cannot claim independent context-blind model sessions or a clean
causal A/B/C ablation. These limits are retained in method.json, REVIEW.md and
comparison_abc.md. A/B/C conclusions do not establish C's superiority.

Observation coverage is computed separately for VAD-supported audio, sampled
visual evidence, narrative support and missing AV evidence. Narrative continuity
never substitutes for visible acting. State transitions can concern outward
style/strategy or uncertainty, rather than a proven categorical emotion switch.
Mouth AUs are not residual nonverbal actions. Head/gaze remain camera-relative;
there is no head–body disentanglement, verified gaze target, or emotion causality.

Existing output directories and completed result files are refused. During an
explicit --prepared completion, the preliminary review is retained as
LOCAL_REVIEW.md and successful input/audio/media files remain intact. All
required JSONs are checked in memory before final assembly. No source datasets
or prior reports are renamed/deleted/overwritten.

`review_video.mp4` retains the original soundtrack. Load
`review_video_states.srt` to see global-time labels, current proposed states
and up to five active event IDs (the JSON retains all). SRT playback times must
be clip-relative; every annotation JSON remains movie-global. Add the selected
start to player time. Inspect REVIEW.md and comparison_abc.md before using
any proposed state or interpretation.

```sh
uv --no-cache run --no-sync python -m pytest tests/test_stage07.py
```

The final review_video_annotated.mp4 adds a right-side state/event panel, including current AV evidence support, without covering the movie frame. It copies the review clip audio stream. The clean review_video.mp4 remains available.

## Gaze-grounded run · pilot_001 (seq_006)

`--gaze-labels` adds human Stage06.3 gaze labels (`gaze_labels.json`) to `stage07_input.json` as
`stage06_3_gaze_references`, with the file's SHA-256. Acting annotations in C may then cite gaze segments
as `gaze` observations; each cited segment must be the annotated character's own and overlap the observation.
A/B ablations cannot use gaze. Without `--gaze-labels` the legacy behaviour is unchanged.

`reports/stage07/tt12300742/analysis_001/` reuses the legacy visual, narrative, state, relation and A/B/C files
unchanged (hashes in `provenance.json`). Only `acting_annotations.json` is new: 14 annotations, one per beat,
that align the gaze labels with subtitle timing. They are assistant-authored proposals; the gaze labels are the
only human-verified input.

```sh
labels=reports/stage06_3/tt12300742/seq_006/labels_001/gaze_labels.json
uv --no-cache run --no-sync python -m oscardp.stage07 prepare --output-dir reports/stage07/tt12300742/pilot_001 \
  --model-dir models/stage06 --gaze-labels "$labels"
uv --no-cache run --no-sync python -m oscardp.stage07 run --prepared --output-dir reports/stage07/tt12300742/pilot_001 \
  --analysis-dir reports/stage07/tt12300742/analysis_001 --model-dir models/stage06 --gaze-labels "$labels"
```

## Face and head evidence · face_head_002

`python -m oscardp.stage07.face_head` turns Stage06.2 per-frame OpenFace output into speech-gated AU episodes and
non-gaze head movements. Outputs are machine evidence (`human_verified=false`).

| Step | Rule |
| --- | --- |
| Speech state per frame | `speaking`: own Stage02 cue and VAD voice; `uncertain`: voice under an `Unknown` cue; else `silent`. ±0.2 s margin |
| Reliable tracks | face ≥150 px, median \|yaw\| ≤35°, OpenFace success ≥90 %. Others are listed but not used |
| Head movement | Only inside human gaze HOLDs, excluding each MOVE and 0.75 s after landing. Excursions from a ±1 s rolling median, ≥ max(3°, 3 × the track's silent jitter): nod (pitch, 0.2–1.0 s, returns), shake (≥2 alternating yaw excursions in 1.5 s), tilt (roll ≥0.3 s), beat (<0.4 s), turn, pitch_shift; HOLD-level posture offsets |
| Eye–head | Per human MOVE: head turn from before the MOVE to 0.75 s after landing; `eyes_only` (<3°), `head_done_at_landing` (≥90 %) or `head_still_moving` |
| AU episodes | Per track and speech state: median and robust SD. Rise ≥ max(0.5, 3 SD) for ≥0.2 s. Upper face in all states; mouth AUs (10/12/14/15/17/20/23) need ≥0.3 s during own speech; AU25/26 are ignored during own speech; mouth AUs ignored under `uncertain` |
| Checks | AU–speech-energy correlation (all vs silent frames); jaw check where 68-point landmarks exist: OpenFace pitch vs a nose-below-eye-line proxy and AU26 |

Head components that overlap in time on different axes are one movement (e.g. `nod+tilt`); a fast component inside
a larger gesture is not counted as a separate beat. `semantic_candidate` marks nod/shake/tilt, the movements that can
carry meaning; beats are speaking style and are not interpreted.

Seq_006 results (`face_head_002/summary.md`):
- Michelle's head moves 11.3°/s while speaking vs 4.1°/s silent inside gaze HOLDs (jitter 0.12°).
- Of 18 gaze shifts with head data, the head turned a median 3.9°: 6 eyes only, 6 head done at landing, 6 still moving.
- Short head movements during speech are not locked to energy peaks (30 % vs 34 % chance).
- AU25 follows speech energy only while speaking; AU26 is weak. Teddy's OpenFace pitch follows AU26 in shot 214
  while rigid points do not, so pitch movements during speech are flagged unless checked.
- 22 head movements on reliable tracks, 14 semantic candidates.

Stage07 `--face-head` adds these as `au` and `head` observations for C only. Assembly derives `shot_ids` for every
annotation and observation from Stage02 Shot times (overlaps under 0.05 s are ignored: millisecond rounding and the
one-frame corrected cuts). The annotated video panel shows the current Shot, active annotation IDs and human gaze.
## Acting interpretation · analysis_003

Annotations follow [the interpretation prompt](prompts/stage07_acting_interpretation_prompt_v1.md). Each beat names the
character's own Stage04.2 objective (`objective_refs`) and a `tactic`; semantic head movements get
`head_gesture_readings`. Assembly checks both when `interpretation_input.json` is present. Current run: `pilot_006`.

```sh
s=reports/stage07/tt12300742
uv --no-cache run --no-sync python -m oscardp.stage07.interpretation_input --run $s/pilot_006 \
  --movie-dir /media/sdu/SiyaoDu/datasets/oscar_movie_analysis/tt12300742 --sequence-id seq_006 \
  --reuse $s/analysis_002 --output-dir $s/analysis_003
# the LLM writes $s/analysis_003/acting_annotations.json from the prompt
uv --no-cache run --no-sync python -m oscardp.stage07 run --prepared --output-dir $s/pilot_006 \
  --analysis-dir $s/analysis_003 --model-dir models/stage06 \
  --gaze-labels reports/stage06_3/tt12300742/seq_006/labels_001/gaze_labels.json \
  --face-head $s/face_head_002/face_head_evidence.json
uv --no-cache run --no-sync python -m oscardp.stage07.review_sheet --run $s/pilot_006 --output-dir $s/review_006
```
