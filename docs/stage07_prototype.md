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
