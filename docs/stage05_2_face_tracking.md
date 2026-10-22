# Stage05.2 face detection and shot-local tracking

Research prototype. This module reads Stage01 boundaries and original video;
it does not modify Stage01–04 or require Stage03's inferred person counts.
It does not load recognition/embedding, AU, gaze, or actor identity models.

## Environment and CLI

Install the optional detector dependency with `uv sync --extra stage05`.
Existing FFmpeg and FFprobe executables are required. OpenCV's CPU DNN backend
runs YuNet; ONNX Runtime and InsightFace are not required. This adds no Torch,
CUDA, tracking-framework, or recognition dependencies.

The model is explicit: supply the local ONNX path with `--model`. If unavailable,
obtain `face_detection_yunet_2023mar.onnx` from OpenCV Zoo's
`models/face_detection_yunet` directory using Git LFS or its binary download,
not the small LFS pointer file. No detector is substituted automatically.

Pilot command (run first with `--dry-run`, then without it):

```bash
uv run --extra stage05 python -m oscardp.stage05 detect-track \
  --movie-id tt12300742 \
  --analysis-root /media/sdu/SiyaoDu/datasets/oscar_movie_analysis \
  --input-root /media/sdu/SiyaoDu/datasets/oscar_movie \
  --shot-ids shot_000209 shot_000210 shot_000211 shot_000214 shot_000217 \
  --model models/yunet/face_detection_yunet_2023mar.onnx \
  --inference-max-size 640
```

For an already prepared environment, `uv run --no-sync python ...` preserves
the installed packages. The exact invocation and measurements for Bugonia are
in `reports/stage05_2/tt12300742/pilot_report.md` after running the pilot.
Default inference uses native resolution; the pilot explicitly uses a 640-pixel
longest edge because YuNet missed large close-up faces at native resolution.
Neither setting changes temporal sampling: every original frame is processed.
X and Y scaling use the actual rounded inference dimensions independently.

`--video` bypasses discovery. Without it, the existing `shots.discovery`
implementation must resolve exactly one IMDb-keyed movie. Duplicate candidates,
missing models, invalid bounds, changed resume inputs, and decode failures fail
explicitly. `--shot-ids` is mandatory. `--device cpu` is the supported device.

## Exact frame and time semantics

Frame indices are global, zero-based; start is inclusive and end is exclusive.
FFmpeg decodes sequentially from frame zero, selects only requested intervals
plus following boundary frames, and streams BGR images into bounded memory.
No approximate seek, transcoding of the movie, TransNet rerun, or FPS conversion
occurs. Prefix and gap frames are decoded internally but never detected or
written as observations. This deliberately trades startup decode cost for
auditable index alignment. Stage01's existing JPEG extractor cannot stream
full-rate annotated video, so this adapter reuses discovery/probing rather
than generating hundreds of JPEG files with that extractor.

Stage01's minimal JSON does not retain the earlier timeline object. If **all**
Stage01 boundary times agree with global frame indices divided by source FPS
within 100 microseconds, use its CFR rule. Every selected source PTS is checked
against that rule (tolerance max(2 ms, 0.1 frame)). Otherwise use decoded PTS
relative to frame zero, with microsecond precision. Both shot boundaries are
checked against Stage01 within 2 ms. A following real frame supplies each
observation's exclusive end time, including VFR. At movie EOF, Stage01 supplies
the final exclusive end. Ambiguous timing fails instead of silently shifting
frames. There is no full-movie preflight timestamp scan.

## Tracking and output interface

Each shot has a fresh tracker. Motion prediction, gated IoU/normalized center
distance, and conservative mutual-best one-to-one assignment associate faces.
Near-tied alternatives break the link and generate an ambiguity warning.
An adjacent-frame thumbnail difference >=0.18 also breaks active associations
and records a visual-discontinuity warning. This conservative safeguard may
split tracks on flashes or rapid motion; it never changes Stage01 shot IDs or
boundaries and does not claim to verify an editorial cut. The threshold is
configurable with `--discontinuity-threshold`.
Unmatched observations create tracks; short losses retain tracks for at most
`--max-missed` frames. Three observed hits confirm a track by default.
Tentative tracks remain visible in output and QC. This geometric baseline can
still switch IDs during crossings; zero ambiguity warnings is not a claim of
zero switches. No appearance or identity evidence is used.

Outputs under `reports/stage05_2/<movie_id>/`:

- `face_detections.jsonl`: one real model detection per row, with global frame,
  movie-relative time, original-pixel `bbox_xywh`, confidence, five landmarks,
  shot-local track ID, and `detection_status: observed`. No predicted rows.
  A frame with no detections has no row; completed checkpoint bounds/counts
  distinguish this from an unprocessed frame.
- `face_tracks.json`: versioned track summaries; observed start and last-observed
  exclusive end, score/size means, detection count, internal missed count,
  valid detection ratio and final tentative/confirmed status.
- `qc_report.json`: per-shot counts, internal-gap detection ratios, short tracks
  (<3 detections), low quality observations (score <0.9 or min box edge <24 px),
  ambiguous links, incomplete shots/errors, detector and model provenance.
- `previews/<shot_id>.mp4`: 960-pixel-wide, approximately 6 FPS QC video with
  actual sampled observations, IDs, scores, shot/frame/time labels. These
  display samples do not change the full-rate tracking input.
- `run_config.json`, `run_metrics.json`, and `checkpoints/`: configuration,
  source file identities, model/Stage01 hashes, timing and recovery evidence.

Missed counts cover gaps between first/last observations, excluding trailing
lost frames. Valid ratio = observed count / that inclusive span. JSON never
contains NaN or Infinity. QC is program output, not measured annotation recall
or tracking accuracy.

`--resume` requires identical inputs/configuration and checks that completed
shot files and previews remain present with recorded sizes. Each shot commits
its JSONL/preview before an atomic completion receipt. Failed/incomplete shots
retry from their beginning; final aggregate files are rebuilt without duplicate
observations. A process lock prevents concurrent writers. The matching config
uses file path/size/mtime plus Stage01/model digests, not a full movie hash or
a formal release manifest. Do not manually modify checkpoint artifacts.
Existing runs require `--resume` or explicit `--overwrite`; the latter replaces
only this implementation's known outputs. Preserve research runs before using it.

Stage05.3 can consume `face_tracks.json` plus observed detections and the source
video to collect face crops for identity research. Stage05.4 can read observed
frame indices, timestamps, original-pixel boxes and landmarks; it must validate
crop quality and cannot treat gaps or predictions as observed faces. Track IDs
do not encode identity and never imply cross-shot links.

## Experimental audited boundary overrides

Use `--boundary-sidecar PATH` to opt into explicitly listed corrections; default
processing continues to use unmodified Stage01 ranges. Unapproved model-reviewed
drafts additionally require `--allow-experimental-boundaries` and a separate
output directory such as `--output-root reports/stage05_2_corrected`. They cannot
overwrite/reuse an original-range run. No source file or review status is changed.
Unaudited selected shots retain original ranges. Conflicts with unselected original
neighbors are printed and recorded; overlapping selected intervals or disagreement
between adjacent corrected cuts fail. Both original/effective temporal objects
are recorded in config, observations, tracks, QC, checkpoints and run metrics.
Sidecar/audit fingerprints participate in resume checks, and each completed
corrected-shot receipt must match its exact processing configuration.

See `stage05_boundary_correction_sidecar_v1.md` for validation and evidence rules.
The proposed ranges for Bugonia were used only in an explicit five-shot experiment.
Its before/after report is under `reports/stage05_2_corrected/tt12300742/comparison/`.

## Validation

`uv run --no-sync python -m pytest tests/test_stage05.py`

Tests include continuous motion, multiple faces/order changes, short loss,
expired/far reentry, ambiguous links, shot reset, anisotropic coordinate mapping,
empty output, JSON fields/finiteness, lossless synthetic CFR pixel/frame alignment,
synthetic VFR timestamps, missing video/model, resume guards, and failed-shot retry.
