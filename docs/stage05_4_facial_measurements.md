# Stage05.4 frame-level OpenFace measurements

This pilot reads corrected Stage05.2 observations and matching tentative
Stage05.3 clusters. It emits AU, head pose, gaze, face/eye landmarks and model
parameters per expected source frame. It does not detect performance events,
AU onset/apex/offset, emotion, intent, gaze targets or verified actor names.

## Run and verify

Use the existing original OpenFace C++ `FeatureExtraction` executable with its
adjacent `model/` and `AU_predictors/` directories. There are no automatic model
downloads, fallback measurements or alternative blendshape models.

```sh
uv sync --inexact --extra stage05-performance --extra stage05-identity --extra test
uv run --no-sync python -m oscardp.stage05 extract-performance \
  --movie-id tt12300742 \
  --tracking-run reports/stage05_2_corrected/tt12300742 \
  --identity-run reports/stage05_3/tt12300742/pilot_001 \
  --shot-ids shot_000209 shot_000210 shot_000211 shot_000214 shot_000217 \
  --openface-binary models/OpenFace/build/bin/FeatureExtraction \
  --au-mode both \
  --output-dir reports/stage05_4/tt12300742/pilot_002
```

`pilot_001` is the completed real run. Every repeat requires a new directory;
there is no overwrite or resume. `--dry-run` performs read-only provenance,
model-file and dynamic-library checks without source decoding or OpenFace
extraction. `--au-mode static|dynamic|both` is explicit; default `both` preserves
two separate runs with static as the primary baseline. `--min-confidence`
defaults to 0.8; this is a tracking gate, not a shared accuracy guarantee.
`--scratch-root` optionally selects an existing scratch parent; scratch videos
are automatically removed after use, including on failure.

```sh
uv run --no-sync python scripts/verify_stage05_performance_pilot.py \
  reports/stage05_4/tt12300742/pilot_001
uv run --no-sync python -m pytest -q
```

The actual environment added pyarrow 23.0.1, matplotlib 3.11.2 and missing
dependencies using constraints preserving numpy 2.5.3, OpenCV 4.14.0,
PyTorch 2.14.0, ONNX 1.23.2 and ONNX Runtime 1.31.0. No C++ system packages,
CUDA or existing model dependencies were replaced. The optional extra does not
require PyTorch or ONNX Runtime for OpenFace inference. `uv.lock` pins packages.

## Input checks and immutable provenance

The Stage05.2 corrected reader verifies movie/video identity, Stage01 and
Boundary Sidecar hashes, ranges, observed detections and Track endpoints.
Stage05.3 must refer to those same four corrected source files and ranges,
model/NPZ Track IDs and a unique cluster assignment per selected Track. The
present pilot accepts tentative or unknown unverified identity references;
`human_verified` and `verified_identity` cannot silently become true. Exactly
one source Track per selected Shot is required. Multi-person selection is not
implemented by this pilot.

All files in the two input run directories, Stage01–04 movie JSONs, the
Boundary Sidecar and audit JSON are hashed before and after processing. Video
identity is checked again; source movies are never rewritten. Original and
effective ranges remain in frame maps and track summaries. Known experimental
boundary status/conflicts are copied unchanged. No Stage04 scene ID is guessed:
`scene_id` and its source are null when no verified mapping is loaded.

## Exact frame transport

The unchanged Stage05 sequential FFmpeg reader supplies every frame of each
selected corrected interval, with zero-based inclusive start/exclusive end.
There is no approximate seek, frame sampling, Stage03 still-image input or QC
video input. Full-resolution BGR frames are encoded in temporary FFV1 level-3
`bgr0` MKV videos; every decoded scratch frame must have the same BGR SHA256 as
the original decoder output. Extra, missing or reordered frames fail.

Each persistent `frame_maps/<shot_id>.json` records:

```
global_frame_index = effective_start_frame + openface_local_frame - 1
```

OpenFace frames are one-based. Its temporary timestamp is separately retained
and checked against `(local_frame-1)/temporary_fps` with 2 ms tolerance for CSV
rounding/transport precision. **Movie time always comes from the original
reader**, never the temporary CFR cadence. CFR retains the already checked
Stage05.2 frame-index clock; VFR retains original relative source PTS, including
the following boundary timestamp. All expected frames remain in output even
when OpenFace omits rows or reports failure. No values are interpolated.

## Original OpenFace execution

Preflight records binary SHA256, declared source CMake version, source Git
commit, CMake cache/compiler flags when available, `ldd`, FFmpeg build information
and SHA256/size manifest for all adjacent models/configuration. Source version
does not attest that a binary was built from an unmodified release; the binary
fingerprint identifies the actual executable. Missing libraries or required
models fail. Individual command logs establish which model modules loaded.

Each Shot is a fresh single-person process:

```
FeatureExtraction -f <verified full-frame scratch.mkv> -out_dir <mode/shot> \
  -2Dfp -3Dfp -pdmparams -pose -aus -gaze \
  -fx <recorded fx> -fy <recorded fy> -cx <recorded cx> -cy <recorded cy>
```

Static adds `-au_static`; dynamic omits it. OpenFace's default video AU
configuration may contain a mixture of static and dynamic AU predictors and
offline calibration. The labels identify **extraction configurations**, not a
claim that every AU uses a dynamic predictor. Raw CSV, OpenFace metadata and
stdout/stderr logs remain under `raw_openface/<mode>/<shot>/`.

CPU execution limits OpenMP to four threads and OpenBLAS to one. No new YuNet or
identity inference runs; OpenFace's own initialization/tracking is intrinsic to
its measurements and is checked against the existing YuNet observations.

## Fields and coordinate conventions

The parser retains all actual numeric columns. It discovers `AUxx_r` intensity
and `AUxx_c` occurrence separately and never invents missing counterparts.
The real pilot emitted 17 intensity and 18 occurrence fields, including
`AU28_c` without `AU28_r`. It also retained six pose fields, eight gaze fields,
68×2 face landmarks, 68×3 model-estimated face landmarks, 56×(2+3) eye landmarks
and 40 PDM parameters. The schema follows the actual CSV, not assumed AU lists.

2D coordinates use full-frame pixels (x right, y down). Pose translation and 3D
landmarks use model-estimated millimetres. Camera origin is the reference;
positive z points away from camera. Rotation is radians, OpenFace's left-handed
positive convention `R = Rx * Ry * Rz` (pitch/yaw/roll). Eye directions are
normalized per-eye camera-reference vectors; angles are
`atan2(mean_gaze.x,-mean_gaze.z)` and `atan2(mean_gaze.y,-mean_gaze.z)`, radians.
No gaze target or calibrated external world frame is inferred.

Real intrinsics are unknown. This pilot explicitly supplies OpenFace's heuristic
`fx=fy=(500*width/640 + 500*height/480)/2`, `cx=width/2`, `cy=height/2`, and records
`camera_calibrated=false`. There is no resizing or 112×112 identity crop in
pose/gaze extraction. Pose/gaze/3D landmarks are not absolute calibrated truth.
Definitions can be checked against the original
[OpenFace output format](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Output-Format)
and [command arguments](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments);
the local C++ source/build fingerprints and model logs are the run-specific
evidence.

## Separate modality QC

OpenFace success, row availability, confidence and modality usability are
separate flags. Nonfinite values become null in structured output with explicit
column/reason lists; original CSV retains the literal source values. Missing
rows become explicit missing statuses, not invented measurements. Non-numeric,
duplicate, extra-face-ID or out-of-range local frame data are rejected.

Spatial correspondence requires facial landmark bbox IoU ≥0.25 with the observed
YuNet box and five-point normalized error ≤0.18 of its diagonal. The five-point
comparison uses eye-region means, nose and mouth corners. A mismatch invalidates
that frame's measurements and marks the Track `failed_face_association`; it is
never forcibly assigned. Spatial agreement is not actor identity verification.

Usability gates are intentionally explicit, uncalibrated research heuristics:

- Common tracking prerequisite: row present, success=1, confidence ≥0.8 and
  source-face spatial correspondence.
- AU: actual intensity/occurrence families exist, finite intensities in [0,5],
  binary occurrences, |Rx|≤0.8 and |Ry|≤1 rad.
- Pose: six finite fields, positive Tz and rotations within ±π.
- Gaze: finite unit-length eye directions (norm 0.9–1.1), finite angles within
  ±π/2, available eye landmarks, eye widths ≥10 px, eye aspect proxies ≥0.12 and
  |Ry|≤0.9 rad. Eye geometry is not independent eye-visibility ground truth;
  when tracking is unusable its dependent eye geometry is not accepted.
- Face landmarks: expected 136/204 finite columns plus common tracking gate.

Rates passing these gates are not accuracy rates. Summaries retain short and
long missing intervals (12-frame classification threshold), nonfinite counts,
face agreement, ranges and adjacent-frame QC jumps (AU >2, pose >0.35 rad, gaze
angle >0.5 rad). Jumps are review flags; values are preserved and no acting
event is created. Confidence is not an AU-specific or gaze-specific calibrated
quality model.

## Output and review

`frame_features.parquet` holds all numerical fields and provenance keys, one row
per frame **and mode**. Its metadata records source hashes, schema version and
main AU mode. `frame_status.jsonl` references exact Parquet rows and holds
reliability/reasons. It is an index, not a second large landmark store.
`track_measurements.json` groups mode-specific QC/ranges/raw CSV references.
`run_config.json`, `model_provenance.json`, `openface_executions.json`, frame maps
and `qc_report.json` keep traceability independent of earlier official outputs.

Comparisons retain each Track/AU's min/max/median/range, first-12-valid median,
paired-frame correlation/RMSE and dynamic-minus-static median. First-12-valid
median is a descriptive initial baseline, **not an identified neutral face**.
Paired comparisons use shared usable source frames only. Plot gaps remain gaps.
Primary static curves, static/dynamic overlays, pose/gaze curves and a source
frame/YuNet/OpenFace landmark Contact Sheet are saved as standalone artifacts.

The verification script checks every expected mode/frame, original/effective
ranges, movie timestamps, tentative identity references, raw CSV values,
JSON/Parquet references and QC counts. It never claims human FACS validation.

Before Stage05.5, resolve the experimental boundary reviews, manually inspect
low-confidence/jump intervals and static/dynamic disagreements, calibrate
domain-specific measurement usability, and specify missing-data handling and
camera limitations. This implementation does not perform Stage05.5 work.
