# AGENTS.md — OscarDP

## 1. Goal

OscarDP runs inside **WSL Ubuntu** and processes movies stored on the Windows `I:` drive.

Current task:

1. Read movies from:

```text
/mnt/i/datasets/oscar_movie
```

2. Use **TransNetV2** to detect shot boundaries.
3. Convert boundaries into shot frame ranges and timestamps.
4. Extract one representative keyframe for each shot.
5. Save QC images for checking suspicious or sampled boundaries.

Do not add face recognition, OpenFace, AU extraction, gaze, shot-scale classification, or camera-motion classification unless explicitly requested.

Stage 2 under `oscardp.script_context` may parse screenplay PDFs, clean SRT
subtitles, align subtitle dialogue to screenplay blocks, and map those results
onto the existing shots. It must remain independent from TransNetV2 Stage 1.
It must never modify `shots.jsonl`, source videos, keyframes, or Stage 1 QC.
An LLM may only repair flagged page structure, select existing IDs for local
low-confidence alignments, or add optional scene annotations. It must never
rewrite the original screenplay, action, or dialogue text.

OpenAI integration is optional. Normal `process-one` must never make a paid API
call. API keys and account/batch identifiers must never enter Git or logs.
Responses may select only subtitle and block IDs supplied by the exact original
review request. Deterministic baseline JSON/JSONL files are immutable during
review application; reviewed files use separate `.llm_reviewed.jsonl` names.
Full-dataset OpenAI or Stage 2 processing is out of scope unless explicitly
authorized.

OpenAI Batch response schemas must be request-specific: allowed block IDs are
exactly the current request's dialogue candidates. Normally preserve screenplay
order. A bounded same-scene backward match may represent repeated or locally
reordered final-film dialogue only when it uses the explicit
`repeated_or_reordered_dialogue` basis; the default maximum backward distance is
three dialogue blocks. Never permit a foreign ID, distant jump, or cross-scene
backward jump. Candidate-recall failures use `uncertain`, not invented IDs.

---

## 2. Paths

Default input:

```text
/mnt/i/datasets/oscar_movie
```

Default output:

```text
/mnt/i/datasets/oscar_movie_processed
```

Use Linux paths only. Do not pass Windows paths such as `I:\datasets\oscar_movie` to WSL tools.

All paths must be configurable through CLI arguments.

---

## 3. Source-data safety

Never modify source movies.

Do not:

- rename, move, delete, or overwrite source files;
- transcode in place;
- write generated files into the source folders.

All outputs must go under `OUTPUT_ROOT`.

---

## 4. Supported videos

Recursively discover:

```text
.mp4
.mkv
.mov
.m4v
.avi
.webm
```

Ignore hidden files, partial downloads, subtitles, PDFs, and images.

Use IMDb ID from the filename or parent folder as `movie_key` when available. Otherwise use a normalized relative-path stem.

---

## 5. Pipeline

```text
discover movies
→ ffprobe video metadata
→ TransNetV2 inference
→ convert transitions to shots
→ extract midpoint keyframes
→ generate QC outputs
→ validate
```

Process one test movie successfully before starting the full batch.

---

## 6. TransNetV2 scope

TransNetV2 is used only for **shot-boundary detection**.

It does not directly predict:

```text
shot_scale
camera_movement
visible_characters
speaker
face identity
AUs
gaze
```

For now, keep:

```json
{
  "shot_scale": null,
  "camera_movement": null
}
```

Prefer the official TransNetV2 implementation. Use the official PyTorch inference path when compatible with the current environment.

---

## 7. Video metadata

Save technical metadata needed for timestamp conversion in:

```text
video_metadata.json
```

Required fields:

```json
{
  "source_video_relpath": "",
  "duration_sec": 0.0,
  "width": 0,
  "height": 0,
  "codec_name": "",
  "fps": 0.0,
  "frame_count": null,
  "is_vfr": false,
  "timestamp_source": "frame_index_cfr"
}
```

Do not assume every movie is exactly 24 fps.

For CFR video:

```text
timestamp_sec = frame_index / fps
```

For VFR video, use decoded PTS timestamps instead of average FPS.

---

## 8. Shot representation

Use zero-based frame indices.

Intervals must use:

```text
start_frame: inclusive
end_frame: exclusive
```

Therefore:

```text
frame_count = end_frame - start_frame
```

Adjacent shots must satisfy:

```text
previous.end_frame == next.start_frame
```

Shot IDs:

```text
shot_000001
shot_000002
shot_000003
```

---

## 9. Output structure

For each movie:

```text
OUTPUT_ROOT/
└── <movie_key>/
    ├── video_metadata.json
    ├── shots.jsonl
    ├── keyframes/
    │   ├── shot_000001.jpg
    │   ├── shot_000002.jpg
    │   └── ...
    ├── qc/
    │   ├── boundaries/
    │   ├── boundary_contact_sheet.jpg
    │   └── qc_summary.json
    └── logs/
        └── process.log
```

Dataset-level manifest:

```text
OUTPUT_ROOT/index.jsonl
```

`shots.jsonl` is the main data file. Do not maintain CSV as a second primary format. Export CSV later only when needed.

Do not generate `transitions.jsonl` by default. Raw transition output may be saved only in debug mode.

---

## 10. Keyframes

Save one representative keyframe per shot.

Default rule:

```text
keyframe_frame = floor((start_frame + end_frame - 1) / 2)
```

Store it as:

```text
keyframes/shot_000001.jpg
```

The midpoint keyframe is used later for browsing, actor checks, and shot-scale analysis.

Do not replace midpoint keyframes with boundary frames.

Extract keyframes in a sequential decode pass. Do not launch one FFmpeg process per shot.

---

## 11. Boundary QC

Boundary QC is separate from representative keyframes.

For a boundary at frame `N`, the useful review pair is:

```text
before = frame N - 1
after  = frame N
```

Example:

```text
qc/boundaries/boundary_000001_before.jpg
qc/boundaries/boundary_000001_after.jpg
```

Do not save every boundary pair by default for the full dataset.

Default QC should save boundary pairs for:

- low-confidence detections;
- very short shots;
- sampled boundaries;
- other suspicious cases.

Support an optional flag:

```text
--save-all-boundary-frames
```

Use full boundary-frame export only for test movies or manual debugging.

`boundary_contact_sheet.jpg` should show sampled or suspicious before/after pairs.

`qc_summary.json` should contain:

```json
{
  "shot_count": 0,
  "mean_shot_duration_sec": 0.0,
  "median_shot_duration_sec": 0.0,
  "very_short_shot_count": 0,
  "missing_keyframes": 0,
  "validation_passed": true
}
```

---

## 12. `shots.jsonl` schema

One JSON object per shot:

```json
{
  "movie_key": "tt32536315",
  "shot_id": "shot_000001",

  "source_video_relpath": "Blue Moon/Blue.Moon.2025.mkv",

  "start_frame": 0,
  "end_frame": 120,
  "frame_count": 120,

  "start_time": "00:00:00.000",
  "end_time": "00:00:05.005",
  "start_sec": 0.0,
  "end_sec": 5.005,
  "duration_sec": 5.005,

  "keyframe_frame": 59,
  "keyframe_time_sec": 2.461,
  "keyframe_relpath": "keyframes/shot_000001.jpg",

  "boundary_before_confidence": null,
  "boundary_after_confidence": 0.98,

  "shot_scale": null,
  "camera_movement": null,

  "model": {
    "name": "TransNetV2",
    "threshold": 0.5
  }
}
```

The numbers above are examples only.

Use valid JSON. Do not output `NaN`, `Infinity`, NumPy scalar types, or Windows paths.

---

## 13. CLI

Provide commands similar to:

```bash
python -m oscardp.shots discover   --input-root /mnt/i/datasets/oscar_movie

python -m oscardp.shots process-one   --video "/mnt/i/datasets/oscar_movie/<movie>"

python -m oscardp.shots process   --input-root /mnt/i/datasets/oscar_movie   --output-root /mnt/i/datasets/oscar_movie_processed

python -m oscardp.shots validate   --output-root /mnt/i/datasets/oscar_movie_processed
```

Useful flags:

```text
--threshold
--device
--resume
--overwrite
--dry-run
--limit
--movie-key
--save-all-boundary-frames
--save-raw-predictions
```

Default behavior should resume completed work rather than overwrite it.

---

## 14. Validation

Before marking a movie complete, verify:

- at least one shot exists;
- shot IDs are unique and sequential;
- `start_frame < end_frame`;
- shots are sorted;
- there are no overlaps;
- there are no unexplained gaps;
- timestamps are monotonic;
- each keyframe lies inside its shot;
- every keyframe file exists;
- the final shot reaches the final decoded frame.

A failed movie must not stop the whole batch. Log the error and continue.

---

## 15. Development order

Implement in this order:

1. movie discovery;
2. ffprobe metadata;
3. TransNetV2 on one movie;
4. transition-to-shot conversion;
5. `shots.jsonl`;
6. midpoint keyframes;
7. QC boundary pairs;
8. validation;
9. resume test;
10. full batch.

Do not run the full dataset until one test movie passes manual inspection.

---

## 16. Later stages

### Stage 2 reviewed-shot QC

In reviewed shot context, `scene_candidates` contains direct subtitle/script
overlap evidence only. A shot assigned with `same_scene_interpolation` must keep
its inherited confidence in `scene.confidence` and use
`scene_candidates: []`; do not manufacture zero-overlap candidates.

Non-anchor sequence audit schema 2.0 distinguishes
`review_target_subtitle_id` from `regression_trigger_subtitle_id`. Preserve both
roles when building human-review packages and never change model decisions as a
side effect of the audit.

### Shot scale

Target labels:

```text
LS
FS
MS
CU
ECU
```

This requires a separate visual classifier or body/face-size analysis.

### Camera movement

Possible labels:

```text
static
pan
tilt
push
pull
tracking
handheld
zoom_in
zoom_out
complex
unknown
```

This requires temporal analysis such as optical flow, feature matching, and global-motion estimation.

A single keyframe is not enough to classify camera movement.
