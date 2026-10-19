# Codex task — Implement OscarDP Stage03 CV Detection v1.0

Repository: https://github.com/SiaDu/OscarDP

Read before modifying: `AGENTS.md`, `docs/stage01_spec.md`, `docs/stage01_frames_upgrade.md`, `docs/stage02_spec_v1.md`, `docs/pipeline_overview.md`, `src/oscardp/shots/frame_samples.py`, `src/oscardp/shots/validation.py`, `src/oscardp/shots/frames_upgrade.py` and existing tests.

Read the four supplied contract files (copy into the repository if not already present):
- `docs/stage03_cv_spec_v1.md`
- `docs/stage04_gpt_spec_v1.md`
- `docs/schemas/stage03_cv_v1.schema.json`
- `docs/schemas/stage04_gpt_v1.schema.json`

Implement **Stage03 only**, adhering strictly to the Stage03 contract. Keep the Stage04 files as a future-stage interface description; do not implement Stage04/05/06 and do not change fixed Stage02 schema or frozen data.

## Models and scope

- Ultralytics YOLO26s (`yolo26s.pt`) with COCO class `0` (`person`) only; default `conf=0.15`, `imgsz=960`, configurable and recorded.
- InsightFace `buffalo_l`, `allowed_modules=['detection']` (SCRFD face detector) using ONNX Runtime; independently run detection on every **distinct** Stage01 sampled image. Default face input size 640, configurable. Preserve returned face detector confidence and 5-point landmarks when available. CPU fallback is allowed only with an explicit warning and accurate provider provenance.
- No object/prop detection, no identity recognition, no face embeddings, no Actor IDs, no person tracking, no VLM, no emotion/AU/gaze, no camera or narrative semantics. Do not force face↔person association when ambiguous; null is valid.
- Preserve original-image **pixel** `xyxy` box coordinates, clipped/validated within image dimensions after preprocessing. Do not report resized-model tensor coordinates. Handle extreme close-ups with Face Detection independent of YOLO.

## Required architecture

Create `src/oscardp/stage03/` with modular CLI, inference, schema validation, input binding, progress/reporting and safe atomic publication. Reuse Stage01 `sample_indices`, `physical_samples`, `frame_samples_v1.json`, manifest selection/checksum routines and hashing; avoid duplicating sampling algorithms. Read sampled JPEGs only — do not rerun FFmpeg or TransNetV2.

CLI target:

`uv run python -m oscardp.stage03 detect --movie-id tt12300742 --analysis-root /media/sdu/SiyaoDu/datasets/oscar_movie_analysis --frozen-manifest /media/sdu/SiyaoDu/datasets/oscardp_releases/oscardp_stage02_v1_0_18movies_20261018/frozen_manifest.json --expected-manifest-sha256 375a58f5f94fbaae1036efa790475b794d7e60130b9686f96e7dfb2fbf44b443 --pilot-start-shot shot_000001 --pilot-end-shot shot_000020 --report-root reports/stage03_pilot/tt12300742 --device cuda`

Pilot range must be inclusive and require a separate explicit report output location; do not publish an official Stage03 JSON for a partial movie.

For full-film processing, support `--full-movie` and `--publish` (two explicit flags). Only with full coverage, schema validation, all JPEG/hash/alias checks and successful inference may `Stage03_<movie_id>.json` be atomically published in the Canonical movie directory. Refuse to overwrite an existing completed Stage03 result without an explicit safe, versioned replacement procedure; preserve old output byte hashes and backups. Do not process unselected movies or every folder automatically.

Also provide `uv run python -m oscardp.stage03 validate --movie-id ... --analysis-root ... --frozen-manifest ... --expected-manifest-sha256 ...` for full output. It must validate deep structural references, source hashes, source image inventory, frame aliases and intra-sample detection association, not merely JSON Schema syntax.

## Data contract

Official `Stage03_<id>.json`: `movie_id`, `stage_version`, `inputs`, `run_id`, `shots[]`. Per shot `shot_id`, `samples[]` in the exact `early`, `middle`, `late` logical order. Per sample: role/frame_index/image_relpath/image_sha256/width_px/height_px/status/persons[]/faces[]. Each person: detection_id, original pixel bbox, raw score. Each face: detection_id, bbox, raw score, 5 landmarks or null, conservative same-sample person_detection_id or null. The supplied Stage03 JSON Schema is normative. Empty arrays mean no detection *in this sampled frame*, not absence from the Shot or movie.

All images and earlier stages are immutable. Bind to the explicitly verified 18-film frozen release for the pilot, but do not claim the upgraded JPEGs are physically frozen inside that release; validate their independent sampling manifest and SHA-256. Frozen `activated_for_pipeline=false` must remain unchanged. Process only user-selected film(s).

## Bug and compatibility audit

Current `src/oscardp/shots/validation.py` official-layout whitelist does not permit Stage03/04 JSON to coexist. Update ONLY the named canonical layout rules needed for the agreed future `Stage03_<id>.json` and `Stage04_<id>.json`, retaining protection against arbitrary files, symlinks, malformed names and corrupt Stage01. Update `AGENTS.md`, whose old text still says Stage01 creates one midpoint JPEG and Stage02 remains unimplemented. Update `README.md`/`docs/pipeline_overview.md` to reflect agreed Stage03/04/05/06 boundaries, not older ones. Keep existing Stage01+Stage02 JSON strictly unchanged.

Use `pyproject.toml` and `uv.lock` for reproducible dependencies. Respect the repo's `torch==2.14.*` pin and existing CUDA installation: never silently downgrade PyTorch. Log the actual model checkpoint SHA-256, package versions, provider, parameters, source hashes, GPU and runtime; store technical provenance under `.internal/<movie_id>/stage03/` or `reports/`, not in the movie root. Avoid committing model weights, screenshots, test input media, and detection outputs to git.

## Engineering and tests

- Preflight with `three_sample` and source-hash checks; fail safely on missing or inconsistent frame metadata, frozen Stage01/02 mismatch, aliases and wrong images.
- Cache by actual physical JPEG content/sample metadata; infer aliased N=1/N=2 roles once, then reuse observed detections without lying about frame count.
- Reuse model instances across many images, stream/chunk images to control VRAM, show progress, emit transparent per-shot errors, use safe partial work products, support interruption and resume.
- No duplicate detection boxes or spurious IDs when rerun; stable ordering and IDs within frame; unit scores in [0,1], finite geometries.
- Generate optional diagnostic images with person/face rectangles for spot checks under report directories only.
- Tests: complete schema/shot inventory, 1-/2-/3-frame alias cases, same-sample associations, no-person-but-face close-up, empty detection, GPU→CPU provider selection, failure/recovery/idempotence, input hash mismatch, pilot isolation, Stage01 validator coexistence, no changes to existing Stage01/02/SRT/shot_img content. Use mocks and small synthetic images; avoid GPU-required CI tests.
- Run `uv run pytest -q`, share changed files/tests and exact pilot command. Do not run the real full movie or any of the other 17 films without user permission.

First ensure the four contract files are in the requested docs paths. Implement Stage03, run tests, and stop. The user will execute the pilot manually and return results for evaluation.
