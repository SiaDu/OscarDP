# OscarDP — Stage03 v1.0: Person and Face CV Detection

**Status:** Proposed contract to commit before implementing Stage03. **Scope:** sampled-frame evidence only; no VLM interpretation, cast identification, cinematic tags, objects, AU, emotion, gaze, or scene construction.

## Position in pipeline

Stage01 supplies shot IDs/boundaries and `shot_img/<shot_id>_{early,middle,late}.jpg` (including physical aliases for 1–2-frame shots). Stage02 supplies independent, frozen text observations; Stage03 must **not use Stage02 to modify or infer visual detections**. Stage04 will combine them with GPT visual and scene annotations; Stage05 will confirm identities and measure face/gaze/AU over actual footage.

## Sources and binding

Inputs: a user-selected **verified frozen manifest**, matching canonical `Stage01_<movie_id>.json`, `.internal/<movie_id>/frame_samples_v1.json`, and image assets. Resolve the chosen frozen release by manifest SHA-256 and its per-movie selected Stage01/Stage02 hashes, but treat imagery as separately verified mutable-workspace assets: check the sampling metadata's Stage01 binding, policy, sample-image hashes and file integrity. Never claim imagery itself is frozen by Stage02's manifest. No automatic activation; no fallback to a different movie or uncontrolled directory discovery. **Existing Stage01/02/SRT/PDF/images are read-only.**

## Official output

Canonical movie directory: `oscar_movie_analysis/<movie_id>/Stage03_<movie_id>.json`. Do not write a formal output there until the entire film is processed and validated. Pilot/partial results must go under a separate report directory, e.g. `reports/stage03_pilot/<movie_id>/`, with explicit coverage. Technical run configuration/logs/diagnostic overlays belong under `.internal/<movie_id>/stage03/` or external `reports/`, not the official movie root.

## Root fields

- `movie_id`: IMDb `tt` identifier.
- `stage_version`: literal `"3.0"`.
- `inputs`: `frozen_release_id`, `frozen_manifest_sha256`, `stage01_sha256`, `stage02_sha256`, `frame_samples_sha256` (SHA-256 hex strings, matching selected sources).
- `run_id`: unique deterministic/auditable processing run identifier; corresponding immutable run manifest stores versions of Ultralytics, ONNX Runtime, InsightFace, exact checkpoint paths/hashes, threshold, image size and hardware.
- `shots`: ordered array with **exactly** the Stage01 shot inventory and shot IDs in full output.

## Per-shot fields

- `shot_id` (required): stable Stage01 ID.
- `samples`: **three logical roles** (`early`, `middle`, `late`) in order; aliases are allowed and must reproduce `frame_samples_v1.json` exactly. The model may infer once per distinct physical image and reuse results for logical aliases.

## Per-sample fields

- `role`: enum `early | middle | late`.
- `frame_index`: integer from authoritative Stage01 sample metadata, never invented from FPS.
- `image_relpath`: relative path in canonical movie directory. Aliases may point to the same physical JPEG.
- `image_sha256`: verified hex SHA-256 from sampling metadata.
- `width_px`, `height_px`: image dimensions.
- `status`: literal `ok` for an official complete output. A valid image with no detections is **ok** with empty arrays, not an error. Pilot/interrupted diagnostics may log errors outside the official output, which may not be published until corrected.
- `persons`: array of per-frame *human* detections, no object classes.
- `faces`: independently detected face instances; **do not gate face detection on YOLO Person success**.

## Person detection (`persons[]`)

- `detection_id`: e.g. `early_p001`, unique within a sample.
- `bbox_xyxy_px`: four finite numbers `[x1, y1, x2, y2]` in ORIGINAL IMAGE PIXEL COORDINATES; `0 <= x1 < x2 <= width_px`, `0 <= y1 < y2 <= height_px`.
- `score`: original YOLO person detection score, number from 0 to 1. Do not label it calibrated probability.

## Face detection (`faces[]`)

- `detection_id`: e.g. `early_f001`.
- `bbox_xyxy_px`: original-image pixel coordinates with same bounds as above.
- `score`: original SCRFD detection score in [0,1].
- `landmarks5_xy_px`: five `[x,y]` points or `null` if unavailable. No invented landmarks.
- `person_detection_id`: a valid person detection ID from **this sample**, or `null` if not reliably associated. Do not discard a valid face when the body detector misses a close-up.

Bounding boxes must preserve image coordinate space even when the model resizes/letterboxes for inference. Box association must remain conservative. Three frames do not establish a stable actor identity or continuous track.

## Prohibitions

No `character_id`, `actor_imdb_id`, emotion, AU, facial expression, gaze target, camera angle, shot scale, object labels, people-object relations, face visibility/occlusion interpretation, or human-verified identity in Stage03. These belong to Stage04/05/06.

## Deterministic validation

Verify frozen and canonical hashes, shot coverage/order/unique IDs, exactly three logical roles, image paths/aliases/frame indices against `frame_samples_v1.json`, real JPEG dimensions/hash, all finite in-range scores and coordinates, globally unique sample-local detection IDs, valid intra-sample face→person references, and required run provenance. Empty detections are legal. Explicit `--movie-id` for pilots and no implicit mass processing. Atomic publication, safe resume, no overwriting an existing valid formal output without explicit versioned promotion.

### Example (illustrative, not actual Bugonia data)

```json
{
  "movie_id": "tt12300742",
  "stage_version": "3.0",
  "inputs": {
    "frozen_release_id": "oscardp_stage02_v1_0_18movies_20261018",
    "frozen_manifest_sha256": "<sha256>",
    "stage01_sha256": "<sha256>",
    "stage02_sha256": "<sha256>",
    "frame_samples_sha256": "<sha256>"
  },
  "run_id": "stage03_bugonia_pilot_v1",
  "shots": [{
    "shot_id": "shot_000001",
    "samples": [{
      "role": "early", "frame_index": 10,
      "image_relpath": "shot_img/shot_000001_early.jpg",
      "image_sha256": "<sha256>", "width_px": 1920, "height_px": 1080,
      "status": "ok",
      "persons": [{"detection_id": "early_p001", "bbox_xyxy_px": [250, 80, 1150, 1060], "score": 0.92}],
      "faces": [{"detection_id": "early_f001", "bbox_xyxy_px": [500, 160, 780, 510], "score": 0.94, "landmarks5_xy_px": null, "person_detection_id": "early_p001"}]
    }]
  }]
}
```

The example omits `middle` and `late` **only for brevity** and includes placeholder hashes; it is **not a valid complete official output**. Pilot artifacts are stored outside the official movie root and explicitly identify their coverage.
