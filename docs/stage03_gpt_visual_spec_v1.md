# OscarDP — Stage03 v1.0: GPT Visual Annotation

**Status:** Proposed data contract; annotation is interactive/manual with ChatGPT GPT-6 High, not a deployed API or CV pipeline. Do **not** implement local YOLO, SCRFD, ArcFace, object detection, face tracking, AU, gaze or emotion here.

## Inputs and source integrity
- Verified explicit Frozen Stage02 Release: `frozen_manifest.json` with Release ID and SHA-256, plus the selected immutable Stage01 and Stage02 hashes. This selects a movie and authenticates textual context, **not** the sampling JPEGs.
- Canonical Stage01 Early/Middle/Late JPEGs and `.internal/<movie_id>/frame_samples_v1.json`; verify sample policy, image files and per-image hashes separately. Short Shot aliases can have repeated image paths; do not fabricate distinct frames.
- Use only the three sampled images as **visual evidence**. Do not let Stage02 screenplay characters become visually present by assertion. Stage02 belongs primarily to Stage04, where textual context is deliberately combined.
- Images may be uploaded directly or accessed through connected Google Drive in small batches. Never claim to have viewed a file not successfully opened.

## Official output and batching
One `oscar_movie_analysis/<movie_id>/Stage03_<movie_id>.json` per **fully covered** film. Intermediate batch fragments are saved under `reports/stage03_pilot/` or other nonofficial workspace paths with explicit movie, Shot range, batch ID, input hashes and annotation version. A future deterministic merge/validator must check exact Stage01 Shot inventory, order, completeness, image hashes, aliases and cross-references before publication. Model output alone is not a PASS, and no automation is implemented by this contract.

**Root fields**: `movie_id`, `stage_version="3.0"`, `inputs` (verified `frozen_release_id`, `frozen_manifest_sha256`, `stage01_sha256`, `stage02_sha256`, `frame_samples_sha256`), `shots[]`. Per-Shot `shot_id` references Stage01. Never duplicate Stage02 narrative, dialogue or temporal fields.

## Per-Shot `samples[]`
Exactly the three logical roles `early`, `middle`, `late`, each with `frame_index`, `image_relpath`, and `image_sha256` bound to Stage01 sample metadata. Two logical roles may refer to the same physical JPEG for very short Shots.

## Per-Shot `visual` (GPT, controlled tags)
| Field | Allowed values |
| --- | --- |
| `shot_scale` | `ECU, CU, MCU, MS, MLS, LS, ELS, mixed, not_applicable, unknown` |
| `camera_angle` | `eye_level, high, low, overhead, unknown` |
| `camera_motion_evidence` | `no_evident_motion, possible_motion, unknown` |
| `primary_subject_position` | `left, center, right, multiple, not_applicable, unknown` |
| `over_the_shoulder` | `true, false, null` |
| `visual_focus` | `person, object, environment, mixed, unknown` |
| `evidence_roles` | Nonempty unique subset of `early, middle, late` |

`shot_scale` is relative to human figures; when none are assessable, use `not_applicable`. `mixed` means a visible change across samples, not uncertainty. Three stills do not establish specific pan/dolly/zoom actions, and still-frame similarity does **not** prove a static camera.

## `subjects[]` (visually distinct people within this Shot)
Each has a **Shot-local** `subject_id` (e.g. `subject_001`), `evidence_roles`, `subject_facing` (`frontal, three_quarter, profile, back, mixed, unknown`), `face_visibility` (`fully_visible, partially_visible, not_visible, mixed, unknown`), `face_quality` (`good, limited, unusable, mixed, not_applicable, unknown`), `occlusion` (`none, partial, heavy, mixed, not_applicable, unknown`), `appearance_mode` (`direct, reflection, screen_or_photo, unknown`). If a person is seen across samples, use one ID only when appearance/position support that link. Do not equate an ID with a verified actor, speaker, or continuous track. Record apparent on-screen people, including background extras where salient, but do not infer off-screen subjects.

## `objects[]` (salient props only)
Each has Shot-local `object_id`, short English `label` (e.g., `letter`, `cup`), `object_framing` (`detail_closeup, prominent, contextual, unknown`), and `evidence_roles`. Objects have **no invented BBox**. It is not necessary to exhaustively enumerate every background object.

## `visible_relations[]`
Each references existing `subject_id` and `object_id`, `relation_type` (`holding, touching, wearing, sitting_on, leaning_on, near, uncertain`) and `evidence_roles`. Only directly observable relations; no imagined transfer action, gaze target, motive or temporal continuity. Empty arrays are valid.

Other per-Shot fields: `visual_summary` (short, strictly visible content, not narrative interpretation), and `review_status` (`unreviewed, model_reviewed, human_verified, human_corrected`). GPT self-checking cannot set a human status.

## Validation and research boundary
The schema rejects extra CV/face fields, Actor IMDb IDs, verified identities, bbox coordinates and unrestricted tags. A future Python validator must additionally enforce all `shot_id` and sample inventory bindings, unique subject/object IDs, existing interaction references, role evidence, no conflicting batch overlaps, and zero invented roles. Uncertain predictions must remain `unknown` or `null` rather than guessed.

Stage04 alone merges actual visual observations with Stage02's scene/character narrative. Stage05 performs real video-based face detection, identity verification and temporal AU/head/gaze analysis; Stage06 addresses explanations and intent.
