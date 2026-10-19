# OscarDP — Stage04 v1.0: GPT Visual & Narrative Annotation

**Status:** Proposed contract for a later stage; do not implement Stage04 during Stage03 development. **Scope:** GPT visually annotates sampled frames and composes film scenes using fixed Stage02 text evidence and Stage03 CV evidence. Character/actor linkage can be **candidate only**, not verified identification; Stage05 owns verified identity, tracking, AU, head pose and gaze. Stage06 owns explainable acting hypotheses.

## Sources and binding

Read a **verified explicitly selected Frozen Stage02 Release**, matching `Stage01_<movie_id>.json`, `Stage02_<movie_id>.json`, `<movie_id>.srt` as applicable, `Stage03_<movie_id>.json` plus original sampled images and `.internal/<movie_id>/frame_samples_v1.json`. Bind **all source SHA-256 hashes and release ID**. The Stage03 source video imagery is supplemental external evidence, NOT claimed to be part of the immutable Stage02 Release. Stage02 screenplay cues are text-derived, NOT automatically observed in shots. Only visual statements grounded in actual viewed frames may be asserted as observed.

Full output: `oscar_movie_analysis/<movie_id>/Stage04_<movie_id>.json`. Partial GPT batches live in a report/workspace location; never publish partial batches as a completed movie. A separate deterministic validator merges batches, checks coverage, then optionally constructs scenes across the **whole film**. GPT annotations are not automatically human-verified.

## Root fields

- `movie_id`: IMDb movie ID.
- `stage_version`: literal `"4.0"`.
- `inputs`: `frozen_release_id`, `frozen_manifest_sha256`, `stage01_sha256`, `stage02_sha256`, `stage03_sha256`, `frame_samples_sha256`.
- `shots`: entire ordered Stage01 shot inventory, each with `shot_id`, `visual`, `subjects`, `objects`, `visible_relations`, `scene_id`, `review_status`.
- `scenes`: full-film scene segmentation, ordered contiguous non-overlapping scene ranges. Stage02 `narrative.scene` is only a textual candidate, not an authoritative boundary.

## Per-shot `visual` (GPT)

- `shot_scale`: `ECU | CU | MCU | MS | MLS | LS | ELS | mixed | not_applicable | unknown`. Person-relative; objects without people may use `not_applicable` even when composition is an insert.
- `camera_angle`: `eye_level | high | low | overhead | unknown`.
- `camera_motion_evidence`: `no_evident_motion | possible_motion | unknown`. Three stills cannot reliably prove a specific dolly, pan, etc.
- `primary_subject_position`: `left | center | right | multiple | not_applicable | unknown`.
- `over_the_shoulder`: `true | false | null` (`null` means visually uncertain).
- `evidence_roles`: non-empty unique subset of `early/middle/late`, showing the frames actually reviewed for this shot-level interpretation.
- `visual_focus`: `person | object | environment | mixed | unknown`.

Do not conflate camera angle, subject orientation and scene narration. Shot summary `mixed` is reserved for **observed change**, not uncertainty. Evidence must reference actual `early/middle/late` images.

## Per-shot `subjects[]` (GPT descriptions of visible human instances)

- `subject_id`: shot-local `subject_001`, unique per shot, not a global actor or face track ID.
- `sample_detection_refs`: array of `{role, person_detection_id|null, face_detection_id|null}` from Stage03; may be empty for a visually evident CV miss; all non-null references must exist.
- `subject_facing`: `frontal | three_quarter | profile | back | mixed | unknown`.
- `face_visibility`: `fully_visible | partially_visible | not_visible | mixed | unknown`.
- `occlusion`: `none | partial | heavy | unknown`.
- `appearance_mode`: `direct | reflection | screen_or_photo | unknown`.
- `identity_candidate`: `{character_name: string|null, actor_imdb_id: string|null, status: "candidate"|"unknown", basis: string|null}`. It may use Stage02 character context and confirmed cast listings as candidate evidence, but must NOT claim identity verification, should not invent IMDb IDs, and must never infer visibility solely from a character mentioned in text.
- `evidence_roles`: non-empty subset of `early/middle/late` when the person is visually observed.

## Per-shot `objects[]` (visually salient props/objects only)

- `object_id`: shot-local `object_001`.
- `label`: concise English noun phrase (e.g., `glass`, `letter`, `table`). **No invented precise BBox**; Stage03 deliberately runs no object detector.
- `evidence_roles`: frames on which this object is visibly present.

## Per-shot `visible_relations[]`

- `subject_id`: reference to the same shot's `subjects`.
- `object_id`: reference to the same shot's `objects`.
- `relation_type`: `holding | touching | wearing | sitting_on | leaning_on | near | uncertain`.
- `evidence_roles`: at least one visible supporting frame. Record only visually defensible observations; do not infer intention, dialogue, inferred gaze, off-screen exchanges or continuous action from three stills.

## Per-shot `scene_id` and review status

- `scene_id`: a valid ID from `scenes[]` in the **full-film published result**. Partial GPT batch work may leave it unassigned until global segmentation.
- `review_status`: `unreviewed | model_reviewed | human_verified | human_corrected`; GPT checking its own output is only `model_reviewed`, not human verification.

## Scene representation (`scenes[]`)

- `scene_id`: `scene_000001`, increasing in chronological order.
- `start_shot_id`, `end_shot_id`: **both inclusive**, contiguous range of Stage01 shot IDs; every Shot belongs to exactly one film scene.
- `location`: string or null.
- `summary`: concise summary grounded in Stage02 and relevant Stage03/04 evidence.
- `boundary_evidence`: array of `{left_shot_id, right_shot_id, reason, basis}`; `basis` is `visual | text | combined`. Adjacent Shot IDs must straddle the candidate boundary. The validator checks adjacency. A visual difference alone cannot prove scene change.
- `review_status`: same statuses as per-shot review.

Narrative beats/objectives/tactics may be added in a **future Stage04 revision** if necessary for research; Stage04 v1.0 must not invent unsupported intent or emotion. Stage05 handles true actor-face linkage and time-series performance measurement; Stage06 explains evidence-linked acting choices.

## Strict validation and batch-workflow requirements

One `Stage04_<movie_id>.json` per whole film with all shots, all source hashes and scene coverage, sorted non-overlapping scenes, no dangling subject/object/detection references, no extra tags outside enums, and explicit reviewed/unknown states. Optional pilot work must have explicit shot range, version, input hashes, and batch ID; merge must reject duplicate or conflicting shots and report missing coverage. Every asserted visual claim has at least one image-role reference; observations that require video (specific camera movements, motion continuity, gaze targeting) must remain unknown or deferred.
