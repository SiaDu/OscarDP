# OscarDP — Stage04 v1.0: Scene & Character Context

**Status:** Proposed contract, not an executable pipeline. Stage04 builds film scene structure and candidate character links using **Stage02 text + Stage03 GPT visual observations**; it does not redo Stage03 cinematography/prop descriptions. Stage05 owns verified identity and continuous face/performance measurements.

## Inputs / provenance
Explicitly select and verify the Frozen Stage02 Release (manifest hash and selected Stage01/Stage02 hashes), Stage03 full output hash, and Stage01 sample metadata hash. Stage03 itself remains a separate visual annotation, and its JPEGs are not falsely claimed as part of the frozen Stage02 release.

Official output: `oscar_movie_analysis/<movie_id>/Stage04_<movie_id>.json` **only after** all Stage01 shots have valid scene assignments and the entire scene partition has been checked. GPT batches/partial scene plans must stay outside official movie directories. Record immutable input hashes; never edit Stage01/02/03.

## Root fields
- `movie_id`, `stage_version="4.0"`.
- `inputs`: `frozen_release_id`, `frozen_manifest_sha256`, `stage01_sha256`, `stage02_sha256`, `frame_samples_sha256`, `stage03_sha256`.
- `shots[]`: ordered, complete inventory of Stage01 shot IDs, each with `shot_id`, `scene_id`, `character_candidates[]`, `context_summary`, `review_status`.
- `scenes[]`: ordered contiguous, nonoverlapping film scenes covering every Shot exactly once.

## Shot character candidates (not identity verification)
Each `character_candidates[]` entry contains `subject_id` referencing that Shot's Stage03 visible person, `character_name` (or null), `status=candidate|unknown`, `basis=visual_context|text_and_visual|unknown`, and `evidence_roles` referencing sampled images. Stage02 `characters_in_script_context` or a subtitle speaker **alone** does not prove that person is visible. Do not include verified Actor IMDb IDs or synthetic face-matching similarity.

`context_summary` summarizes localized narrative context supported by Stage02 and consistent with Stage03 imagery; it is **not** a measured emotion or true AU/gaze observation. Use an empty string if unsupported.

## Scene segmentation
Each `scenes[]` entry: `scene_id`, inclusive `start_shot_id`, inclusive `end_shot_id`, `location` (string|null), concise `summary`, `boundary_evidence[]`, `review_status`.
Each `boundary_evidence[]`: `left_shot_id`, `right_shot_id`, `basis=visual|text|combined`, `reason`. The two IDs must be adjacent and straddle a scene boundary. The first scene may have an empty boundary_evidence list. Stage02 `narrative.scene` is only a text-derived hint, not the authoritative film scene boundary.

Use continuous shots and textual context, not scene breaks asserted from isolated changes in shot scale/camera position. A future validator must check scene ranges, chronological IDs, full coverage and referential integrity. GPT review alone has `review_status=model_reviewed`; human statuses require actual human action.

Stage05 may later link verified characters/actors and extract continuous performances; Stage06 may infer explainable acting from stage-linked evidence. No Stage04 code is implemented by this contract.

## Shot boundary QC dependency

Read any available [Shot Boundary QC v1](shot_boundary_qc_v1.md) as a **separate, optional quality signal**. A Stage01 Shot flagged as under-segmented may contain multiple true editorial Shots; Stage04 must not infer one continuous camera view or a precise scene cut from those three Stage03 images. Stage04 scene IDs remain keyed to original stable Stage01 shot IDs unless a separately approved repair/migration introduces explicit finer segmentation. Flag ambiguity for review rather than silently modifying Stage01/02 or forcing unsupported scene boundaries.
