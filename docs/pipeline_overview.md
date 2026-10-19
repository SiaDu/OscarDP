# Oscar Acting Dataset pipeline

## Workspace and releases

The Canonical Workspace is `/media/sdu/SiyaoDu/datasets/oscar_movie_analysis`.
Each movie folder holds current Stage01/Stage02 JSON, its current SRT, shot images
and optional `stage02_curated/` history. Historical subtitles are zipped to avoid
participating in source discovery. Working revisions preserve old bytes and
provenance before promotion; they do not alter completed Frozen Releases.

Frozen snapshots live under `/media/sdu/SiyaoDu/datasets/oscardp_releases`.
Stage03 interactive GPT annotation needs only Stage01/Stage02 JSONs and accessible sampled JPGs. The user need not supply a Frozen Manifest, hashes or frame-sample metadata. Formal release packaging and the future Stage04 publication contract may independently verify the selected release ID and source hashes. The shared movie-folder layout does not authorize
downstream processing or mutable-workspace fallback. See the [migration and
future-film workflow](dataset_reorganisation.md). No new stage implementation or
change to fixed Stage02 Schema v1.0 / Prompt v1.2 is introduced here.

OscarDP links movies, narrative context, character intent, observable acting, facial and gaze behaviour, and explainable acting annotations for controllable 3D character animation research.

| Stage | Status | Output and purpose |
| --- | --- | --- |
| 01 Shot Detection | Implemented | TransNetV2 boundaries, shot times and Early/Middle/Late images |
| 02 Subtitle & Screenplay Alignment | Source Preflight v1.0 and Validator v1.1.1 implemented; manual annotations available (18 movies) | GPT-6 High annotation generation remains manual. [Preflight](stage02_preflight.md) checks Stage01/SRT/PDF before annotation. Validation separately checks source integrity, annotation consistency and completeness afterwards. [Spec](stage02_spec_v1.md) · [Original Prompt v1.0](prompts/stage02_prompt_v1.txt) · [Prompt v1.1 (archived)](prompts/stage02_prompt_v1_1.txt) · [New-film Prompt v1.2](prompts/stage02_prompt_v1_2.txt) · [Validator](stage02_validation.md) |
| 03 GPT Visual Annotation | Contract drafted; no executable pipeline | Manually inspect Early/Middle/Late frames via ChatGPT; shot-scale, camera evidence, visible people, props and visible interactions; no person/face detector or verified identity. [Spec](stage03_gpt_visual_spec_v1.md) · [Schema](schemas/stage03_gpt_visual_v1.schema.json) · [Prompt](prompts/stage03_gpt_visual_prompt_v1.md) |
| 04 Scene & Character Context | Contract drafted; no executable pipeline | Combine Stage02 text and Stage03 visuals into contiguous film scenes and unverified character candidates. [Spec](stage04_scene_context_spec_v1.md) · [Schema](schemas/stage04_scene_context_v1.schema.json) |
| 05 Actor Identity & Performance Analysis | Planned | Verified actor/character identity, continuous face detection/tracking, Action Units, head pose, gaze and performance events on selected video segments |
| 06 Explainable Acting Annotation | Planned | Evidence-linked acting and intent annotations |

Stage01 implements shot processing. Stage02 v1.0 annotations for 18 movies were generated externally through interactive ChatGPT GPT-6 (High thinking effort), following the linked fixed specification and prompt. Annotation Generation remains manual; the implemented Deterministic Validation CLI is `python -m oscardp.stage02 validate`. It performs Structural and Source Consistency Validation, with source SHA-256 hashes and detailed reports outside the dataset, and does not evaluate screenplay semantics or actual movie visuals. It never modifies earlier outputs or original subtitles. These JSON data files and generated reports are not committed. Stages03–06 remain planned. Future stages should produce separate `StageNN_<movie_id>.json` files in the same IMDb-ID movie directory, reference stable upstream IDs, and record SHA-256 provenance without modifying earlier stages.


**New-film Stage02 gate:** run `uv run python -m oscardp.stage02 preflight --stage01 /path/to/Stage01_tt1234567.json --srt /path/to/tt1234567.srt --screenplay /path/to/tt1234567.pdf --report-root reports/stage02_preflight/tt1234567`. Retain its reviewed `source_preflight.json`; proceed only when `llm_annotation_allowed=true`. The order is Stage01 + SRT + Screenplay → deterministic Source Preflight → PASS / source-bound approved WARNING → manual GPT-6 High semantic annotation → Validator v1.1.1 → separately approved Curated Dataset Release. The preflight is independent of the post-annotation validator and workbook-based inventory audit. Neither performs LLM annotation or activates downstream input. The fixed Stage02 Schema remains v1.0; the spec and v1.2 prompt are unchanged.

Stage01 now uses [three-frame sampling](stage01_spec.md#three-representative-frames-sampling-policy-v1)
with independent internal sampling provenance. Existing frozen-selected movies use
the [image-only frames-upgrade command](stage01_frames_upgrade.md); their JSON,
shot IDs and boundaries remain unchanged. Future Stage03 readiness requires the
explicit `three_sample` validation profile. This Stage01 enhancement does not implement Stage03. The separate Stage03/04 design contracts are documentation only, not executable annotation pipelines. Frozen releases remain immutable.

## Stage03 throughput and shot boundary quality

Stage03 batch size is adaptive; 20 Shots was a pilot choice, **not** a required stop condition. The GPT prompt requests 50–100 Shot work chunks when practical and continuous in-request processing without repeated user “继续” messages. Each completed chunk must be checkpointed as a downloadable batch JSON; existing saved batches must be accessible to resume a later chat. Never substitute filenames/Drive metadata for actually inspected JPGs, or claim automatic background processing. Full-film coverage requires inspecting every available target Shot and validating all results.

Visually discordant Early/Middle/Late samples may identify missed dissolve/fade cuts. Follow [Shot Boundary QC v1.0](shot_boundary_qc_v1.md) and its separate [JSON Schema](schemas/shot_boundary_qc_v1.schema.json). Existing Stage01/02 IDs are frozen; do not alter valid Stage03 batches or quietly revise boundary frames. Stage05 candidate extraction uses the QC sidecar as a review signal, not a blanket exclusion criterion.
