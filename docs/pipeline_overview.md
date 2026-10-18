# Oscar Acting Dataset pipeline

OscarDP links movies, narrative context, character intent, observable acting, facial and gaze behaviour, and explainable acting annotations for controllable 3D character animation research.

| Stage | Status | Output and purpose |
| --- | --- | --- |
| 01 Shot Detection | Implemented | TransNetV2 boundaries, shot times and midpoint images |
| 02 Subtitle & Screenplay Alignment | Source Preflight v1.0 and Validator v1.1.1 implemented; manual annotations available (18 movies) | GPT-6 High annotation generation remains manual. [Preflight](stage02_preflight.md) checks Stage01/SRT/PDF before annotation. Validation separately checks source integrity, annotation consistency and completeness afterwards. [Spec](stage02_spec_v1.md) · [Original Prompt v1.0](prompts/stage02_prompt_v1.txt) · [Prompt v1.1 (archived)](prompts/stage02_prompt_v1_1.txt) · [New-film Prompt v1.2](prompts/stage02_prompt_v1_2.txt) · [Validator](stage02_validation.md) |
| 03 Cinematic Semantic Enrichment | Planned | Shot size, angle, composition, movement, lighting and visible characters |
| 04 Scene & Narrative Sequence Construction | Planned | Film scenes, sequences, goals and conflicts |
| 05 Actor Performance Analysis | Planned | Target actor, face tracking, Action Units, pose, gaze and performance events |
| 06 Explainable Acting Annotation | Planned | Evidence-linked acting and intent annotations |

Stage01 implements shot processing. Stage02 v1.0 annotations for 18 movies were generated externally through interactive ChatGPT GPT-6 (High thinking effort), following the linked fixed specification and prompt. Annotation Generation remains manual; the implemented Deterministic Validation CLI is `python -m oscardp.stage02 validate`. It performs Structural and Source Consistency Validation, with source SHA-256 hashes and detailed reports outside the dataset, and does not evaluate screenplay semantics or actual movie visuals. It never modifies earlier outputs or original subtitles. These JSON data files and generated reports are not committed. Stages03–06 remain planned. Future stages should produce separate `StageNN_<movie_id>.json` files in the same IMDb-ID movie directory, reference stable upstream IDs, and record SHA-256 provenance without modifying earlier stages.


**New-film Stage02 gate:** run `uv run python -m oscardp.stage02 preflight --stage01 /path/to/Stage01_tt1234567.json --srt /path/to/tt1234567.srt --screenplay /path/to/tt1234567.pdf --report-root reports/stage02_preflight/tt1234567`. Retain its reviewed `source_preflight.json`; proceed only when `llm_annotation_allowed=true`. The order is Stage01 + SRT + Screenplay → deterministic Source Preflight → PASS / source-bound approved WARNING → manual GPT-6 High semantic annotation → Validator v1.1.1 → separately approved Curated Dataset Release. The preflight is independent of the post-annotation validator and workbook-based inventory audit. Neither performs LLM annotation or activates downstream input. The fixed Stage02 Schema remains v1.0; the spec and v1.2 prompt are unchanged.
