# Oscar Acting Dataset pipeline

OscarDP links movies, narrative context, character intent, observable acting, facial and gaze behaviour, and explainable acting annotations for controllable 3D character animation research.

| Stage | Status | Output and purpose |
| --- | --- | --- |
| 01 Shot Detection | Implemented | TransNetV2 boundaries, shot times and midpoint images |
| 02 Subtitle & Screenplay Alignment | Manual annotation available (18 movies); deterministic validator implemented | GPT-6 High Shot-Centric annotation generation remains manual. Read-only structure, Stage01 and SRT cross-validation produces separate reports. [Spec](stage02_spec_v1.md) · [Original Prompt v1.0](prompts/stage02_prompt_v1.txt) · [New-film Prompt v1.1](prompts/stage02_prompt_v1_1.txt) · [Validator](stage02_validation.md) |
| 03 Cinematic Semantic Enrichment | Planned | Shot size, angle, composition, movement, lighting and visible characters |
| 04 Scene & Narrative Sequence Construction | Planned | Film scenes, sequences, goals and conflicts |
| 05 Actor Performance Analysis | Planned | Target actor, face tracking, Action Units, pose, gaze and performance events |
| 06 Explainable Acting Annotation | Planned | Evidence-linked acting and intent annotations |

Stage01 implements shot processing. Stage02 v1.0 annotations for 18 movies were generated externally through interactive ChatGPT GPT-6 (High thinking effort), following the linked fixed specification and prompt. Annotation Generation remains manual; the implemented Deterministic Validation CLI is `python -m oscardp.stage02 validate`. It performs Structural and Source Consistency Validation, with source SHA-256 hashes and detailed reports outside the dataset, and does not evaluate screenplay semantics or actual movie visuals. It never modifies earlier outputs or original subtitles. These JSON data files and generated reports are not committed. Stages03–06 remain planned. Future stages should produce separate `StageNN_<movie_id>.json` files in the same IMDb-ID movie directory, reference stable upstream IDs, and record SHA-256 provenance without modifying earlier stages.
