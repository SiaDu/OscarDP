# Oscar Acting Dataset pipeline

OscarDP links movies, narrative context, character intent, observable acting, facial and gaze behaviour, and explainable acting annotations for controllable 3D character animation research.

| Stage | Status | Output and purpose |
| --- | --- | --- |
| 01 Shot Detection | Implemented | TransNetV2 boundaries, shot times and midpoint images |
| 02 Subtitle & Screenplay Alignment | Manual annotation available (18 movies); no executable pipeline | GPT-6 High Shot-Centric dialogue, speaker, screenplay narrative context and tentative performance cues. [Spec](stage02_spec_v1.md) · [Prompt](prompts/stage02_prompt_v1.txt) |
| 03 Cinematic Semantic Enrichment | Planned | Shot size, angle, composition, movement, lighting and visible characters |
| 04 Scene & Narrative Sequence Construction | Planned | Film scenes, sequences, goals and conflicts |
| 05 Actor Performance Analysis | Planned | Target actor, face tracking, Action Units, pose, gaze and performance events |
| 06 Explainable Acting Annotation | Planned | Evidence-linked acting and intent annotations |

Stage 01 is the only executable pipeline in this repository. Stage 02 v1.0 annotations for 18 movies were generated externally through interactive ChatGPT GPT-6 (High thinking effort), following the linked fixed specification and prompt. These JSON data files are not in the repository; automated Stage 02 generation and Stage 01/SRT cross-validation have not yet been implemented. Future pipeline stages should produce separate `StageNN_<movie_id>.json` files in the same IMDb-ID movie directory, reference stable upstream IDs, and record SHA-256 provenance without modifying earlier stages.
