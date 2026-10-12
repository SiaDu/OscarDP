# Oscar Acting Dataset pipeline

OscarDP links movies, narrative context, character intent, observable acting, facial and gaze behaviour, and explainable acting annotations for controllable 3D character animation research.

| Stage | Status | Output and purpose |
| --- | --- | --- |
| 01 Shot Detection | Implemented | TransNetV2 boundaries, shot times and midpoint images |
| 02 Subtitle & Screenplay Alignment | Planned | Dialogue, speaker and screenplay context linked to shots |
| 03 Cinematic Semantic Enrichment | Planned | Shot size, angle, composition, movement, lighting and visible characters |
| 04 Scene & Narrative Sequence Construction | Planned | Film scenes, sequences, goals and conflicts |
| 05 Actor Performance Analysis | Planned | Target actor, face tracking, Action Units, pose, gaze and performance events |
| 06 Explainable Acting Annotation | Planned | Evidence-linked acting and intent annotations |

Each stage writes `StageNN_<movie_id>.json` in the same IMDb-ID movie directory. Downstream files reference earlier IDs and record a SHA-256 of the upstream JSON in their own provenance metadata. They must not modify earlier stages. These contracts describe future work; only Stage 01 is executable now.
