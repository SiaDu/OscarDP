# OscarDP development rules

OscarDP is a modular movie analysis pipeline. Stage01 processing and Stage02 preflight/validation are implemented; Stage02 annotations for 18 films were made through manual GPT. Stages03/04 have design contracts only; Stages05/06 remain planned. Do not implement new stages without an explicit request. Each stage owns an independent, versioned JSON output. Later stages reference stable IDs and never edit completed earlier-stage outputs. Keep source references and hashes traceable.

Inspect existing code before changing it. Reuse working components where appropriate, keep modules small, avoid unnecessary abstractions, and keep CLI paths portable. Update relevant tests when interfaces change. Preserve zero-based frame indices, inclusive starts, exclusive ends, and accurate CFR/VFR timestamps.

Never modify source movies, subtitles, or screenplays. Never delete external dataset directories without explicit authorization. Never run the entire dataset without permission. Never silently overwrite completed outputs. Keep technical provenance and QC outside official per-movie output directories.

Dataset preflight audits and plans are read-only for source data. Apply movie or SRT renames only from an explicitly approved `SAFE` plan with confirmation; retain the journal for verification and rollback. Never rename directories or screenplays as part of preflight.

`data/metadata/movie_titles.json` is the only Stage 01 batch movie-selection file. Batch processing requires `--titles`; its IMDb keys select movies and its values supply titles. Do not add a separate allowlist or exclusion file.

Stage 01 uses movie-first directories named by IMDb ID, stable `shot_000001` IDs, `Stage01_<movie_id>.json`, and Early/Middle/Late samples under `shot_img/` (short-shot aliases; historical single-midpoint outputs remain supported). See `docs/stage01_spec.md` for the contract and `docs/pipeline_overview.md` for the roadmap.


Stage03 is **GPT Visual Annotation from Stage01 sampled images**, not a required local YOLO, SCRFD, Face Recognition or object-detection stage. Never invent BBoxes, face embeddings, verified actors, action units or continuous gaze from three still images. Stage04 builds scene and unverified character context using Stage02+03. Stage05 owns video-based face detection, actor identity and continuous performance measurements. Follow the current Stage03/04 specs and never resurrect deleted CV contracts or model dependencies without explicit approval.

The standalone new-chat Stage03 entry prompt is `docs/prompts/stage03_gpt_visual_prompt_v1.md`. Stage03 requires only user-provided Stage01 JSON, Stage02 JSON and viewable Early/Middle/Late images. Never require manual frozen manifest, precomputed hashes or frame_samples_v1.json from the user for annotation. A GPT output is not a formally verified dataset release; independent local provenance checks may be performed later.
