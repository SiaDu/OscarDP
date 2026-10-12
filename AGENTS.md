# OscarDP development rules

OscarDP is a modular movie analysis pipeline. Stage 01 shot detection is the only active implementation. Stages 02–06 are planned; do not implement them without an explicit request. Each stage owns an independent, versioned JSON output. Later stages reference stable IDs and never edit completed earlier-stage outputs. Keep source references and hashes traceable.

Inspect existing code before changing it. Reuse working components where appropriate, keep modules small, avoid unnecessary abstractions, and keep CLI paths portable. Update relevant tests when interfaces change. Preserve zero-based frame indices, inclusive starts, exclusive ends, and accurate CFR/VFR timestamps.

Never modify source movies, subtitles, or screenplays. Never delete external dataset directories without explicit authorization. Never run the entire dataset without permission. Never silently overwrite completed outputs. Keep technical provenance and QC outside official per-movie output directories.

Dataset preflight audits and plans are read-only for source data. Apply movie or SRT renames only from an explicitly approved `SAFE` plan with confirmation; retain the journal for verification and rollback. Never rename directories or screenplays as part of preflight.

`data/metadata/movie_titles.json` is the only Stage 01 batch movie-selection file. Batch processing requires `--titles`; its IMDb keys select movies and its values supply titles. Do not add a separate allowlist or exclusion file.

Stage 01 uses movie-first directories named by IMDb ID, stable `shot_000001` IDs, `Stage01_<movie_id>.json`, and one midpoint JPEG per shot under `shot_img/`. See `docs/stage01_spec.md` for the contract and `docs/pipeline_overview.md` for the roadmap.
