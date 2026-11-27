# OscarDP agent rules

OscarDP builds an evidence-linked film-acting dataset: shots → dialogue/script → visual and
narrative context → face/gaze measurements → gaze targets → acting interpretation.
Roadmap: `docs/pipeline_overview.md`. Stage specs and prompts: `docs/`.

## Stages

| Stage | Module | Status |
| --- | --- | --- |
| 01 Shots | `oscardp.shots` | Implemented; 18 films. Contract: `docs/stage01_spec.md` |
| 02 Subtitles & screenplay | `oscardp.stage02` | Preflight and validator implemented; 18 films annotated with GPT |
| 03 GPT visual annotation | prompt only | Bugonia (`tt12300742`) only |
| 04, 04.1–04.3 Scene, character, objectives, sequence selection | prompts only | Bugonia only; assistant-authored prototypes |
| 05 Face tracking, identity, OpenFace, events | `oscardp.stage05`, `oscardp.stage05_review` | Pilot on selected Bugonia shots |
| 06.1 Scene layout · 06.2 In-shot head/eye · 06.3 Cross-shot gaze targets | `oscardp.stage06_1/2/3` | Pilot on Bugonia `seq_006` only; code still has seq_006-specific branches |
| 07 Affect & acting interpretation | `oscardp.stage07` | Legacy prototype (formerly named Stage06) |

Do not start a new stage without an explicit request.

## Data safety

- Never modify source movies, subtitles or screenplays.
- Never delete external dataset directories without explicit permission.
- Never run a stage on the whole dataset without permission.
- Never overwrite a completed output; each run writes to a new directory. Frozen releases are immutable.
- Each stage writes its own versioned JSON. Later stages reference stable upstream IDs and never edit earlier outputs.
- Keep provenance (source paths, SHA-256) and QC in `reports/`, never in the per-movie dataset folders.
- The GitHub repo is public. Never commit `reports/`, model weights, movie frames or clips, subtitle text or screenplay text.

## Conventions

- Frame indices are zero-based; ranges are start-inclusive, end-exclusive. Keep CFR/VFR timestamps exact.
- Movie folders are named by IMDb ID; shot IDs are `shot_000001`; outputs are `StageNN_<movie_id>.json`.
- Stage01 samples Early/Middle/Late frames into `shot_img/` (short shots use aliases; old single-midpoint outputs stay supported).
- Stage01 batch selection uses only `data/metadata/movie_titles.json` via `--titles` (keys select movies, values give titles). Do not add allowlists or exclusion files.

## Stage rules

**Stage02 preflight** is read-only for source data. Rename movies or SRTs only from an approved `SAFE` plan after user confirmation, and keep the journal for rollback. Never rename directories or screenplays.

**Stage03** is GPT annotation of Stage01 Early/Middle/Late images, with Stage01/02 JSON as the only other input (prompt: `docs/prompts/stage03_gpt_visual_prompt_v1.md`).
- Do not ask the user for manifests, hashes or `frame_samples_v1.json`.
- Never invent bounding boxes, face embeddings, verified identities, AUs or continuous gaze from still images.
- Work in 50–100-shot checkpoint chunks and keep going without waiting for "继续". A prompt is not a background worker, so never promise a full film in one response. Count a batch as done only when its JSON is accessible.
- Suspected missed cuts go in a separate `Stage03_<movie_id>_shot_boundary_qc.json` (`docs/shot_boundary_qc_v1.md`). Do not change Stage01/02, the Stage03 schema or frozen releases for it. `user_confirmed` requires the user's own video check.

**Stage04** builds scene and unverified character context from Stage02 and Stage03.

**Stage05** owns video-based face detection, identity and continuous measurements. Do not move these into Stage03, and do not restore deleted CV contracts or model dependencies without approval.

**Stage06 vs Stage07 names.** `reports/stage06/`, `models/stage06/`, `stage06_event_preview.json` and `stage06_input_pilot.json` belong to the old acting prototype (now Stage07). Keep their names and schemas and never write into them. New gaze work uses `stage06_1/2/3` and `reports/stage06_1|2|3/`; new acting runs use `reports/stage07/`.

**Assistant-authored evidence** (visual observations, layouts, target candidates, any GPT output) is a hypothesis, not a label or a verified release. Leave human-review fields empty for the user to fill.

## Code

- Read existing code first. Reuse working components, keep modules small, avoid needless abstraction, keep CLI paths portable.
- Update tests when interfaces change. Run them with `uv --no-cache run --no-sync python -m pytest`.

## Docs

- Describe the current method in short, plain sentences. State each limitation once.
- Keep version history as a short changelog at the end; do not append a full section per version.
- Prefer commands and tables to prose.
