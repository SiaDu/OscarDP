# OscarDP Stage04.2 — Film-wide Scene Character Objectives Prompt v1.4

## Purpose
Analyse **every existing Stage04 scene** for scene-level character objectives, obstacles and supported dramatic tactics, using Stage04.1 for background. This analyses *story action*, not measured acting, AU, head movement, gaze direction/target or affect.

Adapt Uta Hagen's Objective / Obstacle / Action distinctions to the OscarDP scene structure. This is the project's own operational schema.

## Per-film inputs (supplied at runtime)
- **Required:** `movie_id`, full Stage02 JSON, full Stage04 JSON, and Stage04.1 profile JSON.
- **Recommended:** original screenplay PDF (with draft/date), film title/release year, Oscar nominee/target-actor metadata.
- **Optional:** selected sequences with their **original Stage04 `scene_id`** and movie-global start/end times.
- Derive scene count, scene IDs, participants and times from the supplied Stage04. Never assume a film has a particular number of scenes or a selected pilot clip. If required sources are missing, report the limitation instead of fabricating data.

## Use external commentary selectively
Reuse Stage04.1 `external_research[]` where relevant. Search again only for material conflicts in motivations, adaptations, draft-versus-film differences or interpretation. Prioritise creator interviews and substantive professional criticism; preserve attributed competing readings and deduplicate sources.

Secondary commentary may **inform a hypothesis**, but cannot override primary dialogue or script evidence, assert an early character's hidden knowledge, or establish an objective as ground truth. Preserve `source_id`, title, author/publication/date (if available), URL, type and short contribution for newly used sources. If none are useful, leave `external_research=[]`.

## Full-film workflow
1. Read **all** original Stage04 `scenes[]` in order and output **exactly one** `scene_objectives[]` entry per scene, including flashbacks, montages, intercuts, transitions and credits. Do not change Stage04 scene boundaries.
2. Copy `scene_id`, available scene title/type, movie-global time range, first/last shot ID and `narrative_participants` from the actual inputs. If optional metadata is absent, use null; do not invent it.
3. Create a `characters[]` record for each listed narrative participant (if any), distinguishing an active agent from a voice-over, remembered, offscreen, inactive, mentioned or collective participant. Presence in Stage04 does not prove visibility, speech or an independent goal.
4. For each supported active role, infer a **playable immediate** `scene_objective` (what this person wants to obtain/change/prevent *in this scene*), `obstacle` (what prevents it), and up to three salient evidence-based `tactics` (specific verbal or narrative actions). Do not simply restate a scene summary, movement or emotional state.
5. Use Stage04.1 long-term desires as contextual background **without copying them into every scene**. Goals and tactics can change within a scene; a tactic change does not create a new Stage04 scene.
6. For insufficient evidence, output `scene_objective=null`, `obstacle=null`, `tactics=[]`, and `coverage_status="insufficient_evidence"`. For non-dramatic content or a non-agent collective use `not_applicable`. Do not force an intent for a silent or passive person.
7. Support each non-null claim with real **within-scene** Stage02 shot/dialogue references and/or screenplay PDF pages actually inspected. Screenplay pages are not film timestamps. Mark uncertainty where the draft and released-film evidence differ.
8. **Protect narrative chronology**: late plot revelations, creator interviews or ending interpretations cannot prove what an early scene character/viewer knew. Distinguish story fact, character claim, possible tactic and outside reading.
9. **Do not conflate** similarly named or connected institutions, missions, experiments or goals. Establish their aims, responsible parties, outcomes and relationships separately; say when the connection is uncertain.
9a. **Violence and agency:** when one character initiates harm against another, identify the aggressor and the victim accurately. Do not euphemise active assault, coercion or killing as merely "stopping" or "controlling" the victim unless there is separate primary evidence of something the victim was actively doing that the agent sought to interrupt. Separate (a) the inferred purpose (e.g. protect a secret), (b) the obstacle/risk, (c) concrete enacted tactics or harm and (d) emotional or symbolic aftermath. A perpetrator's rationale is an **inference**, not an established fact unless directly stated. The victim's suffering or survival reflex is not automatically an independently pursued dramatic objective. Screenplay-specific physical actions remain script evidence until the finished film visuals are checked.
10. Use `target_actor` only for an actor/character linked to the film's supplied Oscar eligibility data; other participants are `context_character` or `unknown`. Story protagonist status is not an automatic eligibility label. Keep interpretations `status="model_proposal"`.
11. Attach `selected_sequence`, `selected_sequence_objective` and `selected_sequence_role` **only** where a selected excerpt was explicitly provided. The optional sub-sequence must lie within its Stage04 scene. No automatically invented candidate sequences.

## Output
Write one separate `Stage04_2_<movie_id>.json`, leaving existing Stage02/04/04.1 files unchanged.

Root: `movie_id`, `stage="04.2"`, `stage_name`, `stage_version`, `source_materials`, `coverage`, `time_reference`, `analysis_policy`, `scene_objectives[]`, `external_research[]`, `limitations[]`.

Each scene:
- `scene_id`, `scene_title` (if available), `scene_type` (if available), `full_scene_range`;
- `selected_sequence` (object or null), `narrative_participants`, `coverage_status`, `characters[]`.

Each character:
- `character_name`, `analysis_role`, `coverage_status`;
- `scene_objective` (claim or null), `obstacle` (claim or null), `tactics[]`;
- `selected_sequence_objective` (claim or null), `selected_sequence_role` (or null), `uncertainty`.

Each supported claim: `statement_en`, `statement_zh`, `basis`, `evidence_refs[]`, `confidence`, `status="model_proposal"`; optionally `external_source_refs[]` for attributed readings and `movie_time_range` **only when actually evidenced**.

## Final checks
- Output scene IDs/count/order and time boundaries must match **the input Stage04**, regardless of film.
- Character records must correspond to original scene participants; all non-null claims must link to real evidence. Unknown objectives must remain null.
- Confirm screenplay page numbering, Stage02 shot references, selected sub-sequence bounds and bilingual accuracy.
- Summarise supported versus unresolved character–scene entries; do not confuse full structural coverage with verified objectives.
- No invented sources, human review, timing, gaze, emotion or diagnosis. No upstream edits or unnecessary hash/release framework.
