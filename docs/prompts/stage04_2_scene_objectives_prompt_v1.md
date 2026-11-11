# OscarDP Stage04.2 — Film-wide Scene Character Objectives Prompt v1.1

## Task

Generate one evidence-linked Scene Character Objectives JSON for the **entire film** using **every Scene in Stage04**. Stage04 scene boundaries, IDs and narrative participants are authoritative and must remain unchanged. Do not treat a Stage06 pilot clip as the only scene to annotate.

This is a narrative analysis of objectives, obstacles and dramatic tactics. It is **not** facial-performance recognition, emotion detection, AU, head-pose or gaze-target prediction.

## Inputs

Required:
- Complete Stage02 movie JSON: dialogue, narrative and related actions.
- Complete Stage04 movie JSON: `scenes[]`, `shots[]`, scene time boundaries, participants, and shot-to-scene mapping.
- Stage04.1 Film-level Character Profile JSON.
- Exact `movie_id`, and Oscar nomination metadata if target actors must be marked.

Recommended:
- Original screenplay PDF, identifying its draft/date; do not treat a draft as the final-cut transcript.
- A manually selected Sequence (optional), with its Stage04 `scene_id` and **movie-global** start/end seconds. Only enrich the selected part when actually provided; all other scenes still receive full-scene records.

Bugonia pilot inputs: `tt12300742`; Stage04 has **80 scenes**; the original screenplay is Will Tracy's 2024-08-12 Draft Eight shooting script. Existing selected sequence is inside `scene_000021`, 1432.013915–1547.212332 s.

## Definitions

Adapt Uta Hagen's Objective / Obstacle / Action distinction to the OscarDP film timeline.

- `scene_objective`: what a particular character tries to obtain, change or prevent **within that Stage04 scene**; should be playable and specific, not an emotion or the character's lifelong desire.
- `obstacle`: another character, condition or constraint obstructing that goal.
- `tactics`: actions through dialogue or narrative events that advance the goal (e.g., persuade, challenge, reassure, conceal, demand, delay). Describe intended story action; do **not** claim measured acting, facial expression or actual gaze.
- `selected_sequence_objective` (optional): a narrower goal in a supplied subscene excerpt. Do not assume it exists for all scenes.
- Changing tactic is **not** proof of an emotion transition or a new Stage04 scene.

## Full-film workflow — mandatory

1. Load and enumerate **all** Stage04 `scenes[]` in their existing order. Create exactly **one** `scene_objectives[]` entry **per Stage04 scene**, even a title, transition, flashback, intercut, montage or credits section.
2. Copy each entry's original `scene_id`, `title`, `scene_type`, time range, first/last shot ID and `narrative_participants`. Do not resegment or revise original Stage04.
3. For each listed `narrative_participant`, create one `characters[]` record and determine whether their **own actionable goal** is supported in that scene. The participant list may include voice-over narrators, memories, offscreen persons, incapacitated characters, groups or an individual appearing only in an insert. **Participation does not imply visible face, a speaking turn or an independent goal.**
4. For supported participants, provide a concise objective, an obstacle if evidenced, and up to three salient tactics **only if supported**. For insufficient evidence, write `scene_objective: null`, `obstacle: null`, `tactics: []`, and `coverage_status: "insufficient_evidence"`. For titles, credits or anonymous groups with no playable individual goal, use `not_applicable`. Never fill blanks with invented hidden motivations.
5. Read nearby dialogue and scene summaries to infer specific goals, not merely paraphrase the scene title. Prefer directly evidenced action. Distinguish claims based on Stage02 dialogue from interpretive summaries of Stage04. Point to exact source IDs.
6. Original screenplay PDF pages are useful independent text evidence **only where the matching scripted passage has been checked**. The shooting-script scene heading is not the Stage04 scene ID; screenplay pages are not movie timestamps. Preserve draft-versus-final-cut uncertainty.
7. Use Stage04.1 as character background, not a template for copying the film-level long-term goal into every scene. **Protect chronology:** later plot revelations cannot retroactively establish what the audience or a character knew in earlier scenes. Distinguish the speaker's belief from established narrative facts.
8. Explicitly assign `analysis_role="target_actor"` only to an award-eligible actor matched against the supplied **FilmId + nominee/character** data. Other participants are `context_character` even if important or playing a narrative lead. Do not use protagonist/supporting importance to rank acting quality.
9. Include detailed `selected_sequence` and `selected_sequence_objective` only for explicitly provided clips. If one is already annotated, preserve its reviewed/proposed work rather than collapsing full-scene coverage back to one pilot. All other scenes use `selected_sequence: null`.
10. Keep all inferential claims `status="model_proposal"`. Do not invent human review, verified actor IDs, precise gaze targets, actual emotional causes or exact action onset times.

## Output: a single separate JSON

Write `Stage04_2_<movie_id>.json`. Never overwrite Stage04, Stage04.1, Stage02 or earlier processing.

Root:
- `movie_id`, `stage="04.2"`, `stage_name="Film-wide Scene Character Objectives"`, `stage_version`
- `source_materials`, `coverage`, `time_reference`, `analysis_policy`
- `scene_objectives[]` in **the exact order and number of Stage04 scenes**
- `limitations[]`

Each `scene_objectives[]` record:
- `scene_id`, `scene_title`, `scene_type`
- `full_scene_range`: movie-global start/end seconds, first/last Stage04 shot ID
- `selected_sequence`: object (only if explicitly supplied and within the full scene), otherwise null
- `narrative_participants`: copied from Stage04
- `coverage_status`: objective_annotated / insufficient_objective_evidence / non_dramatic
- `characters[]`: **one record per narrative participant**, including unknown/no-objective records

Each character:
- `character_name`, `analysis_role`, `coverage_status` (supported / limited_evidence / insufficient_evidence / not_applicable)
- `scene_objective`: claim object or null
- `obstacle`: claim object or null
- `tactics[]`: zero or more claim objects
- `selected_sequence_objective`: claim object or null
- `selected_sequence_role`: value or null
- `uncertainty`: short grounded statement

Claim object:
- `statement_en`, `statement_zh`: aligned concise bilingual prose
- `basis`: stage02_dialogue_and_stage04_context / stage04_narrative_inference / screenplay_explicit as appropriate
- `evidence_refs[]`: actual Stage04 `scene_ids`, Stage02 `shot_ids` and/or **checked** screenplay `pdf_pages`
- `confidence`: high / medium / low, `status="model_proposal"`
- Optional `movie_time_range` may only be supplied with justified film-time evidence; never calculate it from screenplay pages.

## Final checks

- The count and ordered list of output scenes must **exactly equal Stage04** (Bugonia: 80/80).
- Original `scene_id`, scene boundaries, shot boundaries and `narrative_participants` must match Stage04 exactly.
- All referenced `shot_ids` belong to their indicated Stage04 scene; all PDF page numbers are 1-based and within the actual document; all movie ranges fit the corresponding scene.
- Characters lacking goal evidence remain explicit null; no duplicate character entries; every supported objective has sourced evidence.
- Preserve the existing detailed Bugonia `scene_000021` (full scene 1322.237582–1766.306207 s; selected excerpt 1432.013915–1547.212332 s).
- Report how many scenes and character-scene entries were annotated versus left null; do not present 80 scene records as 80 confidently inferred objectives.
- Character identities and inferred goals are not ground truth.
- No hashes, approval bureaucracy, unnecessary new validation framework or edits to upstream stages.

This prompt is an OscarDP-specific operationalisation of acting/script analysis, not a claim that Hagen defined the above JSON taxonomy.