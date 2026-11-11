# OscarDP Stage04.2 — Scene Character Objectives Prompt v1.0

## Task
Analyse scene-level dramatic objectives and tactics for the selected film scene or its selected sub-sequence, using actual screenplay and Stage02/04 evidence. This is narrative intention analysis, NOT affect inference or observed actor performance.

## Required inputs
- Complete Stage02 movie JSON, especially dialogue in and around the selected scene.
- Complete Stage04 movie JSON: the authoritative scene ID, scene start/end, shot IDs, and character associations.
- Stage04.1 Film-level Character Profiles.
- Original screenplay PDF (recommended, and provided in the Bugonia pilot), with its draft/date.
- Exact user-selected scene ID and optional movie-global selection start/end.
- Oscar award/nominee mapping or a supplied target_actor/context_character list.
Pilot: movie tt12300742, scene_000021, full film time 1322.237582–1766.306207, selected 1432.013915–1547.212332, Michelle/Teddy/Don.

## Definitions (following acting/script analysis, adapted from Uta Hagen)
- scene_objective: what this particular character wants to obtain or change in the FULL Stage04 scene; formulate as a playable action, not an emotion.
- obstacle: what prevents the character from achieving that objective.
- tactic: a concrete narrative or dialogue action attempted to pursue the objective (challenge, persuade, reassure, assert authority, threaten, negotiate, etc.), not automatically proof of corresponding visual performance.
- selected_sequence_objective: the character's aim IN THE SELECTED EXCERPT; set separately only to clarify a specific local objective. The selected excerpt may be shorter than the Stage04 scene.
- The character can retain an objective while switching tactics. Do not create a new scene on each edit or dialogue-topic shift.

## Crucial evidence boundaries
1. Use Stage04 existing scene IDs; NEVER resegment the scene or pretend the Stage06 selected excerpt is a new Stage04 scene.
2. Read the full scene AND relevant preceding narrative before analysing the selected excerpt. Align film words/timestamps via Stage02 shots. Do not transfer screenplay timestamps directly to film.
3. A screenplay scene heading is not automatically identical to the Stage04 edited-film scene. For the pilot, the relevant screenplay basement confrontation is marked scene 44; Stage04 calls it scene_000021.
4. Use Stage04.1's long-term desire as context, but DO NOT repeat long-term goals as the current scene objective. Crucially, exclude later-revealed plot knowledge from contemporaneous interpretation of an earlier scene. Distinguish the character's own belief from verified story truth.
5. Output one independently supported objective and obstacle per participating character. If an objective is not demonstrated (for example, a silent context character), say unknown or limited; do not fabricate a hidden intent or speech.
6. Each tactic must point to actual screenplay pages and/or Stage02 movie shot IDs. Where a tactic has a film time interval, derive it from existing Stage02/04 timestamps, not PDF page offsets or guessed timing. Timing is approximate if the evidence is shot-level rather than word-aligned.
7. Do not label emotion, mental state, gaze target/direction, AU, head pose or effect caused by an observed gesture. Those belong to Stage05/Gaze Enrichment/Stage06.
8. The supplied nominee CSV determines target_actor, not film protagonist/supporting-character importance. A context character can have meaningful narrative objectives without becoming the dataset's award-eligible target.
9. A 2024 shooting-script draft and the final film can differ. Do not silently replace film dialogue or invent cross-source alignment. Label unsupported differences.
10. No unsupported causal assertions or medical diagnoses. Every output remains status="model_proposal".

## Output JSON
Write one new Stage04_2_<movie_id>.json without overwriting Stage04 or Stage04.1.
Root: movie_id, stage="04.2", stage_name, stage_version, source_materials, time_reference, analysis_policy, scene_objectives[], limitations[].
Each entry in scene_objectives[]:
- scene_id, scene_title, full_scene_range (start/end/first_shot_id/last_shot_id).
- selected_sequence (start/end/source/first_overlapping_shot/last_overlapping_shot/selection_is_whole_scene), or null when no excerpt specified.
- narrative_participants, characters[].
Each character:
- character_name, analysis_role, scene_objective, obstacle, tactics[], selected_sequence_objective (nullable), selected_sequence_role, uncertainty.
- objective/obstacle/tactic records: statement_en, statement_zh, basis, confidence, evidence_refs, status="model_proposal".
- Each optional movie_time_range is movie-global seconds and must remain inside the selected Stage04 scene.
- Do not require a separate tactic in every shot; be explicit when the target is an attentive listener, an offscreen participant, or has insufficient evidence.

## Mandatory pilot-specific checks
- Michelle: the selected 115s runs from reclaiming the floor to presenting legal and corporate consequences and offering a mutual deal. The first Teddy reaction after the full speech is OUTSIDE the selected interval.
- Teddy: visible in the scene and addressed by Michelle, but do not invent acceptance of her bargain or independent speaking tactics during a period mostly spent listening.
- Don: visible in some sampled shots and follows Teddy's directions, but has little independent dialogue in the selected 115s. Do not invent strong motivations or a low-intelligence diagnosis.
- Compare screenplay pages 24–30 with Stage02 shots 202–253. Movie times come from Stage02/04, which cover the full scene; selected excerpt overlaps shots 211–223.
- The final-act Michelle revelation in Stage04.1 does NOT authorise post-hoc motive assertions in this early interrogation sequence.
- Confirm target mapping: Michelle / Emma Stone is the sole matching Oscar acting nominee in the supplied CSV; Teddy and Don are context.

## Final check
Validate JSON syntax, exact IDs, nonempty evidence per claim, selected time within full scene, no out-of-range tactics, bilingual descriptions and honest uncertainty. Keep Stage01–06 pre-existing outputs untouched. No unnecessary hash/release bureaucracy or extra project framework.