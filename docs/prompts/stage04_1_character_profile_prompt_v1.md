# OscarDP Stage04.1 — Film-level Character Profile Prompt v1.0

## Task
Create an evidence-linked film-level narrative character profile. This is CHARACTERISATION, not observed acting, facial motion or an emotion timeline.

## Required inputs
- Complete Stage02 movie JSON: dialogue, narrative, action and optional performance cues.
- Complete Stage04 movie JSON: authoritative movie scenes, scene IDs, shot-to-scene mapping and unverified character links.
- Original screenplay PDF when available. Record screenplay writer, draft, date and any discrepancies with the completed film.
- Oscar acting nominees CSV, movie IMDb ID and intended character names.
- Pilot: Bugonia tt12300742, Michelle, Teddy, Don.

## Research framing
Adapt Uta Hagen's given circumstances, relationships, objectives and obstacles and John Truby's desire/need and conflict concepts into these FIVE OscarDP fields. This is a project-specific operationalisation, not a verbatim standard taxonomy:
1. identity_context — identity and meaningful circumstances.
2. beliefs_disposition — core beliefs and recurring tendencies with cross-scene evidence.
3. dramatic_desire — long-term desire or need; allow changing goals and narrative spans.
4. relationships — important interpersonal ties, dependence, opposition or power.
5. dramatic_conflict — substantial contradictions and obstacles in the character's arc.

Return only 1–3 meaningful claims per field. Unknown information must remain unknown; do not invent adjective inventories.

## Evidence and narrative-time rules
- Ground every substantive claim in exact source locations: the 1-based PDF page number of the screenplay, the existing Stage04 scene ID and, when useful, Stage02 shot IDs. The PDF page number is NOT the page number printed in the screenplay body.
- Mark how each claim is supported: screenplay_explicit, narrative_inference, insufficient_source_information, or evidence_boundary. Merely seeing the Stage02 label screenplay_explicit is not independent PDF verification.
- A dated shooting-script draft is not a guaranteed transcript of the released film. Keep film shot timing and Stage04 scene boundaries authoritative, and disclose unsupported screenplay/film alignments.
- Film-level analysis may include the ending and a later identity revelation. Record first_explicitly_revealed_in_stage04_scene on late-revealed claims. Do not leak those claims as audience-known information into earlier scene objectives.
- Distinguish what a character sincerely believes, what the character claims to others, and what is established by the narrative. No clinical diagnoses or IQ judgments.
- Exclude AU, Head, Gaze, facial expressivity, recurring actor gestures (such as hair-touching), emotion labels and actor performance techniques. These require the later video analysis.
- Award selection: read the user CSV, handle pipe-delimited FilmId fields, and match the exact film ID with its nominee ID and character, rather than guessing from protagonist status. Pilot mapping in the provided CSV: Emma Stone, Michelle, Actress in a Leading Role, nominee (not winner). Teddy and Don are context_character, not inferior acting samples.
- All candidate character identities and interpretive claims remain provisional.

## Output JSON
Write one new file Stage04_1_<movie_id>.json; do not overwrite Stage04.
Root: movie_id, stage="04.1", stage_name, stage_version, source_materials, eligibility_policy, claims_policy, profiles[], limitations[].
Each profile: character_name, profile_scope, analysis_role (target_actor/context_character), actor_nominee (object or null), and the FIVE fields above.
Each field contains claim objects with:
- statement_en, statement_zh: equivalent brief English and Chinese claims.
- basis, confidence (high/medium/low), status="model_proposal".
- evidence_refs: list of sourced references with source="screenplay_pdf" and pdf_pages, source="stage04" and scene_ids, or source="stage02" and shot_ids.
- dramatic_desire may have narrative_scope (start_scene_id, end_scene_id, note).
- late-revealed claims may have first_explicitly_revealed_in_stage04_scene.

## Check before saving
Verify source IDs actually exist, evidence pages fall inside the PDF, film IMDb IDs and nominee IDs match, source status is honest and claims are bilingual. Report important uncertainty, screenplay vs film discrepancies and limited coverage. No hashes, large validation framework, new model installation or modifications to Stage01–04.

Specific Bugonia warning: the August 12, 2024 Draft Eight / Pink Shooting Script contains a major final-act revelation concerning Michelle. Stage04.1 may record it with a late-reveal marker. Stage04.2 for the earlier basement scene must NOT explain earlier dialogue as though this later reveal were already public knowledge.