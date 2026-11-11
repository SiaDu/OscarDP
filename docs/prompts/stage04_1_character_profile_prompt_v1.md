# OscarDP Stage04.1 — Film-level Character Profile Prompt v1.2

## Purpose
Create concise, evidence-linked **film-level narrative character profiles**. This concerns a character's circumstances, beliefs, goals, relationships and dramatic conflicts—not observed acting, facial motion, gaze, AU or emotion detection.

Use Uta Hagen's given circumstances, relationships, objectives and obstacles, and John Truby's desire/need/conflict, as theoretical inspiration. The five output fields below are **OscarDP's own research schema**, not a verbatim taxonomy from either author.

## Per-film inputs (supplied at runtime)
- **Required:** `movie_id`, complete Stage02 JSON and Stage04 JSON.
- **Recommended:** film title and release year, original screenplay PDF (identify draft and date), and Oscar acting-nominee metadata for target-actor identification.
- **Characters:** explicit `target_characters`, if given. Otherwise select the film's matched Oscar acting-nominated characters and major interacting characters supported by Stage02/04. Do not fabricate an actor–character mapping.
- If a source is missing, state the limitation and leave unsupported claims unresolved; do not pretend to have read it.

Do not hard-code any film, character, award year, source page, scene ID, scene count or plot detail into this prompt.

## External research
When browsing is available, search the film title/year and relevant character/story interpretations **before finalising profiles**. Use a small, diverse set of relevant sources; prioritise:
1. Director, screenwriter or other firsthand creator interviews.
2. Substantive professional film criticism.
3. Other audience interpretations, clearly attributed as opinions.

Read actual sources; record `source_id`, title, author (if known), publication, publication date (if known), URL, `source_type` (`creator_interview`, `professional_criticism`, `other_interpretation`) and a short paraphrased contribution in `external_research[]`. Deduplicate overlapping viewpoints. Do not invent sources or citations.

Creator commentary can illuminate creative intent but does not establish the sole valid reading. Film criticism is **secondary interpretation**, never proof of dialogue, a gesture, objective, gaze target or exact shot timing. Where major readings conflict, attribute each and preserve the disagreement. If browsing is unavailable, set `external_research=[]` and note this limitation.

## Five dimensions (per character)
1. `identity_context`: identity, relevant personal history and social position.
2. `beliefs_disposition`: repeatedly supported beliefs and tendencies, not transient moods or diagnoses.
3. `dramatic_desire`: long-term desires and needs; allow multiple goals and changes across narrative phases.
4. `relationships`: meaningful interpersonal ties, dependence, power and opposition.
5. `dramatic_conflict`: major internal/external contradictions and obstacles.

Prefer a few meaningful, supported claims per dimension, but **do not impose a fixed claim count** when the narrative has distinct phases. Use empty arrays for unsupported dimensions; never fill fields for symmetry.

Each claim includes `statement_en`, `statement_zh`, `basis`, `evidence_refs[]`, `confidence` (`high`/`medium`/`low`) and `status="model_proposal"`. Use `basis` values `screenplay_explicit`, `stage02_dialogue`, `stage04_context` or `narrative_inference` as appropriate. Optional `external_source_refs[]` may inform **interpretation**, never replace primary evidence. For evolving goals, include optional `narrative_scope` with original Stage04 start/end scene IDs; for a revelation, include optional `first_explicitly_revealed_in_stage04_scene`.

## Evidence and interpretation rules
- **Distinguish** source-established events, a character's claims, the character's beliefs, analyst inference and attributed third-party interpretation. Avoid treating a claim within the story as independent fact.
- Link primary claims to existing Stage02 shot/dialogue IDs, Stage04 scene IDs, and/or **actually checked** screenplay PDF pages (1-based PDF page number, not printed script page). Only use `screenplay_explicit` after reading the original script.
- A screenplay draft may differ from the edited film. Use Stage04 IDs and Stage02 movie timestamps for the film; do not infer them from screenplay scene numbers or pages.
- **Protect narrative chronology:** film-level profiles may include a later revelation, but must mark when it becomes established. Never retroactively assume an earlier scene's participants or audience knew it. If a character's private knowledge is uncertain, state that uncertainty.
- **Disambiguate overlapping organisations, projects, experiments and causal chains**. Where two activities are related, record their distinct stated aims, agents, methods, outcomes and evidence before interpreting their relationship. Do not collapse them into one event or assert independence without support.
- No clinical diagnosis, intelligence judgment or stable movement/gaze habit inferred from dialogue or scene summaries.
- Determine `analysis_role` from exact nominee/film/character mapping when available: `target_actor` for award-eligible performance; `context_character` for other relevant characters; `unknown` when unresolved. Story prominence and award category are not acting-quality scores.

## Output
Write **one separate file** `Stage04_1_<movie_id>.json`; never overwrite Stage04.

Root: `movie_id`, `stage="04.1"`, `stage_name`, `stage_version`, `source_materials`, `eligibility_policy`, `claims_policy`, `profiles[]`, `external_research[]`, `limitations[]`.

Each profile: `character_name`, `profile_scope`, `analysis_role`, `actor_nominee` (object or null), and the five dimensions above.

Check the actual source IDs, PDF pages, character/nominee mappings, bilingual consistency, narrative chronology and competing interpretations. Keep all inferred results provisional. Do not modify prior stages or build a separate release/hash bureaucracy.
