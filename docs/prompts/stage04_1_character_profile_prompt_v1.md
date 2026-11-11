# OscarDP Stage04.1 — Film-level Character Profile Prompt v1.1

## Goal
Generate a concise, evidence-linked **full-film character profile**, separating screenplay facts, film-story claims, the character's beliefs, research interpretation and third-party commentary. Use Uta Hagen (given circumstances, relationships, objectives/obstacles) and John Truby (desire/need/conflict) as theoretical inspiration, not as a purported standard JSON taxonomy. Do not infer facial acting, AU, head pose, gaze, stable expression habits or clinical diagnoses.

## Required inputs
- Complete Stage02 film JSON (dialogue, narrative, screenplay-based cues).
- Complete Stage04 film JSON (canonical edited-film scene IDs and narrative participants).
- Original screenplay PDF where available, identifying its draft/date.
- Oscar acting nominees CSV / exact IMDb movie ID to determine target_actor versus context_character.
- Explicit characters to annotate. Bugonia pilot: MICHELLE, TEDDY, DON.

## Web research: use when browsing is available
Before final interpretation, search the film title/year together with character analyses and ending explanations. **Prioritise source quality**:
1. Director/screenwriter interviews and other firsthand production commentary.
2. Substantive reviews or analysis in established film publications (e.g. BFI Sight and Sound, RogerEbert.com).
3. Other reviews or audience readings only as explicitly attributed viewpoints, never as narrative fact.
Use a small diverse sample, normally 3–8 useful sources per film, rather than copying consensus or repeatedly searching each claim. Read the source; do not fabricate pages, URLs, quotes or interviews.

For each useful external source save title, author (if known), publication, date (if known), URL, type (creator_interview / professional_criticism / other_interpretation) and a brief paraphrased contribution. Add these to an optional root `external_research[]`, with `source_id`. Every interpretation drawn from them must be **explicitly attributed**, and competing readings may coexist. An interview describes authorial intention, **not proof of the sole valid interpretation**. An external review cannot independently verify a screenplay gesture, exact shot timing, eye target or the character's objective.

If browsing is unavailable, leave `external_research=[]` and identify the limitation. Do not silently invent web citations.

## The five profile fields
1. `identity_context` — identities, history, social position.
2. `beliefs_disposition` — recurring stated beliefs and supported tendencies.
3. `dramatic_desire` — long-term desires/needs, allowing multiple chronological spans.
4. `relationships` — important interpersonal relationships.
5. `dramatic_conflict` — major internal and external contradictions/obstacles.

Keep 1–3 substantial claims per field. Each is a claim object:
- `statement_en`, `statement_zh` (equivalent concise statements);
- `basis` (screenplay_explicit / narrative_inference / insufficient_source_information / evidence_boundary);
- `evidence_refs[]`: actual 1-based screenplay PDF page numbers and/or Stage04 scene IDs / Stage02 shot IDs;
- `confidence` (high/medium/low), `status="model_proposal"`.
- Optional `external_source_refs[]`: IDs of external sources informing an **interpretation**, not replacing primary evidence.
- For evolving desires: optional `narrative_scope` with start/end scene IDs.
- For later revelations: `first_explicitly_revealed_in_stage04_scene`.

## Essential distinctions
- **Source fact vs character claim vs analyst inference vs critical interpretation**: do not collapse into one truth. Record chronology and whose perspective each assertion represents. Film-level profiles can include late revelations, but later knowledge must not leak into earlier-scene interpretation.
- A draft shooting script is not the edited movie. Stage04 scenes and Stage02 film times govern film alignment; screenplay PDF pages are separate text evidence.
- **Never conflate nested or related experiments / institutions / aims.** For Bugonia specifically distinguish:
  (A) Auxolith's **opioid-withdrawal drug clinical trial** involving Sandy, reported complications, corporate response and compensation (shooting-script PDF pp. 53–54, 64);
  (B) the **Andromedan programme to alter humanity's inherited traits** to protect humanity/Earth, Sandy's claimed status as a test subject, and a late report of near-total programme failure (PDF pp. 88–89, 100–101).
  They may be connected in story logic; **do not assert they are completely independent or precisely identical without direct support**. Keep distinct goals, responsible institutions, outcomes and evidential status. A corporate trial's failure is NOT the same claim as failure of the species-level programme.
- The final-act extraterrestrial reveal must not retroactively make early dialogue a verified expression of secret knowledge.
- Oscar award category is eligibility context, not a hierarchy of acting quality. For supplied Bugonia CSV: Emma Stone / Michelle is target_actor; Teddy and Don are context_character.

## Deliverable
One UTF-8 `Stage04_1_<movie_id>.json`, **without changing Stage04**. Root: movie_id, stage="04.1", stage_name, stage_version, source_materials, eligibility_policy, claims_policy, profiles[], limitations[], optional external_research[].
Each profile: character_name, profile_scope, analysis_role, actor_nominee (or null), five fields.

Before saving check actual scene IDs, PDF pages and award matching, bilingual consistency, chronology, distinct experiments and honest uncertainty. Do not add hash/release bureaucracy or modify upstream stages.
