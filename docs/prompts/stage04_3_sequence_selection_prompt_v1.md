# OscarDP Stage04.3 — LLM-assisted Performance Sequence Candidate Selection Prompt v1.0

## Purpose

For **any film in the Oscar acting dataset**, propose a **manageable, diverse and evidence-linked shortlist of performance sequences** featuring award-eligible target performers. Use existing narrative/shot metadata and selective online research. This stage is a **low-cost LLM retrieval and ranking pass**, not a continuous-video analysis or a validated performance dataset.

Never hard-code a particular film, actor, character, scene count, scene ID, plot, timestamp or nominated clip into this prompt. Obtain every such detail from runtime inputs.

## Inputs

Required:
- `movie_id`, complete Stage02 (dialogue/narrative), Stage03 (sampled-shot visual features), Stage04 (canonical scene/shot/time mapping), Stage04.1 (film-level character profile), Stage04.2 (scene objectives).
- Award target mapping, normally the acting-nominee CSV or a verified film/actor/character relationship; report unresolved mappings.

Recommended:
- Film title/year, production screenplay PDF (version/date; textual context only).
- Optional prior Stage05/06 pilot selections and their exact film-global time boundaries; they are **not human-approved** merely because they exist.
- Optional runtime review budget (maximum candidate count, preferred durations, priority first-batch count).

If a required source is absent, report the limitation rather than invent the contents. Stage02/04 film timing—not screenplay page numbers or web clips—is the authority for local sequence boundaries.

## External research — targeted and verifiable

Research the current film, target actor and their **performance**, seeking:
1. Official Academy nomination/award records, **only to verify film-level eligibility**; an acting nomination does **not** designate an official nominated scene.
2. Verified distributor/official film clips and substantive firsthand director, performer, writer, editor or cinematographer interviews discussing particular scenes or performance decisions.
3. Detailed established film criticism highlighting specific confrontations, interactions, reactions or acting choices.
4. Other analyses as lower-confidence discovery hints, explicitly attributed.

Read each useful source; record title, author, publisher, date (if known), stable URL and type. Store a brief **paraphrase** and label `scene_support_scope`: `specific_scene`, `scene_family`, `general_film_style`, `production_context` or `film_level_award_only`. Do not claim a source precisely identifies a scene or provides a movie-global time when it does not. No fabricated interviews, quotations, Oscar clips, URLs or timestamps. Avoid promotional and mentions-count bias; repeated copies of one review do not constitute independent support.

External sources are **recall/context signals**, not proof of outstanding acting, character's inner emotion, specific gaze or objectively measured performance. External text must never replace internal film/script evidence. If browsing is unavailable, proceed via internal discovery and set `external_research=[]` with a limitation.

## Two independent discovery channels

**A — Internal, before popularity cues:** Scan every Stage04 scene with Stage02–04.2. Find potentially valuable target-actor sequences containing:
- Immediate scene goals, changes of tactic, negotiation, power shifts, interpersonal conflict or revealing dialogue.
- Listening, hesitant response, quiet reading/observation, controlled conversation and nonverbal baselines.
- Interaction with other characters, not merely the target's speaking time.

**B — External:** Use the above reputable commentary to recall distinctive scenes. Align its **semantic descriptions** to actual Stage02 dialogue, Stage04 scene summaries and Stage04.2 goals. If no unambiguous alignment exists, record an unresolved external lead rather than fabricating a segment.

Merge and deduplicate. Mark a candidate's `discovery_channel` as `internal_discovery`, `external_highlighted` or `both`. Do not prefer an over-discussed dramatic clip just because it appears in many reviews; retain a diversity of low- and high-intensity situations.

## Coarse time localisation and visual feasibility

1. Enumerate **all** Stage04 scene IDs for a traceable `scene_screening[]` table, even when not shortlisted.
2. Generate contiguous within-scene candidate sequences; a sequence may include multiple adjacent Stage04 shots, but must not cross the scene boundary silently.
3. Normally use **existing Stage04 shot start/end times**. If a prior user-provided pilot has explicit verified movie-global boundaries inside shots, preserve those times and mark their provenance. **Do not infer local timestamps from review text, screenplay pages or a trailer**.
4. Each candidate lists `scene_id`, `start_seconds`, `end_seconds`, `duration_seconds`, `first_shot_id`, `last_shot_id` and all `shot_ids`.
5. Stage03 sees sampled frames, **not the entire shot**. Use its shot scale, face visibility/quality and Stage04's *unverified* target-character candidates only for an initial `visual_suitability` estimate. Distinguish confidence in the shot/scene boundary from identity and continuous-face usability. Do not assert gaze direction, exact eye targets, AU, head-pose values, affect or verified actor identity.
6. Candidate segments should be short enough for manual video review, typically around **20–180 seconds** when the evidence and narrative units permit. Long Stage04 scenes may yield several non-overlapping candidate segments. Do not cut through a meaningful dialogue beat solely to hit a fixed duration.
7. Prefer a **manageable pool** instead of blanket processing the whole film. Mark priority `high`/`medium`/`low` according to expected annotation opportunity and footage feasibility, not an acting-quality rating. Explicitly preserve lower-priority external showpieces whose primary salience is production spectacle rather than observable acting.
8. Create a smaller `review_plan.first_batch_sequence_ids` representing diverse interaction types; leave the rest on hold. **No new Stage05 or gaze processing is authorised by creating candidates.**

## Required output

One independent UTF-8 JSON: `Stage04_3_<movie_id>.json`. Do not change Stage01–04.2 or previous review/artifact outputs.

Root fields:
- `movie_id`, `movie_name` (if known), `stage="04.3"`, `stage_name`, `stage_version`, `generated_on`;
- `target_performances[]` (actor, character, award evidence, exact eligibility scope and identity status);
- `inputs`, `policy`, `external_research[]`, `coverage`, `scene_screening[]`, `sequence_candidates[]`, `review_plan`, `limitations[]`.

Each `scene_screening[]` entry: `scene_id`, whether the target is a Stage04 narrative participant, screening decision and IDs of selected candidates, possibly empty.

Each `sequence_candidates[]` entry:
- `sequence_id`, `scene_id`, target actor/character, concise bilingual or Chinese title;
- exact shot-derived movie-global boundaries, inclusive shot ID list and `boundary_basis`;
- `priority`, `discovery_channel`, `acting_opportunity_tags[]`;
- `selection_reason_zh`, `acting_observation_to_review_zh` (opportunities, **not measured events**);
- `primary_evidence`: Stage04 scene, Stage04.2 character/goal, actual in-range Stage02 dialogue-shot IDs, sampled Stage03/04 character-candidate visual-shot IDs;
- `external_evidence[]`: source IDs and an honest role such as `candidate_recall_or_context_only`;
- `visual_suitability`: estimated sampled-shot coverage and evidence limitations;
- `limitations_zh`, `status="model_candidate"`, `human_review_status="not_reviewed"`;
- optionally `previous_stage06_pilot` for known reused excerpts.

Do not silently upgrade candidate to `human_approved` or imply human review happened. Store source metadata once at the root and refer to source IDs, not duplicate long reviews inside every candidate.

## End-of-run quality checks

- The ordered `scene_screening[]` list exactly matches input Stage04 IDs/count.
- Every candidate's target actor/character is supported by the runtime Oscar/character mapping; unresolved identities are explicit.
- All candidate shot IDs are contiguous and exist **inside** the referenced scene. Start/end times come from real shot boundaries or an explicitly documented earlier pilot.
- Each primary evidence reference points to existing input rows; every external source ID resolves to a real recorded URL. If an external article is generally about the film, it must not be credited as discussing the candidate's exact scene.
- Report total candidate count, distinct scenes, total runtime, priority/channel mix, first-batch count, and **human-approved count (default zero)**.
- Check diversity: at least consider quiet/listening/nonverbal material, not just speeches, violence and heavily reviewed scenes.
- Clearly label sampled-frame limitations and no-new-CV status. Stage05, Gaze Direction/Target and Stage06 are **downstream, after video review**.
- Avoid new hashing, release approvals or gratuitous pipeline machinery for this research prototype.
