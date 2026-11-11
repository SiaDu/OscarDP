# OscarDP Stage04.2 — Film-wide Scene Character Objectives Prompt v1.2

## Goal
Analyse **every existing Stage04 Scene** for each listed narrative participant's immediate objective, obstacle and supported narrative tactics. This is script/character ACTION analysis, not emotional or visual acting recognition. Adapt Uta Hagen's Objective / Obstacle / Action framework into this OscarDP representation.

## Required inputs
Complete Stage02 JSON, Stage04 JSON, Stage04.1 JSON, movie IMDb ID and nomination/target metadata. Read the original screenplay PDF when provided, specifying its draft/date. Optional explicitly selected Scene/Sequence with movie-global seconds. Bugonia: 80 Stage04 scenes; optional detailed sub-sequence in scene_000021 (1432.013915–1547.212332 s).

## External research, selectively
If available, consult Stage04.1 `external_research[]`; browse again **only for disputed motivation, story-level ambiguity, adaptation/draft differences or important conflicting interpretations**. Prioritise creator interviews, then established film criticism. Record each new useful source's title, author/publication, date, URL, type and short interpretation, deduplicating Stage04.1 sources. Record competing readings rather than picking a purported consensus.

**Do not use an online ending explanation to overwrite the screenplay, Stage02 dialogue, Stage04 boundaries, an actor's expressed beliefs, or a character's knowledge at an earlier time.** A critic's reading is an attributed contextual hypothesis, not ground truth for `scene_objective`. Every supported objective must still cite primary film/script evidence. Add optional `external_research[]` and `external_source_refs[]` to interpretive notes when helpful; leave empty when no search or no useful source.

## Per-scene process
1. Enumerate ALL Stage04 `scenes[]` in order, producing exactly one output scene record for every original Scene ID, including flashbacks, titles, montage, intercuts and credits.
2. Preserve Stage04's scene title/type, first/last shot, exact movie-global start/end and `narrative_participants`. Do not create new scenes or reinterpret screenplay scene numbers as film Scene IDs.
3. Create one `characters[]` entry per Stage04 narrative participant. Participation may mean voice-over, memory, offscreen presence or an inactive captive; it does NOT guarantee a face, speech or independent objective.
4. Distinguish the character's long-term desire (Stage04.1) from a **playable immediate** `scene_objective` (what they attempt to obtain/change/prevent here); `obstacle` (what prevents it); and `tactics` (specific supported dialogue/story actions). A change of tactic does not change the Stage04 Scene.
5. An unobserved goal is **unknown**, not presumed. With weak evidence use `scene_objective=null`, `obstacle=null`, `tactics=[]`, `coverage_status="insufficient_evidence"`; for purely non-dramatic scenes or non-agent collectives use `not_applicable`.
6. Anchor supported statements in actual within-scene Stage02 shot/dialogue IDs and/or original screenplay's 1-based PDF page numbers. Do not fabricate timings from PDF pages. A screenplay draft may differ from the released film; record uncertainty.
7. **Timeline / epistemic rule:** do not use a final-act reveal as proof of what early-scene characters or viewers knew. Separate objectively established plot facts from character claims, tactics and critic interpretation.
8. **Nested-experiment distinction:** in Bugonia distinguish Auxolith's opioid-withdrawal drug trial involving Sandy from the claimed Andromedan human-gene reform programme and its final council assessment. Do not merge their goals or outcomes or infer exact institutional equivalence merely from critical commentary.
9. Use nominee CSV for target_actor; supporting scene participants remain context_character regardless of narrative importance. All interpretive labels remain `model_proposal`.
10. Only for explicitly provided sub-sequences attach `selected_sequence`, `selected_sequence_objective` and `selected_sequence_role`. For Bugonia preserve the detailed scene_000021/115-second pilot while outputting all 80 full-scene records.

## Output
One separate UTF-8 `Stage04_2_<movie_id>.json`; never overwrite Stage04 or Stage04.1.

Root: movie_id, stage="04.2", stage_name, stage_version, source_materials, coverage, time_reference, analysis_policy, scene_objectives[], limitations[], optional external_research[].

Scene: scene_id, scene_title, scene_type, full_scene_range, selected_sequence (object or null), narrative_participants, coverage_status, characters[].

Character: character_name, analysis_role, coverage_status, scene_objective (claim or null), obstacle (claim or null), tactics[], selected_sequence_objective (claim or null), selected_sequence_role, uncertainty.

Claim: statement_en/statement_zh, basis, evidence_refs[], confidence, status="model_proposal"; optional external_source_refs[] (supporting perspective ONLY) and movie_time_range where actually grounded. Use at most 1–3 salient tactics; do not invent an action for every shot.

## Checks
Check scene count/order and original boundaries against Stage04 (Bugonia 80/80); check each participant, PDF page and Stage02 shot ID; confirm positive evidence for non-null claims, true nulls for insufficient/no-action cases and chronology; report supported vs unknown counts. No emotion, gaze target, AU or medically diagnostic inferences; no invented reviews or links, hash bureaucracy or new validation system.

**Bugonia caution:** screenplay scene 44 corresponds to the long Stage04 scene_000021; the 115-second selected excerpt ends before Teddy's reaction to Michelle's major proposed deal. Don is mostly silent; do not infer a detailed strategy from silence.
