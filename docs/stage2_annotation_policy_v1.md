# OscarDP Screenplay–Final Film Dialogue Alignment Annotation Policy v1

**Canonical repository copy.** This is an exact copy of the frozen policy at
`/mnt/g/datasets/oscar_movie_processed/tt27847051/review/openai/gold_adjudication_stage253/annotation_policy_v1.md`.
Source SHA-256: `0ac78ef566d1e84198528fd706d6d31e241ed6c37444c87aed0e2de4e34b74c3`.
The dataset original remains the immutable experiment artifact.

**Status:** FROZEN v1 for the `tt27847051` pilot adjudication.  
**Cross-film status:** provisional; apply consistently, but reopen only when a new film exposes a genuinely new annotation regime.

## Core unit

The annotation unit is the **screenplay dialogue turn realized by a final-film subtitle**, not general scene-level semantic similarity. A match should preserve the same proposition or clearly the same communicative/dramatic turn.

## 1. Exact / near-exact dialogue

**Policy:** `match`. Prefer the most specific source turn when several candidates are plausible. Screenplay order is evidence, not a reason to choose a weaker textual/semantic match.

## 2. Paraphrased dialogue

**Policy:** `match` when wording changes but the proposition/communicative act is preserved. Mere thematic relatedness is insufficient.

## 3. Expanded or contracted final-film dialogue

**Policy:** `match` when expansion/contraction still realizes the same screenplay turn. If the film introduces an independent new fact, demand, reaction, or proposition, annotate that added material `no_match`.

**Pilot examples:**  
- Positive: `subtitle_000711` → `scene_UNNUMBERED_056_dialogue_001`.  
- Negative: `subtitle_000700`, `subtitle_000712`, `subtitle_000715` → `no_match`.

## 4. Repeated dialogue

**Policy:** multiple final-film subtitles may map to the same screenplay block. Repetition is allowed. If a later screenplay turn is a substantially more specific lexical/semantic realization, prefer that specific turn rather than forcing reuse of an earlier block.

## 5. Reordered dialogue

**Policy:** final-film order may differ from screenplay order. Local backward/forward movement is a quality diagnostic, not a correctness rule. Choose the best turn-level correspondence.

**Pilot example:** `subtitle_000634` → `scene_UNNUMBERED_053_dialogue_015`.

## 6. Subtitle fragmentation

**Policy:** several subtitles may jointly realize one screenplay turn, and short fragments may map to that same turn when local context makes the relation clear.

**Pilot example:** `subtitle_000085` is the contracted fragment of the screenplay question in `dialogue_025`.

## 7. Vocatives / isolated names

**Policy:** attach a standalone vocative/name to an adjacent substantive screenplay turn only when temporal context and discourse function make that attachment clear. Do not select a distant block solely because it contains a similar name. If attachment or speaker is genuinely unresolved, use human review/`ambiguous`.

**Pilot example:** `subtitle_000698` attaches to `scene_UNNUMBERED_055_dialogue_001`.

## 8. Courtesy words / fillers

**Policy:** added greetings, courtesy phrases, fillers, acknowledgements, or social niceties are `no_match` when no screenplay turn expresses that proposition. Do not absorb them into a nearby socially related line.

**Pilot example:** `subtitle_000655` → `no_match`.

## 9. Improvised / added dialogue

**Policy:** `no_match` when the film introduces a new proposition not present in screenplay dialogue, even if it occurs inside the same scene or actor speech.

**Pilot examples:** `subtitle_000700`, `000712`, `000713`, `000715`.

## 10. Graphic / telegram / title / sign text

**Policy:** visible text is **not dialogue**. Annotate `no_match` in the dialogue-alignment task even when it is semantically caused by or related to nearby dialogue. A future action/graphic alignment layer may represent it separately.

**Pilot examples:** `subtitle_000823`–`000825`.

## 11. Narration / voice-over

**Policy:** spoken narration/voice-over may `match` when the screenplay explicitly represents it as a dialogue/V.O. block. On-screen written text must not be treated as narration.

## 12. Candidate-recall failure

**Policy:** if the correct screenplay turn is identifiable but absent from the supplied candidate set, use `uncertain` with empty block IDs. Never invent a block ID.

**Pilot reference cases:** `subtitle_000042`, `000043`, `000182`.

## 13. Multiple plausible screenplay blocks

**Policy:** prefer the block with the strongest specific turn-level correspondence. If available subtitle/context cannot distinguish repeated nearby blocks or speaker attribution, mark the case `ambiguous` for video/speaker adjudication rather than changing gold to optimize model score.

**Pilot unresolved case:** `subtitle_000693`.

## 14. One subtitle spanning multiple dialogue blocks

**Policy:** a `match` may contain multiple ordered source dialogue blocks when one subtitle genuinely spans those turns. Do not add unrelated blocks merely to improve semantic coverage.

## 15. Several subtitles mapping to one screenplay block

**Policy:** explicitly allowed. Long screenplay speeches may be fragmented into many final-film subtitles; each subtitle may map to the same block when it realizes part of that turn.

## Decision hierarchy

1. Is the subtitle spoken dialogue? If not (graphic/title/telegram/sign), `no_match`.
2. Is there a supplied screenplay block that expresses the same proposition or clearly the same communicative turn?
   - Yes → `match` the most specific block(s).
   - No → continue.
3. Is an identifiable likely screenplay target missing from the candidate set?
   - Yes → `uncertain`.
   - No → `no_match`.
4. If several nearby blocks remain genuinely indistinguishable with the available evidence → human `ambiguous`; do not alter gold solely to improve evaluation.

## Gold-change rule

A frozen gold label may change only after explicit adjudication with source context. Model agreement alone is never evidence sufficient to change gold.
