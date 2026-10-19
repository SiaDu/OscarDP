# OscarDP — Stage03 GPT Visual Annotation v1.0 (interactive prompt)

Read and follow `docs/stage03_gpt_visual_spec_v1.md` and `docs/schemas/stage03_gpt_visual_v1.schema.json`.

I will supply a batch manifest, explicit source hashes, and the actual Early/Middle/Late images (possibly through a connected Google Drive folder). Confirm which images were **really accessible**. Do not assert that Drive files or images were viewed unless retrieved. Do not invent unseen image content.

For each requested Shot, examine the available three images (short shots may alias), fill the **exact** Stage03 schema:
- Fixed shot-level `visual` labels and evidence roles.
- Shot-local `subjects[]`: facing, visibility, quality, occlusion, appearance mode; no Actor or Character IDs.
- Salient `objects[]` and directly visible `visible_relations[]`.
- Short `visual_summary` without story, emotion, intentions, exact gaze or unsupported camera motion.
- `review_status` may be `unreviewed` or `model_reviewed`, never human without human action.

Do not generate any BBoxes, person/face detector confidence, ArcFace embeddings or verified identities. Do not copy Stage02 screenplay action into visually observed claims. If visually uncertain, use `unknown`/null or empty arrays.

Output a machine-readable JSON **batch** with the requested Shot IDs and all source/coverage bindings. Never label a partial batch as the complete `Stage03_<movie_id>.json`. Full-film publication requires separately verifying coverage, hashes, sample references and the exact Stage03 schema. Prefer consistent IDs within each shot and make no cross-shot identity assumptions.
