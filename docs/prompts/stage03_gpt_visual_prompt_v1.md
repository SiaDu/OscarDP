# OscarDP — Stage03 GPT Visual Annotation v1.0

**Standing instruction for a new ChatGPT conversation.**

Repository: https://github.com/SiaDu/OscarDP  
Minimal user request: **“做 OscarDP Stage03。Google Drive：<文件夹链接>。尽可能一次做完。”**

## 1. Read the current contract and user data

Fetch these files from GitHub **main**:
- `docs/stage03_gpt_visual_spec_v1.md`
- `docs/schemas/stage03_gpt_visual_v1.schema.json`
- `docs/shot_boundary_qc_v1.md` and `docs/schemas/shot_boundary_qc_v1.schema.json`
- This standing prompt.

Use the connected Google Drive tool on the user's supplied folder. Find `Stage01_<movie_id>.json`, `Stage02_<movie_id>.json`, all accessible `shot_<id>_early/middle/late.jpg` files, **and any existing Stage03 batch JSON files**. Search subfolders if necessary. The user need not separately supply a Frozen Manifest, hashes, `frame_samples_v1.json`, a batch manifest, original movie, SRT, PDF or local detectors. When existing Stage03 batches are absent, do not assume they exist; begin at the first unprocessed Shot that is actually evidenced by accessible work.

**Crucial:** File listing, names, Drive search snippets and thumbnails do not alone prove actual full image inspection. Load/view each image that is used as visual evidence. If the connector cannot display actual image content, stop inventing output; state the access limitation and request the smallest remedy (e.g., upload source images/ZIP in chat). Do not claim that all 1,100 shots have been inspected when only a sample was examined.

Stage01 controls shot IDs/boundaries; Stage02 is text-derived context, **not proof** of observed people, props, behaviour or shot scale. Do not infer visible performance from the screenplay. No BBoxes, CV confidence, actor/character identification, emotion, AUs, precise gaze, imagined between-frame motion or ungrounded cinematic details.

## 2. Perform the annotation

For every actually reviewed Shot, examine its Early/Middle/Late sample content (account for documented 1–2-frame aliases). Produce exactly the Stage03 schema's `visual`, `subjects[]`, `objects[]`, `visible_relations[]`, `visual_summary`, `review_status` and `samples[]` data. Use only fixed enumerations. Do not hallucinate missing images. Use `unknown`, `null` or empty arrays when evidence is limited. `review_status=model_reviewed` is the maximum absent a human check.

**Boundary QC:** If the three samples depict incompatible visual scenes inside one Stage01 Shot, add an issue to a **separate** Shot Boundary QC JSON conforming to `docs/schemas/shot_boundary_qc_v1.schema.json`. GPT may flag `suspected_undersegmentation`, but the actual number/location of cuts and whether a transition is a fade or dissolve cannot be verified from three stills alone. Preserve existing Stage01/Stage02 shot IDs and the Stage03 Schema. User-confirmed corrections must be recorded with `review_status=user_confirmed` and an explicit user-report basis; do not upgrade GPT-only observations to confirmation.

## 3. High-throughput, no manual “continue” between internal batches

**There is NO hard 20-shot batch limit.** A prior 20-shot Pilot is a checkpoint, not the maximum.

1. Inventory all Stage01 shots and existing Stage03 batches that are actually accessible. Validate the existing batch IDs/ranges and against the current Stage03 Schema when possible; never silently overwrite a pre-existing completed shot. If an earlier batch is only on the user's local computer, another chat cannot see it unless the user places it in Drive or attaches it.
2. Start with the **next unprocessed consecutive Shot**. Use adaptive **50–100 Shot work chunks** (150–300 logical JPG roles) when resource access and attention allow; shrink the chunk rather than lowering evidence quality. A chunk size is a work target, not a promise that all images can be read.
3. **After finishing a chunk, persist an independently valid batch JSON and any QC sidecar, then immediately proceed to the next accessible consecutive chunk in the same user request**, without asking the user to type “继续” after every 20 or 50 Shots. Multiple batch files may be delivered in one response. If writing files to connected Drive is supported and authorized, retain checkpoints there; otherwise attach the created files to the chat. Do not pretend a completed chat message runs further work in the background.
4. When a technical/task execution limit, lost image access, or context limit prevents further inspection, stop truthfully: provide **all actually completed batch files**, last completed Shot ID, next unprocessed Shot ID, inaccessible images/issues and what remains. Never claim you processed an unfinished range. A new user turn may then resume from saved artifacts.
5. Process the **whole movie** in one request **only if** every image was in fact accessible and reviewed and the complete output can be validated. If all Stage01 Shots are covered, merge validated batches into one `Stage03_<movie_id>.json` without a root `batch` field, and report full-film coverage. Otherwise keep `Stage03_<movie_id>_batch_NNN.json` with exact inclusive ranges.

Previous batch numbering must not be reset if a validated prior `batch_001` is available. For Bugonia `tt12300742`, when the actual existing `Stage03_tt12300742_batch_001.json` (shot_000001–shot_000020) is accessible and confirmed, resume from `shot_000021` as `batch_002`. Do not assume this earlier file lives in Drive merely because the user completed it in a different chat.

## 4. Output and validation

- Create real downloadable JSON files, not a pasted sample, when file tools are available.
- Validate each partial/full JSON against `docs/schemas/stage03_gpt_visual_v1.schema.json` with a JSON Schema validator when available, plus Shot ordering, ID uniqueness, genuine image references, valid relation refs and evidence roles.
- Keep the Shot Boundary QC file separate from Stage03. If no anomalies were found, no QC sidecar is necessary. Existing annotations need not be rewritten solely to register a QC issue.
- Provide a compact summary: shots reviewed **this run**, total distinct cumulative Shots based on verified available batches, saved filenames, QC issues, next unprocessed Shot, and any validation or image-access limitation.
- Do **not** modify GitHub, local source movies, Stage01/02 or Frozen Releases during annotation.

**Important operational limit:** This is an interactive workflow. Do as much actual work as supported within the current request, but GitHub prompts cannot force ChatGPT to run asynchronously after its answer, automatically reopen future sessions, or guarantee 3,300 image reads/1,100 Shot annotations in a single request. For an unattended end-to-end operation, a separate Work-mode or automated workflow is required.
