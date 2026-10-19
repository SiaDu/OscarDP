# OscarDP — Shot Boundary QC v1.0 (Stage01/Stage03 advisory sidecar)

This document defines **optional QC records** for missed editorial cuts, crossfades/dissolves, and visually heterogeneous Stage01 shots surfaced during GPT review. The official contract is [`schemas/shot_boundary_qc_v1.schema.json`](schemas/shot_boundary_qc_v1.schema.json).

## Why keep it separate?

TransNetV2 may merge multiple true cinematic shots across a gradual transition. Stage03 examines three sampled images and may flag incompatible views. This observation **does not** grant permission to change Stage01 boundaries, renumber shot IDs, or change frozen Stage02 dialogue/scene mapping. The Stage03 v1.0 output schema stays unchanged so previously validated batches (including Bugonia batch_001) remain valid.

Store the issues as `Stage03_<movie_id>_shot_boundary_qc.json` in working reports or near the GPT batch outputs, **outside the immutable Frozen Release**. Multiple processing sessions may append/merge records by (movie ID, Shot ID) after reviewing conflicts. A sidecar is not a formal shot segmentation correction and does not record imaginary cut frames.

## Evidence and review levels

- `model_suspected` / `basis=sampled_images`: only the three sampled JPGs were inspected; assert `visual_discontinuity` or `suspected_undersegmentation`, **not confirmed cuts**.
- `user_confirmed` / `basis=user_video_report`: the user reports checking the source video and confirming the editorial issue; identify it as user testimony, not as independent video inspection by GPT.
- `human_video_verified` / `basis=human_video_review`: exact source-video evidence was actually inspected by a human. Precise boundary frame numbers still require explicit measurement.
- `reported_transition=fade_or_dissolve` is deliberately broad when the exact edit effect cannot be distinguished. Do not claim a specific cut time or a number of shots without sufficient supporting evidence.

`performance_selection=needs_review` is the safe default for ambiguous cases. Use `avoid_without_video_check` when a known mixed Shot should not be blindly passed as one contiguous actor-performance clip. **Do not globally exclude all under-segmented Shots**; Stage05 should check the targeted interval in the source video if the Shot contains possible actor evidence. `no_action` is for cases where the issue has no direct downstream consequence. Visual/narrative importance is not assessed by the absence of faces.

## Bugonia Pilot 001: user-confirmed report

The Stage03 `shot_000011` annotations describe a pink flower (Early), hive opening and bee (Middle), and a wide field with hive boxes/building (Late). The user reports that the original footage actually contains **three** editorial shots joined by gradual fade/dissolve-like transitions. This is a **user report**; no exact frame boundaries have been independently remeasured.

**Example issue record (illustrative documentation; not a generated data file committed to this code repository):**

```json
{
  "movie_id": "tt12300742",
  "qc_version": "1.0",
  "issues": [
    {
      "shot_id": "shot_000011",
      "issue_type": "confirmed_undersegmentation",
      "review_status": "user_confirmed",
      "evidence_roles": [
        "early",
        "middle",
        "late"
      ],
      "observation": "Early: pink flower close-up; Middle: bees at hive opening; Late: field with beehives and a small building. The user reports that the original video contains three shots joined by gradual transitions.",
      "reported_transition": "fade_or_dissolve",
      "estimated_true_shot_count": 3,
      "exact_boundaries_verified": false,
      "performance_selection": "avoid_without_video_check",
      "basis": "user_video_report",
      "notes": "User-reported three actual shots; exact transition subtype and cut frame indices were not independently measured. Keep the existing Stage01 Shot ID and Stage03 batch intact."
    }
  ]
}
```

Other visually discontinuous Shots should be flagged in future Stage03 passes without requiring the user to inspect every single case immediately. Do not alter completed Stage03 image observations solely because a QC issue exists.

## QC validation

A JSON Schema validator can check field structure, but it cannot confirm editorial cuts. Independently ensure each `shot_id` exists in Stage01, no duplicate/conflicting QC entries, evidence roles refer to actually viewed images, and any `user_confirmed` evidence genuinely came from the user (rather than LLM inference). Keep source proof and optional future boundary repairs as separate, auditable material.
