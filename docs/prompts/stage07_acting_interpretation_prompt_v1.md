# OscarDP Stage07 — Acting Interpretation Prompt v1

**用途**：为一个已选 Sequence 写表演解读（`acting_annotations.json`）。把 Stage04 的叙事上下文（角色档案、场景目标、战术）
和可观察的表演（人工 gaze、机器 AU 和头动、台词）在同一条时间轴上连起来。任何 LLM 都可以执行；输入和输出都存档。

**输入**：`python -m oscardp.stage07.interpretation_input` 生成的 `interpretation_input.json`（同目录下还有沿用的旧分支文件）。
输出写到同一目录，再用 `python -m oscardp.stage07 run --prepared --analysis-dir <该目录>` 组装和校验。

## 证据规则

| 来源 | 地位 |
| --- | --- |
| `gaze`（含 `reviewer_notes`） | 人工标签，可直接当观察 |
| `au_episodes`、`head_movements` | 未审核的机器测量；只来自可信近景轨迹 |
| 本人说话时的嘴部 AU | 弱证据（已按说话规则过滤，仍可能含口型）；说话时不考虑 AU25/26 |
| `head_movements` 中的 `beat` | 说话节奏，属于说话风格，不解读语义 |
| `scene_objectives`、`characters`、`sequence`、`legacy_visual_notes` | 早先 LLM/助手的假设，不是事实 |
| 缺失的证据（`gaze_unavailable`、`face_unreliable_tracks`） | 是覆盖缺口，不是"没有发生" |

不得使用剧本中这一场之后才揭示的信息解释当下动机。

## 实际发给模型的 Prompt

你是熟悉表演理论和电影表演的分析者。下面的 JSON 是一个电影片段的全部证据。请按角色把片段切成**节拍（beat）**，
说明每个节拍里角色**想要什么**（场景目标）、**用什么战术**，以及这些战术**如何体现在可观察的行为上**。

1. **切节拍**：边界来自台词意义的转折、注视模式的变化或镜头切换。同一角色的节拍不重叠，一般 3–20 秒。
   每个有证据的角色、每个镜头都要考虑；没有可观察证据的角色可以不写。
2. **每个节拍写**：
   - `beat`：一句短标题。
   - `objective_refs`：该角色自己在 `objective_refs` 里的目标 ID（至少一个）。
   - `tactic`：可表演的动作，用动词短语（如"要求发言权""把后果说成预测""拉上 Don""先让步再收回主导"）。
     对应 Stage04.2 的 `tactics` 时照用其意思；新提出的战术要在 `interpretation` 里说明依据。
   - `observation`：先写看得到、测得到的事，每条引用证据 ID（`gaze` / `au` / `head` / `video` / `narrative` 用台词的 `srt_id`）。
   - `interpretation`：战术如何通过注视、面部、头部和台词体现；说明哪些证据支持、哪些只是相容。
   - `alternative_interpretation`：至少一个合理的不同读法。
   - `uncertainty`：证据缺口和可靠性问题。
3. **语义头动**：对每个 `semantic_candidate=true` 的头部动作写 `meaning`（emphasis / negation / affirmation /
   attitude / other，加一句说明），以及 `alternatives`、`uncertainty`。说话者看动作落在哪句话、哪个词附近；
   听者看对方此刻在说什么。
4. **不要**：编造证据 ID；仅凭 AU 判断情绪类别；把节奏性的 beat 当语义动作；把 Stage04 的假设当事实。

## 输出结构

```json
{
  "status": "model_proposal",
  "authoring_method": "<模型、日期、本 prompt 版本>",
  "annotations": [{
    "annotation_id": "C_annotation_001", "status": "model_proposal", "character_candidate": "MICHELLE",
    "start": 1449.031, "end": 1458.082, "beat": "…", "objective_refs": ["scene_000021:MICHELLE:scene_objective"],
    "tactic": "…", "state_ids": ["C_state_002"], "event_ids": [],
    "observation": [{"source_modality": "gaze", "start": 1456.0, "end": 1458.08, "description": "…",
                     "evidence_ids": ["shot_000215_michelle_g005"]}],
    "interpretation": "…", "evidence": ["…"], "alternative_interpretation": ["…"], "uncertainty": ["…"]
  }],
  "head_gesture_readings": [{"evidence_id": "shot_000221_track_002_head_003", "meaning": "negation: …",
                             "alternatives": ["…"], "uncertainty": ["…"]}]
}
```

`state_ids` 取与节拍重叠的同角色 `affect_states`。组装时代码会检查：证据 ID 存在、角色和时间对得上、
`objective_refs` 属于该角色、`tactic` 非空、头动解读只针对语义候选；并自动补上 `shot_ids`。

## Changelog

- v1（2026-11-28）：首版。Bugonia `seq_006` 由 Claude 在对话中执行，输出 `reports/stage07/tt12300742/analysis_003`。
