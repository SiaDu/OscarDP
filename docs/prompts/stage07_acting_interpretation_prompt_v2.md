# OscarDP Stage07 — Acting Interpretation Prompt v2

**用途**：为一个已选 Sequence 写表演解读（`acting_annotations.json`）。v2 根据 Bugonia `seq_006` 第一轮人工审核
（`review_006`：15 条里 4 YES / 9 PARTLY / 2 NO）修改：基于 gaze 的判断成立，基于 AU 和"分析者知识"的判断不成立。

**输入**：`python -m oscardp.stage07.interpretation_input` 生成的 `interpretation_input.json`（v2 可选加 `--gaze-stats`
和 `--review`）。输出写到同一目录，再用 `python -m oscardp.stage07 run --prepared --analysis-dir <该目录>` 组装和校验。

## 证据规则

| 来源 | 地位 |
| --- | --- |
| `gaze`（含 `reviewer_notes`）、`gaze_relations` | 人工标签及由它算出的关系，可直接当观察 |
| `review_feedback` | 审核者对上一版的意见；其中的描述是人工观察，优先于机器证据 |
| `head_movements`（`semantic_candidate=true`）| 未审核的机器测量；可以解读含义 |
| `utterance_head_style`、`beat` | 说话时的头动风格（按句：速度相对沉默的倍数、动作类型），只描述风格，不逐个解读 |
| `au_episodes` | **只作提示**：说明哪里有变化，不能据此命名情绪；只有和 gaze、头动、台词或审核意见一致时才写进观察 |
| `scene_objectives`、`characters`、`sequence`、`legacy_visual_notes` | 早先 LLM/助手的假设，不是事实 |
| 缺失的证据 | 是覆盖缺口，不是"没有发生" |

AU 已知会出错的情况：眨眼和视线转移被读成眼睑收紧（眼周 AU 在这些时段已屏蔽）；整段持续的表情被每个镜头自己的基线抵消；
细微变化低于阈值；抿嘴（AU24）、努嘴（AU18/22）、呼吸等测不到。

## 实际发给模型的 Prompt

你是熟悉表演理论和电影表演的分析者。下面的 JSON 是一个电影片段的全部证据。请按角色把片段切成**节拍（beat）**，
说明每个节拍里角色**想要什么**、**用什么战术**，以及这些战术**如何体现在可观察的行为上**。

1. **切节拍**：边界来自台词意义的转折、注视模式的变化或镜头切换。同一角色的节拍不重叠，一般 3–20 秒。
   有 `review_feedback` 时，沿用上一版的节拍编号和时间，除非审核意见要求改动。
2. **每个节拍写**：`beat`、`objective_refs`（该角色自己的目标 ID）、`tactic`（动词短语）、`observation`（每条引用证据 ID）、
   `interpretation`、`alternative_interpretation`（至少一个）、`uncertainty`。
3. **角色只知道当时知道的事**：动机只能用角色此刻能看到、听到的信息解释。Stage04 里"谁最动摇"之类的分析者结论，
   不能当作角色已经知道的事；如果用到，要说明是角色从对方行为里推断的，并给出那个行为。
4. **注视关系**：听者看向第三人时，先查 `gaze_relations`（跟随视线、共同注意、对视），再考虑"受到威胁"等心理解释。
   时间顺序以标签为准；审核者备注里提到可能漏标的注视，要写进 `uncertainty`。
5. **头动**：说话时连续的头动按句写成风格（例如"说服时头动是沉默时的 3 倍，混合点头和歪头"），不逐个解读。
   只对孤立的 `semantic_candidate` 写 `head_gesture_readings`：`meaning`（emphasis / negation / affirmation /
   attitude / other，加一句说明）、`alternatives`、`uncertainty`。孤立 = 沉默时的动作，或该句里唯一的语义候选；
   同一句里有两个以上、紧贴 gaze 转移（头带着或跟着眼睛）、或审核者描述为连续动作的，按风格处理。
   机器类型（如 shake）和审核者看到的不同时，以审核者为准，机器类型写进 `alternatives`。
6. **表情**：不从 AU 推情绪。描述表情时用审核者的观察或可见的组合（如"抬眉、睁眼、张嘴"），并说明是否有 AU 支持。
7. **不要**：编造证据 ID；把节奏性的 beat 当语义动作；把 Stage04 的假设当事实；把上一版被否定的读法换个说法再写一遍。

## 输出结构

与 v1 相同（`status`、`authoring_method`、`annotations[]`、`head_gesture_readings[]`）。`authoring_method` 写明模型、
日期、本 prompt 版本，以及是否读过 `review_feedback`。组装时代码会检查：证据 ID 存在、角色和时间对得上、
`objective_refs` 属于该角色、`tactic` 非空、头动解读只针对语义候选；并自动补上 `shot_ids`。

## Changelog

- v2（2026-11-28）：AU 降为提示，眼周 AU 屏蔽；加角色知识限制、注视关系、按句头动风格和审核意见输入。
  Bugonia `seq_006` 输出 `reports/stage07/tt12300742/analysis_004`。
- v1（2026-11-28）：首版，`analysis_003`。
