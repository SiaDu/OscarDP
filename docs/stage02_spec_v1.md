# OscarDP — Stage02 Shot-Centric Semantic Annotation v1.0

**状态：固定格式（2026-10-12）**。参照 `Stage02_tt0318374_LLM_pilot.json`（The Cooler，前 154 shots）。这是一份**文本来源的叙事与表演线索标注**，并非电影画面视觉标注。

## 1. 输入、输出、角色

每部电影提供：`Stage01_<movie_id>.json`（shot ID 和起止秒数）、`<movie_id>.srt`（电影实际对白与音频字幕）、电影 screenplay PDF（对齐剧本上下文及说话者）。可选：已经存在的 Stage02 audit JSON，供核对，但**不能替代重新阅读原始 SRT/PDF**。

- **SRT 是台词的唯一正文来源**；不能用 PDF 中原版、删改前的对白覆盖电影字幕。
- **PDF 是语义对齐、场景、人物、动作和剧本表演提示的来源**。允许电影删改、剧本重排和电影独有内容；不能强迫一对一匹配。
- **Stage01 是 shot ID 与时间的唯一来源**；不得更改其边界或裁掉片头片尾。
- 正式仅输出完整影片的 `Stage02_<movie_id>.json`；允许内部按剧情分段分析，但默认不向用户交付分块文件，也不要求逐段确认。若确实无法完成全片，才输出如实标注覆盖范围的 `Stage02_<movie_id>_part001.json`，绝不冒充完整结果。
- 原始 SRT/PDF/Stage01 不移动、不修改。如有旧版对齐审计资料，可供核对，但当前流程只生成一份正式 Stage02 JSON；不在其中重复页码、匹配证据或未分配单元。

## 2. 根对象

固定 `movie_id`、`movie_name`、`shots`。**部分范围文件**允许增加 `pilot`，至少包括 `review_start`、`review_end`（exclusive）和 `shot_count`；完整影片正式文件可以不带 `pilot`。

## 3. 每个 Shot 的固定字段

| 字段 | 数据类型 | 规定 |
|---|---|---|
| `shot_id` | string | 与 Stage01 完全一致，如 `shot_000020` |
| `temporal` | object | `start`, `end` 秒；与 Stage01 完全一致。Stage02 不必重复帧号 |
| `dialogue` | array | 来自原始 SRT 的实际台词，未出现则 `[]` |
| `narrative` | object | 剧本/SRT 支持的情境语义，字段见下文 |
| `tags` | string[] | 简短、尽可能一致的 `snake_case` 文本叙事主题标签；无依据时 `[]` |

### 3.1 `dialogue[]`

每条有：`srt_indices`（integer[]，保留原始字幕编号）、`speaker`（剧本或上下文支持的角色名，无法判断用 `null`）、`text`（**只取实际 SRT**）。同一个角色连续的多个 cue 可合并一句，索引全部保留。一个 SRT 跨多个 Shot，在每个重叠 Shot 中出现**相同索引与原文**；混合说话者时可以保留整句并加可选 `voice_overlap: true`，不得强行分配说话者。

### 3.2 `narrative`

固定六个字段：

- `scene`: 简洁的叙事场景名称，**暂定剧本情境，不等于 Stage04 的最终场景分割**。
- `location`: 剧本支持的地点；不明确可为 `null`。
- `time_of_day`: 例如 `Day` / `Night` / `Unknown`（无法可靠判断时用 `Unknown`）。
- `interior_exterior`: 例如 `Interior` / `Exterior` / `Interior / Exterior transition` / `Unknown`。
- `characters_in_script_context`: 相关剧情人物数组；**不等同于此 Shot 画面里实际可见的人物**。
- `action`: `[{"text": "...", "tag": "snake_case"}]`；只对能合理定位到这个 Shot 的动作/事件给出简短描述，不能把整段剧本事件灌入每个镜头；否则用 `[]`。

### 3.3 可选字段（仅有内容时出现）

- `performance_cues`: `[{"character": "BERNIE", "category": "gaze_aversion", "description": "...", "basis": "screenplay_explicit", "verification": "screenplay_text_only_not_video_observation"}]`。`basis` 可为 `screenplay_explicit` 或 `llm_inferred`；两者都**不是**已经从视频验证的表演；意图/情绪缺乏证据时不生成。`category` 用不含人物名字的通用 `snake_case` 分类，保持跨电影一致。
- `sound_cues`: `[{"srt_index": 1, "text": "[music]"}]`；SRT 中注明的音效、环境声等，非电影图像观察。
- `song_lyrics`: `[{"srt_index": 2, "text": "♪ ... ♪"}]`；SRT 中的歌词，不放入普通人物对白。
- `dialogue[]` 条目可选 `voice_overlap: true`，仅针对混合说话者的原始字幕。

不允许添加顶层 `unassigned_subtitle_cues`、`unassigned_screenplay_units` 或 Shot 内 `match_evidence`、PDF `page`、`matched_subtitle_cue_ids`、文本相似度评分。

## 4. JSON 示例（源自 The Cooler Pilot，缩短展示）

```json
{
  "movie_id": "tt0318374",
  "movie_name": "The Cooler",
  "shots": [
    {
      "shot_id": "shot_000020",
      "temporal": { "start": 246.96, "end": 250.32 },
      "dialogue": [
        {"srt_indices": [32], "speaker": "NATALIE", "text": "Fuck, Dewar's and water, please."},
        {"srt_indices": [33], "speaker": "BERNIE", "text": "Hi, Natalie."}
      ],
      "narrative": {
        "scene": "Bernie meets Natalie",
        "location": "Shangri-La casino bar",
        "time_of_day": "Night",
        "interior_exterior": "Interior",
        "characters_in_script_context": ["BERNIE", "NATALIE", "DORIS"],
        "action": [{"text": "Bernie greets Natalie and reminds her of his name.", "tag": "bernie_greets_natalie"}]
      },
      "tags": ["social_interaction", "introduction", "workplace", "first_contact", "hesitation"],
      "performance_cues": [
        {
          "character": "BERNIE",
          "category": "initial_eye_contact",
          "description": "The screenplay describes Bernie trying to catch Natalie's eye; she initially does not look toward him.",
          "basis": "screenplay_explicit",
          "verification": "screenplay_text_only_not_video_observation"
        }
      ]
    }
  ]
}
```

## 5. 语义标注标准

1. 模型需**真正阅读**相关连续剧本与字幕上下文，理解改写对白、人物关系、时间/场景改变、剧本与成片顺序差异；不能把词面模糊匹配冒充语义标注。
2. 一定先识别剧情块，再映射到 Shot；剧情块**不是**正式 Stage04 scene。
3. SRT 时间与 Stage01 时间重叠是台词归属的确定性依据；并非每句台词只对应一个 Shot。
4. 剧本明确写出的 gaze、emotion、intent 等可以记 `performance_cues`，并标明 `screenplay_explicit`。推断必须写 `llm_inferred`；弱推断宁可不填。
5. 未上传完整电影或连续画面，**不能**标注真实 shot size、camera angle、camera movement、构图，或把剧本行为表述为“电影已观察到”。摄影视觉信息属于 Stage03。
6. 同场景中连续静默镜头可保留适当共同 `scene`/`location`；不要重复编造 action、emotion、gaze。
7. 字幕歌词、音效独立于 `dialogue`；未归属的来源材料不强行塞进 Stage02。
8. 保留英文原始人名、台词和标签用英文；JSON 必须是合法 UTF-8，不应包含 Markdown 注释。

## 6. 跨窗口手工工作流（不用付费 API）

**一个新对话框处理一部电影，默认一次性完成整部影片。** 上传本规范及该电影的 Stage01 JSON、SRT 和 PDF（旧版审计数据可选）。模型先确认文件可读、Shot 总数及时间范围、SRT 条数，再以连续剧情为依据在**内部**进行分块阅读、语义判断和跨段一致性检查；内部可以分段，但**不得把分段变成用户必须逐次确认的操作**，不得默认要求用户每隔 10–15 分钟发送「继续」。

**只交付一份完整 `Stage02_<movie_id>.json`**。必须覆盖 Stage01 的所有 Shot，包括没有对白的 Shot；输出前逐项校验唯一且无遗漏的 Shot ID、时间与 Stage01 完全一致、`srt_indices` 都真实存在、台词没有被 PDF 覆盖、没有未经证实的视听/眼神标签。可在最终回复中简述检查结果及有代表性的语义歧义，不要求另外输出多份中间文件。

**例外：不能在本次任务内真实完成时。** 必须说明实际完成的连续 Shot ID/时间范围、未完成范围与原因，仅保存已完成的 `Stage02_<movie_id>_partNNN.json`；**不得用自动填充、复制相邻镜头语义或未核验的匹配来假装全片完成**。如要在其他对话继续，应附本规范、原始输入和已完成部分，并从下一个未处理 Shot 接续。若文件不可读或前段上下文丢失，应直接说明，不得凭空补齐。

## 7. 新对话框可复制的 Prompt

> 请以我上传的《OscarDP Stage02 Shot-Centric Semantic Annotation v1.0》为**固定 Schema**，从本次上传的 `Stage01_<movie_id>.json`、电影原始 `.srt` 和 screenplay `.pdf` 为依据，给该电影做真正的语言模型语义对齐与 Shot-level Annotation。请阅读连续剧情语义而非仅使用 Python 字符串相似度；Python 只允许用来解析文件、按时间关联 Shot、生成及校验 JSON。
>
> **SRT 是实际对白唯一正文来源**；剧本负责场景、人物、说话者、剧情、表演线索以及对白语义定位，不得替换 SRT 原文。剧本与成片可能删改、重排，无法确认请留空。不得臆测电影中实际可见的人、镜头运动、拍摄景别或已发生的 gaze；剧本表演线索须标记来源。
>
> 每个 Shot 输出 `shot_id`、`temporal`、`dialogue`、`narrative`、`tags`，可选 `performance_cues`、`sound_cues`、`song_lyrics`。遵守本规范的字段和命名规则。**不要生成密集的原始 `screenplay_units`、PDF 页码、match evidence 和 unassigned 数组。**
>
> **请一次性处理整部电影**，从第一个 Shot 到最后一个 Shot，无须我手动指定范围。可以在内部按连续剧情分段阅读和理解，但要自动衔接、保持上下文一致，**不要每 10–15 分钟停下来让我发送「继续」**。只有实际进行过 LLM 语义判断的内容才能作为完成的标注。
>
> 核对 Shot 总数、时间、SRT 编号与全文可读性；最终只输出一份**真实可下载**的 `Stage02_<movie_id>.json`，并校验 Shot 无缺失、无重复、时间与 Stage01 完全一致、台词只来自 SRT。如果确实无法一次完成，不可编造未处理内容或声称已完成全片；请如实报告覆盖范围，保存实际完成的 `Stage02_<movie_id>_part001.json` 并解释原因。这是异常退路，不是默认分批交付方式。
