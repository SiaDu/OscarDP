# OscarDP Stage06.1 — Scene Layout Prompt v1.0

> **用途**：Codex 实现指令 + VLM 分析规范。先针对 Bugonia `seq_006` 建立大致场景布局和各镜头机位；后续 Stage06.2/06.3 才使用眼头方向估计注视对象。
>
> **只做空间信息。** 不做 gaze direction、gaze target、gaze behaviour、情绪或表演意图判断。本文件是新 Stage06.1 的规范，**不是**旧 `src/oscardp/stage06/`（现规划为 Stage07）的修改授权。

## 1. 目标和原则

对一个已经选出的电影 Sequence，实际查看多个 Shot 的 Early/Middle/Late 图片，输出：
1. **人物大致站位**：谁坐在哪里、谁站在谁附近、身体大致面向哪边；
2. **摄影机大致机位**：每个 Shot 从哪个区域拍摄、主要拍摄谁、与参考人物轴线的关系；
3. **布局是否变化**：如果有人明显走位或镜头轴线发生变化，标出变化；无法判断就用 `unknown`。

不做精确三维重建，也不推断真实摄影机标定参数。归一化俯视坐标只用于**画示意图**，不是有真实尺度的测量结果。

优先依据真正打开并看过的图像，不得仅凭 Stage03 `visual_summary`、Stage04 人物候选或剧情脑补空间信息。Stage03/04 只能提供提示，不是已核实的人物真值。

所有标签名以直白易懂为原则，例如 `people`、`camera_position`、`possible_targets`；不引入复杂图网络、证据评分模型或管理系统。

## 2. Pilot 输入

第一版只处理：
- `movie_id = tt12300742`（Bugonia）
- `sequence_id = seq_006`
- `scene_id = scene_000021`
- 电影全局时间：`1432.013915–1547.212332` 秒
- 13 个镜头：`shot_000211` 至 `shot_000223`（以 Stage04.3 实际 `shot_ids` 列表为准）
- 已知可能人物：`MICHELLE`、`TEDDY`、`DON`；**不是逐镜头已核实身份**。

先从实际本地输入路径查找 Stage01/Stage03/Stage04/Stage04.3 JSON 和 Stage01 生成的 Shot JPG。**不得假设**本地数据目录一定存在，也不要修改它们。

优先检查空间信息较多的参考镜头：
- `shot_000212`：MS，Teddy 面部及 Michelle 前景，利于判断相对位置；
- `shot_000218`：MS，Teddy 与 Don 同框线索；
- `shot_000222`：MLS，三人同框候选，利于建立整体布局。

这些说明来自已有候选标注，仅供优先检索；**必须看图片核对**，尤其 `shot_000222` 的人物映射仍有未确认部分。随后覆盖 `seq_006` 的所有 13 个镜头。需要更早的建立镜头时，允许在 `scene_000021` 内找相邻 Shot，但不要把该 Sequence 之外的镜头混进 13 个待分析镜头记录。

每个 Shot 优先读取 middle，必要时看 early/late 判断构图变化、人物走位和可能的镜头内部移动。若三张采样帧相互矛盾，说明不能根据稀疏采样声称整镜头空间稳定。

## 3. 实际发给 VLM 的分析 Prompt

你是一名熟悉电影镜头语言的视觉分析助手。请根据我**提供并实际可见**的连续电影 Shot 图片，为这一 Sequence 建立**粗略俯视场景布局**，并说明各 Shot 的摄影机大致位于什么区域。

请按以下步骤：

A. 根据 MS/MLS/LS/OTS 和多人同框画面确定人物身份候选、相对位置、坐姿/站姿、身体朝向。把人物位置画在同一个示意平面。不同 Shot 中对应的人物应使用同一个名称。无法核实身份或空间关系时明确写 `UNKNOWN`/`unknown`/`null`。

B. 利用正反打、OTS、画面空间方向、场景物体及连续镜头，估计每个 Shot 的摄影机位置和大致拍摄对象。**不要直接把屏幕左/右等同于现实空间左/右**。同一个布局要使用一致坐标系，不能逐镜头任意旋转示意图。

C. 考虑电影的 **180° 轴线原则**：常规正反打一般保持同一人物关系轴线的一侧，因此屏幕方向通常连续；但这只是一条先验，不是铁律。三人戏可能切换人物关系轴线，也可能通过中性镜头或运动换轴。判断轴线时必须注明是哪两个人构成的轴线，以及相对于哪个参考镜头；未知就输出 `unknown`，不要为了符合规则强行安放摄影机。

D. 识别明显走位。如果人物的相对站位发生变化，分成新的 `layout`；若无法确认变化，保持 `unknown`，**不要自动认定整段 Sequence 人物始终静止**。如果只有镜头变化但人物未变化，则尽量复用一个 `layout`。

E. 对无法从画面支持的布局、身体朝向、机位和人物身份用 `null` 或 `unknown`，并用一两句中文指出缺什么证据。不要为美观填假坐标。简单保留证据 Shot ID，不做复杂数值打分。

**禁止推断**：
- 谁正在看谁、Gaze Target、眼球方向、Eye/Head Gaze、Look Away；
- 人物情绪、思考、回避、表演意图；
- 真实摄影机米制坐标、精确 yaw/pitch/focal length；
- 由说话者自动推断身体朝向或注视目标。

`facing_to` 指身体大致朝向谁，**不是注视谁**；`camera_subject` 指摄影机拍摄对象，**不是人物 gaze target**。

输出以下指定 JSON，不添加自造的类别；必要补充解释写在 `note`。

## 4. 固定字段和允许的标签

### 4.1 Layout / 人物

| 字段 | 类型／允许的值 | 说明 |
| --- | --- | --- |
| `layout_id` | `layout_001` 等 | 本段内编号；相对站位变化时新建 |
| `start_shot_id`, `end_shot_id` | 真实 Shot ID | 布局适用范围；只是候选 |
| `reference_shots` | 真实 Shot ID 数组 | 真正看过、用于建立该布局的镜头 |
| `axis_pair` | `["MICHELLE","TEDDY"]` 等两人数组，或 `null` | 主轴线是**哪两个人**；无足够证据则 null |
| `person` | `MICHELLE` / `TEDDY` / `DON` / `UNKNOWN` | 不确认身份时 UNKNOWN，不凭剧情强指认 |
| `position` | `[x,y]` 或 `null` | 统一俯视平面的粗略归一化坐标，x/y 在 0–1，最多一位小数；不表示真实距离 |
| `posture` | `sitting` / `standing` / `moving` / `unknown` | 肢体状态；只在实际画面支持时填写 |
| `facing_to` | `MICHELLE` / `TEDDY` / `DON` / `OTHER` / `UNKNOWN` | **身体**大致朝向的对象；非 eye gaze |
| `position_quality` | `clear` / `rough` / `unknown` | 相对位置是否有足够画面依据，非几何精度 |
| `layout_status` | `stable` / `changed` / `unknown` | 是否有足够信息判定相对站位稳定/变化 |
| `evidence_shots` | 真实 Shot ID 数组 | 视觉支持来源 |
| `note` | 简短中文字符串 | 未确认内容、假设和限制 |

### 4.2 每个 Shot 的摄影机

| 字段 | 类型／允许的值 | 说明 |
| --- | --- | --- |
| `shot_id` | 真实 Shot ID | 必须覆盖该 Sequence 全部 13 个 Shot |
| `layout_id` | 某个已输出的 `layout_id`，或 `null` | 布局不能确认时 null |
| `camera_position` | `[x,y]` 或 `null` | **同一俯视坐标系**下示意机位；不足以判断时 null |
| `camera_subject` | `MICHELLE` / `TEDDY` / `DON` / `GROUP` / `OTHER` / `UNKNOWN` | 镜头主要拍摄谁；非 gaze |
| `axis_pair` | 两个人物名组成的数组，或 `null` | 当前镜头用来判断 180° 关系的轴线 |
| `axis_reference_shot` | 真实 Shot ID，或 `null` | `axis_side` 比较对象 |
| `axis_side` | `same` / `opposite` / `on_axis` / `unknown` | 相对于 `axis_reference_shot` 的同侧/异侧/轴上/未知 |
| `camera_quality` | `clear` / `rough` / `unknown` | 机位**相对关系**是否有依据，不是标定精度 |
| `layout_status` | `stable` / `changed` / `unknown` | 该镜头是否支持沿用布局 |
| `evidence_shots` | 真实 Shot ID 数组 | 只放实际查看过的视觉证据 |
| `note` | 简短中文字符串 | 简述判断或不确定性 |

若 `axis_side` 为 `same`/`opposite`/`on_axis`，`axis_pair` 和 `axis_reference_shot` 必须有效；在两个镜头未使用相同人物轴线时，不得比较同侧/异侧。

**不要**把 `camera_position` 当成由 VLM 可靠恢复的精确参数。若只知道“从 Teddy 一侧拍 Michelle”，但无法在俯视平面放置具体点，设 `camera_position=null`，把已知关系写在 `note`。

### 4.3 整体状态

- `status`：固定 `model_candidate`。
- `review_status`：新输出固定 `pending`；只有用户实际人工检查后才可另行更新为 `approved` / `rejected`，不能靠 Codex 或 VLM 自评升级。
- 不继承 Stage04.3 的选段审批状态；`seq_006` 本身仍是待审候选，即使已有 Stage05 或旧 Stage06 原型。

## 5. JSON 输出样式

下面只表示字段结构；**不要把示例空值直接当成完成的推断结果**。

```json
{
  "movie_id": "tt12300742",
  "sequence_id": "seq_006",
  "scene_id": "scene_000021",
  "start_s": 1432.013915,
  "end_s": 1547.212332,
  "layouts": [
    {
      "layout_id": "layout_001",
      "start_shot_id": "shot_000211",
      "end_shot_id": "shot_000223",
      "reference_shots": ["shot_000212", "shot_000218", "shot_000222"],
      "axis_pair": null,
      "people": [
        {
          "person": "MICHELLE",
          "position": null,
          "posture": "unknown",
          "facing_to": "UNKNOWN",
          "position_quality": "unknown",
          "evidence_shots": [],
          "note": "待实际查看图片后填入"
        }
      ],
      "layout_status": "unknown",
      "note": "仅展示字段结构"
    }
  ],
  "shots": [
    {
      "shot_id": "shot_000211",
      "layout_id": null,
      "camera_position": null,
      "camera_subject": "UNKNOWN",
      "axis_pair": null,
      "axis_reference_shot": null,
      "axis_side": "unknown",
      "camera_quality": "unknown",
      "layout_status": "unknown",
      "evidence_shots": [],
      "note": "待实际查看图片后填入"
    }
  ],
  "status": "model_candidate",
  "review_status": "pending"
}
```

正式结果必须含**全部 13 个 Shot**，只使用真实输入中存在的 ID 和电影全局时间。结构示例中没有填满的部分不视为已经完成。

## 6. 可视化和人工检查

输出到独立目录，例如 `reports/stage06_1/tt12300742/seq_006/`：

1. `scene_layout.json`：经过可用图片推断的真实候选结果；如果 VLM/图片不可用，**不要写伪造的结果 JSON**。
2. `scene_layout.svg`：依据 JSON 画简单俯视图。人物圆点+姓名；有证据才画身体方向箭头、摄影机及轴线。多个布局分开画。缺机位就不画。显著写 `Approximate layout — not calibrated`。
3. `review.md`：说明 VLM 名称/运行方式、读取的图像数、参考镜头、人物未确认之处，以及全部 13 个 Shot 的机位/轴线/布局状态和需要人工核对的项目。

SVG 只是**原结果的可视化**，不得在绘图步骤为 null 坐标偷偷填默认位置。

验收重点：
- 人物相对站位和同框、OTS 证据是否一致；
- 13 个镜头是否有合理的相机区域/轴线关系或明确 unknown；
- 人物走位/无法判断是否被保留；
- 没有把剧情猜测、camera subject、body facing 当成 gaze target。

## 7. 给 Codex 的实现边界

先阅读仓库 `AGENTS.md`、`docs/pipeline_overview.md`、`docs/stage06_prototype.md` 和本 Prompt，检查已有 Stage01/03/04 数据接口、JPG 路径及真实可用的 VLM 能力。

- 新建**独立** Stage06.1 原型，不覆盖 `src/oscardp/stage06/` 旧情绪模块、`models/stage06/`、`reports/stage06/`、历史 tests 或 Stage01–05 的任何输出。
- 不修改原始电影/图片/字幕/剧本；只读取已有 JSON/JPG。
- 不训练模型、不重建三维、不默认下载大型模型、不添加数据库、复杂哈希/版本管理、任务队列或额外审批流程。
- 优先复用项目现成可用的视觉推理方式。**如果没有真正能看图片的 VLM 调用能力**，实现最小的本地输入准备（图片路径/可查看的联系表、Shot 清单）及已有候选 JSON 的验证/绘图功能，明确报告缺少 VLM 的阻塞；不能假装程序调用了 VLM，也不能拿 Stage03 文本直接冒充本轮图片推断。
- 如 Codex 当前环境确实可以直接检查图像并生成视觉推断，允许按本 Prompt 实际逐镜头完成 Pilot；报告具体查看了哪些文件。
- 使用现有 `uv`/Python 项目约定与可复用工具。除非确有必要，不新添依赖。
- 新结果写到独立目录，绝不覆盖已有已完成输出；如目标目录存在，先报告并换新输出路径或停止。
- 可编写少量必要测试：标签集合、引用的 Shot/Layout ID、坐标范围、全部 Shot 覆盖、axis 同侧判断具备参考镜头。
- 完成后只汇报：修改/新增文件、实际使用的图像和 VLM、生成的 JSON/SVG/review 路径、无法判断的 Shot、如何运行。**不要继续 Stage06.2/06.3**。

**额外说明：** 本 Prompt 是单 Sequence 的 Pilot 规范。若实现可复用的 CLI，运行时应从 Stage04.3 读取 Sequence 信息，而不是把电影 ID、时间或 13 个 Shot 硬编码进核心推断代码；Pilot 专用默认参数可以单独保留。
