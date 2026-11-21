# OscarDP Stage06.1 — Scene Layout Prompt v1.2

> **v1.2 改动**：继承 v1.1 的布局、机位点、camera_group 与审查状态，新增「静态布局假设」与「基于可见空间线索的摄影机候选区域（Camera Region）」。**目标不是提高有坐标的镜头数量，而是让原本机位未知的特写尽可能获得有证据的区域约束。**
>
> **用途**：Codex 实现指令 + VLM 分析规范。先针对 Bugonia `seq_006` 建立大致场景布局和各镜头机位；后续 Stage06.2/06.3 才使用眼头方向估计注视对象。
>
> **只做空间信息。** 不做 gaze direction、gaze target、gaze behaviour、情绪或表演意图判断。本文件是新 Stage06.1 的规范，**不是**旧 `src/oscardp/stage06/`（现规划为 Stage07）的修改授权。

## 1. 目标和原则

对一个已经选出的电影 Sequence，实际查看多个 Shot 的 Early/Middle/Late 图片，输出：
1. **人物大致站位**：谁坐在哪里、谁站在谁附近、身体大致面向哪边；
2. **摄影机大致机位**：每个 Shot 从哪个区域拍摄、主要拍摄谁、与参考人物轴线的关系；
3. **布局是否变化**：如果有人明显走位或镜头轴线发生变化，标出变化；无法判断就用 `unknown`。
4. **复用重复机位**：识别构图、透视、背景固定物与演员视角非常相似的不同 Shot，并归入候选 `camera_group`。同组不表示摄影机坐标完全一致。
5. **Camera Region**：在已知人物相对位置的同一示意坐标系中，结合躯干视角、同画面前后景、屏幕排列和明确的轴线信息，推断摄影机的**候选区域**；不能据此宣称精确位姿或真实距离。

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

D. 识别明显走位。如果人物的相对站位发生变化，分成新的 `layout`；若无法确认变化，保持 `unknown`，**不要自动认定整段 Sequence 人物始终静止**。如果只有镜头变化但人物未变化，则尽量复用一个 `layout`。本轮 Bugonia v1.2 另有一项**明确的研究工作假设**：在没有相反视觉证据时，可以暂时沿用三人相对站位不变的 `layout_001`，但必须记录 `assumed_static`，不能改写成已经视觉证明的 `stable`。

E. 检查重复机位：优先比较 `shot_000213/000217/000219/000221` 的 Michelle 特写，及 `shot_000214/000220` 的 Teddy 特写。它们只是**待验证的机位分组候选**，不要直接当成相同机位。结合背景固定物、透视、机位高度和人物侧面角度判断；只有图像证据足够时才使用相同 `camera_group`。人物名字相同或镜头相邻**不够**；单纯裁切/焦距改变与真实机位变化要允许不确定。

F. 提取**实际画面支持的最小机位线索**：
- 人物躯干/肩膀是正面、斜前、侧面还是背面（`view_of_torso`），**不是脸或眼球朝向**；躯干不可见或转动明显时输出 `unknown`。
- 同一 Shot 中，人物 A 是否明显遮挡人物 B / A 在 B 前面（`depth_order`），不要仅凭跨镜头景别或画面占比判定距离。
- 同一 Shot 中可见两个人物的画面左右次序（`screen_order`）。
- 现有轴线关系与 `camera_group` 作为额外、可放弃的软约束。

G. 将这些信息与人物示意坐标一起，用简单的二维网格/扇区**计算 Camera Region**。只保留与有依据的线索相容的网格区域，不要求单一连通多边形。若线索相互矛盾、镜头缺少足够躯干线索或区域超出当前示意范围，应标记 `null` 并解释；**绝不自动在可行区域中心填一个假的 `camera_position`**。

H. 对无法从画面支持的布局、身体朝向、机位和人物身份用 `null` 或 `unknown`，并用一两句中文指出缺什么证据。不要为美观填假坐标。简单保留证据 Shot ID，不做复杂数值打分。

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
| `layout_id` | 某个已输出的 `layout_id`，或 `null` | 可在显式 Pilot 静态假设下引用布局，但不能因此宣称已观测稳定 |
| `layout_basis` | `observed` / `assumed_static` / `unknown` | 直接视觉支持 / Pilot 静态布局假设 / 没有依据 |
| `camera_position` | `[x,y]` 或 `null` | **同一俯视坐标系**下示意机位；不足以判断时 null |
| `camera_clues` | 见 §4.3 | VLM 从真实图片提取的最小空间线索 |
| `camera_region` | 见 §4.3，或 `null` | 合法网格候选区域，不是摄影机坐标真值 |
| `camera_group` | `camera_001` 等，或 `null` | 可复用的候选相似机位组；**允许无坐标仍有分组** |
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

### 4.3 摄影机线索与候选区域（新增）

本轮沿用 v1.1 的布局示意坐标。对 Bugonia `seq_006`，若用户采用「三人没有明显位移」作为 Pilot 工作假设，输出根字段 `layout_assumption: "static_for_pilot"`。它**不是** `review_status=approved`，不是已经对 115 秒连续视频逐帧核实，也不自动改变已有 `layout_status`。

每个 Shot 新增 `layout_basis`：
- `observed`：该 Shot/参照镜头有直接可见的空间支持；
- `assumed_static`：复用了同一布局，但本 Shot 不足以直接验证；
- `unknown`：无法沿用或不属于该布局。

`layout_id` 可以在 `assumed_static` 下引用 `layout_001`。当出现实际可见的明显走位/不同场景，不能继续套用静态假设，应创建新布局或返回 null。

每个 Shot 的 `camera_clues` 为：

| 字段 | 允许的值 | 规则 |
| --- | --- | --- |
| `view_person` | MICHELLE / TEDDY / DON / UNKNOWN | 可以观察到躯干的主要人物 |
| `view_of_torso` | `front`, `front_left`, `front_right`, `left_side`, `right_side`, `back_left`, `back_right`, `back`, `unknown` | **以人物自身的身体左/右为准**，不是屏幕左右；不能由头部朝向替代 |
| `depth_order` | `[{"nearer":"TEDDY","farther":"DON"}]` 或 `[]` | 仅限**同一 Shot** 可靠前后景/遮挡证据；人物不应相同 |
| `screen_order` | `[{"left":"MICHELLE","right":"TEDDY"}]` 或 `[]` | 仅限**同一 Shot** 可见两人；不是世界坐标中的左右 |
| `camera_height` | `low` / `eye_level` / `high` / `unknown` | 相对于主体的机位高度；**不作为俯视位置约束** |
| `note` | 简短中文 | 指明来源画面和线索是否可能误判 |

`camera_region` 类型为 `null` 或下面的结构：

```json
{
  "grid_step": 0.1,
  "cells": [[0.2, 0.7], [0.3, 0.7]],
  "region_quality": "rough",
  "used_clues": ["view_of_torso", "depth_order"],
  "note": "示意坐标系中的候选网格，不代表真实米制摄影机位置。"
}
```

- `cells` 是**符合线索的网格中心**，不是精确摄影机位置；可以不连通。`grid_step` Pilot 固定 0.1；至少一格才输出结构，否则 `null`。
- `region_quality`：`clear` / `rough` / `unknown`；不要把几何网格可行性误称为真实机位测量精度。
- `used_clues` 只允许 `view_of_torso`, `depth_order`, `screen_order`, `axis_side`。保留参与约束的线索名称。
- 如果只有 `camera_subject`、`camera_height`、人物台词或弱 `camera_group`，没有足够区分性的视觉线索，则 `camera_region=null`。
- `view_of_torso` 只有在躯干可见且 `facing_to` 能可靠提供身体方向时才可用。不能把 Michelle 的正脸方向直接当作躯干朝向，不能把 `facing_to=TEDDY` 当作 gaze target。
- `depth_order` 更适合从明确前后景/遮挡建立**近于谁**的约束；只看一人脸在画面内更大，不构成跨 Shot 真实深度约束。
- `screen_order` 应按候选摄影机及画面拍摄方向进行粗略投影判断，必须让代码说明正反方向约定；无可用拍摄朝向时该条线索不得硬筛。
- `axis_side` 只有轴线对与参考 Shot 都有效、且该参考机位已有可解释的同侧关系时才用于网格筛选。不能借 180° 法则默认所有镜头同侧。
- 同一 `camera_group` 可作区域一致性的**复核提示**；但组内只是 rough 时，不自动求交也不把某个 Shot 的区域强行复制给其他 Shot。
- 统一示意网格限定在目前布局坐标的 [0,1] × [0,1]；如果线索指向图外，必须明确这是**示意视域不足**，不得把网格外的真实机位可能性当成几何矛盾，更不能虚构边界外位置。
- 线索产生冲突时，记录冲突并把 `camera_region=null` 或放宽最弱线索；不能偷偷丢掉互相矛盾的输入后宣布高置信度结果。

以上是一个粗略候选区域的**几何可行性检查**，不是 camera pose estimation 的真实标定。不要上来设计优化器、SLAM 或大规模不确定性校准。

### 4.4 重复机位组 `camera_groups[]`

该字段用于跨镜头共享**粗略机位方向关系**，而非确定精确摄影机位置：

| 字段 | 类型／允许的值 | 说明 |
| --- | --- | --- |
| `camera_group` | `camera_001`、`camera_002` 等 | 候选分组 ID |
| `shot_ids` | 真实 Shot ID 数组 | 至少 2 个镜头才建立分组；单镜头的 `shots[].camera_group=null` |
| `reference_shot` | 属于本组的真实 Shot ID | 分组参考 |
| `group_quality` | `clear` / `rough` / `unknown` | 对机位相似性的把握程度 |
| `evidence_shots` | 实际看过的 Shot ID 数组 | 图像证据 |
| `note` | 简短中文 | 相似的固定背景、视角等；以及不能确认的部分 |

同组的 `camera_position` **不要求自动继承**。如果无法定位，只填 `camera_group` 并保留 `camera_position=null`。不同镜头属于同组时，可以复用“相似拍摄方向”作为后续 Stage06.3 的候选上下文，**不能把同组视为摄影机真实外参相同**。

`camera_subject` 仅描述画面拍摄对象。画机位箭头时，仅当摄影机和被拍对象均有可信的示意坐标，才计算并画“摄影机 → 被拍对象”的箭头；否则在图例列出“`camera_001` 看向 MICHELLE”之类的文字，**不在地图上捏造箭头起点**。

### 4.5 整体状态

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
  "layout_assumption": "static_for_pilot",
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
  "camera_groups": [],
  "shots": [
    {
      "shot_id": "shot_000211",
      "layout_id": "layout_001",
      "layout_basis": "assumed_static",
      "camera_position": null,
      "camera_clues": {
        "view_person": "UNKNOWN",
        "view_of_torso": "unknown",
        "depth_order": [],
        "screen_order": [],
        "camera_height": "unknown",
        "note": "待检查真实图片"
      },
      "camera_region": null,
      "camera_group": null,
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

正式结果必须含**全部 13 个 Shot**，只使用真实输入中存在的 ID 和电影全局时间。结构示例中没有填满的部分不视为已经完成。`camera_groups` 未建立时用空数组；不要把这份样例当真实推断。新增的 `layout_basis=assumed_static` 是**明确的 Pilot 假设**，不能默认为验证通过。

## 6. 可视化和人工检查

输出到独立目录，例如 `reports/stage06_1/tt12300742/seq_006/`：

1. `scene_layout.json`：经过可用图片推断的真实候选结果；如果 VLM/图片不可用，**不要写伪造的结果 JSON**。
2. `scene_layout.svg`：依据 JSON 画简单俯视图。人物圆点+姓名；有证据才画身体方向箭头、摄影机及轴线。多个布局分开画。**位置已知且拍摄对象位置已知时绘制从相机指向对象的拍摄方向箭头**；否则不画假箭头。对于只有 `camera_group`、没有坐标的镜头，**在图旁单独列出分组及 Shot ID**，不要假画在俯视图中。v1.2 要以**浅色半透明网格/区域**绘制 `camera_region.cells`，按 camera_group 或 Shot ID 标注；若多区域重叠要能区分。空区域不作图，不把区域中心当成真实摄影机位置。显著写 `Approximate layout — not calibrated`。
3. `review.md`：说明 VLM 名称/运行方式、读取的图像数、参考镜头、人物未确认之处，以及全部 13 个 Shot 的机位/轴线/布局状态和需要人工核对的项目。**逐组解释 `camera_groups[]` 为什么是相似机位；明确哪些只是粗略候选**。v1.2 对每个 Shot 加入 `layout_basis`、可用空间线索、候选区域是否存在、哪项线索冲突或被排除，并区分「看过图像」和「假设沿用站位」。

SVG 只是**原结果的可视化**，不得在绘图步骤为 null 坐标偷偷填默认位置。

验收重点：
- 人物相对站位和同框、OTS 证据是否一致；
- 13 个镜头是否有合理的相机区域/轴线关系或明确 unknown；
- 相似机位是否可以正确分组，且无定位机位可通过旁边列表检查；
- 特写的布局关联是否不再与机位坐标绑定；Camera Region 是否有真实可见的证据，以及空区域能否老实保留；
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
- 可编写少量必要测试：标签集合、引用的 Shot/Layout/Camera Group ID、坐标范围、全部 Shot 覆盖、axis 同侧判断具备参考镜头、camera group 至少 2 Shot、不重复引用、示意图不为 null 机位画点。v1.2 另测：`layout_basis` 枚举、`camera_clues` 合法性、网格中心边界/重复单元、没有约束不能凭空生成区域、null 机位不能被 SVG 捏造为点、旧 v1.1 JSON 仍能正常读取。
- 完成后只汇报：修改/新增文件、实际使用的图像和 VLM、生成的 JSON/SVG/review 路径、无法判断的 Shot、如何运行。**不要继续 Stage06.2/06.3**。

**额外说明：** 本 Prompt 是单 Sequence 的 Pilot 规范。若实现可复用的 CLI，运行时应从 Stage04.3 读取 Sequence 信息，而不是把电影 ID、时间或 13 个 Shot 硬编码进核心推断代码；Pilot 专用默认参数可以单独保留。


## 8. v1.0 Pilot → v1.1 最小迁移要求

已有 Bugonia `seq_006` 第一轮结果（人工核对前）实际呈现：13 个 Shot 中只有 3 个 `camera_position` 非空（`000212`、`000218`、`000222`）；`000215`、`000223` 虽无坐标但有相对 222 的同轴侧候选信息。**这不是程序错误，也不是要求把剩下的 10 个位置硬补齐。**

Codex 应只做以下增量工作：
1. **沿用**原来已形成的人物布局和现有机位坐标，不因为新字段而自动重算覆盖；
2. 让 VLM 实际比较可能重复的 CU 组（先看 `213/217/219/221` 和 `214/220`），按可见证据标 `camera_group` 或 `null`；不做自动“同人物 = 同机位”的规则；
3. 新 JSON 中加入 `camera_groups[]`、各 `shots[].camera_group`。保持兼容旧结果：旧文件若没有新字段，读取时按空列表/null 处理；
4. SVG 给有位置依据的机位画面向拍摄对象的箭头；对无法定位但可分组的镜头，在图旁列出组成员，不编造坐标；
5. 更新 `review.md`，保留旧推断及其不确定性；明确这次新增了哪些分组、依赖哪些图像；
6. 优先允许**直接读取第一轮 JSON 并做增量更新**，无需从零重跑全部空间 VLM。不要覆盖老 Pilot 结果，输出到新的 `seq_006_v1_1` 等独立目录；
7. 只修小范围代码和必要的验证/测试。不要引入 VGGT、MASt3R、COLMAP、Gaze-LLE 或 Stage06.2/06.3。

达成条件：镜头相似性分组可供下一阶段使用；图能表达已知机位及未定位的组；Unknown 不被消除。**不以摄影机坐标数量增加作为质量指标。**


## 9. v1.1 → v1.2 Pilot 增量迁移（必须执行）

**已检查的真实 v1.1 结果：** `layout_001` 中 MICHELLE 为 `[0.2,0.5]`，TEDDY 为 `[0.8,0.5]`，DON 为 `[0.6,0.3]`；`layout_status=unknown`。现有显式摄影机坐标仅 `shot_000212=[0.3,0.7]`、`shot_000218=[0.6,0.8]`、`shot_000222=[0.4,0.9]`，其余为 null。复用 `camera_001: 217/219/221` 和 `camera_002: 214/220`，原先 213 不被强行并入 001。上述数据只是**既有模型候选**，不是人工真值。

执行顺序：
1. 先对照本地现有 v1.1 JSON 和真实联系表/原图，**不改写**既有的 3 个机位、人物坐标、2 个 camera group 或原来的轴侧判断。新文件只能添加字段、附加候选结果与适当的假设链接；旧结果完整保留。
2. 将 `static_for_pilot` 作为**可选且显式启用**的单 Sequence 工作假设；因此原先无 `layout_id` 的特写可附上 `layout_001` 且 `layout_basis=assumed_static`；原有 `layout_status`、`review_status` **不能因此升级**。其他场景或批量电影默认不得自动采用静态假设。
3. VLM 复核有条件的 CU：尤其 `211,213,214,216,217,219,220,221`，另复核与其相关的同框/背景参照 `212,218,222`。**不保证**每个镜头都能提取可靠 `view_of_torso`。已经看过的 JPG 可复用其实际记录，新的推断只认真实看到的图像证据。
4. 提取最小 `camera_clues`，运行可复现的简易网格过滤逻辑；当某一条线索不能转换为可靠几何限制，就放弃该条，不偷偷补虚假信息。网格只是对现有示意布局的一致性检查，不是真实三维场景重建。
5. 把 `camera_region` 绘制成浅色区域，保留原始 3 个机位点，保留未定位的 Camera Group 列表。若重画引起 SVG 过于杂乱，可按组或按 Shot 分开生成简单示意，不添加复杂前端。
6. 新结果写到 `reports/stage06_1/tt12300742/seq_006_v1_2/` 或其他新路径，不覆盖 v1.0/v1.1。更新 `docs/stage06_1_scene_layout.md` 增加 v1.2 的运行命令及输出约定；不修改旧 Stage06 情绪代码。
7. 验收不看“非空 `camera_position` 数量”，而看机位区域约束有无来源、是否与正反打/遮挡一致，以及未知位置能否保持未知。不要引入 VGGT、MASt3R、COLMAP、Gaze-LLE，也不推进 Stage06.2/6.3。
