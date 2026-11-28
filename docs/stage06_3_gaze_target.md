# Stage06.3 v1.0 · Cross-shot Gaze Target Inference

本次 Pilot 覆盖 Bugonia `tt12300742`、`seq_006` 的 13 个 Shot。
角色表保存 Stage03/04、Stage05.3 匿名聚类和已检查原片帧的证据。聚类只在各自
批次内解释，边界残留和碎片保留独立 Track。角色和目标均为候选。

每帧按 Track、上游行号和电影全局 PTS 对齐最新 Eye Observability QC。只有 Gaze 可用
掩码改变的 Track 会在内存重算 Stable Gaze。仍使用 Stage06.2 的双眼及均值 3° 容差、
连续至少 0.25 秒条件。没有有效 Gaze 的区间用独立头向，沿用连续段切分并用 12° 头向
稳定容差。眼位、短时眼动和 Blink 作为伴随观测保存。

每个候选镜头的机位网格和旧机位点作为不同假设。以 Stage06.1 的名义取景方向建立
水平投影，另检查 ±12° 取景偏转和归一化平面站位扰动。对每个人分别检查 Gaze 32°、
头向 46° 的水平相容范围，记录相容假设数和角差。这里的比例是敏感度检查；
没有标定实际机位、世界轴向、机身俯仰或演员眼高。命名主候选需两种机位来源
都满足至少 67% 相容（有该来源时），且对另一人有 15° 中位差距。头向保持候选集合，
无可用机位的相似组保留上下文。对白、身体朝向和反打只参与先验。

Hold/Switch 只比较相邻、有命名目标的 Gaze-supported 段。跨镜头还要求两段接近
各自剪辑边界、同一角色有视觉候选支持、同一空间布局。人物不可测的间隔单独输出
`coverage_gaps`。时间戳为正片全局秒。旧实验性边界仍保持待审状态。

## 复现

需要仓库现有的源结果和 Pilot 已检查的原片证据包。可用新的输出目录再次运行。

```sh
seq=reports/stage06_2/tt12300742/seq_006
pilot=reports/stage06_3/tt12300742/seq_006/pilot_001
uv --no-cache run --no-sync python -m oscardp.stage06_3 \
  --layout reports/stage06_1/tt12300742/seq_006_v1_2/scene_layout.json \
  --full-sequence "$seq/full_sequence_001" \
  --eye-qc "$seq/eye_observability_fix_001" \
  --movie-dir /media/sdu/SiyaoDu/datasets/oscar_movie_analysis/tt12300742 \
  --visual-evidence "$pilot/visual_evidence/reviewed_frames.json" \
  --output-dir reports/stage06_3/tt12300742/seq_006/pilot_replay_001
```

## v1.1 · Relative camera orientation

v1.1 直接复放 `pilot_001` 的 230 个区间和角色映射。新约束由实际原片帧逐项记录，
分为同框投影顺序、相对深度、身体视角和重复构图。223 的床轴投影使用既有平面代理；
217、219、221 的相似构图只支持相对观察。机位候选点及网格沿用 Stage06.1，
保留通过筛选的各个分支及示意平面方位范围。Gaze 与头向的角度容差沿用 v1.0。
只要某个备选人物仍有相容假设，就保留该候选。对比表另固定这条规则，
单独比较是否应用新机位过滤。

AWAY 和 OTHER 需区间内独立视觉方向证据。原片中可见的物件和可疑区域有单独的
复核记录。审核帧按原片 PTS 和 Track 有效性从区间内选取，标注首、中、尾抽样角色。
所有候选均待人工复核。CSV 中 `human_target` 留空，允许填写
`MICHELLE / TEDDY / DON / AWAY / OTHER / UNKNOWN / UNCLEAR`。

```sh
base=reports/stage06_3/tt12300742/seq_006
uv --no-cache run --no-sync python -m oscardp.stage06_3.orientation_pilot \
  --baseline "$base/pilot_001" \
  --layout reports/stage06_1/tt12300742/seq_006_v1_2/scene_layout.json \
  --eye-qc reports/stage06_2/tt12300742/seq_006/eye_observability_fix_001 \
  --orientation-evidence "$base/pilot_002/review_contact_sheets/orientation/observations.json" \
  --output-dir "$base/pilot_002_replay"
```

## v1.2 · Event-driven Look States

每条已测 Track 独立分析。以 Stage06.2 稳定方向建立初态，逐项核对 Eye Shift、
Weak Eye Motion、Adjustment 和头眼事件的稳定前后态。复用 Stage06.2 的 3°、
0.25 s 稳定判据和 6° 持续方向差，以及已有头向、连续段和间隔规则。短暂离开后
返回、校验失败、Blink、缺少稳定前后态都留在事件记录中。只有受支持的持续变化
才开新方向状态；缺测段标为 Unknown，可用数据恢复时重新建立状态。向量只在
同一镜头的相机坐标内比较，独立保留头向和实际相机 Gaze。

按 Look State 抽取原片帧。运动前、后帧分别标记，并明确是否在当前状态里。
逐状态的实际画面观察记录在 `visual_evidence/observations.json`，原片索引记录
帧号、全局 PTS、Track 框和散列。目标候选分别保存视觉支持、空间相容性和冲突。
人工表允许 MICHELLE、TEDDY、DON、AWAY、OTHER、UNKNOWN、UNCLEAR；标签留空。

```sh
seq=reports/stage06_2/tt12300742/seq_006
base=reports/stage06_3/tt12300742/seq_006
uv --no-cache run --no-sync python -m oscardp.stage06_3.look_pilot \
  --full-sequence "$seq/full_sequence_001" --eye-qc "$seq/eye_observability_fix_001" \
  --baseline-v1 "$base/pilot_001" --baseline-v2 "$base/pilot_002" \
  --layout reports/stage06_1/tt12300742/seq_006_v1_2/scene_layout.json \
  --observations "$base/pilot_003/visual_evidence/observations.json" \
  --output-dir "$base/pilot_003_replay"
```

若需重新逐帧检查，运行同一入口时省略 `--observations`，输出到临时目录以抽取
PTS 样本，填写逐状态观察后再发布。正式审核入口为
`review/<shot_id>/<track_id 的短编号>/review.md`。

## v1.3 · Gaze Target Hypothesis Evaluation

Bugonia `tt12300742` 的 `seq_006/pilot_004` 直接读取 `pilot_003/look_states.json` 的
44 个状态和 PTS 样本。Look State 的划分、方向、QC 和 Eye Motion 均沿用冻结值。
原片抽样包及每张 JPEG 的散列随运行核对。

- **Visual-only** 读取新的交互视觉观察，记录注视者、候选、可见线索、局限和帧引用。
  检查输入包含原片、Track 范围内的脸和眼区裁切及上下镜头。检查时使用画面中的人物关系，
  未提供数值布局、几何匹配或旧排序。212 的预期目标在实验请求中已知。
- **Geometry-only** 复用 Stage06.1 布局、相对机位约束、Stage06.2 冻结方向及 Track 身份候选。
  沿用 32° Gaze、46° 头向的空间检查。优先假设至少有一个分支达到 67% 相容，且在
  每个已建模分支中的相容样本数不低于对手、角差中位数领先至少 15°。
  这些数值描述示意平面扰动下的相容性。机位、镜头俯仰和演员眼高尚未标定。
- **Fusion** 在持有方向的画面与稳定测量给出同一优先目标时，提高其支持等级。
  如果仅有头向且视觉保留多人并列，融合继续保留该并列。来源分歧时保留双方候选，
  报告首选冲突及候选空间相容性。若稳定 Gaze 的空间证据与纯上下文假设冲突，
  可以优先排序空间假设。缺少测量时，仍可给出画面方向或纯上下文假设。

`preferred_target` 可以为 `null`。候选和排名由 `MICHELLE / TEDDY / DON / AWAY / OTHER`
组成，`target_ranking` 用内层数组表示并列。确定程度还体现在 `evidence_level` 和
`decision_status`。没有足够证据确认时，`selected_target` 为 `UNKNOWN`。视觉确认需至少两张
状态内帧显示独立落点且无竞争候选。几何确认沿用 v1.2 的 Gaze、唯一相容候选和
所有分支的检验，并要求稳定方向及支持的身份映射。融合确认还要求两种方法各自
达到确认条件且命中同一目标。无独立端点依据的 AWAY / OTHER 不进入融合确认。

已保存的交互视觉记录为 `pilot_004/visual_observations.json`。用同一记录生成新的回放目录。

```sh
base=reports/stage06_3/tt12300742/seq_006
uv --no-cache run --no-sync python -m oscardp.stage06_3.hypothesis_pilot run \
  --baseline "$base/pilot_003" \
  --layout reports/stage06_1/tt12300742/seq_006_v1_2/scene_layout.json \
  --orientation "$base/pilot_002/camera_orientation_constraints.json" \
  --observations "$base/pilot_004/visual_observations.json" \
  --output-dir "$base/pilot_004_replay"
```

三份 `*_predictions.json` 各保留全部 44 个状态。 `comparison.md` 给出目标覆盖、
一致与冲突、候选排序变动和全部状态的排序。 `review.md` 链接按 Shot、Track、Look State
组织的 18 个审核样本。可先从 `review/blind.md` 查看原片并填写
`review/blind_labels.csv`，或使用含预测列的 `human_review.csv`。
人工目标可填 `MICHELLE / TEDDY / DON / AWAY / OTHER / UNCLEAR`，确定程度和备注为
审核记录字段。填写后可执行评测。

```sh
base=reports/stage06_3/tt12300742/seq_006/pilot_004
uv --no-cache run --no-sync python -m oscardp.stage06_3.hypothesis_pilot evaluate \
  --pilot "$base" --labels "$base/review/blind_labels.csv"
```

评测单独计数空标签和 `UNCLEAR`，两者均不参与指标。Top-1 Accuracy 的分母是
有效人工标签中有 Preferred 的状态，Prediction Coverage 的分母是全部有效人工标签。
Candidate Recall 检查完整候选集合。准确率–覆盖率按所有 Preferred、排除纯上下文的
Preferred、`supported_hypothesis` 和 Selected 分别统计。没有有效人工标签时只输出标签状态。

## v1.4 · Gaze timeline

v1.3 的审核只看 Look State，短暂转头、瞥视、回避和测不到人脸的段落都没有进入审核表。
v1.4 只请人标注视行为：每个镜头、每个角色一条时间线，一行表示"从 `start_s` 起看向哪里"，
持续到下一行开始。`gaze` 取 MICHELLE / TEDDY / DON / OTHER（看向的人或物），
DOWN / UP / SIDE / AVERT / CLOSED（无明确目标的表演性注视），BLINK（视线延续）或 UNCLEAR。
`start_s` 是眼睛落到新目标的第一帧。头部运动、转移时长和普通眨眼不标，
它们由测量和人工切换时刻计算。连续相同的 `gaze` 自动合并。

机器只提供候选切分点，每个单元预填一行：

- **MOVE**：同一 Track 的头眼运动事件，间隔短于 0.25 s（最短 Look State）的合并为一段。
  排除眨眼、回到原方向的微调（`adjustment_or_same_direction`）和超过 2 s 的单个事件；
  后者多为漂移或小脸头姿噪声，只在所在状态上注明 `long motion`。
- MOVE 把 pilot_003 的状态切成 `look_NNN_p1`、`_p2`…，短于 0.25 s 的边角并入相邻 MOVE。
  状态分为 **HOLD**、**BLINK**（`blink_gap`）和 **UNMEASURED**。整个状态的目标候选只计在含稳定方向锚点的那一段。
- **GAP**：某角色在镜头里没有可测人脸、且长于 0.25 s 的段，列出其中未测量的 Track 碎片。

评测按时间比对：各方法的首选目标与人工时间线重合的秒数（准确率、覆盖率、候选召回），
以及人工切换时刻在 ±0.25 s 内是否有机器切分点或 MOVE 起点。
每个镜头、每个角色输出一个带正片时间、机器单元和注视者框的片段。

```sh
base=reports/stage06_3/tt12300742/seq_006
uv --no-cache run --no-sync python -m oscardp.stage06_3.behavior_pilot run \
  --look-pilot "$base/pilot_003" --hypothesis-pilot "$base/pilot_004" \
  --full-sequence reports/stage06_2/tt12300742/seq_006/full_sequence_001 \
  --unit-labels "$base/pilot_006/review/units.csv" --output-dir "$base/pilot_007"
uv --no-cache run --no-sync python -m oscardp.stage06_3.behavior_pilot evaluate --pilot "$base/pilot_007"
```

不给 `--unit-labels` 时转写 v1.3 盲审标签。审核说明见 `pilot_007/review/README.md`。
审核人最终没有填 `pilot_007` 的时间线，而是在 `pilot_006/review/units.csv` 上直接按 MOVE/HOLD 重标，见下节。

## Human gaze labels · labels_001

seq_006 的 gaze 结果用人工标签，供 Stage07 使用。机器单元只做评测。

| 规则 | 说明 |
| --- | --- |
| 读法 | 只按时间读审核表，忽略 `unit_id` 列（审核人复用了一些行） |
| 结构 | MOVE 在眼睛落点结束，后接同目标的 HOLD；BLINK 单列；眨眼也可作为 MOVE/HOLD 上的标记 |
| 目标 | MICHELLE / TEDDY / DON，或表演性方向 DOWN / UP / LEFT / RIGHT / CLOSED / UNFOCUSED / ROLL（原文保留在 `form_text`，左右按注视者自己的方向） |
| 修正 | MOVE 与同目标 HOLD 之间的空档由 HOLD 提前开始补齐；相邻同目标 HOLD 合并。每处修正写入 `corrections` |
| 不可标 | 看不清眼睛的角色记为 `not_annotatable`（218、222 的 DON） |
| 时间 | 正片秒，左闭右开，约 ±1 帧 |

评测（`evaluation.json`，容差 ±0.25 s）：45 次人工转移中，机器 MOVE 找到 30 次，原始事件找到 37 次。
48 个机器 MOVE 中 23 个对上转移，22 个落在人工 HOLD 内。含头部信号的 MOVE 中位比人工早 0.37 s 开始、
晚 0.75 s 结束，因为头在眼睛落点后还在转。HOLD 目标：fused 准确率 1.0、覆盖率 0.23。
探索性观察（未进评测）：近景里虹膜在眼眶中的横向位置能分开 TEDDY 和 DON，OpenFace 视线角度分不开；
这是后续自动检测的方向。

```sh
base=reports/stage06_3/tt12300742/seq_006
uv --no-cache run --no-sync python -m oscardp.stage06_3.gaze_labels \
  --sheet "$base/pilot_006/review/units.csv" \
  --full-sequence reports/stage06_2/tt12300742/seq_006/full_sequence_001 \
  --units "$base/pilot_007" --output-dir "$base/labels_001" \
  --approval "reviewer confirmed in conversation, 2026-11-27"
```
