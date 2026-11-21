# Stage06.2 镜头内头眼方向分析 v1.1

Pilot 读取 Stage05.4 主模式 Parquet、帧状态、Track 汇总及同源 Stage05.5 事件。
Stage04.3 提供选段镜头列表，Stage04 的单一人物候选可与同镜头单一 corrected
Track 对应。身份名称是候选。使用现有 `stage05-performance` 的
NumPy、PyArrow、Matplotlib，无需重新执行 OpenFace 或 Stage06.1。

```sh
m=tt12300742
movie_dir=/media/sdu/SiyaoDu/datasets/oscar_movie_analysis/$m
uv --no-cache run --no-sync python -m oscardp.stage06_2 \
  --stage04 "$movie_dir/Stage04_$m.json" --sequences "$movie_dir/Stage04_3_$m.json" \
  --tracking-run "reports/stage05_2_corrected/$m" --identity-run "reports/stage05_3/$m/pilot_001" \
  --measurement-run "reports/stage05_4/$m/pilot_001" --event-run "reports/stage05_5/$m/pilot_001" \
  --output-dir "reports/stage06_2/$m/seq_006/shot_000217/pilot_002" --review-frames
```

默认运行 `seq_006` 的 `shot_000217`。加 `--review-frames` 时用 FFmpeg/FFprobe
从 Stage05 记录的原片按 PTS 抽少量对照帧。重复运行须选择新的输出目录。

以 Stage05.4 记录的 OpenFace 源码
[3d4b5cf8](https://github.com/TadasBaltrusaitis/OpenFace/tree/3d4b5cf8d96138be42bed229447f36cbb09a5a29)
为准。[GetPose](https://github.com/TadasBaltrusaitis/OpenFace/blob/3d4b5cf8d96138be42bed229447f36cbb09a5a29/lib/local/LandmarkDetector/src/LandmarkDetectorFunc.cpp#L56)
用头模的 3D 点做 PnP，得到头模到相机的旋转。
[RotationHelpers.h](https://github.com/TadasBaltrusaitis/OpenFace/blob/3d4b5cf8d96138be42bed229447f36cbb09a5a29/lib/local/Utilities/include/RotationHelpers.h#L47)
给出 `R = Rx @ Ry @ Rz`，按 CSV 弧度直接代入源码符号。
[GazeEstimation.cpp](https://github.com/TadasBaltrusaitis/OpenFace/blob/3d4b5cf8d96138be42bed229447f36cbb09a5a29/lib/local/GazeAnalyser/src/GazeEstimation.cpp#L87)
对模型偏移左乘同一个矩阵，并在相机坐标下计算各眼单位向量。相机轴为 +X 向右、
+Y 向下、+Z 远离相机。头部前向用 `R @ [0,0,-1]`，各眼头相对方向用
`R.T @ g`，综合向量为两单位向量的归一化和。左／右沿用 OpenFace 编号 0／1。
绘图 yaw/pitch 分别为 `atan2(x,-z)` 和 `atan2(y,-z)`，度数随各曲线坐标系
定义。事件头部量是完整 SO(3) 净转角（含 roll），Gaze 量是单位向量净夹角。

保留原始向量、角度、源门槛、眼睑几何代理量和原始时间。Stage05.5 眼闭合候选
的整段被保守屏蔽，候选覆盖区间可能包含仍睁眼的帧。v1.1 把双眼绝对夹角
超过 12° 从硬拒绝改成 `binocular_disagreement` 诊断标记。这是本项目的
启发式提示，其他源门槛与向量有效性检查仍适用。双眼运动增量明显矛盾时，
事件仍归为 `uncertain`。无效帧、不连续帧号、源时间不连续或超过 0.25 秒
的间隔不做插值。眼信号不足时，单独可检测的头转候选归为 `uncertain`。

OpenFace 的 CSV 角度由原始双眼向量均值计算。本模块先归一化每眼向量再融合，
随后以向量计算方向变化。Review 分别报告 CSV 舍入差和融合定义的数值差。
v1.1 图中独立绘制左右眼和综合头部相对轨迹，并标出闭合区间、诊断线与候选事件。

沿用 Stage05.5 时间中值（半径 0.06 s）和滞后阈值。Head、整体 Gaze、
Eye-relative 的高／低种子门槛分别为 15/6、26/10、30/12 °/s。各信号净变化
至少 6°，至少 3 帧和 0.12 s，稳定端取 ≤3°。眼相对变化独立检测，整体
Gaze 只补充种子。双眼局部运动差与抖动门槛写入 JSON。净变化规则可能漏掉
短时、低幅或区间内来回的变化。

输出 `gaze_frames.parquet`、`gaze_events.json`、`gaze_review.png`、
`review.md` 和可选 `review_frames.jpg`。事件使用正片全局秒，开始帧包含，
末端是最后一帧的 source next PTS。`valid_frame_ratio` 是可用于头眼分解的
双眼帧比例。

```sh
uv --no-cache run --no-sync python -m pytest tests/test_stage06_2.py
```

## v1.2 · Eye Position 与稳定方向

复用上面的命令，把输出改为 `pilot_003`，并增加
`--eye-position --track-id shot_000217_track_001`。保留 `--review-frames`
可输出从原片 PTS 核对的眼部叠图。此模式仅用于 Bugonia 的这个 Track。

OpenFace 的每眼局部 0–7 点属于虹膜拟合轮廓（见源码
`GazeEstimation.cpp::GetPupilPosition`），其均值是模型中心代理。
眼角使用眼模配置映射的 8、14 点，按画面 x 排序；两轴均以眼角连线长度
归一化。转换补偿平面内旋转与尺度，无法消除斜视角、非刚性眼睑遮挡或
错误的虹膜拟合。闭合候选、无效 tracking/landmarks/pose、低开度、较大偏航
或拟合中心落出眼睑时，该眼位置为空。可见度字段是 QC 代理。

位置用有效连续段内的局部中值（0.05 s）。位置跨度不超过眼宽的 3% 且
持续至少 0.25 s 才产生眼位稳定候选。相机方向稳定还要求两眼和综合
向量在该段各自变化不超过 3°。相邻稳定段的方向差至少 6°、双眼支持
一致且间隔不超过 0.55 s 才标方向变化候选。不跨无效段比较；明显姿态
变化下的眼位候选可能被降为 unknown。附加头眼运动分类保留在
`motion_events.json` 中供对照。代表帧是未来复核或估计器调用清单，
没有执行新模型或测速。Pilot 保留原约 24 FPS。

## MediaPipe Iris 同帧对照

使用已安装的 `stage06-iris` 环境和本地 `models/mediapipe/face_landmarker.task`。
复用上面命令中的六组 Stage04/05 路径，入口换成
`uv --no-cache run --no-sync python -m oscardp.stage06_2.iris_comparison`，
传入 `--model models/mediapipe/face_landmarker.task` 和
`--output-dir reports/stage06_2/tt12300742/seq_006/mediapipe_comparison_001`。
不需要 Stage06.2 的 `--eye-position`、`--track-id` 或 `--review-frames`。
入口锁定 Bugonia 217 的三个时间窗，校验源视频 PTS 后在完整 RGB 图像上使用
Face Landmarker IMAGE 模式。叠图的琥珀色为 MediaPipe，青色为 OpenFace。
输出 `review.md` 的数值表自动生成，实际原片判读需另行记录。

## v1.3 · MediaPipe 眼位时间线

使用 `oscardp.stage06_2.mediapipe_pilot`，沿用上面同帧对照的模型和六组
Stage04/05 输入参数。另加 `--shot-id shot_000217`，并把输出根改为
`reports/stage06_2/tt12300742/seq_006/mediapipe_pilot_001`。每个镜头写入
独立子目录，已存在的子目录会被拒绝。检查 217 后可用相同命令和模型依次
运行 `shot_000211` 和 `shot_000214`。不要给旧版 Stage06.2 主入口传参。

MediaPipe 使用已纠偏 Stage05.2 观测框，四边各扩展 25%，裁剪到真实图像。
每帧以 Stage05.4 的全局时间和原片 PTS 匹配。模型在 RGB 裁切图上独立运行，
坐标按裁切框的宽、高和原点映射回原帧，再用眼角连线归一化。输出
`left/right` 表示画面左／右，分别对应 MediaPipe 的人物右／左；
x 向画面右，y 向下，两轴均以各自眼角距离为单位。OpenFace 数值仍提供
头姿、AU、相机坐标的 Gaze 和分解后的头相对 Gaze。眼位代理不是三维方向。

QC 检查跟踪关联、眼顺序、眼裂开度、虹膜中心是否落在眼睑内、源闭眼候选及
较大头部偏航（绝对值超过 0.5 rad）。相邻帧移动 ≥0.08 眼宽标跳变，
另一眼没有 ≥0.04 眼宽的同步变化时屏蔽该步的眼位，但保留原拟合点。
这些阈值用来发现明显错误，无法校准遮挡、透视压缩或真实虹膜误差。
事件沿用 v1.2 的 0.05 s 局部中值及 0.03 眼宽稳定容差；新增单眼支持标记
与双眼水平同向检查。缺失或眼支持改变切断比较，单眼候选不能视作双眼确认。
代表帧可定期复核长稳定段，计时包括解码、推理和制图，没有运行替代 Gaze 模型。

## v1.4 · 独立证据、弱眼动与定期代表帧

只从 v1.3 的三个 Parquet 和对应 JSON 重放。无需安装或启动模型，也不解码原片。

```sh
root=reports/stage06_2/tt12300742/seq_006
uv --no-cache run --no-sync python -m oscardp.stage06_2.evidence_replay \
  --source "$root/mediapipe_pilot_001" --output-dir "$root/mediapipe_pilot_002"
```

- 紧凑的 `evidence_frames.parquet` 用真实电影 PTS、原始帧号和上游行号连接旧数值；
  分别记录左右眼位、相机 Gaze、头姿的可用性与拒绝原因。相机 Gaze 按原
  Stage05 门槛重算，并以双眼及综合方向在有效连续段内各自满足 3° 和至少
  0.25 秒为稳定条件。MediaPipe 眼位和头相对分解不是这个条件的前提。
- 严格 Eye Shift 和原 Adjustment 门槛保留。弱眼动检查同眼前后三帧的持续
  水平位移，阈值 0.020 眼宽，首尾时间跨度至少 0.16 秒、至多 0.55 秒。
  已知明显跳变、闭合、跟踪失败或该眼缺失会中断。对侧存在至少 0.010
  眼宽的反向持续位移时拒绝。候选标明支持眼和对侧限制，仍可能受透视及拟合偏差
  影响；不改变独立的方向变化判定。
- 保留状态／事件选帧，并按正片 PTS 增加 0.5 秒检查窗。窗口内优先复用已选帧，
  否则优先用多模态帧。保存可用和不可用模态，明显闭合和跟踪失败不作代表；
  无合格帧的窗口记为 missing。所有文件写入新目录。

## Bugonia seq_006 全段补齐

十三镜头以 Stage04.3 列表为准。211、214、217 的 v1.4 原件和 v1.3 逐帧表
直接复用；保留实验性纠偏待审状态。新增镜头用原 Stage01 边界。本批先以原分辨率
YuNet 跟踪，再只对 213、218、219、220、221 用已有 640 配置补测覆盖不足的部分。
Stage05.3 的匿名聚类按批次命名，不能用相同编号跨批次合并。无足够 ArcFace 样本
的 Track 仍可保留数值，未确认的碎片不送进 OpenFace。

Stage05.2 跟踪命令（使用 Bash）：

```sh
m=tt12300742
analysis=/media/sdu/SiyaoDu/datasets/oscar_movie_analysis
video=/media/sdu/SiyaoDu/datasets/oscar_movie/${m}_Bugonia/$m.mp4
uv --no-cache run --no-sync python -m oscardp.stage05 detect-track \
  --movie-id "$m" --analysis-root "$analysis" --video "$video" \
  --model models/yunet/face_detection_yunet_2023mar.onnx \
  --shot-ids shot_000{212,213,215,216,218,219,220,221,222,223} \
  --output-root reports/stage05_2_seq006_001
uv --no-cache run --no-sync python -m oscardp.stage05 detect-track \
  --movie-id "$m" --analysis-root "$analysis" --video "$video" \
  --model models/yunet/face_detection_yunet_2023mar.onnx --inference-max-size 640 \
  --shot-ids shot_000{213,218,219,220,221} \
  --output-root reports/stage05_2_seq006_fallback_001
```

以下是本批 Stage05.3–05.5 的调用方式（每次运行选择新输出目录）。两组分别为：

- native: 212、215、216、222、223（跟踪目录 `reports/stage05_2_seq006_001/tt12300742`）。
- fallback: 213、218、219、220、221（跟踪目录 `reports/stage05_2_seq006_fallback_001/tt12300742`）。

```sh
# 先设置 tracking、ids、measurement、events 为相应的新目录，
# 将 shell 位置参数设置为一组 shot_000NNN。
m=tt12300742
uv --no-cache run --no-sync python -m oscardp.stage05 associate-identities \
  --movie-id "$m" --stage01-boundaries --source-run "$tracking" --shot-ids "$@" \
  --model models/insightface/models/buffalo_l/w600k_r50.onnx --output-dir "$ids"
uv --no-cache run --no-sync python -m oscardp.stage05 extract-performance \
  --movie-id "$m" --stage01-boundaries --multi-track --tracking-run "$tracking" \
  --identity-run "$ids" --shot-ids "$@" --openface-binary models/OpenFace/build/bin/FeatureExtraction \
  --output-dir "$measurement"
uv --no-cache run --no-sync python -m oscardp.stage05 detect-events \
  --movie-id "$m" --measurement-run "$measurement" --shot-ids "$@" --output-dir "$events"
```

多人 Track 使用同尺寸原坐标的隔离帧，既不平移也不缩放。竞争人脸与目标检测框
重叠时屏蔽该帧；未观测帧不测量。每 Track 的 FFV1 传输帧散列与原帧散列分开，
核对传输往返以及 OpenFace 五点与该 Track 的观测一致性。新 Stage05.4 只测量各
Track 自身时间跨度，Stage05.5 按 Track 独立检测；参数沿用既有默认值。

```sh
m=tt12300742
d=/media/sdu/SiyaoDu/datasets/oscar_movie_analysis/$m
root=reports/stage06_2/$m/seq_006
uv --no-cache run --no-sync python -m oscardp.stage06_2.full_sequence \
  --stage04 "$d/Stage04_$m.json" --sequences "$d/Stage04_3_$m.json" \
  --corrected "reports/stage05_2_corrected/$m" --reuse "$root/mediapipe_pilot_002" \
  --reuse-positions "$root/mediapipe_pilot_001" --model models/mediapipe/face_landmarker.task \
  --batch "reports/stage05_2_seq006_001/$m" "reports/stage05_3/$m/seq006_native_001" \
    "reports/stage05_4/$m/seq006_native_001" "reports/stage05_5/$m/seq006_native_001" \
  --batch "reports/stage05_2_seq006_fallback_001/$m" "reports/stage05_3/$m/seq006_fallback_001" \
    "reports/stage05_4/$m/seq006_fallback_001" "reports/stage05_5/$m/seq006_fallback_001" \
  --output-dir "$root/full_sequence_001"
```

汇总按正片 PTS 排列，保留旧纠偏与原 Stage01 边界之间可能的单帧重叠，
不跨剪辑或 Track 合并角速度、事件或身份。同框 Track 数不等于人数。
