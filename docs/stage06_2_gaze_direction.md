# Stage06.2 镜头内头眼方向分析 v1.0

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
  --output-dir "reports/stage06_2/$m/seq_006/shot_000217/pilot_001" --review-frames
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
的整段被保守屏蔽，候选覆盖区间可能包含仍睁眼的帧。双眼夹角超过 12° 时
保留逐眼数值与 `*_qc_rejected` 诊断均值，综合可信列为 null。无效帧、
不连续帧号、源时间不连续或超过 0.25 秒的间隔不做插值。眼信号不足时，
单独可检测的头转候选归为 `uncertain`。

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
