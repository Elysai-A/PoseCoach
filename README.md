# PoseCoach

# PoseCoach

一个基于 MediaPipe Pose 的智能健身动作分析系统。
本项目针对高位下拉动作，实现人体姿态提取、动作计数、左右协调分析以及动作质量可视化评价。

## 最快演示：使用仓库已有 CSV

建议使用 Python 3.12，新建虚拟环境，然后在项目目录运行：

```bash
python -m venv .venv
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
python analyze_reps_3d.py --output-dir demo_output --plots
python compare_rep_shape.py --rep 1 --output demo_output/rep_1.png
python -m unittest discover -s tests -v
```

已有 `lat_pulldown_2d_3d.csv` 包含 959 条有效角度记录，离线结果为 **15 次完整动作**。运行 CSV 分析只需要 NumPy；画图额外需要 Matplotlib，不需要模型、摄像头或 MediaPipe 推理。测试包含视频组件，需要全部依赖。

输出：

- `rep_analysis.csv`：原来的10列指标 + 同步评分、不对称原因、质量分和数据状态。
- `normalized_curves.csv`：每个 rep 101 个点，进度 0%～100%，包含左右角度和幅值归一化曲线。
- `analysis_settings.json`：记录输入路径、配置、计数、缺口数，便于解释与复现。
- `angle_timeline.png`：左右肘角时间曲线、分段区间和计数阈值。
- `rep_scores.png`：逐次动作分数及组成，`!` 表示建议检查的动作。
- `reps/rep_001.png` 等：单次动作角度/归一化曲线，含整体、下拉和回程同步分。

不加 `--output-dir` 时，分析脚本仍写项目目录中的 `rep_analysis.csv`，会覆盖旧分析快照。推荐演示时使用单独输出目录。仓库原来的三个 CSV 样本保留不变。

## 分析自己的录制视频

原始 `lat_pulldown_neutral.mp4` 和模型未随仓库提供。将自己的视频放进项目目录，下载官方模型到 `pose_landmarker_lite.task`：

<https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task>

```bash
python lat_pulldown_3d_demo.py --video lat_pulldown_neutral.mp4 --model pose_landmarker_lite.task --output demo_output/video_angles.csv --headless
python analyze_reps_3d.py --input demo_output/video_angles.csv --output-dir demo_output/video_report --plots
```

移除 `--headless` 可以在自己电脑上显示视频窗口。`compare_rep_shape.py --show` 可显示单次曲线窗口。原来的 `pose_demo.py` 和 `test_camera.py` 保留摄像头功能，必须在有本地摄像头的电脑上验证。

Linux 上 MediaPipe 原生库可能还需要系统图形运行库，即使选择 CPU / headless 也可能需要：

```bash
sudo apt-get update
sudo apt-get install libgles2 libegl1 libgl1
```

这些是 Debian/Ubuntu 命令；Windows/macOS 不要运行它们。Python 依赖版本是在云端 Linux + Python 3.12 环境实际使用的版本，自己电脑仍需核对安装与窗口/摄像头是否正常。不要同时安装 `opencv-python`、`opencv-contrib-python` 和 headless 变体，它们共享 `cv2` 命名空间。本项目使用 `opencv-contrib-python`。

在只读的云端 home 下，可设置 `MPLCONFIGDIR` 为可写目录（例如项目的 `work/mpl`）；本地正常用户目录一般不需要。

## 数据流与入口

```text
录制视频
  → lat_pulldown_3d_demo.py（MediaPipe关键点 → 2D/3D肘角 → 角度CSV）
  → analyze_reps_3d.py（7帧平滑 → 分段 → 原ROM/时长 → 新指标）
      → motion_metrics.py（统一进度曲线与评分）
      → CSV / 配置JSON
      → visualize_reps.py（PNG）
```

现有 CSV 是不依赖视频素材的演示入口。`count_reps_3d.py`、`plot_angles.py`、`plot_2d_3d.py` 仍可按原方法独立使用；新功能以 `analyze_reps_3d.py` 的结果为准。旧计数脚本没有新增的跟踪缺口保护。

## 评分规则（演示用，尚未用人工标签校准）

1. 每个 rep 共用时间进度 `p = (t - start) / (end - start)`，插值到101个点。
2. 每侧幅值 `(angle - min) / ROM`，保留真实时间差，不分别对齐左右最低点。
3. 曲线平均绝对差 `E` 是曲线间面积除以进度长度，范围0～1；同步分 `S = 100 × (1 - E)`。下拉和回程使用同一个整体最低点划分，分别计算。
4. ROM 相对差 `R = |ROM_L - ROM_R| / max(ROM_L, ROM_R)`。
5. 最低点相对时差 `T = |t_L - t_R| / rep_duration`。
6. ROM 平衡分 `B = 100 × (1 - R)`；最低点时序分 `D = 100 × max(0, 1 - T/0.25)`。
7. 最终质量分 `Q = 0.50S + 0.30B + 0.20D`。名称保留 quality，含义是当前可测的左右协调质量，不能证明完整的高位下拉技术正确。

默认不对称提示：`R > 0.20`、`T > 0.10`、整体 `E > 0.15`、下拉或回程 `E > 0.20`，任一条件满足即提示，所有原因分别保存。同步分主要表示曲线形状/时序相似程度；ROM差单独评估。两个分量可能同时受延迟影响，所以它们不是独立统计量。

50/30/20 是明确的设计权重：整体过程占主要比重，ROM单独反映幅度，最低点反映可直接解释的关键事件。25%、20%、10% 等是可调整的演示阈值，没有把它们当成教练或医学标准，也没有使用机器学习。下拉/回程时长保留展示，没有已校准的目标节奏，因此不凭空加入快慢惩罚。

可用一个 JSON 文件修改阈值（其余保留默认值）：

```json
{"rom_difference_ratio": 0.25, "bottom_time_ratio": 0.10}
```

```bash
python analyze_reps_3d.py --config settings.json --output-dir demo_output --plots
```

## 数据不足与分段边界

- 原分段语义保留：最后一个顶部样本开始下拉，首次重新越过顶部阈值结束；不是解剖意义上的完整伸展平台。
- 输入必须有 `time,left_3d,right_3d`，时间递增、值有限，肘角在0～180°；错误输入会清楚报错，不自动猜测或修复。
- 全程左右10%～90%百分位幅度都不足5°时，不把小范围跟踪抖动归一化成完整动作；配置JSON中记录 `low_session_motion`。
- 不完整的首尾动作不计数，低于0.5秒的重复完成事件按原计数规则抑制。
- 大于 `max(0.20秒, 3×全视频中位采样间隔)` 的缺口会重置分段和平滑历史，不跨越丢失区间制造动作。
- 单次动作不足5点、任一侧ROM不足5°、有过大缺口或最低点不在内部时，状态为 `insufficient_data`，评分留空，不能当成完美同步。
- MediaPipe提取器仍只记录左右都可靠的帧。开始/结尾遗漏或连续丢失时，不能恢复看不到的动作。等间隔缺帧可能无法仅由CSV识别。


详细的初始检查、逐文件变更、测试边界和公式解释见 `DELIVERY_REPORT_ZH.md`。
