"""保留原来的3D分段/ROM算法，并增量加入每次动作的曲线指标。"""

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from motion_metrics import assess_rep, validate_config

ROOT = Path(__file__).resolve().parent
BASE_FIELDS = [
    "rep", "start_time", "bottom_time", "end_time", "left_rom", "right_rom",
    "rom_difference", "bottom_time_difference", "pull_time", "return_time",
]
EXTRA_FIELDS = [
    "rom_difference_ratio", "left_bottom_time", "right_bottom_time",
    "bottom_time_difference_ratio", "sample_count", "max_gap_seconds",
    "data_status", "data_warning", "curve_mae", "pull_curve_mae", "return_curve_mae",
    "synchronization_score", "pull_synchronization_score", "return_synchronization_score",
    "asymmetry_result", "asymmetry_reasons", "rom_balance_score",
    "bottom_timing_score", "quality_score",
]
CURVE_FIELDS = ["rep", "progress_percent", "left_angle", "right_angle",
                "left_normalized", "right_normalized"]


def validate_data(times, left, right):
    if len(times) < 2 or len(times) != len(left) or len(times) != len(right):
        raise ValueError("Need at least two paired time/left/right samples")
    if not all(math.isfinite(v) for values in (times, left, right) for v in values):
        raise ValueError("CSV values must be finite numbers")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("CSV time must be strictly increasing")
    if times[0] < 0 or any(v < 0 or v > 180 for v in left + right):
        raise ValueError("Time must be >= 0; elbow angles must be between 0 and 180")


def load_angle_csv(path):
    times, left, right = [], [], []
    with open(path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"time", "left_3d", "right_3d"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Input CSV needs time, left_3d and right_3d columns")
        for line_number, row in enumerate(reader, 2):
            try:
                values = [float(row[name]) for name in ("time", "left_3d", "right_3d")]
            except (ValueError, TypeError) as error:
                raise ValueError(f"Invalid/missing value on CSV line {line_number}") from error
            times.append(values[0])
            left.append(values[1])
            right.append(values[2])
    validate_data(times, left, right)
    return times, left, right


def moving_average(data, window_size=7):

    result = []

    for i in range(len(data)):

        start = max(
            0,
            i - window_size + 1
        )

        window = data[start:i + 1]

        result.append(
            sum(window) / len(window)
        )

    return result

def percentile(data, percent):

    sorted_data = sorted(data)

    index = int(
        (len(sorted_data) - 1)
        * percent
    )

    return sorted_data[index]

def normalize(value, low, high):

    if high == low:
        return 0

    value = (
        (value - low)
        / (high - low)
    )

    return max(
        0,
        min(1, value)
    )

def prepare_motion(times, left_3d, right_3d, gap_limit, minimum_rom=5.0):
    # 原7帧移动平均、10/90百分位幅值归一化保持不变。
    # 大缺口后清空平滑历史，以免把缺口两边当成连续运动。
    left_smooth, right_smooth = [], []
    block_start = 0
    for i in range(1, len(times) + 1):
        if i == len(times) or times[i] - times[i - 1] > gap_limit + 1e-9:
            left_smooth.extend(moving_average(left_3d[block_start:i]))
            right_smooth.extend(moving_average(right_3d[block_start:i]))
            block_start = i
    left_low = percentile(left_smooth, 0.10)
    left_high = percentile(left_smooth, 0.90)
    right_low = percentile(right_smooth, 0.10)
    right_high = percentile(right_smooth, 0.90)
    left_normalized = [normalize(v, left_low, left_high) for v in left_smooth]
    right_normalized = [normalize(v, right_low, right_high) for v in right_smooth]
    combined = [(left + right) / 2 for left, right in zip(left_normalized, right_normalized)]
    # 百分位归一化也会放大静止姿态的小抖动，先检查是否有足够幅度。
    low_session_motion = max(left_high - left_low, right_high - right_low) < minimum_rom
    if low_session_motion:
        combined = [0.0] * len(times)
    return left_smooth, right_smooth, combined, low_session_motion


def segment_reps(times, combined, gap_limit):
    reps = []

    stage = "waiting"

    last_top_index = None

    rep_start_index = None


    last_rep_time = -999

    for i, value in enumerate(combined):

        # 长时间跟踪丢失：重新等候完整动作，不跨越缺口计数。
        if i > 0 and times[i] - times[i - 1] > gap_limit + 1e-9:
            stage = "waiting"
            last_top_index = None
            rep_start_index = None

        # 顶部
        if value > 0.70:

            if stage == "bottom":

                rep_end_index = i

                # 和原 count_reps_3d.py 相同的 0.5s 防重复规则。
                if times[i] - last_rep_time > 0.5:
                    reps.append((rep_start_index, rep_end_index))
                    last_rep_time = times[i]

                stage = "top"

            else:

                stage = "top"

            last_top_index = i


        # 底部
        elif value < 0.30:

            if (
                stage == "top"
                and last_top_index is not None
            ):

                rep_start_index = (
                    last_top_index
                )

                stage = "bottom"
    return reps


def analyze_data(times, left_3d, right_3d, config=None):
    validate_data(times, left_3d, right_3d)
    config = validate_config(config)
    expected_interval = statistics.median(b - a for a, b in zip(times, times[1:]))
    gap_limit = max(config["maximum_gap_seconds"],
                    config["gap_interval_multiplier"] * expected_interval)
    left_smooth, right_smooth, combined, low_session_motion = prepare_motion(
        times, left_3d, right_3d, gap_limit, config["minimum_rom_degrees"])
    reps = segment_reps(times, combined, gap_limit)

    analysis_results = []


    for rep_number, (
        start_index,
        end_index
    ) in enumerate(
        reps,
        start=1
    ):

        left_segment = (
            left_smooth[
                start_index:
                end_index + 1
            ]
        )

        right_segment = (
            right_smooth[
                start_index:
                end_index + 1
            ]
        )

        combined_segment = (
            combined[
                start_index:
                end_index + 1
            ]
        )


        # -------------------------
        # 左右运动幅度 ROM
        # -------------------------

        left_max = max(
            left_segment
        )

        left_min = min(
            left_segment
        )

        right_max = max(
            right_segment
        )

        right_min = min(
            right_segment
        )


        left_rom = (
            left_max - left_min
        )

        right_rom = (
            right_max - right_min
        )

        rom_difference = abs(
            left_rom - right_rom
        )


        # -------------------------
        # 找左右最低点
        # -------------------------

        left_min_local = (
            left_segment.index(
                left_min
            )
        )

        right_min_local = (
            right_segment.index(
                right_min
            )
        )


        left_bottom_index = (
            start_index
            + left_min_local
        )

        right_bottom_index = (
            start_index
            + right_min_local
        )


        left_bottom_time = (
            times[left_bottom_index]
        )

        right_bottom_time = (
            times[right_bottom_index]
        )


        bottom_time_difference = abs(
            left_bottom_time
            - right_bottom_time
        )


        # -------------------------
        # 整体动作最低点
        # -------------------------

        bottom_local_index = (
            combined_segment.index(
                min(combined_segment)
            )
        )

        bottom_index = (
            start_index
            + bottom_local_index
        )


        start_time = (
            times[start_index]
        )

        bottom_time = (
            times[bottom_index]
        )

        end_time = (
            times[end_index]
        )


        pull_time = (
            bottom_time - start_time
        )

        return_time = (
            end_time - bottom_time
        )


        analysis_results.append(dict(zip(BASE_FIELDS, [
            rep_number,
            start_time,
            bottom_time,
            end_time,
            left_rom,
            right_rom,
            rom_difference,
            bottom_time_difference,
            pull_time,
            return_time
        ])))

    normalized_curves = []
    for result, (start, end) in zip(analysis_results, reps):
        metrics, curves = assess_rep(
            times[start:end + 1], left_smooth[start:end + 1], right_smooth[start:end + 1],
            result["bottom_time"], config, expected_interval,
        )
        result.update(metrics)
        normalized_curves.extend({"rep": result["rep"], **row} for row in curves)
    return analysis_results, normalized_curves, {
        "times": times, "left_smooth": left_smooth, "right_smooth": right_smooth,
        "combined": combined, "reps": reps,
        "gap_limit_seconds": gap_limit,
        "low_session_motion": low_session_motion,
        "tracking_gap_count": sum(b - a > gap_limit + 1e-9 for a, b in zip(times, times[1:])),
    }


def save_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(input_path, output_dir, config=None):
    config = validate_config(config)
    times, left, right = load_angle_csv(input_path)
    results, curves, motion = analyze_data(times, left, right, config)
    output_dir = Path(output_dir)
    save_csv(output_dir / "rep_analysis.csv", BASE_FIELDS + EXTRA_FIELDS, results)
    save_csv(output_dir / "normalized_curves.csv", CURVE_FIELDS, curves)
    # 保存评分配置，展示结果可以追溯到具体规则。
    (output_dir / "analysis_settings.json").write_text(json.dumps({
        "input_csv": str(Path(input_path).resolve()), "sample_count": len(times),
        "rep_count": len(results), "config": config,
        "tracking_gap_count": motion["tracking_gap_count"],
        "gap_limit_seconds": motion["gap_limit_seconds"],
        "low_session_motion": motion["low_session_motion"],
        "segmentation": "7 sample trailing mean; 10/90 percentiles; top>0.70, bottom<0.30; cooldown>0.5s",
        "quality_formula": "sync_weight*sync + rom_weight*ROM_balance + bottom_weight*bottom_timing",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"检测到动作次数: {len(results)}")
    if motion["low_session_motion"]:
        print("整体运动幅度不足：不将静止姿态的小抖动计为动作")
    if motion["tracking_gap_count"]:
        print(f"跟踪缺口: {motion['tracking_gap_count']}（不跨缺口计数）")
    for row in results:
        score = row["quality_score"]
        score_text = "无法可靠评估" if score is None else f"{score:.1f}/100"
        print(f"Rep {row['rep']:2}: ROM L={row['left_rom']:.1f}° R={row['right_rom']:.1f}°; "
              f"quality={score_text}; {row['asymmetry_result']} "
              f"{row['asymmetry_reasons'] or row['data_warning']}")
    print(f"结果已保存: {output_dir}")
    return results, curves, motion


def main():
    parser = argparse.ArgumentParser(description="PoseCoach offline rep analysis")
    parser.add_argument("--input", type=Path, default=ROOT / "lat_pulldown_2d_3d.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, help="JSON overrides for demo thresholds/weights")
    parser.add_argument("--plots", action="store_true", help="Save headless PNG visualizations")
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
        if config is not None and not isinstance(config, dict):
            raise ValueError("Config JSON must be an object")
        results, curves, motion = run_analysis(args.input, args.output_dir, config)
        if args.plots:
            from visualize_reps import save_visualizations
            save_visualizations(results, curves, motion, args.output_dir)
    except (OSError, ValueError) as error:
        parser.exit(1, f"PoseCoach: {error}\n")


if __name__ == "__main__":
    main()
