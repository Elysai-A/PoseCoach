"""每次动作的曲线比较。只使用插值、平均差和明确的评分规则。"""

import numpy as np


# 演示用启发式阈值，不是医学标准。可用 --config JSON 修改。
DEFAULT_CONFIG = {
    "rom_difference_ratio": 0.20,
    "bottom_time_ratio": 0.10,
    "curve_mae": 0.15,
    "phase_mae": 0.20,
    "minimum_rom_degrees": 5.0,
    "minimum_samples": 5,
    "maximum_gap_seconds": 0.20,
    "gap_interval_multiplier": 3.0,
    "bottom_score_zero_ratio": 0.25,
    "sync_weight": 0.50,
    "rom_weight": 0.30,
    "bottom_weight": 0.20,
}


def validate_config(overrides=None):
    config = DEFAULT_CONFIG.copy()
    if overrides:
        unknown = set(overrides) - set(config)
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        config.update(overrides)
    for key, value in config.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number")
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if config["minimum_samples"] < 3 or int(config["minimum_samples"]) != config["minimum_samples"]:
        raise ValueError("minimum_samples must be an integer >= 3")
    if abs(sum(config[k] for k in ("sync_weight", "rom_weight", "bottom_weight")) - 1) > 1e-9:
        raise ValueError("Quality weights must sum to 1")
    return config


def normalize_motion_curve(times, angles, samples=101):
    """统一时间进度，再统一幅值。左右必须使用相同的 times！"""
    times = np.asarray(times, dtype=float)
    angles = np.asarray(angles, dtype=float)
    if len(times) < 2 or len(times) != len(angles):
        raise ValueError("At least two paired time/angle samples are required")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(angles)):
        raise ValueError("Time/angle samples must be finite")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Times must be strictly increasing")
    if not isinstance(samples, int) or samples < 3:
        raise ValueError("samples must be an integer >= 3")
    progress = (times - times[0]) / (times[-1] - times[0])
    grid = np.linspace(0, 1, samples)
    resampled_angles = np.interp(grid, progress, angles)
    rom = float(np.max(angles) - np.min(angles))
    # 恒定角度不具有可比较的运动曲线，不把它当成完美同步。
    normalized = None if rom == 0 else (resampled_angles - np.min(angles)) / rom
    return grid, resampled_angles, normalized


def curve_mean_error(progress, left, right, start=0.0, end=1.0):
    """两条曲线之间的面积 / 区间长度，表示平均幅值差（0～1）。"""
    if not 0 <= start < end <= 1:
        raise ValueError("Curve interval must satisfy 0 <= start < end <= 1")
    inside = (progress > start) & (progress < end)
    x = np.concatenate(([start], progress[inside], [end]))
    errors = np.abs(np.interp(x, progress, left) - np.interp(x, progress, right))
    return float(np.trapezoid(errors, x) / (end - start))


def assess_rep(times, left, right, bottom_time, config=None, expected_interval=None):
    """返回指标字典和可导出的101点曲线，不修改原ROM/时间计算。"""
    config = validate_config(config)
    grid, left_angles, left_curve = normalize_motion_curve(times, left)
    _, right_angles, right_curve = normalize_motion_curve(times, right)
    left_rom = max(left) - min(left)
    right_rom = max(right) - min(right)
    duration = times[-1] - times[0]
    rom_ratio = abs(left_rom - right_rom) / max(left_rom, right_rom) if max(left_rom, right_rom) else 0.0
    left_bottom = times[int(np.argmin(left))]
    right_bottom = times[int(np.argmin(right))]
    bottom_ratio = abs(left_bottom - right_bottom) / duration
    intervals = np.diff(times)
    nominal_interval = expected_interval if expected_interval is not None else float(np.median(intervals))
    gap_limit = max(config["maximum_gap_seconds"], config["gap_interval_multiplier"] * nominal_interval)

    reasons = []
    if len(times) < config["minimum_samples"]:
        reasons.append("too_few_samples")
    if min(left_rom, right_rom) < config["minimum_rom_degrees"]:
        reasons.append("low_rom")
    if float(max(intervals)) > gap_limit + 1e-9:
        reasons.append("tracking_gap")
    if not times[0] < bottom_time < times[-1]:
        reasons.append("invalid_bottom")
    metrics = {
        "rom_difference_ratio": rom_ratio,
        "left_bottom_time": left_bottom,
        "right_bottom_time": right_bottom,
        "bottom_time_difference_ratio": bottom_ratio,
        "sample_count": len(times),
        "max_gap_seconds": float(max(intervals)),
        "data_status": "insufficient_data" if reasons else "ok",
        "data_warning": ";".join(reasons),
        "curve_mae": None,
        "pull_curve_mae": None,
        "return_curve_mae": None,
        "synchronization_score": None,
        "pull_synchronization_score": None,
        "return_synchronization_score": None,
        "asymmetry_result": "insufficient_data",
        "asymmetry_reasons": "",
        "rom_balance_score": None,
        "bottom_timing_score": None,
        "quality_score": None,
    }
    curves = []
    for i, progress in enumerate(grid):
        curves.append({
            "progress_percent": float(progress * 100),
            "left_angle": float(left_angles[i]),
            "right_angle": float(right_angles[i]),
            "left_normalized": None if left_curve is None else float(left_curve[i]),
            "right_normalized": None if right_curve is None else float(right_curve[i]),
        })
    if reasons:
        return metrics, curves

    bottom_progress = (bottom_time - times[0]) / duration
    errors = {
        "curve_mae": curve_mean_error(grid, left_curve, right_curve),
        "pull_curve_mae": curve_mean_error(grid, left_curve, right_curve, end=bottom_progress),
        "return_curve_mae": curve_mean_error(grid, left_curve, right_curve, start=bottom_progress),
    }
    metrics.update(errors)
    metrics["synchronization_score"] = 100 * (1 - errors["curve_mae"])
    metrics["pull_synchronization_score"] = 100 * (1 - errors["pull_curve_mae"])
    metrics["return_synchronization_score"] = 100 * (1 - errors["return_curve_mae"])

    flags = []
    if rom_ratio > config["rom_difference_ratio"]:
        flags.append("rom_difference")
    if bottom_ratio > config["bottom_time_ratio"]:
        flags.append("bottom_timing")
    if errors["curve_mae"] > config["curve_mae"]:
        flags.append("whole_curve")
    if errors["pull_curve_mae"] > config["phase_mae"]:
        flags.append("pull_sync")
    if errors["return_curve_mae"] > config["phase_mae"]:
        flags.append("return_sync")
    metrics["asymmetry_result"] = "asymmetric" if flags else "symmetric"
    metrics["asymmetry_reasons"] = ";".join(flags)
    metrics["rom_balance_score"] = 100 * (1 - rom_ratio)
    metrics["bottom_timing_score"] = 100 * max(0, 1 - bottom_ratio / config["bottom_score_zero_ratio"])
    metrics["quality_score"] = (
        config["sync_weight"] * metrics["synchronization_score"]
        + config["rom_weight"] * metrics["rom_balance_score"]
        + config["bottom_weight"] * metrics["bottom_timing_score"]
    )
    return metrics, curves
