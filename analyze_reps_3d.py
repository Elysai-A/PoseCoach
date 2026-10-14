import csv


# =========================
# 1. 读取 3D 数据
# =========================

times = []
left_3d = []
right_3d = []

with open(
    "lat_pulldown_2d_3d.csv",
    "r"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        times.append(
            float(row["time"])
        )

        left_3d.append(
            float(row["left_3d"])
        )

        right_3d.append(
            float(row["right_3d"])
        )


# =========================
# 2. 移动平均
# =========================

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


left_smooth = moving_average(
    left_3d
)

right_smooth = moving_average(
    right_3d
)


# =========================
# 3. 百分位数
# =========================

def percentile(data, percent):

    sorted_data = sorted(data)

    index = int(
        (len(sorted_data) - 1)
        * percent
    )

    return sorted_data[index]


left_low = percentile(
    left_smooth,
    0.10
)

left_high = percentile(
    left_smooth,
    0.90
)

right_low = percentile(
    right_smooth,
    0.10
)

right_high = percentile(
    right_smooth,
    0.90
)


# =========================
# 4. 归一化
# =========================

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


left_normalized = [
    normalize(
        value,
        left_low,
        left_high
    )
    for value in left_smooth
]

right_normalized = [
    normalize(
        value,
        right_low,
        right_high
    )
    for value in right_smooth
]


combined = [
    (left + right) / 2
    for left, right in zip(
        left_normalized,
        right_normalized
    )
]


# =========================
# 5. 把动作切成一下一下
# =========================

reps = []

stage = "waiting"

last_top_index = None

rep_start_index = None


for i, value in enumerate(combined):

    # 顶部
    if value > 0.70:

        if stage == "bottom":

            rep_end_index = i

            reps.append(
                (
                    rep_start_index,
                    rep_end_index
                )
            )

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


# =========================
# 6. 分析每一次动作
# =========================

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


    analysis_results.append([
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
    ])


# =========================
# 7. 输出结果
# =========================

print()
print(
    "检测到动作次数:",
    len(analysis_results)
)

print()


for result in analysis_results:

    (
        rep,
        start,
        bottom,
        end,
        left_rom,
        right_rom,
        rom_diff,
        sync_diff,
        pull_time,
        return_time
    ) = result


    print(
        f"第 {rep} 次"
    )

    print(
        f"  左 ROM: {left_rom:.1f}°"
    )

    print(
        f"  右 ROM: {right_rom:.1f}°"
    )

    print(
        f"  ROM 差: {rom_diff:.1f}°"
    )

    print(
        f"  左右最低点时间差: "
        f"{sync_diff:.3f}s"
    )

    print(
        f"  下拉时间: "
        f"{pull_time:.2f}s"
    )

    print(
        f"  回程时间: "
        f"{return_time:.2f}s"
    )

    print()


# =========================
# 8. 保存分析结果
# =========================

with open(
    "rep_analysis.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "rep",
        "start_time",
        "bottom_time",
        "end_time",
        "left_rom",
        "right_rom",
        "rom_difference",
        "bottom_time_difference",
        "pull_time",
        "return_time"
    ])

    writer.writerows(
        analysis_results
    )


print(
    "每次动作分析已保存到 "
    "rep_analysis.csv"
)