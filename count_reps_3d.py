import csv
import matplotlib.pyplot as plt


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

def moving_average(
    data,
    window_size=7
):

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
    left_3d,
    window_size=7
)

right_smooth = moving_average(
    right_3d,
    window_size=7
)


# =========================
# 3. 计算百分位数
# 用于过滤极端错误值
# =========================

def percentile(
    data,
    percent
):

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
# 最低点接近 0
# 顶部接近 1
# =========================

def normalize(
    value,
    low,
    high
):

    if high == low:
        return 0

    result = (
        (value - low)
        / (high - low)
    )

    # 限制到 0~1
    result = max(
        0,
        min(1, result)
    )

    return result


left_normalized = []
right_normalized = []

for angle in left_smooth:

    left_normalized.append(
        normalize(
            angle,
            left_low,
            left_high
        )
    )


for angle in right_smooth:

    right_normalized.append(
        normalize(
            angle,
            right_low,
            right_high
        )
    )


# =========================
# 5. 合并左右数据
# =========================

combined = []

for left, right in zip(
    left_normalized,
    right_normalized
):

    combined.append(
        (left + right) / 2
    )


# =========================
# 6. 状态机自动计数
# =========================

rep_count = 0

stage = "waiting"

rep_times = []

last_rep_time = -999


for i in range(len(combined)):

    value = combined[i]
    current_time = times[i]


    # -------------------------
    # 手臂在顶部
    # -------------------------

    if value > 0.70:

        # 如果之前已经拉到底
        if stage == "bottom":

            # 防止短时间重复计数
            if (
                current_time
                - last_rep_time
                > 0.5
            ):

                rep_count += 1

                rep_times.append(
                    current_time
                )

                last_rep_time = (
                    current_time
                )

                print(
                    "第",
                    rep_count,
                    "次完成，时间:",
                    round(
                        current_time,
                        2
                    ),
                    "秒"
                )

        stage = "top"


    # -------------------------
    # 拉到底部
    # -------------------------

    elif value < 0.30:

        if stage == "top":

            stage = "bottom"


# =========================
# 7. 输出结果
# =========================

print()
print("======================")
print("高位下拉次数:", rep_count)
print("======================")


# =========================
# 8. 画计数曲线
# =========================

plt.figure(
    figsize=(13, 7)
)

plt.plot(
    times,
    combined,
    label="Combined 3D Motion"
)


# 顶部阈值
plt.axhline(
    0.70,
    linestyle="--",
    label="Top Threshold"
)


# 底部阈值
plt.axhline(
    0.30,
    linestyle="--",
    label="Bottom Threshold"
)


# 标记每一次动作完成
for rep_time in rep_times:

    plt.axvline(
        rep_time,
        linestyle=":"
    )


plt.xlabel("Time (s)")
plt.ylabel("Normalized Motion")

plt.title(
    "Lat Pulldown - Automatic Rep Counting"
)

plt.legend()

plt.grid()

plt.show()