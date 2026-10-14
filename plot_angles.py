import csv
import matplotlib.pyplot as plt


# =========================
# 1. 准备数据
# =========================

times = []
left_angles = []
right_angles = []


# =========================
# 2. 读取 CSV
# =========================

with open(
    "lat_pulldown_angles.csv",
    "r"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        times.append(
            float(row["time"])
        )

        left_angles.append(
            float(row["left_elbow_angle"])
        )

        right_angles.append(
            float(row["right_elbow_angle"])
        )


# =========================
# 3. 移动平均函数
# =========================

def moving_average(data, window_size=5):

    smoothed = []

    for i in range(len(data)):

        start = max(
            0,
            i - window_size + 1
        )

        window = data[start:i + 1]

        average = sum(window) / len(window)

        smoothed.append(average)

    return smoothed


# =========================
# 4. 平滑左右肘角度
# =========================

left_smooth = moving_average(
    left_angles,
    window_size=5
)

right_smooth = moving_average(
    right_angles,
    window_size=5
)


# =========================
# 5. 绘图
# =========================

plt.figure(
    figsize=(12, 6)
)

# 原始左肘
plt.plot(
    times,
    left_angles,
    label="Left Raw",
    alpha=0.3
)

# 平滑左肘
plt.plot(
    times,
    left_smooth,
    label="Left Smooth"
)

# 原始右肘
plt.plot(
    times,
    right_angles,
    label="Right Raw",
    alpha=0.3
)

# 平滑右肘
plt.plot(
    times,
    right_smooth,
    label="Right Smooth"
)


# =========================
# 6. 图表设置
# =========================

plt.xlabel("Time (s)")
plt.ylabel("Elbow Angle (degrees)")

plt.title(
    "Lat Pulldown - Raw vs Smoothed Elbow Angle"
)

plt.legend()
plt.grid()


# =========================
# 7. 显示
# =========================

plt.show()