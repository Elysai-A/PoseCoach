import csv
import matplotlib.pyplot as plt


times = []

left_2d = []
right_2d = []

left_3d = []
right_3d = []


# =========================
# 读取 CSV
# =========================

with open(
    "lat_pulldown_2d_3d.csv",
    "r"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        times.append(
            float(row["time"])
        )

        left_2d.append(
            float(row["left_2d"])
        )

        right_2d.append(
            float(row["right_2d"])
        )

        left_3d.append(
            float(row["left_3d"])
        )

        right_3d.append(
            float(row["right_3d"])
        )


# =========================
# 简单移动平均
# =========================

def moving_average(data, window_size=5):

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


left_2d_smooth = moving_average(left_2d)
right_2d_smooth = moving_average(right_2d)

left_3d_smooth = moving_average(left_3d)
right_3d_smooth = moving_average(right_3d)


# =========================
# 绘图
# =========================

plt.figure(
    figsize=(13, 7)
)

plt.plot(
    times,
    left_2d_smooth,
    label="Left 2D"
)

plt.plot(
    times,
    right_2d_smooth,
    label="Right 2D"
)

plt.plot(
    times,
    left_3d_smooth,
    label="Left 3D"
)

plt.plot(
    times,
    right_3d_smooth,
    label="Right 3D"
)


plt.xlabel("Time (s)")
plt.ylabel("Elbow Angle (degrees)")

plt.title(
    "Lat Pulldown - 2D vs 3D Elbow Angle"
)

plt.legend()
plt.grid()

plt.show()