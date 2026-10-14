import csv
import matplotlib.pyplot as plt


# 我们先研究第几次动作
target_rep = 1


# =========================
# 1. 找到这一次动作的时间范围
# =========================

with open("rep_analysis.csv", "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        if int(row["rep"]) == target_rep:

            start_time = float(row["start_time"])
            end_time = float(row["end_time"])

            break


print("分析第", target_rep, "次动作")
print("开始时间:", start_time)
print("结束时间:", end_time)
times = []
left_angles = []
right_angles = []


with open(
    "lat_pulldown_2d_3d.csv",
    "r"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        time = float(row["time"])

        if start_time <= time <= end_time:

            times.append(time)

            left_angles.append(
                float(row["left_3d"])
            )

            right_angles.append(
                float(row["right_3d"])
            )
def normalize(data):

    minimum = min(data)
    maximum = max(data)

    result = []

    for value in data:

        normalized = (
            (value - minimum)
            / (maximum - minimum)
        )

        result.append(normalized)

    return result


left_normalized = normalize(
    left_angles
)

right_normalized = normalize(
    right_angles
)
plt.figure(
    figsize=(10, 5)
)

plt.plot(
    times,
    left_normalized,
    label="Left"
)

plt.plot(
    times,
    right_normalized,
    label="Right"
)

plt.xlabel("Time (s)")
plt.ylabel("Normalized Motion")

plt.title(
    f"Rep {target_rep} - Left vs Right"
)

plt.legend()

plt.grid()

plt.show()

