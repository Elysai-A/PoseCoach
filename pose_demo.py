import cv2
import mediapipe as mp
import time
import math


# =========================
# 计算三个点之间的夹角
# =========================

def calculate_angle(a, b, c):
    """
    计算三个点形成的夹角
    b 是角度的顶点
    """

    angle1 = math.atan2(
        a[1] - b[1],
        a[0] - b[0]
    )

    angle2 = math.atan2(
        c[1] - b[1],
        c[0] - b[0]
    )

    angle = math.degrees(angle2 - angle1)

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


# =========================
# 1. 创建姿态识别器
# =========================

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="pose_landmarker_lite.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)

landmarker = PoseLandmarker.create_from_options(options)


# =========================
# 2. 我们需要绘制的骨架连接
# =========================

POSE_CONNECTIONS = [

    # 肩膀与躯干
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),

    # 左臂
    (11, 13),
    (13, 15),

    # 右臂
    (12, 14),
    (14, 16),

    # 左腿
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),

    # 右腿
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
]


# =========================
# 3. 打开摄像头
# =========================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

print("姿态识别已启动，按 q 退出")

last_timestamp = 0


# =========================
# 4. 不断处理摄像头画面
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("无法读取摄像头画面")
        break


    # OpenCV 使用 BGR
    # MediaPipe 使用 RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # 转成 MediaPipe 图像
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # VIDEO 模式需要时间戳
    timestamp_ms = int(
        time.perf_counter() * 1000
    )

    if timestamp_ms <= last_timestamp:
        timestamp_ms = last_timestamp + 1

    last_timestamp = timestamp_ms


    # =========================
    # 5. AI 姿态识别
    # =========================

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # =========================
    # 6. 如果检测到人体
    # =========================

    if result.pose_landmarks:

        landmarks = result.pose_landmarks[0]

        # 获取左腿三个关键点
        left_hip = landmarks[23]
        left_knee = landmarks[25]
        left_ankle = landmarks[27]

        height, width, _ = frame.shape


        # =========================
        # 计算左膝角度
        # =========================

        # 左髋坐标
        hip_point = (
            int(left_hip.x * width),
            int(left_hip.y * height)
        )

        # 左膝坐标
        knee_point = (
            int(left_knee.x * width),
            int(left_knee.y * height)
        )

        # 左脚踝坐标
        ankle_point = (
            int(left_ankle.x * width),
            int(left_ankle.y * height)
        )


        # 计算左膝角度
        left_knee_angle = calculate_angle(
            hip_point,
            knee_point,
            ankle_point
        )


        # 把角度显示在膝盖旁边
        cv2.putText(
            frame,
            f"{left_knee_angle:.1f}",
            knee_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        # -------------------------
        # 绘制关键点
        # -------------------------

        for landmark in landmarks:

            # 如果这个点可信度太低，不画
            if landmark.visibility < 0.5:
                continue

            # 如果关键点跑出画面，也不画
            if not (
                0 <= landmark.x <= 1
                and 0 <= landmark.y <= 1
            ):
                continue

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )


        # -------------------------
        # 绘制骨架连线
        # -------------------------

        for start, end in POSE_CONNECTIONS:

            start_landmark = landmarks[start]
            end_landmark = landmarks[end]

            # 两个点都必须足够可信
            if (
                start_landmark.visibility < 0.5
                or end_landmark.visibility < 0.5
            ):
                continue

            # 两个点都必须在画面里
            if not (
                0 <= start_landmark.x <= 1
                and 0 <= start_landmark.y <= 1
                and 0 <= end_landmark.x <= 1
                and 0 <= end_landmark.y <= 1
            ):
                continue

            start_point = (
                int(start_landmark.x * width),
                int(start_landmark.y * height)
            )

            end_point = (
                int(end_landmark.x * width),
                int(end_landmark.y * height)
            )

            cv2.line(
                frame,
                start_point,
                end_point,
                (255, 0, 0),
                2
            )


    # =========================
    # 7. 显示最终画面
    # =========================

    cv2.imshow(
        "PoseCoach - Pose Detection",
        frame
    )


    # 按 q 退出
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================
# 8. 释放资源
# =========================

cap.release()
landmarker.close()
cv2.destroyAllWindows()