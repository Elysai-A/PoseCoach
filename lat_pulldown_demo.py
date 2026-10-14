import cv2
import mediapipe as mp
import math
import csv


# =========================
# 计算三个点形成的夹角
# =========================

def calculate_angle(a, b, c):

    angle1 = math.atan2(
        a[1] - b[1],
        a[0] - b[0]
    )

    angle2 = math.atan2(
        c[1] - b[1],
        c[0] - b[0]
    )

    angle = math.degrees(
        angle2 - angle1
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


# =========================
# 判断三个关键点是否可信
# =========================

def landmarks_visible(
    a,
    b,
    c,
    threshold=0.6
):

    return (
        a.visibility >= threshold
        and b.visibility >= threshold
        and c.visibility >= threshold
    )


# =========================
# 过滤突然跳变的角度
# =========================

def filter_angle(
    current_angle,
    previous_angle,
    max_jump=35
):

    # 当前帧没有可靠角度
    if current_angle is None:
        return previous_angle

    # 第一帧没有历史数据
    if previous_angle is None:
        return current_angle

    # 如果一帧突然变化太大
    if abs(
        current_angle - previous_angle
    ) > max_jump:

        return previous_angle

    return current_angle


# =========================
# 1. 创建姿态识别器
# =========================

BaseOptions = mp.tasks.BaseOptions

PoseLandmarker = (
    mp.tasks.vision.PoseLandmarker
)

PoseLandmarkerOptions = (
    mp.tasks.vision.PoseLandmarkerOptions
)

RunningMode = (
    mp.tasks.vision.RunningMode
)

options = PoseLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=
        "pose_landmarker_lite.task"
    ),

    running_mode=RunningMode.VIDEO,

    num_poses=1
)

landmarker = (
    PoseLandmarker.create_from_options(
        options
    )
)


# =========================
# 2. 骨架连接
# =========================

POSE_CONNECTIONS = [

    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),

    (23, 25),
    (25, 27),

    (24, 26),
    (26, 28),
]


# =========================
# 3. 打开视频
# =========================

video_path = (
    "lat_pulldown_neutral.mp4"
)

cap = cv2.VideoCapture(
    video_path
)

if not cap.isOpened():

    print("无法打开视频")

    exit()


fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:
    fps = 30


delay = int(
    1000 / fps
)


print("视频已打开")
print("视频帧率:", fps)
print("按 q 退出")


# =========================
# 4. 数据初始化
# =========================

frame_index = 0

angle_records = []

previous_left_angle = None
previous_right_angle = None


# =========================
# 5. 逐帧处理
# =========================

while True:

    ret, frame = cap.read()

    if not ret:

        print("视频播放结束")

        break


    # OpenCV BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    mp_image = mp.Image(

        image_format=
        mp.ImageFormat.SRGB,

        data=rgb_frame
    )


    # 当前视频时间
    timestamp_ms = int(
        frame_index
        * 1000
        / fps
    )

    frame_index += 1


    # 姿态识别
    result = (
        landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )
    )


    # =========================
    # 检测到人体
    # =========================

    if result.pose_landmarks:

        landmarks = (
            result.pose_landmarks[0]
        )

        height, width, _ = (
            frame.shape
        )


        # =========================
        # 获取左右手臂关键点
        # =========================

        left_shoulder = landmarks[11]
        left_elbow = landmarks[13]
        left_wrist = landmarks[15]

        right_shoulder = landmarks[12]
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]


        # =========================
        # 转换为像素坐标
        # =========================

        left_shoulder_point = (
            int(left_shoulder.x * width),
            int(left_shoulder.y * height)
        )

        left_elbow_point = (
            int(left_elbow.x * width),
            int(left_elbow.y * height)
        )

        left_wrist_point = (
            int(left_wrist.x * width),
            int(left_wrist.y * height)
        )


        right_shoulder_point = (
            int(right_shoulder.x * width),
            int(right_shoulder.y * height)
        )

        right_elbow_point = (
            int(right_elbow.x * width),
            int(right_elbow.y * height)
        )

        right_wrist_point = (
            int(right_wrist.x * width),
            int(right_wrist.y * height)
        )


        # =========================
        # 计算原始角度
        # =========================

        left_raw_angle = None
        right_raw_angle = None


        # 左手关键点可信才计算
        if landmarks_visible(
            left_shoulder,
            left_elbow,
            left_wrist
        ):

            left_raw_angle = (
                calculate_angle(
                    left_shoulder_point,
                    left_elbow_point,
                    left_wrist_point
                )
            )


        # 右手关键点可信才计算
        if landmarks_visible(
            right_shoulder,
            right_elbow,
            right_wrist
        ):

            right_raw_angle = (
                calculate_angle(
                    right_shoulder_point,
                    right_elbow_point,
                    right_wrist_point
                )
            )


        # =========================
        # 过滤异常跳变
        # =========================

        left_elbow_angle = (
            filter_angle(
                left_raw_angle,
                previous_left_angle
            )
        )

        right_elbow_angle = (
            filter_angle(
                right_raw_angle,
                previous_right_angle
            )
        )


        # 更新上一帧
        if left_elbow_angle is not None:

            previous_left_angle = (
                left_elbow_angle
            )


        if right_elbow_angle is not None:

            previous_right_angle = (
                right_elbow_angle
            )


        # =========================
        # 保存数据
        # =========================

        if (
            left_elbow_angle is not None
            and
            right_elbow_angle is not None
        ):

            current_time = (
                timestamp_ms / 1000
            )

            angle_records.append([
                current_time,

                left_raw_angle
                if left_raw_angle is not None
                else "",

                right_raw_angle
                if right_raw_angle is not None
                else "",

                left_elbow_angle,
                right_elbow_angle
            ])


        # =========================
        # 左上角显示过滤后的角度
        # =========================

        if left_elbow_angle is not None:

            cv2.putText(
                frame,
                f"LEFT: {left_elbow_angle:.1f}",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 255),
                4
            )


        if right_elbow_angle is not None:

            cv2.putText(
                frame,
                f"RIGHT: {right_elbow_angle:.1f}",
                (50, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 255),
                4
            )


        # =========================
        # 绘制关键点
        # =========================

        for landmark in landmarks:

            if landmark.visibility < 0.5:
                continue

            if not (
                0 <= landmark.x <= 1
                and
                0 <= landmark.y <= 1
            ):
                continue


            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )


            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


        # =========================
        # 绘制骨架
        # =========================

        for start, end in POSE_CONNECTIONS:

            start_landmark = (
                landmarks[start]
            )

            end_landmark = (
                landmarks[end]
            )


            if (
                start_landmark.visibility < 0.5
                or
                end_landmark.visibility < 0.5
            ):
                continue


            if not (
                0 <= start_landmark.x <= 1
                and
                0 <= start_landmark.y <= 1
                and
                0 <= end_landmark.x <= 1
                and
                0 <= end_landmark.y <= 1
            ):
                continue


            start_point = (
                int(
                    start_landmark.x
                    * width
                ),
                int(
                    start_landmark.y
                    * height
                )
            )


            end_point = (
                int(
                    end_landmark.x
                    * width
                ),
                int(
                    end_landmark.y
                    * height
                )
            )


            cv2.line(
                frame,
                start_point,
                end_point,
                (255, 0, 0),
                3
            )


    # =========================
    # 缩小用于显示
    # =========================

    height, width = (
        frame.shape[:2]
    )

    max_width = 960
    max_height = 720


    scale = min(
        max_width / width,
        max_height / height,
        1.0
    )


    display_width = int(
        width * scale
    )

    display_height = int(
        height * scale
    )


    display_frame = cv2.resize(
        frame,
        (
            display_width,
            display_height
        )
    )


    cv2.imshow(
        "PoseCoach - Lat Pulldown",
        display_frame
    )


    if (
        cv2.waitKey(delay)
        & 0xFF
        == ord("q")
    ):
        break


# =========================
# 6. 保存CSV
# =========================

with open(
    "lat_pulldown_angles.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "time",
        "left_raw",
        "right_raw",
        "left_elbow_angle",
        "right_elbow_angle"
    ])

    writer.writerows(
        angle_records
    )


print("过滤后的角度数据已保存")


# =========================
# 7. 释放资源
# =========================

cap.release()

landmarker.close()

cv2.destroyAllWindows()