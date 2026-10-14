import cv2
import mediapipe as mp
import math
import csv


# =========================
# 1. 计算二维夹角
# =========================

def calculate_angle_2d(a, b, c):

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
# 2. 计算三维夹角
# =========================

def calculate_angle_3d(a, b, c):

    # 从肘部 b 指向肩膀 a
    vector1 = (
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2]
    )

    # 从肘部 b 指向手腕 c
    vector2 = (
        c[0] - b[0],
        c[1] - b[1],
        c[2] - b[2]
    )

    dot_product = (
        vector1[0] * vector2[0]
        + vector1[1] * vector2[1]
        + vector1[2] * vector2[2]
    )

    length1 = math.sqrt(
        vector1[0] ** 2
        + vector1[1] ** 2
        + vector1[2] ** 2
    )

    length2 = math.sqrt(
        vector2[0] ** 2
        + vector2[1] ** 2
        + vector2[2] ** 2
    )

    if length1 == 0 or length2 == 0:
        return None

    cos_angle = (
        dot_product
        / (length1 * length2)
    )

    # 避免浮点误差
    cos_angle = max(
        -1.0,
        min(1.0, cos_angle)
    )

    angle = math.degrees(
        math.acos(cos_angle)
    )

    return angle


# =========================
# 3. 转换二维坐标
# =========================

def to_pixel(landmark, width, height):

    return (
        int(landmark.x * width),
        int(landmark.y * height)
    )


# =========================
# 4. 转换三维世界坐标
# =========================

def to_world(landmark):

    return (
        landmark.x,
        landmark.y,
        landmark.z
    )


# =========================
# 5. 判断关键点是否可信
# =========================

def arm_visible(
    shoulder,
    elbow,
    wrist,
    threshold=0.6
):

    return (
        shoulder.visibility >= threshold
        and elbow.visibility >= threshold
        and wrist.visibility >= threshold
    )


# =========================
# 6. 创建 MediaPipe
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

landmarker = PoseLandmarker.create_from_options(
    options
)


# =========================
# 7. 打开视频
# =========================

video_path = "lat_pulldown_neutral.mp4"

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

frame_index = 0

records = []

print("2D / 3D 对照分析已启动")
print("视频帧率:", fps)


# =========================
# 8. 处理视频
# =========================

while True:

    ret, frame = cap.read()

    if not ret:
        print("视频播放结束")
        break


    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    timestamp_ms = int(
        frame_index * 1000 / fps
    )

    frame_index += 1


    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # 必须同时有 2D 和 3D 数据
    if (
        result.pose_landmarks
        and result.pose_world_landmarks
    ):

        landmarks = (
            result.pose_landmarks[0]
        )

        world_landmarks = (
            result.pose_world_landmarks[0]
        )

        height, width, _ = frame.shape


        # =========================
        # 左臂
        # =========================

        left_shoulder = landmarks[11]
        left_elbow = landmarks[13]
        left_wrist = landmarks[15]


        # =========================
        # 右臂
        # =========================

        right_shoulder = landmarks[12]
        right_elbow = landmarks[14]
        right_wrist = landmarks[16]


        left_2d = None
        right_2d = None

        left_3d = None
        right_3d = None


        # =========================
        # 左侧计算
        # =========================

        if arm_visible(
            left_shoulder,
            left_elbow,
            left_wrist
        ):

            left_2d = calculate_angle_2d(
                to_pixel(
                    left_shoulder,
                    width,
                    height
                ),
                to_pixel(
                    left_elbow,
                    width,
                    height
                ),
                to_pixel(
                    left_wrist,
                    width,
                    height
                )
            )

            left_3d = calculate_angle_3d(
                to_world(
                    world_landmarks[11]
                ),
                to_world(
                    world_landmarks[13]
                ),
                to_world(
                    world_landmarks[15]
                )
            )


        # =========================
        # 右侧计算
        # =========================

        if arm_visible(
            right_shoulder,
            right_elbow,
            right_wrist
        ):

            right_2d = calculate_angle_2d(
                to_pixel(
                    right_shoulder,
                    width,
                    height
                ),
                to_pixel(
                    right_elbow,
                    width,
                    height
                ),
                to_pixel(
                    right_wrist,
                    width,
                    height
                )
            )

            right_3d = calculate_angle_3d(
                to_world(
                    world_landmarks[12]
                ),
                to_world(
                    world_landmarks[14]
                ),
                to_world(
                    world_landmarks[16]
                )
            )


        # 四个值都存在才记录
        if (
            left_2d is not None
            and right_2d is not None
            and left_3d is not None
            and right_3d is not None
        ):

            current_time = (
                timestamp_ms / 1000
            )

            records.append([
                current_time,
                left_2d,
                right_2d,
                left_3d,
                right_3d
            ])


            # =========================
            # 左上角显示
            # =========================

            cv2.putText(
                frame,
                f"2D L:{left_2d:.1f} R:{right_2d:.1f}",
                (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 255),
                4
            )

            cv2.putText(
                frame,
                f"3D L:{left_3d:.1f} R:{right_3d:.1f}",
                (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 255),
                4
            )


    # =========================
    # 缩小显示
    # =========================

    height, width = frame.shape[:2]

    scale = min(
        960 / width,
        720 / height,
        1.0
    )

    display_frame = cv2.resize(
        frame,
        (
            int(width * scale),
            int(height * scale)
        )
    )


    cv2.imshow(
        "PoseCoach - 2D vs 3D",
        display_frame
    )


    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break


# =========================
# 9. 保存数据
# =========================

with open(
    "lat_pulldown_2d_3d.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "time",
        "left_2d",
        "right_2d",
        "left_3d",
        "right_3d"
    ])

    writer.writerows(
        records
    )


print("2D / 3D 数据已保存")


# =========================
# 10. 释放资源
# =========================

cap.release()
landmarker.close()
cv2.destroyAllWindows()