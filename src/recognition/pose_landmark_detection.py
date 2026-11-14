
import cv2
import mediapipe as mp
import time
from pathlib import Path

# --------------------------------------------------
# 1. Locate the project and model
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Pose model not found: {MODEL_PATH}"
    )


# --------------------------------------------------
# 2. Set up MediaPipe
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# --------------------------------------------------
# 3. Configure pose detection
# --------------------------------------------------

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.6,
    min_pose_presence_confidence=0.6,
    min_tracking_confidence=0.6
)


# --------------------------------------------------
# 4. Select only important upper-body landmarks
# --------------------------------------------------

IMPORTANT_POINTS = {
    0: "Nose",
    11: "Left Shoulder",
    12: "Right Shoulder",
    13: "Left Elbow",
    14: "Right Elbow",
    15: "Left Wrist",
    16: "Right Wrist"
}

# Connect shoulders, elbows and wrists
UPPER_BODY_CONNECTIONS = [
    (11, 12),  # Shoulder to shoulder
    (11, 13),  # Left shoulder to elbow
    (13, 15),  # Left elbow to wrist
    (12, 14),  # Right shoulder to elbow
    (14, 16)   # Right elbow to wrist
]

# Minimum visibility required to draw a landmark
VISIBILITY_THRESHOLD = 0.6


# --------------------------------------------------
# 5. Open webcam
# --------------------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")


# --------------------------------------------------
# 6. Start pose detection
# --------------------------------------------------

try:

    with PoseLandmarker.create_from_options(options) as landmarker:

        print("GestureBridge upper-body detection started.")
        print("Press P to print landmark information.")
        print("Press Q to quit.")

        while True:

            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            # Mirror webcam
            frame = cv2.flip(frame, 1)

            height, width, _ = frame.shape

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Convert to MediaPipe image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Timestamp
            timestamp_ms = time.monotonic_ns() // 1_000_000

            # Detect pose
            results = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            pose_detected = False
            visible_points = {}

            if results.pose_landmarks:

                pose = results.pose_landmarks[0]

                # ------------------------------------------
                # 7. Check important landmark visibility
                # ------------------------------------------

                for index, name in IMPORTANT_POINTS.items():

                    point = pose[index]

                    visibility = point.visibility

                    if (
                        visibility >= VISIBILITY_THRESHOLD
                        and 0 <= point.x <= 1
                        and 0 <= point.y <= 1
                    ):

                        visible_points[index] = point

                # ------------------------------------------
                # 8. Draw reliable connections only
                # ------------------------------------------

                for start, end in UPPER_BODY_CONNECTIONS:

                    if start in visible_points and end in visible_points:

                        point1 = visible_points[start]
                        point2 = visible_points[end]

                        x1 = int(point1.x * width)
                        y1 = int(point1.y * height)

                        x2 = int(point2.x * width)
                        y2 = int(point2.y * height)

                        cv2.line(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            3
                        )

                # ------------------------------------------
                # 9. Draw reliable landmark points
                # ------------------------------------------

                for index, point in visible_points.items():

                    x = int(point.x * width)
                    y = int(point.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        6,
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        frame,
                        IMPORTANT_POINTS[index],
                        (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 0),
                        1
                    )

                pose_detected = len(visible_points) > 0

            # ------------------------------------------
            # 10. Display detection status
            # ------------------------------------------

            if pose_detected:

                status = (
                    f"Visible upper-body points: "
                    f"{len(visible_points)}/7"
                )

                color = (0, 255, 0)

            else:

                status = "No reliable upper-body points"
                color = (0, 0, 255)

            cv2.putText(
                frame,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            cv2.imshow(
                "GestureBridge - Upper Body Detection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # ------------------------------------------
            # 11. Print landmark information
            # ------------------------------------------

            if key == ord("p"):

                print("\n---------- UPPER BODY REPORT ----------")

                if results.pose_landmarks:

                    pose = results.pose_landmarks[0]

                    for index, name in IMPORTANT_POINTS.items():

                        point = pose[index]

                        print(
                            name,
                            "| X:", round(point.x, 3),
                            "| Y:", round(point.y, 3),
                            "| Visibility:",
                            round(point.visibility, 3)
                        )

                else:

                    print("No pose detected.")

                print("---------------------------------------\n")

            # Quit
            if key == ord("q"):
                break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print("Upper-body detection stopped.")
