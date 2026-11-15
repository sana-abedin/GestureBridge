
import cv2
import mediapipe as mp
import time
from pathlib import Path

# --------------------------------------------------
# 1. Locate project and model files
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HAND_MODEL = PROJECT_ROOT / "models" / "hand_landmarker.task"
FACE_MODEL = PROJECT_ROOT / "models" / "face_landmarker.task"
POSE_MODEL = PROJECT_ROOT / "models" / "pose_landmarker_lite.task"

for model in [HAND_MODEL, FACE_MODEL, POSE_MODEL]:
    if not model.exists():
        raise FileNotFoundError(f"Model not found: {model}")


# --------------------------------------------------
# 2. MediaPipe setup
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions


# --------------------------------------------------
# 3. Configure hand model
# --------------------------------------------------

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# --------------------------------------------------
# 4. Configure face model
# --------------------------------------------------

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(FACE_MODEL)
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True
)


# --------------------------------------------------
# 5. Configure pose model
# --------------------------------------------------

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(POSE_MODEL)
    ),
    running_mode=RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.6,
    min_pose_presence_confidence=0.6,
    min_tracking_confidence=0.6
)


# --------------------------------------------------
# 6. Hand connections
# --------------------------------------------------

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# --------------------------------------------------
# 7. Upper-body pose connections
# --------------------------------------------------

POSE_CONNECTIONS = [
    (11, 12),  # Shoulders
    (11, 13),  # Left shoulder to elbow
    (13, 15),  # Left elbow to wrist
    (12, 14),  # Right shoulder to elbow
    (14, 16)   # Right elbow to wrist
]

IMPORTANT_POSE_POINTS = {
    0: "Nose",
    11: "Left Shoulder",
    12: "Right Shoulder",
    13: "Left Elbow",
    14: "Right Elbow",
    15: "Left Wrist",
    16: "Right Wrist"
}

VISIBILITY_THRESHOLD = 0.6


# --------------------------------------------------
# 8. Open webcam
# --------------------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")


# --------------------------------------------------
# 9. Run all three models
# --------------------------------------------------

try:

    with HandLandmarker.create_from_options(hand_options) as hand_detector:

        with FaceLandmarker.create_from_options(face_options) as face_detector:

            with PoseLandmarker.create_from_options(pose_options) as pose_detector:

                print("GestureBridge full detection started.")
                print("Hands + Face + Upper Body")
                print("Press P to print detection information.")
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

                    # Create MediaPipe image
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame
                    )

                    # One shared timestamp for all three models
                    timestamp_ms = time.monotonic_ns() // 1_000_000

                    # ------------------------------------------
                    # 10. Detect hands
                    # ------------------------------------------

                    hand_results = hand_detector.detect_for_video(
                        mp_image,
                        timestamp_ms
                    )

                    # ------------------------------------------
                    # 11. Detect face
                    # ------------------------------------------

                    face_results = face_detector.detect_for_video(
                        mp_image,
                        timestamp_ms
                    )

                    # ------------------------------------------
                    # 12. Detect upper-body pose
                    # ------------------------------------------

                    pose_results = pose_detector.detect_for_video(
                        mp_image,
                        timestamp_ms
                    )

                    hand_count = len(hand_results.hand_landmarks)
                    face_count = len(face_results.face_landmarks)
                    pose_count = len(pose_results.pose_landmarks)

                    # ------------------------------------------
                    # 13. Draw hand landmarks
                    # ------------------------------------------

                    for hand_landmarks in hand_results.hand_landmarks:

                        for start, end in HAND_CONNECTIONS:

                            x1 = int(hand_landmarks[start].x * width)
                            y1 = int(hand_landmarks[start].y * height)

                            x2 = int(hand_landmarks[end].x * width)
                            y2 = int(hand_landmarks[end].y * height)

                            cv2.line(
                                frame,
                                (x1, y1),
                                (x2, y2),
                                (0, 255, 0),
                                2
                            )

                        for landmark in hand_landmarks:

                            x = int(landmark.x * width)
                            y = int(landmark.y * height)

                            cv2.circle(
                                frame,
                                (x, y),
                                4,
                                (0, 0, 255),
                                -1
                            )

                    # ------------------------------------------
                    # 14. Draw face landmarks
                    # ------------------------------------------

                    for face_landmarks in face_results.face_landmarks:

                        for landmark in face_landmarks:

                            x = int(landmark.x * width)
                            y = int(landmark.y * height)

                            if 0 <= x < width and 0 <= y < height:

                                cv2.circle(
                                    frame,
                                    (x, y),
                                    1,
                                    (255, 0, 255),
                                    -1
                                )

                    # ------------------------------------------
                    # 15. Draw upper-body pose landmarks
                    # ------------------------------------------

                    visible_pose_points = {}

                    if pose_count > 0:

                        pose = pose_results.pose_landmarks[0]

                        for index, name in IMPORTANT_POSE_POINTS.items():

                            point = pose[index]

                            if (
                                point.visibility >= VISIBILITY_THRESHOLD
                                and 0 <= point.x <= 1
                                and 0 <= point.y <= 1
                            ):
                                visible_pose_points[index] = point

                        # Draw upper-body connections
                        for start, end in POSE_CONNECTIONS:

                            if (
                                start in visible_pose_points
                                and end in visible_pose_points
                            ):

                                point1 = visible_pose_points[start]
                                point2 = visible_pose_points[end]

                                x1 = int(point1.x * width)
                                y1 = int(point1.y * height)

                                x2 = int(point2.x * width)
                                y2 = int(point2.y * height)

                                cv2.line(
                                    frame,
                                    (x1, y1),
                                    (x2, y2),
                                    (255, 255, 0),
                                    3
                                )

                        # Draw important pose points
                        for index, point in visible_pose_points.items():

                            x = int(point.x * width)
                            y = int(point.y * height)

                            cv2.circle(
                                frame,
                                (x, y),
                                6,
                                (255, 255, 0),
                                -1
                            )

                    # ------------------------------------------
                    # 16. Display detection counts
                    # ------------------------------------------

                    cv2.putText(
                        frame,
                        f"Hands: {hand_count}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Faces: {face_count}",
                        (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Pose points: {len(visible_pose_points)}/7",
                        (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2
                    )

                    # Show webcam
                    cv2.imshow(
                        "GestureBridge - Full Detection",
                        frame
                    )

                    key = cv2.waitKey(1) & 0xFF

                    # ------------------------------------------
                    # 17. Print detection report
                    # ------------------------------------------

                    if key == ord("p"):

                        print("\n========== GESTUREBRIDGE REPORT ==========")

                        print("Hands detected:", hand_count)
                        print("Faces detected:", face_count)
                        print("Poses detected:", pose_count)

                        for index, hand in enumerate(
                            hand_results.hand_landmarks
                        ):
                            print(
                                f"Hand {index + 1}: {len(hand)} landmarks"
                            )

                        for index, face in enumerate(
                            face_results.face_landmarks
                        ):
                            print(
                                f"Face {index + 1}: {len(face)} landmarks"
                            )

                        if pose_count > 0:

                            print("\nUpper-body landmark visibility:")

                            pose = pose_results.pose_landmarks[0]

                            for index, name in IMPORTANT_POSE_POINTS.items():

                                print(
                                    name,
                                    ":",
                                    round(pose[index].visibility, 3)
                                )

                        print("==========================================\n")

                    # Quit
                    if key == ord("q"):
                        break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print("GestureBridge full detection stopped.")
