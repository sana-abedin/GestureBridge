
import cv2
import mediapipe as mp
import time
from pathlib import Path

# --------------------------------------------------
# 1. Find the project folder and model files
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HAND_MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
FACE_MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"

if not HAND_MODEL_PATH.exists():
    raise FileNotFoundError(f"Hand model not found: {HAND_MODEL_PATH}")

if not FACE_MODEL_PATH.exists():
    raise FileNotFoundError(f"Face model not found: {FACE_MODEL_PATH}")


# --------------------------------------------------
# 2. Set up MediaPipe Tasks
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
RunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions


# --------------------------------------------------
# 3. Configure the hand detection model
# --------------------------------------------------

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(HAND_MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# --------------------------------------------------
# 4. Configure the face detection model
# --------------------------------------------------

face_options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(FACE_MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True
)


# --------------------------------------------------
# 5. Hand landmark connections
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
# 6. Open webcam
# --------------------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")


# --------------------------------------------------
# 7. Run both models together
# --------------------------------------------------

try:

    with HandLandmarker.create_from_options(hand_options) as hand_landmarker:

        with FaceLandmarker.create_from_options(face_options) as face_landmarker:

            print("GestureBridge combined detection started.")
            print("Press P to print detection information.")
            print("Press Q to quit.")

            while True:

                success, frame = camera.read()

                if not success:
                    print("Could not read webcam frame.")
                    break

                # Mirror the webcam image
                frame = cv2.flip(frame, 1)

                height, width, _ = frame.shape

                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                # Convert frame to MediaPipe image
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                # Use one timestamp for both models
                timestamp_ms = time.monotonic_ns() // 1_000_000

                # Detect hands
                hand_results = hand_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                # Detect face
                face_results = face_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                hand_count = len(hand_results.hand_landmarks)
                face_count = len(face_results.face_landmarks)


                # --------------------------------------------------
                # 8. Draw hand landmarks
                # --------------------------------------------------

                for hand_landmarks in hand_results.hand_landmarks:

                    # Draw connections
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

                    # Draw landmark points
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


                # --------------------------------------------------
                # 9. Draw face landmarks
                # --------------------------------------------------

                for face_landmarks in face_results.face_landmarks:

                    for landmark in face_landmarks:

                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        cv2.circle(
                            frame,
                            (x, y),
                            1,
                            (255, 0, 255),
                            -1
                        )


                # --------------------------------------------------
                # 10. Display detection status
                # --------------------------------------------------

                cv2.putText(
                    frame,
                    f"Hands detected: {hand_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Faces detected: {face_count}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 255),
                    2
                )

                # Display webcam window
                cv2.imshow(
                    "GestureBridge - Combined Detection",
                    frame
                )

                key = cv2.waitKey(1) & 0xFF


                # --------------------------------------------------
                # 11. Print information when P is pressed
                # --------------------------------------------------

                if key == ord("p"):

                    print("\n---------- DETECTION REPORT ----------")

                    print("Hands detected:", hand_count)
                    print("Faces detected:", face_count)

                    for index, hand in enumerate(
                        hand_results.hand_landmarks
                    ):
                        print(
                            f"Hand {index + 1}: "
                            f"{len(hand)} landmarks"
                        )

                    for index, face in enumerate(
                        face_results.face_landmarks
                    ):
                        print(
                            f"Face {index + 1}: "
                            f"{len(face)} landmarks"
                        )

                    print("--------------------------------------\n")


                # Quit when Q is pressed
                if key == ord("q"):
                    break

finally:

    camera.release()
    cv2.destroyAllWindows()

    print("GestureBridge combined detection stopped.")
