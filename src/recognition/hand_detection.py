
# ==========================================
# GestureBridge - Real-Time Hand Detection
# Using OpenCV and MediaPipe
# ==========================================

import cv2
import mediapipe as mp
import time
from pathlib import Path


# STEP 1: Locate the trained MediaPipe model file

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"


# STEP 2: Check whether the model file exists

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Hand detection model not found: {MODEL_PATH}"
    )


# STEP 3: Set up MediaPipe Hand Landmarker

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)


# STEP 4: Define hand connections

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17),
    (17, 18), (18, 19), (19, 20)
]


# STEP 5: Open the laptop webcam

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")


print("GestureBridge hand detection started!")
print("Show your hand to the camera.")
print("Press Q to exit.")


# STEP 6: Create the MediaPipe detector

try:
    with HandLandmarker.create_from_options(options) as landmarker:

        # STEP 7: Continuously capture webcam frames

        while True:

            success, frame = camera.read()

            if not success:
                print("Could not read camera frame.")
                break

            # Mirror the webcam image
            frame = cv2.flip(frame, 1)

            # Get image dimensions
            height, width, _ = frame.shape

            # Convert OpenCV BGR image to RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Convert the image into MediaPipe format
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Generate a timestamp for this video frame
            timestamp_ms = time.monotonic_ns() // 1_000_000

            # Detect hand landmarks
            results = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            # STEP 8: Draw landmarks if a hand is detected

            if results.hand_landmarks:

                for hand_landmarks in results.hand_landmarks:

                    # Draw connections between landmarks
                    for start_index, end_index in HAND_CONNECTIONS:

                        start_point = hand_landmarks[start_index]
                        end_point = hand_landmarks[end_index]

                        start = (
                            int(start_point.x * width),
                            int(start_point.y * height)
                        )

                        end = (
                            int(end_point.x * width),
                            int(end_point.y * height)
                        )

                        cv2.line(
                            frame,
                            start,
                            end,
                            (0, 255, 0),
                            2
                        )

                    # Draw the 21 landmark points
                    for landmark in hand_landmarks:

                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 0, 255),
                            -1
                        )

                cv2.putText(
                    frame,
                    "Hand Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "No Hand Detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            # STEP 9: Display the final webcam frame

            cv2.imshow(
                "GestureBridge - Hand Detection",
                frame
            )

            # Exit when Q is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:

    # STEP 10: Release resources

    camera.release()
    cv2.destroyAllWindows()

    print("GestureBridge hand detection stopped.")
