
# ==========================================
# GestureBridge - Hand Landmark Coordinates
# ==========================================

import cv2
import mediapipe as mp
import time
import math
from pathlib import Path


# STEP 1: Locate the MediaPipe model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"


# STEP 2: Check the model file

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )


# STEP 3: Configure MediaPipe

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


# STEP 4.1: Normalize hand landmarks

def normalize_landmarks(hand_landmarks):

    # Get wrist landmark (Landmark 0)

    wrist = hand_landmarks[0]

    wrist_x = wrist.x
    wrist_y = wrist.y
    wrist_z = wrist.z

    # Calculate distances from wrist

    distances = []

    for landmark in hand_landmarks:

        distance = math.hypot(
            landmark.x - wrist_x,
            landmark.y - wrist_y
        )

        distances.append(distance)

    # Find largest distance

    scale = max(distances)

    # Avoid division by zero

    if scale == 0:
        scale = 1.0

    # Calculate normalized coordinates

    normalized_points = []

    for landmark in hand_landmarks:

        normalized_x = (landmark.x - wrist_x) / scale
        normalized_y = (landmark.y - wrist_y) / scale
        normalized_z = (landmark.z - wrist_z) / scale

        normalized_points.append({
            "x": normalized_x,
            "y": normalized_y,
            "z": normalized_z
        })

    return normalized_points


# STEP 5: Open the webcam

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

print("GestureBridge coordinate extraction started!")
print("Press P to print landmark coordinates.")
print("Press Q to exit.")


# STEP 6: Start hand detection

try:

    with HandLandmarker.create_from_options(options) as landmarker:

        while True:

            # Capture webcam frame

            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            # Mirror the image

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

            # Generate video timestamp

            timestamp_ms = time.monotonic_ns() // 1_000_000

            # Detect hands

            results = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            detected_hands = len(results.hand_landmarks)

            # Create fresh storage for this frame

            hand_data = []

            # Display hand count

            if detected_hands == 0:
                status_text = "No Hand Detected"
            else:
                status_text = f"Hands Detected: {detected_hands}"

            cv2.putText(
                frame,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            # STEP 7: Process each detected hand

            if results.hand_landmarks:

                for hand_index, hand_landmarks in enumerate(
                    results.hand_landmarks
                ):
                                        # Get handedness information
                    hand_label = f"Hand {hand_index + 1}"

                    if results.handedness:

                        detected_label = (
                            results.handedness[hand_index][0].category_name
                        )

                        # Provisional label swap from earlier testing

                        if detected_label == "Left":
                            hand_label = "Right"

                        elif detected_label == "Right":
                            hand_label = "Left"

                        else:
                            hand_label = detected_label

                    # Extract original landmarks

                    landmark_points = []

                    for landmark in hand_landmarks:

                        landmark_points.append({
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z
                        })

                    # Generate normalized landmarks

                    normalized_points = normalize_landmarks(
                        hand_landmarks
                    )

                    # Store original and normalized coordinates

                    hand_data.append({
                        "hand_index": hand_index,
                        "hand_label": hand_label,
                        "landmarks": landmark_points,
                        "normalized_landmarks": normalized_points
                    })

                    # Draw hand connections

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

                    # Draw landmark points

                    for landmark_number, landmark in enumerate(
                        hand_landmarks
                    ):

                        x = int(landmark.x * width)
                        y = int(landmark.y * height)

                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 0, 255),
                            -1
                        )

                    # Display hand label

                    cv2.putText(
                        frame,
                        f"{hand_label} Hand",
                        (20, 80 + hand_index * 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2
                    )

            # STEP 8: Display instructions

            cv2.putText(
                frame,
                "Press P: Print Coordinates | Q: Exit",
                (20, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Display webcam

            cv2.imshow(
                "GestureBridge - Landmark Coordinates",
                frame
            )

            # Read keyboard input

            key = cv2.waitKey(1) & 0xFF

            # STEP 9: Print coordinates when P is pressed

            if key == ord('p'):

                if hand_data:

                    for hand in hand_data:

                        print("\n================================")
                        print(hand["hand_label"], "COORDINATES")
                        print("================================")

                        print(
                            "Total landmarks detected:",
                            len(hand["landmarks"])
                        )

                        # Print original coordinates

                        print("\nORIGINAL COORDINATES")

                        for index, point in enumerate(
                            hand["landmarks"]
                        ):

                            print(
                                f"Landmark {index:02d}: "
                                f"X={point['x']:.4f}, "
                                f"Y={point['y']:.4f}, "
                                f"Z={point['z']:.4f}"
                            )

                        # Print normalized coordinates

                        print("\nNORMALIZED COORDINATES")

                        for index, point in enumerate(
                            hand["normalized_landmarks"]
                        ):

                            print(
                                f"Landmark {index:02d}: "
                                f"X={point['x']:.4f}, "
                                f"Y={point['y']:.4f}, "
                                f"Z={point['z']:.4f}"
                            )

                else:
                    print("No hand detected. Show your hand first.")

            # Exit when Q is pressed

            if key == ord('q'):
                break
            
finally:

    # Release webcam and close windows

    camera.release()
    cv2.destroyAllWindows()

    print("Coordinate extraction stopped.")


# ==========================================
# End of GestureBridge Landmark Coordinates
# ==========================================
