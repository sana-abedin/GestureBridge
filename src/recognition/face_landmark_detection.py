
import cv2
import mediapipe as mp
import time
from pathlib import Path

# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Locate the face landmark model
MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"

# Check whether the model file exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Face model not found at: {MODEL_PATH}"
    )

# MediaPipe Face Landmarker setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Configure the face detector
options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1,
    output_face_blendshapes=True
)

# Open the webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Could not open webcam.")

# Create the face landmarker
with FaceLandmarker.create_from_options(options) as landmarker:

    print("Face landmark detection started.")
    print("Press P to print face landmark information.")
    print("Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("Could not read webcam frame.")
            break

        # Mirror the webcam image
        frame = cv2.flip(frame, 1)

        height, width, _ = frame.shape

        # Convert BGR image to RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Convert image into MediaPipe format
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Get current timestamp
        timestamp_ms = time.monotonic_ns() // 1_000_000

        # Detect facial landmarks
        results = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        face_count = len(results.face_landmarks)

        # If a face is detected
        if face_count > 0:

            for face_landmarks in results.face_landmarks:

                # Draw facial landmark points
                for landmark in face_landmarks:

                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        frame,
                        (x, y),
                        1,
                        (0, 255, 0),
                        -1
                    )

            cv2.putText(
                frame,
                "Face Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:
            cv2.putText(
                frame,
                "No Face Detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        # Display the webcam feed
        cv2.imshow(
            "GestureBridge - Face Landmark Detection",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # Print landmark information when P is pressed
        if key == ord("p"):

            print("\nFace count:", face_count)

            if face_count > 0:
                first_face = results.face_landmarks[0]

                print(
                    "Total facial landmarks:",
                    len(first_face)
                )

                print("\nFirst 5 landmark coordinates:")

                for index, landmark in enumerate(first_face[:5]):
                    print(
                        index,
                        "X:", landmark.x,
                        "Y:", landmark.y,
                        "Z:", landmark.z
                    )

                # Display facial expression blendshape scores
                if results.face_blendshapes:
                    print("\nFacial blendshape information:")

                    for item in results.face_blendshapes[0][:10]:
                        print(
                            item.category_name,
                            ":",
                            round(item.score, 3)
                        )

        # Quit when Q is pressed
        if key == ord("q"):
            break

# Release webcam and close windows
camera.release()
cv2.destroyAllWindows()

print("Face landmark detection stopped.")
