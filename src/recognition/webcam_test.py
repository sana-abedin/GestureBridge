
import cv2

# Step 1: Access the laptop webcam
camera = cv2.VideoCapture(0)

# Step 2: Check whether the webcam opened successfully
if not camera.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam started successfully!")
print("Press Q to close the webcam.")

# Step 3: Continuously capture video frames
while True:

    success, frame = camera.read()

    # Step 4: Check whether a frame was captured
    if not success:
        print("Error: Could not read camera frame.")
        break

    # Step 5: Flip the image horizontally
    frame = cv2.flip(frame, 1)

    # Step 6: Display the webcam video
    cv2.imshow("GestureBridge - Webcam Test", frame)

    # Step 7: Close when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 8: Release the webcam
camera.release()

# Step 9: Close all OpenCV windows
cv2.destroyAllWindows()

print("Webcam closed successfully.")