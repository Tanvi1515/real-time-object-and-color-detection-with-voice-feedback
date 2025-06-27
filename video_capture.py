import cv2

# Open webcam (Use 0 for default camera, 1 for external camera)
cap = cv2.VideoCapture(0)

# Reduce frame resolution for faster processing
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Increase FPS (optional, some webcams may not support higher FPS)
cap.set(cv2.CAP_PROP_FPS, 30)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit if no frame is captured

    cv2.imshow("Live Video", frame)  # Show real-time video

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
