import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO(r"C:\Users\dell\OneDrive\Desktop\ctrl+alt+win\driverdrowsiness\yolov8s.pt")  # Update the path if needed

# Open webcam
cap = cv2.VideoCapture(0)  # Change to 1 if external webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 on the frame
    results = model(frame)

    # Display results
    for r in results:
        frame = r.plot()

    cv2.imshow("YOLO Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
