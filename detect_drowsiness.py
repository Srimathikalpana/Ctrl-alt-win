from ultralytics import YOLO
import cv2

# Load YOLO model
model = YOLO("best.pt")  # Ensure the correct path to best.pt

# Define class names from your dataset
class_names = {0: "Droopy-eye-lids", 1: "Face-dropping", 2: "Yawning"}

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Run YOLO detection
    results = model(frame)

    drowsy_detected = False  # Flag to track drowsiness

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Get class ID
            label = class_names.get(class_id, "Unknown")  # Get class label
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Check if detected class is related to drowsiness
            if label in ["Droopy-eye-lids", "Yawning", "Face-dropping"]:
                drowsy_detected = True

    # Display status text
    status_text = "DROWSY" if drowsy_detected else "AWAKE"
    color = (0, 0, 255) if drowsy_detected else (0, 255, 0)  # Red for drowsy, Green for awake
    cv2.putText(frame, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Driver Monitoring System", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
