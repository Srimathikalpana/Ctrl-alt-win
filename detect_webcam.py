import cv2
import numpy as np
import winsound  # For alarm sound (Windows)
import os  # For cross-platform sound support
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO(r"C:\Users\sri\Desktop\Ideathon& Hacksphere\Ideathon\Ctrl-alt-win\best.pt")

# Class names from the dataset
class_names = {
    0: 'Attentive eye', 1: 'Drowsy eye', 2: 'Eyeclosed', 3: 'Open-Mouth', 4: 'Yawn',
    5: 'asleep', 6: 'close', 7: 'closed', 8: 'noYawn', 9: 'open', 10: 'yawn'
}

# Assign a unique color to each class
np.random.seed(42)  # Fix seed for consistent colors
colors = {i: tuple(np.random.randint(100, 255, 3).tolist()) for i in class_names.keys()}

# Labels that indicate the person is asleep
asleep_labels = {'asleep', 'Eyeclosed', 'closed','Face-dropping'}

# Open webcam
cap = cv2.VideoCapture(0)
alarm_on = False  # Alarm state

def play_alarm():
    """Plays a warning beep when drowsiness is detected."""
    frequency = 2500  # Frequency in Hz
    duration = 1000   # Duration in milliseconds
    try:
        # Windows beep
        winsound.Beep(frequency, duration)
    except:
        # Linux/macOS alternative
        os.system('play -nq -t alsa synth 1 sine {}'.format(frequency))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model.predict(frame, save=False, conf=0.5)  # Adjust confidence threshold if needed

    # Check if drowsiness is detected
    drowsy_detected = False

    # Loop through detected objects
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class index
            label = class_names.get(cls, "Unknown")

            # Check if detected label is related to drowsiness
            if label in asleep_labels:
                drowsy_detected = True

            # Choose color based on class
            color = colors.get(cls, (0, 255, 0))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background for visibility
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
            
            # Draw text label
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0), 2)

    # Trigger alarm if drowsy state is detected
    if drowsy_detected and not alarm_on:
        alarm_on = True
        play_alarm()
    elif not drowsy_detected:
        alarm_on = False  # Reset alarm state when awake

    # Display the output
    cv2.imshow("Drowsiness Detection", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

