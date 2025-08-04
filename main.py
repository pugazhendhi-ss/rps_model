from ultralytics import YOLO
import cv2

# Load your trained YOLO model
model_path = "D:/pyprojects/rps_model/ml_model/rps_yolov8n.pt"
model = YOLO(model_path)

# Class names (edit if needed based on your dataset order)
class_names = ["Paper", "Rock", "Scissors"]

# Open webcam (0 is default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'Q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Run inference on current frame
    results = model(frame, verbose=False)

    # Visualize results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            cls_id = int(box.cls[0])               # Class ID
            conf = float(box.conf[0])              # Confidence

            label = f"{class_names[cls_id]} {conf:.2f}"
            color = (0, 255, 0)  # Green box

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Show the frame with boxes
    cv2.imshow("RPS Detector", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
