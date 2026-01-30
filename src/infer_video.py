from ultralytics import YOLO
import cv2
import time

# Load edge-friendly model
model = YOLO("yolov8n.pt")

# Video path (you can use any road video)
video_path = "data/Indian_vehicle_dataset/9c64085f-1c36-48f8-8924-1e8b310ab39d.mp4"
cap = cv2.VideoCapture(video_path)

vehicle_classes = ["car", "motorcycle", "bus", "truck"]

prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO inference
    results = model(frame, conf=0.4, verbose=False)
    detections = results[0].boxes

    count = 0
    for box in detections:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        if cls_name in vehicle_classes:
            count += 1

    # FPS calculation
    curr_time = time.time()
    fps = int(1 / (curr_time - prev_time)) if prev_time else 0
    prev_time = curr_time

    # Traffic density logic
    if count <= 5:
        density = "LOW"
        color = (0, 255, 0)
    elif count <= 15:
        density = "MEDIUM"
        color = (0, 255, 255)
    else:
        density = "HIGH"
        color = (0, 0, 255)

    # Overlay text
    cv2.putText(frame, f"Vehicles: {count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Density: {density}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"FPS: {fps}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Smart City Traffic Analytics", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
