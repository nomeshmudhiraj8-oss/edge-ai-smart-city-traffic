

from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

img_path = "data/Indian_vehicle_dataset/20210430_19_00_07_000_M0qQbjq7mhasETXjCQGvnoZxK3a2_T_4160_3120.jpg"

# Run inference
results = model(img_path, conf=0.4)

# Get detections
detections = results[0].boxes

# Vehicle classes from COCO
vehicle_classes = ["car", "motorcycle", "bus", "truck"]

count = 0

for box in detections:
    cls_id = int(box.cls[0])
    cls_name = model.names[cls_id]
    if cls_name in vehicle_classes:
        count += 1

print(f"🚗 Total Vehicles Detected: {count}")

# Show image
results[0].show()

