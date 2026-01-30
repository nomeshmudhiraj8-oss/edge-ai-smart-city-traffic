import streamlit as st
import cv2
import time
from ultralytics import YOLO

st.set_page_config(page_title="Smart City Traffic Analytics", layout="wide")

st.title("🚦 Smart City Traffic Analytics (Edge AI)")
st.write("Real-time vehicle detection and traffic density analysis on low-power devices")

# Load model
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

vehicle_classes = ["car", "motorcycle", "bus", "truck"]

video_path = st.text_input(
    "Enter video path",
    "data/Indian_vehicle_dataset/9c64085f-1c36-48f8-8924-1e8b310ab39d.mp4"
)

start = st.button("Start Analysis")

if start:
    cap = cv2.VideoCapture(video_path)

    frame_placeholder = st.empty()
    stats_placeholder = st.empty()

    prev_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4, verbose=False)
        detections = results[0].boxes

        count = 0
        for box in detections:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            if cls_name in vehicle_classes:
                count += 1

        curr_time = time.time()
        fps = int(1 / (curr_time - prev_time)) if prev_time else 0
        prev_time = curr_time

        if count <= 5:
            density = "LOW"
        elif count <= 15:
            density = "MEDIUM"
        else:
            density = "HIGH"

        cv2.putText(frame, f"Vehicles: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Density: {density}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")

        stats_placeholder.markdown(
            f"""
            ### 📊 Live Stats
            - 🚗 Vehicles Detected: **{count}**
            - 🚦 Traffic Density: **{density}**
            - ⚡ FPS: **{fps}**
            """
        )

    cap.release()
