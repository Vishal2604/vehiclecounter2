import cv2
import numpy as np
from openvino.runtime import Core
from ultralytics.utils.ops import xywh2xyxy
import time
from sort import Sort  # Import SORT tracker

# Load OpenVINO model
core = Core()
model_path = "yolov8l_openvino_model.xml"
model = core.read_model(model_path)
compiled_model = core.compile_model(model, "GPU")

# Load video
video_path = "cars.mp4"  # Change to your video file
cap = cv2.VideoCapture(video_path)

# Read input shape
input_layer = compiled_model.input(0)
input_shape = input_layer.shape
img_size = (input_shape[2], input_shape[3])

# Class names for YOLOv8 (filtering for vehicles)
class_names = ["person", "bicycle", "car", "motorbike", "bus", "truck"]
vehicle_classes = ["car", "motorbike", "bus", "truck"]

# Initialize SORT tracker
tracker = Sort()
vehicle_counted = set()  # Track counted vehicle IDs

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess image
    input_image = cv2.resize(frame, img_size)
    input_image = input_image.transpose((2, 0, 1))  # HWC to CHW
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32)

    # Inference
    start_time = time.time()
    output = compiled_model([input_image])[compiled_model.outputs[0]]
    end_time = time.time()

    detections = []

    # Post-process output
    for detection in output:
        x, y, w, h, conf, cls = detection[:6]
        if conf > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = xywh2xyxy(np.array([x, y, w, h])).astype(int)[0]
            class_name = class_names[int(cls)]
            
            if class_name in vehicle_classes:
                detections.append([x1, y1, x2, y2, conf])  # Append detection for tracking

    # Track vehicles using SORT
    tracked_objects = tracker.update(np.array(detections))

    for obj in tracked_objects:
        x1, y1, x2, y2, obj_id = obj.astype(int)
        if obj_id not in vehicle_counted:
            vehicle_counted.add(obj_id)  # Count new vehicles

        # Draw bounding box and ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display vehicle count
    cv2.putText(frame, f"Vehicles Counted: {len(vehicle_counted)}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show FPS
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display frame
    cv2.imshow("Vehicle Counter (SORT)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
