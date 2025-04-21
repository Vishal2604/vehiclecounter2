# import numpy as np
# import cv2
# import cvzone
# import math
# from openvino.runtime import Core
# from sort import Sort
# import os

# # Load OpenVINO Model for GPU
# ie = Core()
# # model_path = "yolov8.xml"  # Path to OpenVINO IR model

# # model_path = "yolov8.xml"  # Ensure correct filename
# # model_path = "C:/Users/Nilesh Singh/OneDrive/Desktop/VehicleCounter/Yolo-Weights/yolov8n.pt"  # Ensure correct filename
# model_path = "C:/Users/Nilesh Singh/OneDrive/Desktop/VehicleCounter/yolov8n_openvino_model/yolov8n.xml"  # Ensure correct filename
# print(f"Checking model path: {model_path}")

# if not os.path.exists(model_path):
#     raise FileNotFoundError(f"🚨 Model file not found at: {os.path.abspath(model_path)}")

# compiled_model = ie.compile_model(model_path, "GPU")

# # Video Capture
# cap = cv2.VideoCapture("C:/Users/Nilesh Singh/OneDrive/Desktop/VehicleCounter/vehiclecounter/assets/cars.mp4")  # For Video

# # Class Names
# # classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat"]
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"
#               ]

# # Load Mask
# mask = cv2.imread("C:/Users/Nilesh Singh/OneDrive/Desktop/VehicleCounter/vehiclecounter/assets/mask-1280-720.png")

# # Tracking
# tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
# limits = [400, 297, 673, 297]
# totalCount = []

# while cap.isOpened():
#     success, img = cap.read()
#     if not success:
#         break

#     imgRegion = cv2.bitwise_and(img, mask) if mask is not None else img

#     # Preprocessing for OpenVINO
#     input_blob = cv2.resize(imgRegion, (640, 640))  # Resize to model input size
#     input_blob = np.transpose(input_blob, (2, 0, 1))  # HWC to CHW
#     input_blob = np.expand_dims(input_blob, axis=0).astype(np.float32) / 255.0

#     # Inference on GPU
#     results = compiled_model([input_blob])[compiled_model.outputs[0]]

#     detections = np.empty((0, 5))
#     # for detection in results[0]:
#     #     x1, y1, x2, y2, conf, cls = detection[:6]
#     #     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
#         # if classNames[int(cls)] in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
#     #         currentArray = np.array([x1, y1, x2, y2, conf])
#     #         detections = np.vstack((detections, currentArray))
#     # for r in results:
#     #     boxes = r.boxes
#     #     for box in boxes:
#     # for detection in results[0]:
#     for detection in results:
#         # Bounding Box
#         # x1, y1, x2, y2 = box.xyxy[0]
#         x1, y1, x2, y2, conf, cls = detection[:6]
#         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#         # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
#         w, h = x2 - x1, y2 - y1

#         # Confidence
#         # conf = math.ceil((box.conf[0] * 100)) / 100
#         # Class Name
#         # cls = int(box.cls[0])
#         # cls = int(cls[0])
#         # currentClass = classNames[cls]
#         currentClass = classNames[int(cls)]

#         if currentClass == "car" or currentClass == "truck" or currentClass == "bus" \
#                 or currentClass == "motorbike" and conf > 0.3:
#             # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
#             #                    scale=0.6, thickness=1, offset=3)
#             # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
#             currentArray = np.array([x1, y1, x2, y2, conf])
#             detections = np.vstack((detections, currentArray))




#     resultsTracker = tracker.update(detections)
    
#     # Draw Line
#     cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    
#     for result in resultsTracker:
#         x1, y1, x2, y2, obj_id = map(int, result)
#         w, h = x2 - x1, y2 - y1
#         cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
#         cvzone.putTextRect(img, f' {int(obj_id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

#         cx, cy = x1 + w // 2, y1 + h // 2
#         cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

#         if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
#             if totalCount.count(obj_id) == 0:
#                 totalCount.append(obj_id)
#                 cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

#     cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
#     cv2.imshow("Image", img)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import numpy as np
import cv2
import cvzone
import math
from openvino.runtime import Core
from sort import Sort

# Load OpenVINO Model for GPU
ie = Core()
# model_path = "yolov8.xml"  # Path to OpenVINO IR model
model_path = "C:/Users/Nilesh Singh/OneDrive/Desktop/VehicleCounter/yolov8n_openvino_model/yolov8n.xml"
compiled_model = ie.compile_model(model_path, "GPU")

# Video Capture
cap = cv2.VideoCapture("C:/Users/Nilesh Singh/OneDrive/Desktop/VehicleCounter/vehiclecounter/assets/cars.mp4")

# Class Names
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


# Load Mask
mask = cv2.imread("C:/Users/Nilesh Singh/OneDrive/Desktop/VehicleCounter/vehiclecounter/assets/mask-1280-720.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limits = [400, 297, 673, 297]
totalCount = []

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    imgRegion = cv2.bitwise_and(img, mask) if mask is not None else img

    # Preprocessing for OpenVINO
    input_blob = cv2.resize(imgRegion, (640, 640))  # Resize to model input size
    input_blob = np.transpose(input_blob, (2, 0, 1))  # HWC to CHW
    input_blob = np.expand_dims(input_blob, axis=0).astype(np.float32) / 255.0

    # Inference on GPU
    results = compiled_model([input_blob])[compiled_model.outputs[0]]

    detections = np.empty((0, 5))
    for detection in results[0]:
        x1, y1, x2, y2, conf, cls = detection[:6]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        if classNames[int(cls)] in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
            currentArray = np.array([x1, y1, x2, y2, conf])
            detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    
    # Draw Line
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
    
    for result in resultsTracker:
        x1, y1, x2, y2, obj_id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f' {int(obj_id)}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if totalCount.count(obj_id) == 0:
                totalCount.append(obj_id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# cap.release()
# cv2.destroyAllWindows()
