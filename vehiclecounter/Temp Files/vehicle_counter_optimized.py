import cv2
import numpy as np
from openvino.runtime import Core
import cvzone
import math
from sort import Sort
import time

def initialize_model():
    # Initialize OpenVINO Runtime
    ie = Core()
    
    # Load YOLO model - using yolov8n (nano) for better speed
    model = ie.read_model("C:/Users/Nilesh Singh/OneDrive/Desktop/VehicleCounter/vehiclecounter/Temp Files/yolov8n.xml")
    
    # Get available devices - prioritize GPU
    available_devices = ie.available_devices
    device = "GPU.0" if "GPU.0" in available_devices else "CPU"
    print(f"Using device: {device}")
    
    # Compile the model for the selected device
    compiled_model = ie.compile_model(model=model, device_name=device)
    
    output_layer = compiled_model.output(0)
    return compiled_model, output_layer

def process_frame(frame, compiled_model, output_layer):
    # Preprocess frame
    input_image = cv2.resize(frame, (640, 640))
    input_image = input_image / 255.0
    input_image = input_image.transpose((2, 0, 1))
    input_image = np.expand_dims(input_image, 0).astype(np.float32)
    
    # Run inference
    results = compiled_model([input_image])[output_layer]
    return results

def main():
    # Initialize video capture
    cap = cv2.VideoCapture("C:/Users/Nilesh Singh/OneDrive/Desktop/VehicleCounter/vehiclecounter/assets/cars.mp4")
    
    # Initialize model
    compiled_model, output_layer = initialize_model()
    
    # Initialize tracker
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    
    # Load mask and graphics
    mask = "C:/Users/Nilesh Singh/OneDrive/Desktop/VehicleCounter/vehiclecounter/assets/mask-1280-720.png"
    imgGraphics = cv2.imread("C:/Users/Nilesh Singh/OneDrive/Desktop/VehicleCounter/vehiclecounter/assets/graphics.png", cv2.IMREAD_UNCHANGED)
    
    # Counter setup
    limits = [400, 297, 673, 297]
    totalCount = []
    
    # Class names
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck"]  # shortened for brevity
    
    # FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    while True:
        success, img = cap.read()
        if not success:
            break
            
        # Update FPS
        fps_counter += 1
        if fps_counter % 30 == 0:
            fps = 30/(time.time() - fps_start_time)
            fps_start_time = time.time()
        
        # Apply mask
        # imgRegion = cv2.bitwise_and(img, mask)
        
        # Overlay graphics
        img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
        
        # Process frame and get detections
        # results = process_frame(imgRegion, compiled_model, output_layer)
        results = process_frame(img, compiled_model, output_layer)
        
        detections = np.empty((0, 5))
        
        # Process detections
        for detection in results[0]:
            confidence = detection[4]
            if confidence > 0.3:
                class_id = int(detection[5])
                if class_id in [2, 5, 7]:  # car, bus, truck
                    x1, y1, x2, y2 = (detection[:4] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])).astype(int)
                    currentArray = np.array([x1, y1, x2, y2, confidence])
                    detections = np.vstack((detections, currentArray))
        
        # Update tracker
        resultsTracker = tracker.update(detections)
        
        # Draw counting line
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
        
        # Process tracking results
        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            
            # Draw bounding box
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                             scale=2, thickness=3, offset=10)
            
            # Calculate center point
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            
            # Check if vehicle crossed the line
            if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                if totalCount.count(id) == 0:
                    totalCount.append(id)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
        
        # Display count and FPS
        cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
        cv2.putText(img, f'FPS: {int(fps)}', (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 