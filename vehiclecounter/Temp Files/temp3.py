import numpy as np
import cv2
import cvzone
import math
import time
from openvino.runtime import Core
from sort import Sort  # Using your existing SORT implementation

# Initialize OpenVINO
ie = Core()

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute new unpadded dimensions
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    
    # Calculate padding
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    # Divide padding into 2 sides
    dw /= 2
    dh /= 2
    
    # Resize
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    # Calculate top, bottom, left, right padding
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    # Add padding
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return im, r, (dw, dh)

class YoloDetector:
    def __init__(self, model_path, device="GPU", conf_threshold=0.3):
        # Load the YOLOv8 model through OpenVINO
        self.model = ie.read_model(model_path)
        # Load the model on the specified device (GPU by default)
        self.compiled_model = ie.compile_model(model=self.model, device_name=device)
        
        # Get input and output layers
        self.input_layer = next(iter(self.compiled_model.inputs))
        self.output_layers = list(self.compiled_model.outputs)
        
        # Model parameters
        self.input_size = self.input_layer.shape[2:]  # HW format
        self.conf_threshold = conf_threshold
        
        # Class names
        self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]
        
        # Target classes for vehicle detection
        self.target_classes = ["car", "truck", "bus", "motorbike"]
        self.target_indices = [self.classNames.index(cls) for cls in self.target_classes]
        
        # Debug model output shape
        print("Model input shape:", self.input_layer.shape)
        print("Model output layers:", len(self.output_layers))
        for i, output in enumerate(self.output_layers):
            print(f"Output layer {i} shape:", output.shape)
    
    def preprocess(self, image):
        # Save original image dimensions
        self.orig_shape = image.shape[:2]  # (H, W)
        
        # Preprocess the image to match model input size
        preprocessed_img, self.scale, self.pad = letterbox(image, new_shape=self.input_size)
        
        # Convert to the format expected by the model (NCHW)
        blob = cv2.cvtColor(preprocessed_img, cv2.COLOR_BGR2RGB)
        blob = blob.transpose(2, 0, 1)  # HWC to CHW
        blob = np.expand_dims(blob, axis=0).astype(np.float32)
        blob /= 255.0  # Normalize
        
        return blob
    
    def process_yolov8_output(self, output):
        """Process YOLOv8 OpenVINO model output (detection format)"""
        detections = np.empty((0, 5))  # Format: x1, y1, x2, y2, conf

        # Check the shape of the output to determine format
        if len(output.shape) == 3:  # YOLOv8 detection output format [1, num_boxes, 84]
            # Last dimension contains box coordinates, objectness score, and class scores
            for i in range(output.shape[1]):
                box = output[0, i, :]
                confidence = box[4]  # Object confidence
                
                if confidence >= self.conf_threshold:
                    # Check if any class probability exceeds threshold
                    class_scores = box[5:]
                    class_id = np.argmax(class_scores)
                    class_conf = class_scores[class_id]
                    
                    # Filter for vehicle classes
                    if class_id in self.target_indices and class_conf > self.conf_threshold:
                        # Get scaled coordinates
                        x, y, w, h = box[0:4]
                        
                        # Convert from center coordinates to corner coordinates and scale to original image
                        x1 = int((x - w/2) / self.scale - self.pad[0])
                        y1 = int((y - h/2) / self.scale - self.pad[1])
                        x2 = int((x + w/2) / self.scale - self.pad[0])
                        y2 = int((y + h/2) / self.scale - self.pad[1])
                        
                        # Clip to image boundaries
                        x1, y1 = max(0, x1), max(0, y1)
                        x2 = min(self.orig_shape[1], x2)
                        y2 = min(self.orig_shape[0], y2)
                        
                        # Add detection
                        current_detection = np.array([x1, y1, x2, y2, class_conf])
                        detections = np.vstack((detections, current_detection))
        else:
            # Handle alternative YOLOv8 output format if needed
            print(f"Unexpected output shape: {output.shape}")
            
        return detections
    
    def detect(self, image):
        # Preprocess image
        input_blob = self.preprocess(image)
        
        # Perform inference
        outputs = self.compiled_model([input_blob])
        
        # Get the detection output (format depends on YOLOv8 version and export options)
        # Try to find the detection output among the model outputs
        detections = np.empty((0, 5))
        class_names = []
        
        # Debug outputs
        for output_name, output_data in outputs.items():
            print(f"Output {output_name}: shape {output_data.shape}")
            
            # Try to process this output
            try:
                detections = self.process_yolov8_output(output_data)
                # If we got detections, we found the right output
                if len(detections) > 0:
                    break
            except Exception as e:
                print(f"Error processing output {output_name}: {e}")
                continue
        
        # Return detections and dummy class names (we'll use the IDs from tracker)
        return detections, ["vehicle"] * len(detections)

def main():
    # Video capture - replace with camera number for webcam (e.g., 0)
    cap = cv2.VideoCapture("C:/Users/Nilesh Singh/OneDrive/Desktop/VehicleCounter/vehiclecounter/assets/cars.mp4")
    
    # Set resolution to 1280x720 for better performance (if using webcam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Check if export model exists, if not use the direct ONNX conversion
    import os
    model_path = "yolov8n_openvino_model/yolov8n.xml"
    
    if not os.path.exists(model_path):
        print("OpenVINO model not found. Converting YOLO model to OpenVINO format...")
        try:
            from ultralytics import YOLO
            # Load and export the model
            yolo_model = YOLO("yolov8n.pt")  # Using smaller nano model for better speed
            yolo_model.export(format="openvino")
            model_path = "yolov8n_openvino_model/yolov8n.xml"
            print(f"Model exported to {model_path}")
        except Exception as e:
            print(f"Error exporting model: {e}")
            print("Please download the YOLOv8n OpenVINO model manually.")
            return
    
    print(f"Using model: {model_path}")
    
    try:
        # Create detector
        detector = YoloDetector(model_path, device="GPU", conf_threshold=0.3)
        
        # Load mask for ROI detection
        mask_path = "C:/Users/Nilesh Singh/OneDrive/Desktop/VehicleCounter/vehiclecounter/assets/mask-1280-720.png"
        mask = cv2.imread(mask_path) if os.path.exists(mask_path) else None
        if mask is None:
            print(f"Warning: Mask not found at {mask_path}. Continuing without mask.")
        
        # Load and prepare graphics overlay
        graphics_path = "C:/Users/Nilesh Singh/OneDrive/Desktop/VehicleCounter/vehiclecounter/assets/graphics.png"
        imgGraphics = cv2.imread(graphics_path, cv2.IMREAD_UNCHANGED) if os.path.exists(graphics_path) else None
        if imgGraphics is None:
            print(f"Warning: Graphics overlay not found at {graphics_path}. Continuing without graphics.")
        
        # Initialize tracker
        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        
        # Define counting line
        limits = [400, 297, 673, 297]
        totalCount = []
        
        # FPS calculation variables
        prev_time = 0
        fps_history = []
        
        # Process first frame to verify everything works
        ret, first_frame = cap.read()
        if not ret:
            print("Error reading video file")
            return
            
        print("Processing first frame to verify setup...")
        imgRegion = cv2.bitwise_and(first_frame, mask) if mask is not None else first_frame
        first_detections, _ = detector.detect(imgRegion)
        print(f"First frame detection count: {len(first_detections)}")
        
        # Reset video capture
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time) if prev_time > 0 else 0
            prev_time = current_time
            fps_history.append(fps)
            if len(fps_history) > 30:  # Average over last 30 frames
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)
            
            # Read frame
            success, img = cap.read()
            if not success:
                print("End of video or error reading frame")
                break
            
            # Apply mask for region of interest
            imgRegion = cv2.bitwise_and(img, mask) if mask is not None else img
            
            # Overlay graphics
            if imgGraphics is not None:
                try:
                    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
                except Exception as e:
                    print(f"Error overlaying graphics: {e}")
            
            # Detect objects
            try:
                detections, _ = detector.detect(imgRegion)
            except Exception as e:
                print(f"Error during detection: {e}")
                detections = np.empty((0, 5))
            
            # Update tracker
            resultsTracker = tracker.update(detections)
            
            # Draw counting line
            cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
            
            # Process tracking results
            for result in resultsTracker:
                x1, y1, x2, y2, id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Calculate width and height
                w, h = x2 - x1, y2 - y1
                
                # Draw bounding box and ID
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                cvzone.putTextRect(img, f' {int(id)}', (max(0, x1), max(35, y1)),
                                  scale=2, thickness=3, offset=10)
                
                # Calculate center point
                cx, cy = x1 + w // 2, y1 + h // 2
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                
                # Check if vehicle crossed the counting line
                if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                    if totalCount.count(id) == 0:
                        totalCount.append(id)
                        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
            
            # Display vehicle count
            cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
            
            # Display FPS
            cv2.putText(img, f"FPS: {avg_fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow("Image", img)
            
            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()