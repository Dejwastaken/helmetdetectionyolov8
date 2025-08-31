import cv2
import math
import cvzone
from ultralytics import YOLO
import numpy as np
from collections import Counter
import os
import sys
import argparse
import onnxruntime as ort
import time

def get_model_choice():
    print('\n' + '='*50)
    print('🤖 MODEL SELECTION')
    print('='*50)
    print('1. PyTorch/YOLO models (.pt files)')
    print('2. ONNX models - Normal (.onnx files)')
    print('3. ONNX models - Simplified (.onnx files)')
    print('='*50)
    
    while True:
        choice = input('Choose model type (1, 2, or 3): ').strip()
        if choice == '1':
            return 'pytorch'
        elif choice == '2':
            return 'onnx_normal'
        elif choice == '3':
            return 'onnx_simplified'
        else:
            print('❌ Invalid choice. Please enter 1, 2, or 3.')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Helmet Detection Video Processing')
    parser.add_argument('--model-type', choices=['pytorch', 'onnx_normal', 'onnx_simplified'],
                       help='Choose model type: pytorch (.pt), onnx_normal (.onnx), or onnx_simplified (.onnx). If not provided, will prompt for selection.')
    parser.add_argument('--video-name', 
                       help='Video name (without .mp4 extension). If not provided, will prompt for input.')
    return parser.parse_args()

def print_static_progress(current, total, start_time):
    """Print a static progress bar that overwrites the previous line"""
    percent = (current / total) * 100
    bar_length = 40
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + '░' * (bar_length - filled_length)
    
    elapsed_time = time.time() - start_time
    fps = current / elapsed_time if elapsed_time > 0 else 0
    eta = (total - current) / fps if fps > 0 else 0
    
    # Format time
    def format_time(seconds):
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds//60:.0f}m {seconds%60:.0f}s"
        else:
            return f"{seconds//3600:.0f}h {(seconds%3600)//60:.0f}m"
    
    # Use sys.stdout to ensure no newline and proper flushing
    progress_line = f"\r📊 [{bar}] {percent:.1f}% | {current:4d}/{total} frames | {fps:.1f} fps | ETA: {format_time(eta):>8}"
    sys.stdout.write(progress_line)
    sys.stdout.flush()

args = parse_arguments()

# Get model choice if not provided via command line
if args.model_type is None:
    model_type = get_model_choice()
else:
    model_type = args.model_type

print('\n' + '='*60)
print('🔍 HELMET DETECTION SYSTEM')
print('='*60)
print(f'🤖 Model type: {model_type.upper()}')
print('='*60)

if args.video_name:
    video_name = args.video_name
else:
    print('\n📹 VIDEO SELECTION')
    print('-' * 30)
    video_name = input("Enter the video name (without .mp4): ")

# Initialize video capture
video_path = f"media/{video_name}.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties for output
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create video writer for output
output_path = f"media/output/{video_name}_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Get total frame count for progress bar
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Load models based on selected type
if model_type == 'pytorch':
    print("🔥 Loading PyTorch models...")
    vehicle_model = YOLO("models/py/yolov8m.pt")
    helmet_model = YOLO("models/py/helmet_detection_trained.pt")
    use_onnx = False
    print("✅ PyTorch models loaded successfully")
    
elif model_type == 'onnx_normal':
    print("⚡ Loading ONNX models (Normal) with pure ONNX runtime...")
    vehicle_onnx_path = "models/onnx/yolov8m.onnx"
    helmet_onnx_path = "models/onnx/helmet_detection_trained.onnx"
    
    try:
        # Load ONNX models using pure ONNX runtime
        vehicle_model = ort.InferenceSession(vehicle_onnx_path)
        helmet_model = ort.InferenceSession(helmet_onnx_path)
        use_onnx = True
        print(f"✅ ONNX Normal models loaded successfully with pure runtime")
        print(f"  🚗 Vehicle: {vehicle_onnx_path}")
        print(f"  🪖 Helmet: {helmet_onnx_path}")
    except Exception as e:
        print(f"❌ Error loading ONNX Normal models: {e}")
        print("Falling back to PyTorch models...")
        vehicle_model = YOLO("models/py/yolov8m.pt")
        helmet_model = YOLO("models/py/helmet_detection_trained.pt")
        use_onnx = False
        
elif model_type == 'onnx_simplified':
    print("🚀 Loading ONNX models (Simplified) with pure ONNX runtime...")
    vehicle_onnx_path = "models/onnx/yolov8m_simplified.onnx"
    helmet_onnx_path = "models/onnx/helmet_detection_trained_simplified.onnx"
    
    try:
        # Load ONNX models using pure ONNX runtime
        vehicle_model = ort.InferenceSession(vehicle_onnx_path)
        helmet_model = ort.InferenceSession(helmet_onnx_path)
        use_onnx = True
        print(f"✅ ONNX Simplified models loaded successfully with pure runtime")
        print(f"  🚗 Vehicle: {vehicle_onnx_path}")
        print(f"  🪖 Helmet: {helmet_onnx_path}")
    except Exception as e:
        print(f"❌ Error loading ONNX Simplified models: {e}")
        print("Falling back to PyTorch models...")
        vehicle_model = YOLO("models/py/yolov8m.pt")
        helmet_model = YOLO("models/py/helmet_detection_trained.pt")
        use_onnx = False

# Define class names for helmet detection
classNames = ['With Helmet', 'Without Helmet']

MOTORCYCLE_CLASS = 3  # motorcycle in COCO
BICYCLE_CLASS = 1     # bicycle in COCO

# Polygon coordinates from user (HTML image map)
polygon = np.array([[2141,1031], [700,700], [700,5], [15,8], [9,1685], [1810,1691]], np.int32)
polygon = polygon.reshape((-1, 1, 2))

print(f"\n🎬 Processing video: {video_path}")
print(f"📁 Output will be saved to: {output_path}")
print(f"📐 Video properties: {width}x{height} @ {fps}fps")
print(f"🎞️ Total frames: {total_frames}")
print(f"🤖 Using {model_type.upper()} models")
print()

frame_count = 0
total_helmets_detected = 0
total_without_helmets_detected = 0
total_motorcycles_detected = 0
total_bicycles_detected = 0
helmet_colors_detected = Counter()  # Track helmet colors
motorcycle_colors_detected = Counter()  # Track motorcycle colors

def detect_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Mask out very low saturation/brightness for colored pixels
    color_mask = (hsv[:,:,1] > 50) & (hsv[:,:,2] > 50)
    hue_values = hsv[:,:,0][color_mask]

    # Check for black / dark pixels
    dark_mask = hsv[:,:,2] < 50  # low brightness
    dark_ratio = np.sum(dark_mask) / (img.shape[0] * img.shape[1])

    if dark_ratio > 0.5:  # If most pixels are dark, consider it black
        return "Black", None

    if len(hue_values) == 0:
        return "Unknown", None

    # Dominant hue
    hist, bins = np.histogram(hue_values, bins=180, range=[0,180])
    dominant_hue = np.argmax(hist)
    h = dominant_hue

    # Map hue → color
    if (h >= 0 and h <= 10) or (h >= 170 and h <= 179):
        color_name = "Red"
    elif h >= 11 and h <= 20:
        color_name = "Orange"
    elif h >= 21 and h <= 30:
        color_name = "Yellow"
    elif h >= 35 and h <= 85:
        color_name = "Green"
    elif h >= 90 and h <= 130:
        color_name = "Blue"
    elif h >= 131 and h <= 160:
        color_name = "Purple"
    elif h >= 161 and h <= 170:
        color_name = "Pink"
    else:
        color_name = "Unknown"


    return color_name, h

def preprocess_onnx_input(img, input_size=(640, 640)):
    """Preprocess image for ONNX model inference"""
    # Resize image while maintaining aspect ratio
    h, w = img.shape[:2]
    scale = min(input_size[0]/w, input_size[1]/h)
    new_w, new_h = int(w*scale), int(h*scale)
    
    # Resize and pad
    resized = cv2.resize(img, (new_w, new_h))
    
    # Create padded image
    padded = np.full((*input_size, 3), 114, dtype=np.uint8)
    y_offset = (input_size[1] - new_h) // 2
    x_offset = (input_size[0] - new_w) // 2
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    # Convert BGR to RGB and normalize
    padded = padded[:, :, ::-1]  # BGR to RGB
    padded = padded.astype(np.float32) / 255.0
    
    # Transpose to NCHW format
    padded = np.transpose(padded, (2, 0, 1))
    padded = np.expand_dims(padded, axis=0)
    
    return padded, scale, x_offset, y_offset

def postprocess_onnx_output(output, scale, x_offset, y_offset, conf_threshold=0.25):
    """Postprocess ONNX model output (assumes NMS is already applied)"""
    detections = []
    
    # Output format: [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, conf, class]
    if len(output.shape) == 3:
        output = output[0]  # Remove batch dimension
    
    for detection in output:
        if len(detection) >= 6:
            x1, y1, x2, y2, conf, cls = detection[:6]
            
            if conf >= conf_threshold:
                # Scale back to original image coordinates
                x1 = (x1 - x_offset) / scale
                y1 = (y1 - y_offset) / scale
                x2 = (x2 - x_offset) / scale
                y2 = (y2 - y_offset) / scale
                
                detections.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)])
    
    return detections

def run_onnx_inference(session, img, conf_threshold=0.25):
    """Run inference using pure ONNX runtime"""
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    input_size = (input_shape[2], input_shape[3])  # Height, Width
    
    # Preprocess
    processed_img, scale, x_offset, y_offset = preprocess_onnx_input(img, input_size)
    
    # Run inference
    outputs = session.run(None, {input_name: processed_img})
    
    # Postprocess
    detections = postprocess_onnx_output(outputs[0], scale, x_offset, y_offset, conf_threshold)
    
    return detections

# Initialize progress tracking
start_time = time.time()
print("🚀 Starting video processing...")
print()  # Single blank line before progress bar starts

while True:
    success, img = cap.read()
    if not success:
        break
    
    frame_count += 1
    
    # Update static progress bar every 5 frames or on last frame
    if frame_count % 5 == 0 or frame_count == total_frames:
        print_static_progress(frame_count, total_frames, start_time)
    
    # Initialize detection arrays for every frame
    motorcycles_detected = []
    bicycles_detected = []
    helmet_detections_found = 0
    
    # Run detection every 3 frames for optimal balance of performance and smoothness
    if frame_count % 3 == 0:
        # Vehicle detection with optimized confidence thresholds
        if use_onnx:
            # Pure ONNX runtime inference with optimized thresholds
            vehicle_detections = run_onnx_inference(vehicle_model, img, conf_threshold=0.2)
        else:
            # PyTorch YOLO wrapper inference
            vehicle_results = vehicle_model(img, stream=True, conf=0.45, iou=0.5, verbose=False)
            vehicle_detections = []
            for r in vehicle_results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cls = int(box.cls[0])
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        vehicle_detections.append([x1, y1, x2, y2, conf, cls])

        # Process vehicle detections
        for detection in vehicle_detections:
            x1, y1, x2, y2, conf, cls = detection
            
            # Calculate center of bounding box
            cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
            
            # Check if center is inside polygon
            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                if cls == MOTORCYCLE_CLASS:
                    motorcycles_detected.append((x1, y1, x2, y2, conf))
                    total_motorcycles_detected += 1
                    
                    # Extract motorcycle ROI and detect color
                    motorcycle_roi = img[y1:y2, x1:x2]
                    if motorcycle_roi.size > 0:  # Ensure ROI is not empty
                        color_name, hue_value = detect_color(motorcycle_roi)
                        motorcycle_colors_detected[color_name] += 1
                        
                        # Draw motorcycle bounding box in blue
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cvzone.putTextRect(img, f'Motorcycle {conf:.2f}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1)
                        cvzone.putTextRect(img, f'Color: {color_name}', (max(0, x1), max(55, y1-20)), scale=0.6, thickness=1)
                    else:
                        # Draw motorcycle bounding box in blue (fallback)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cvzone.putTextRect(img, f'Motorcycle {conf:.2f}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1)
                    
                elif cls == BICYCLE_CLASS:
                    bicycles_detected.append((x1, y1, x2, y2, conf))
                    total_bicycles_detected += 1
                    
                    # Draw bicycle bounding box in orange
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cvzone.putTextRect(img, f'Bicycle {conf:.2f}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1)
        
        # Helmet detection with optimized confidence thresholds  
        if use_onnx:
            # Pure ONNX runtime inference - more sensitive for safety
            helmet_detections = run_onnx_inference(helmet_model, img, conf_threshold=0.15)
        else:
            # PyTorch YOLO wrapper inference
            helmet_results = helmet_model(img, stream=True, conf=0.35, iou=0.5, verbose=False)
            helmet_detections = []
            for r in helmet_results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = round(box.conf[0].item(), 2)
                        cls = int(box.cls[0])
                        helmet_detections.append([x1, y1, x2, y2, conf, cls])
        
        # Process helmet detections
        for detection in helmet_detections:
            x1, y1, x2, y2, conf, cls = detection
            cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
            
            # Check if center is inside polygon
            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                if cls == 0:  # With helmet
                    total_helmets_detected += 1
                    
                    # Draw green rectangle for with helmet
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cvzone.putTextRect(img, f'With Helmet {conf:.2f}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1)
                    
                elif cls == 1:  # Without helmet
                    total_without_helmets_detected += 1
                    helmet_detections_found += 1
                    # Draw red rectangle for without helmet
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cvzone.putTextRect(img, f'Without Helmet {conf:.2f}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1)

    # Suppress debug output to keep progress bar static
    # if frame_count % 60 == 0 and helmet_detections_found > 0:
    #     print(f"\n  🚨  Without helmet detections found: {helmet_detections_found}")
            
    # Draw the polygon on the frame for visualization
    cv2.polylines(img, [polygon], isClosed=True, color=(0,255,255), thickness=2)
    
    # Write frame to output video
    out.write(img)

# Final progress update and completion message
print_static_progress(total_frames, total_frames, start_time)
print()  # New line after progress bar
print("✅ Processing complete!")
print(f"🎉 Video processing finished in {time.time() - start_time:.1f} seconds")

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video processing complete! Output saved to: {output_path}")
print(f"Total frames processed: {frame_count}")
print(f"\n📊 Detection Summary:")
print(f"   🏍️  Motorcycles detected: {total_motorcycles_detected}")
print(f"   🚲  Bicycles detected: {total_bicycles_detected}")
print(f"   🪖  With Helmets detected: {total_helmets_detected}")
print(f"   🚨  Without Helmets detected: {total_without_helmets_detected}")

print(f"\n🎨 Motorcycle Colors Detected:")
if motorcycle_colors_detected:
    for color, count in motorcycle_colors_detected.most_common():
        print(f"   🏍️  {color}: {count}")
else:
    print(f"   No motorcycle colors detected")

print(f"\n🎨 Detection Box Colors:")
print(f"   🔵 Blue = Motorcycles")
print(f"   🟠 Orange = Bicycles") 
print(f"   🟢 Green = With Helmet")
print(f"   🔴 Red = Without Helmet")