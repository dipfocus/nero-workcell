#!/usr/bin/env python3
"""
Test YOLO for detecting all objects.
Used for debugging and checking what the camera can detect.
"""
import sys
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cameras import create_camera
from ultralytics import YOLO

# COCO class names
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def main():
    print("Initializing camera...")
    camera = create_camera('realsense', width=640, height=480)
    camera.start()
    
    print("Loading YOLO model...")
    checkpoint_dir = PROJECT_ROOT / 'checkpoints'
    model_path = checkpoint_dir / 'yolo11x.pt'
    model = YOLO(str(model_path))
    
    print("\nStarting all-object detection...")
    print("Press 'q' to quit\n")
    
    while True:
        frame_data = camera.read_frame()
        color_image = frame_data['color']
        
        if color_image is None:
            continue
        
        # Detect all objects
        results = model(color_image, verbose=False)
        
        # Render results
        vis_image = color_image.copy()
        detected_objects = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf < 0.3:  # Low-confidence threshold
                    continue
                
                # Get class name
                class_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
                
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw box
                color = (0, 255, 0)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label text
                text = f"{class_name} ({conf:.2f})"
                cv2.putText(vis_image, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                detected_objects.append((class_name, conf, cls_id))
        
        # Show detected object list
        if detected_objects:
            print(f"\rDetected {len(detected_objects)} objects: ", end="")
            for obj_name, obj_conf, obj_id in detected_objects:
                print(f"{obj_name}(ID:{obj_id}, {obj_conf:.2f}) ", end="")
            print(" " * 20, end="")  # Clear extra trailing chars
        
        cv2.imshow('YOLO All Objects Detection', vis_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.stop()
    cv2.destroyAllWindows()
    print("\n\nTest finished")

if __name__ == '__main__':
    main()
