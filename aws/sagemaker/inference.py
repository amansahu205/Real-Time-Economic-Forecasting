"""
SageMaker Inference Script for YOLO Object Detection
This runs on the SageMaker endpoint and processes images.
"""

import os
import io
import json
import base64
import numpy as np
from PIL import Image

# Model will be loaded globally
model = None


def model_fn(model_dir):
    """Load the YOLO model."""
    from ultralytics import YOLO
    
    model_path = os.path.join(model_dir, 'best.pt')
    if not os.path.exists(model_path):
        # Fallback to pretrained
        model_path = 'yolov8n.pt'
    
    model = YOLO(model_path)
    return model


def input_fn(request_body, request_content_type):
    """Process input image."""
    if request_content_type in ['application/x-image', 'image/jpeg', 'image/png']:
        image = Image.open(io.BytesIO(request_body))
        return np.array(image)
    elif request_content_type == 'application/json':
        data = json.loads(request_body)
        if 'image_base64' in data:
            image_bytes = base64.b64decode(data['image_base64'])
            image = Image.open(io.BytesIO(image_bytes))
            return np.array(image)
    
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Run YOLO detection."""
    import cv2
    
    # Run inference
    results = model(input_data, conf=0.25, verbose=False)[0]
    
    # Extract detections
    detections = []
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls_id = int(box.cls[0].cpu().numpy())
        cls_name = model.names[cls_id]
        
        detections.append({
            'class': cls_name,
            'class_id': cls_id,
            'confidence': round(conf, 3),
            'bbox': [int(x1), int(y1), int(x2), int(y2)]
        })
    
    # Create annotated image
    annotated = results.plot()
    
    # Encode annotated image to base64
    _, buffer = cv2.imencode('.jpg', annotated)
    annotated_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        'detections': detections,
        'total_count': len(detections),
        'annotated_image': annotated_base64
    }


def output_fn(prediction, response_content_type):
    """Format output."""
    return json.dumps(prediction)
