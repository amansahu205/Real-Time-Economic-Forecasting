#!/usr/bin/env python3
"""
Tiled Object Detection Module

Provides tiled detection functionality for high-resolution satellite imagery.
Splits images into overlapping tiles, runs detection, and merges results.

Usage:
    from tiled_detector import TiledDetector
    
    detector = TiledDetector(model, tile_size=1024, overlap=128)
    annotated_img, detections = detector.process_image(image)
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from ultralytics import YOLO


class TiledDetector:
    """
    Tiled object detection for high-resolution images.
    
    Splits large images into overlapping tiles, runs detection on each tile,
    merges results with NMS to remove duplicates, and returns annotated image.
    """
    
    def __init__(self, model: YOLO, tile_size: int = 1024, overlap: int = 128):
        """
        Initialize tiled detector.
        
        Args:
            model: YOLO model instance
            tile_size: Size of each tile (default: 1024)
            overlap: Overlap between tiles in pixels (default: 128)
        """
        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap
        self.stride = tile_size - overlap
    
    def create_tiles(self, image: np.ndarray) -> List[Dict]:
        """
        Split image into overlapping tiles.
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            List of tile dictionaries with image data and position info
        """
        h, w = image.shape[:2]
        tiles = []
        
        # Calculate number of tiles needed
        rows = (h - self.overlap) // self.stride + 1
        cols = (w - self.overlap) // self.stride + 1
        
        for row in range(rows):
            for col in range(cols):
                # Calculate tile boundaries
                y1 = row * self.stride
                x1 = col * self.stride
                y2 = min(y1 + self.tile_size, h)
                x2 = min(x1 + self.tile_size, w)
                
                # Adjust if at edge to maintain tile size
                if y2 == h and y2 - y1 < self.tile_size:
                    y1 = max(0, h - self.tile_size)
                if x2 == w and x2 - x1 < self.tile_size:
                    x1 = max(0, w - self.tile_size)
                
                tile = image[y1:y2, x1:x2]
                
                tiles.append({
                    'image': tile,
                    'row': row,
                    'col': col,
                    'x': x1,
                    'y': y1,
                    'width': x2 - x1,
                    'height': y2 - y1
                })
        
        return tiles
    
    def detect_on_tiles(self, tiles: List[Dict], conf: float, iou: float) -> List[Dict]:
        """
        Run detection on each tile and convert to global coordinates.
        
        Args:
            tiles: List of tile dictionaries
            conf: Confidence threshold
            iou: IOU threshold for detection
            
        Returns:
            List of detections with global coordinates
        """
        all_detections = []
        
        for tile_info in tiles:
            tile_img = tile_info['image']
            
            # Run detection on tile
            results = self.model(
                tile_img,
                conf=conf,
                iou=iou,
                imgsz=640,
                device=0,
                verbose=False
            )[0]
            
            boxes = results.boxes
            
            if len(boxes) > 0:
                # Convert tile coordinates to global coordinates
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    
                    # Adjust to global coordinates
                    global_x1 = x1 + tile_info['x']
                    global_y1 = y1 + tile_info['y']
                    global_x2 = x2 + tile_info['x']
                    global_y2 = y2 + tile_info['y']
                    
                    all_detections.append({
                        'bbox': [global_x1, global_y1, global_x2, global_y2],
                        'confidence': float(boxes.conf[i].cpu().numpy()),
                        'class_id': int(boxes.cls[i].cpu().numpy()),
                        'class_name': self.model.names[int(boxes.cls[i].cpu().numpy())],
                        'tile_row': tile_info['row'],
                        'tile_col': tile_info['col']
                    })
        
        return all_detections
    
    def nms_global(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Apply Non-Maximum Suppression across all tiles to remove duplicates.
        
        Args:
            detections: List of detection dictionaries
            iou_threshold: IOU threshold for NMS
            
        Returns:
            Filtered list of detections
        """
        if not detections:
            return []
        
        # Group by class
        by_class = {}
        for det in detections:
            cls = det['class_name']
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(det)
        
        final_detections = []
        
        for cls, dets in by_class.items():
            # Convert to numpy arrays for NMS
            boxes = np.array([d['bbox'] for d in dets])
            scores = np.array([d['confidence'] for d in dets])
            
            # Apply NMS using OpenCV
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                scores.tolist(),
                score_threshold=0.0,
                nms_threshold=iou_threshold
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                for idx in indices:
                    final_detections.append(dets[idx])
        
        return final_detections
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        # Color map for different classes
        colors = {
            'ship': (0, 255, 0),           # Green
            'harbor': (255, 0, 0),         # Blue
            'large-vehicle': (0, 165, 255), # Orange
            'small-vehicle': (0, 255, 255), # Yellow
            'plane': (255, 0, 255),        # Magenta
            'car': (0, 255, 0),            # Green
            'vehicle': (0, 255, 0),        # Green
            'truck': (0, 165, 255),        # Orange
            'building': (255, 0, 0),       # Blue
            'parking': (255, 255, 0),      # Yellow
        }
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cls_name = det['class_name']
            conf = det['confidence']
            
            color = colors.get(cls_name, (255, 255, 255))
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{cls_name} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated
    
    def process_image(
        self, 
        image: np.ndarray, 
        conf: float = 0.25, 
        iou: float = 0.45
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process full image with tiling.
        
        Args:
            image: Input image
            conf: Confidence threshold
            iou: IOU threshold
            
        Returns:
            Tuple of (annotated_image, detection_stats)
        """
        # Create tiles
        tiles = self.create_tiles(image)
        
        # Detect on each tile
        detections = self.detect_on_tiles(tiles, conf, iou)
        
        # Apply global NMS to remove duplicates
        detections = self.nms_global(detections, iou)
        
        # Draw detections
        annotated = self.draw_detections(image, detections)
        
        # Calculate statistics
        class_counts = {}
        for det in detections:
            cls = det['class_name']
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        stats = {
            'total_detections': len(detections),
            'class_counts': class_counts,
            'num_tiles': len(tiles),
            'detections_per_tile': len(detections) / len(tiles) if tiles else 0,
            'confidence_scores': [d['confidence'] for d in detections]
        }
        
        return annotated, stats


if __name__ == "__main__":
    # Example usage
    print("TiledDetector module loaded successfully!")
    print("\nUsage:")
    print("  from tiled_detector import TiledDetector")
    print("  detector = TiledDetector(model, tile_size=1024, overlap=128)")
    print("  annotated, stats = detector.process_image(image)")
