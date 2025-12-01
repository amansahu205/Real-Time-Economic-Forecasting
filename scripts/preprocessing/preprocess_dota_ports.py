"""
DOTA Dataset Preprocessing for Port Activity Detection
Converts DOTA oriented bounding boxes to YOLO format
Tiles large images and filters port-relevant classes
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
from collections import defaultdict

# Paths
DOTA_BASE = Path('data/raw/satellite/dota')
OUTPUT_BASE = Path('data/raw/satellite/dota/ports')

# Port-relevant classes from DOTA
# DOTA classes: plane, ship, storage-tank, baseball-diamond, tennis-court,
#               basketball-court, ground-track-field, harbor, bridge,
#               large-vehicle, small-vehicle, helicopter, roundabout,
#               soccer-ball-field, swimming-pool
PORT_CLASSES = {
    'ship': 0,
    'harbor': 1,
    'large-vehicle': 2,
    'small-vehicle': 3,
    'storage-tank': 4,
}

# Tiling configuration
TILE_SIZE = 1024  # Larger than 640 to preserve context
OVERLAP = 200     # Overlap between tiles to avoid cutting objects

def parse_dota_annotation(anno_file):
    """Parse DOTA annotation file (oriented bounding boxes)"""
    annotations = []
    
    with open(anno_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        
        # Parse coordinates (8 values)
        try:
            coords = [float(p) for p in parts[:8]]
            class_name = parts[8]
            difficulty = int(parts[9]) if len(parts) > 9 else 0
            
            # Only keep port-relevant classes
            if class_name in PORT_CLASSES:
                annotations.append({
                    'coords': coords,  # [x1, y1, x2, y2, x3, y3, x4, y4]
                    'class': class_name,
                    'class_id': PORT_CLASSES[class_name],
                    'difficulty': difficulty
                })
        except (ValueError, IndexError):
            continue
    
    return annotations

def oriented_box_to_bbox(coords):
    """Convert oriented bounding box (4 corners) to axis-aligned bbox"""
    # coords: [x1, y1, x2, y2, x3, y3, x4, y4]
    x_coords = [coords[i] for i in range(0, 8, 2)]
    y_coords = [coords[i] for i in range(1, 8, 2)]
    
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    
    return x_min, y_min, x_max, y_max

def bbox_to_yolo(bbox, img_width, img_height):
    """Convert bbox to YOLO format (normalized x_center, y_center, width, height)"""
    x_min, y_min, x_max, y_max = bbox
    
    x_center = (x_min + x_max) / 2.0 / img_width
    y_center = (y_min + y_max) / 2.0 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    # Clip to valid range
    x_center = max(0, min(1, x_center))
    y_center = max(0, min(1, y_center))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return x_center, y_center, width, height

def is_bbox_in_tile(bbox, tile_x, tile_y, tile_size):
    """Check if bbox center is within tile boundaries"""
    x_min, y_min, x_max, y_max = bbox
    
    # Center of bbox
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    
    # Check if center is in tile
    if (tile_x <= cx < tile_x + tile_size and 
        tile_y <= cy < tile_y + tile_size):
        return True
    return False

def clip_bbox_to_tile(bbox, tile_x, tile_y, tile_size):
    """Clip bbox to tile boundaries and convert to tile-relative coords"""
    x_min, y_min, x_max, y_max = bbox
    
    # Clip to tile
    x_min_clipped = max(x_min, tile_x)
    y_min_clipped = max(y_min, tile_y)
    x_max_clipped = min(x_max, tile_x + tile_size)
    y_max_clipped = min(y_max, tile_y + tile_size)
    
    # Convert to tile-relative coordinates
    x_min_rel = x_min_clipped - tile_x
    y_min_rel = y_min_clipped - tile_y
    x_max_rel = x_max_clipped - tile_x
    y_max_rel = y_max_clipped - tile_y
    
    return x_min_rel, y_min_rel, x_max_rel, y_max_rel

def tile_image_and_annotations(image_path, annotations, output_images_dir, output_labels_dir, 
                                base_name, tile_size=TILE_SIZE, overlap=OVERLAP):
    """Tile large image and corresponding annotations"""
    
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ⚠️  Failed to read image: {image_path}")
        return 0, 0
    
    img_height, img_width = img.shape[:2]
    
    tiles_created = 0
    objects_kept = 0
    
    # Calculate stride
    stride = tile_size - overlap
    
    # Generate tiles
    for y in range(0, img_height, stride):
        for x in range(0, img_width, stride):
            # Ensure we don't go out of bounds
            tile_x = min(x, img_width - tile_size) if x + tile_size > img_width else x
            tile_y = min(y, img_height - tile_size) if y + tile_size > img_height else y
            
            # Skip if we've already processed this tile
            if x > 0 and tile_x == x - stride:
                continue
            if y > 0 and tile_y == y - stride:
                continue
            
            # Extract tile
            tile = img[tile_y:tile_y+tile_size, tile_x:tile_x+tile_size]
            
            # Find annotations in this tile
            tile_annotations = []
            for anno in annotations:
                bbox = oriented_box_to_bbox(anno['coords'])
                
                # Check if object is in this tile
                if is_bbox_in_tile(bbox, tile_x, tile_y, tile_size):
                    # Clip bbox to tile
                    bbox_clipped = clip_bbox_to_tile(bbox, tile_x, tile_y, tile_size)
                    
                    # Convert to YOLO format
                    yolo_bbox = bbox_to_yolo(bbox_clipped, tile_size, tile_size)
                    
                    tile_annotations.append({
                        'class_id': anno['class_id'],
                        'bbox': yolo_bbox
                    })
                    objects_kept += 1
            
            # Only save tile if it has annotations
            if tile_annotations:
                tile_name = f"{base_name}_tile_{tile_y}_{tile_x}"
                
                # Save image
                tile_img_path = output_images_dir / f"{tile_name}.png"
                cv2.imwrite(str(tile_img_path), tile)
                
                # Save labels
                tile_label_path = output_labels_dir / f"{tile_name}.txt"
                with open(tile_label_path, 'w') as f:
                    for anno in tile_annotations:
                        class_id = anno['class_id']
                        x_c, y_c, w, h = anno['bbox']
                        f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                
                tiles_created += 1
    
    return tiles_created, objects_kept

def process_split(split_name, dota_base, output_base, tile_size=TILE_SIZE, overlap=OVERLAP):
    """Process a dataset split (train/val/test)"""
    
    print(f"\n{'='*70}")
    print(f"Processing {split_name.upper()}")
    print(f"{'='*70}")
    
    # Paths
    images_dir = dota_base / split_name / 'images'
    labels_dir = dota_base / split_name / 'labelTxt'
    
    output_images_dir = output_base / split_name / 'images'
    output_labels_dir = output_base / split_name / 'labels'
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all label files
    if not labels_dir.exists():
        print(f"  ⚠️  Labels directory not found: {labels_dir}")
        return {'images': 0, 'tiles': 0, 'objects': 0, 'skipped': 0}
    
    label_files = list(labels_dir.glob('*.txt'))
    
    print(f"Found {len(label_files)} annotation files")
    
    total_images = 0
    total_tiles = 0
    total_objects = 0
    skipped_images = 0
    
    class_counts = defaultdict(int)
    
    for label_file in tqdm(label_files, desc=f"  {split_name}"):
        # Parse annotations
        annotations = parse_dota_annotation(label_file)
        
        # Skip if no port-relevant objects
        if not annotations:
            skipped_images += 1
            continue
        
        # Count classes
        for anno in annotations:
            class_counts[anno['class']] += 1
        
        # Find corresponding image
        image_name = label_file.stem + '.png'
        image_path = images_dir / image_name
        
        if not image_path.exists():
            print(f"  ⚠️  Image not found: {image_name}")
            skipped_images += 1
            continue
        
        # Tile image and annotations
        tiles_created, objects_kept = tile_image_and_annotations(
            image_path, annotations, output_images_dir, output_labels_dir,
            label_file.stem, tile_size, overlap
        )
        
        if tiles_created > 0:
            total_images += 1
            total_tiles += tiles_created
            total_objects += objects_kept
    
    stats = {
        'images': total_images,
        'tiles': total_tiles,
        'objects': total_objects,
        'skipped': skipped_images,
        'class_counts': dict(class_counts)
    }
    
    print(f"\n  ✅ Processed {total_images} images into {total_tiles} tiles")
    print(f"  📦 Total objects: {total_objects}")
    print(f"  ⏭️  Skipped {skipped_images} images (no port objects)")
    print(f"\n  Class distribution:")
    for class_name, count in class_counts.items():
        print(f"    {class_name:15s}: {count:6d}")
    
    return stats

def create_yaml(output_base, all_stats):
    """Create YAML configuration file"""
    
    yaml_content = f"""# DOTA Dataset - Port Activity Detection
# Focused on maritime and port infrastructure

path: {output_base.absolute().as_posix()}

train: train/images
val: val/images
test: test/images  # Note: DOTA test set has no labels (competition set)

nc: {len(PORT_CLASSES)}
names: {list(PORT_CLASSES.keys())}

# Classes:
# 0: ship - Vessels in port or at sea
# 1: harbor - Port infrastructure, docks, piers
# 2: large-vehicle - Port trucks, cranes, heavy equipment
# 3: small-vehicle - Cars in port area
# 4: storage-tank - Fuel/cargo storage tanks

# Dataset statistics
# Tile size: {TILE_SIZE}x{TILE_SIZE}
# Overlap: {OVERLAP}px
"""
    
    yaml_path = output_base / 'ports_dota.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n✅ YAML file created: {yaml_path}")
    return yaml_path

# Main execution
def main():
    print("="*70)
    print("DOTA Dataset Preprocessing for Port Activity")
    print("="*70)
    print(f"Source: {DOTA_BASE}")
    print(f"Output: {OUTPUT_BASE}")
    print(f"Tile size: {TILE_SIZE}x{TILE_SIZE}")
    print(f"Overlap: {OVERLAP}px")
    print(f"Classes: {list(PORT_CLASSES.keys())}")
    print("="*70)
    
    all_stats = {}
    
    # Process each split
    for split in ['train', 'val']:
        stats = process_split(split, DOTA_BASE, OUTPUT_BASE, TILE_SIZE, OVERLAP)
        all_stats[split] = stats
    
    # Create YAML
    yaml_path = create_yaml(OUTPUT_BASE, all_stats)
    
    # Final summary
    print("\n" + "="*70)
    print("🎉 DOTA PREPROCESSING COMPLETE!")
    print("="*70)
    
    print("\nDataset Summary:")
    print("-" * 70)
    
    for split, stats in all_stats.items():
        print(f"\n{split.upper()}:")
        print(f"  Original images with port objects: {stats['images']}")
        print(f"  Tiles created: {stats['tiles']}")
        print(f"  Total objects: {stats['objects']}")
        print(f"  Skipped images: {stats['skipped']}")
    
    # Overall class distribution
    print("\n" + "="*70)
    print("Overall Class Distribution:")
    print("-" * 70)
    
    total_class_counts = defaultdict(int)
    for split, stats in all_stats.items():
        for class_name, count in stats.get('class_counts', {}).items():
            total_class_counts[class_name] += count
    
    total_objects = sum(total_class_counts.values())
    for class_name in PORT_CLASSES.keys():
        count = total_class_counts.get(class_name, 0)
        pct = (count / total_objects * 100) if total_objects > 0 else 0
        print(f"{class_name:15s}: {count:8d} ({pct:5.1f}%)")
    
    print("-" * 70)
    print(f"{'TOTAL':15s}: {total_objects:8d} (100.0%)")
    
    print("\n" + "="*70)
    print("Next Steps:")
    print("="*70)
    print(f"1. Dataset ready at: {OUTPUT_BASE}")
    print(f"2. Config file: {yaml_path}")
    print("3. Update notebook to train with DOTA ports dataset")
    print("4. Expected mAP: 60-75% (better than xView ports!)")
    print("="*70)

if __name__ == '__main__':
    main()
