"""
Create True 2-Class Retail Dataset
Filters only car (0) and equipment (5) from existing retail data
Remaps to: 0=car, 1=equipment
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
SOURCE_DIR = Path('data/raw/satellite/xview-economic-simple/retail')
TARGET_DIR = Path('data/raw/satellite/xview-economic-simple/retail_2class')

# Class mapping: old_id -> new_id
# We only keep car (0) and equipment (5)
CLASS_MAPPING = {
    0: 0,  # car -> car
    5: 1,  # equipment -> equipment
}

CLASS_NAMES = ['car', 'equipment']

print("="*70)
print("Creating 2-Class Retail Dataset")
print("="*70)
print(f"Source: {SOURCE_DIR}")
print(f"Target: {TARGET_DIR}")
print(f"Classes: {CLASS_NAMES}")
print("="*70)

# Create target directories
for split in ['train', 'val', 'test']:
    (TARGET_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
    (TARGET_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

# Process each split
stats = {'train': {}, 'val': {}, 'test': {}}

for split in ['train', 'val', 'test']:
    print(f"\nProcessing {split}...")
    
    source_labels = SOURCE_DIR / split / 'labels'
    source_images = SOURCE_DIR / split / 'images'
    target_labels = TARGET_DIR / split / 'labels'
    target_images = TARGET_DIR / split / 'images'
    
    if not source_labels.exists():
        print(f"  ⚠️  {split} labels not found, skipping...")
        continue
    
    label_files = list(source_labels.glob('*.txt'))
    
    kept_images = 0
    kept_objects = 0
    skipped_images = 0
    
    for label_file in tqdm(label_files, desc=f"  {split}"):
        # Read annotations
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # Filter and remap classes
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            
            old_class = int(parts[0])
            
            # Only keep car (0) or equipment (5)
            if old_class in CLASS_MAPPING:
                new_class = CLASS_MAPPING[old_class]
                # Remap class ID
                parts[0] = str(new_class)
                new_lines.append(' '.join(parts) + '\n')
                kept_objects += 1
        
        # Only keep images that have at least one object
        if new_lines:
            # Copy image
            image_name = label_file.stem + '.jpg'
            source_image = source_images / image_name
            
            if source_image.exists():
                shutil.copy2(source_image, target_images / image_name)
                
                # Write filtered labels
                with open(target_labels / label_file.name, 'w') as f:
                    f.writelines(new_lines)
                
                kept_images += 1
            else:
                print(f"    ⚠️  Image not found: {image_name}")
        else:
            skipped_images += 1
    
    stats[split] = {
        'images': kept_images,
        'objects': kept_objects,
        'skipped': skipped_images
    }
    
    print(f"  ✅ Kept {kept_images} images with {kept_objects} objects")
    print(f"  ⏭️  Skipped {skipped_images} images (no car/equipment)")

# Create YAML file
yaml_content = f"""# xView Economic Dataset - Retail Activity (2 Classes Only)
path: {TARGET_DIR.absolute().as_posix()}

train: train/images
val: val/images
test: test/images

nc: 2
names: {CLASS_NAMES}

# Focus on retail-specific objects
description: Car and Equipment detection for retail activity monitoring
"""

yaml_path = TARGET_DIR / 'retail_2class.yaml'
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print("\n" + "="*70)
print("Summary")
print("="*70)

for split in ['train', 'val', 'test']:
    if stats[split]:
        print(f"{split.upper():6s}: {stats[split]['images']:5d} images, "
              f"{stats[split]['objects']:6d} objects, "
              f"{stats[split]['skipped']:4d} skipped")

print("="*70)
print(f"✅ YAML file created: {yaml_path}")
print("="*70)

# Calculate class distribution
print("\nCalculating class distribution...")
class_counts = {0: 0, 1: 0}

for split in ['train', 'val']:
    label_dir = TARGET_DIR / split / 'labels'
    if label_dir.exists():
        for label_file in label_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1

total = sum(class_counts.values())
print("\nClass Distribution:")
print("-" * 50)
for class_id, count in class_counts.items():
    pct = (count / total * 100) if total > 0 else 0
    print(f"Class {class_id} ({CLASS_NAMES[class_id]:12s}): {count:6d} ({pct:5.1f}%)")
print("-" * 50)
print(f"TOTAL:                    {total:6d} (100.0%)")

print("\n" + "="*70)
print("✅ 2-Class Retail Dataset Created!")
print("="*70)
print(f"\nNext steps:")
print(f"1. Train with: {yaml_path.name}")
print(f"2. Expected performance: 65-75% mAP on car, 40-50% on equipment")
print("="*70)
