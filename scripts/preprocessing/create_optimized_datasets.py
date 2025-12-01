"""
Create Optimized Activity-Specific Datasets
Filters and remaps classes for better training performance
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Base paths
SOURCE_BASE = Path('data/raw/satellite/xview-economic-simple')
TARGET_BASE = Path('data/raw/satellite/xview-economic-simple')

# Dataset configurations
DATASETS = {
    'industrial_2class': {
        'source': 'industrial',
        'class_mapping': {
            4: 0,  # warehouse -> warehouse
            5: 1,  # equipment -> equipment
        },
        'class_names': ['warehouse', 'equipment'],
        'description': 'Warehouse + Equipment for industrial activity monitoring'
    },
    'city_3class': {
        'source': 'city',
        'class_mapping': {
            0: 0,  # car -> car
            1: 1,  # truck -> truck
            4: 2,  # warehouse -> warehouse
        },
        'class_names': ['car', 'truck', 'warehouse'],
        'description': 'Car + Truck + Warehouse for urban monitoring'
    }
}

def create_dataset(dataset_name, config):
    """Create a filtered and remapped dataset"""
    
    print("\n" + "="*70)
    print(f"Creating {dataset_name.upper()}")
    print("="*70)
    print(f"Description: {config['description']}")
    print(f"Source: {config['source']}")
    print(f"Classes: {config['class_names']}")
    print(f"Mapping: {config['class_mapping']}")
    print("="*70)
    
    source_dir = SOURCE_BASE / config['source']
    target_dir = TARGET_BASE / dataset_name
    
    # Create target directories
    for split in ['train', 'val', 'test']:
        (target_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (target_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Process each split
    stats = {'train': {}, 'val': {}, 'test': {}}
    
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split}...")
        
        source_labels = source_dir / split / 'labels'
        source_images = source_dir / split / 'images'
        target_labels = target_dir / split / 'labels'
        target_images = target_dir / split / 'images'
        
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
                
                # Only keep classes in mapping
                if old_class in config['class_mapping']:
                    new_class = config['class_mapping'][old_class]
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
        print(f"  ⏭️  Skipped {skipped_images} images (no relevant classes)")
    
    # Create YAML file
    yaml_content = f"""# xView Economic Dataset - {dataset_name}
path: {target_dir.absolute().as_posix()}

train: train/images
val: val/images
test: test/images

nc: {len(config['class_names'])}
names: {config['class_names']}

# {config['description']}
"""
    
    yaml_path = target_dir / f'{dataset_name}.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    # Print summary
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
    class_counts = {i: 0 for i in range(len(config['class_names']))}
    
    for split in ['train', 'val']:
        label_dir = target_dir / split / 'labels'
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
        print(f"Class {class_id} ({config['class_names'][class_id]:12s}): {count:6d} ({pct:5.1f}%)")
    print("-" * 50)
    print(f"TOTAL:                    {total:6d} (100.0%)")
    
    return yaml_path, stats, class_counts

# Main execution
print("="*70)
print("Creating Optimized Activity-Specific Datasets")
print("="*70)
print("This will create clean, focused datasets for:")
print("  1. Industrial: Warehouse + Equipment")
print("  2. City: Car + Truck + Warehouse")
print("="*70)

all_results = {}

for dataset_name, config in DATASETS.items():
    yaml_path, stats, class_counts = create_dataset(dataset_name, config)
    all_results[dataset_name] = {
        'yaml': yaml_path,
        'stats': stats,
        'class_counts': class_counts
    }

# Final summary
print("\n" + "="*70)
print("🎉 ALL DATASETS CREATED SUCCESSFULLY!")
print("="*70)

print("\nDataset Summary:")
print("-" * 70)

for dataset_name, results in all_results.items():
    total_images = sum(s['images'] for s in results['stats'].values() if s)
    total_objects = sum(s['objects'] for s in results['stats'].values() if s)
    
    print(f"\n{dataset_name.upper()}:")
    print(f"  YAML: {results['yaml']}")
    print(f"  Images: {total_images}")
    print(f"  Objects: {total_objects}")
    print(f"  Class distribution:", end="")
    for class_id, count in results['class_counts'].items():
        pct = (count / sum(results['class_counts'].values()) * 100) if sum(results['class_counts'].values()) > 0 else 0
        print(f" {count} ({pct:.1f}%)", end="")
    print()

print("\n" + "="*70)
print("Next Steps:")
print("="*70)
print("1. Update notebook configuration to use new datasets")
print("2. Train models with optimized class distributions")
print("3. Expected performance improvements:")
print("   - Industrial: 50-65% mAP (vs <20% with 6-class)")
print("   - City: 60-70% mAP (vs <20% with 6-class)")
print("="*70)
