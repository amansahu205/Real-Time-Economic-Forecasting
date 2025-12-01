#!/usr/bin/env python3
"""
Process Satellite Data with Tiled Detection

Unified script to process all satellite imagery (ports and retail) using
the tiled detection method for optimal accuracy.

Datasets:
- Google Earth Ports: Port_of_LA, Port_of_hongkong, Port_of_Tanjung_priok, Port_of_Salalah
- Google Earth Retail: Mall_of_america

Usage:
    # Process all ports
    python process_satellite_data.py --dataset ports
    
    # Process retail
    python process_satellite_data.py --dataset retail
    
    # Process specific location
    python process_satellite_data.py --dataset ports --location Port_of_LA
    
    # Process everything
    python process_satellite_data.py --dataset all
"""

from pathlib import Path
import sys
import cv2
import argparse
from tqdm import tqdm
from ultralytics import YOLO

# Import modules
sys.path.append(str(Path(__file__).parent))
from src.detection.tiled_detector import TiledDetector
from src.detection.annotation_manager import StructuredAnnotationManager


# Paths
PROJECT_ROOT = Path(r"D:\MS\UMD\Courses\Fall-2025\DATA-650\Real-Time-Economic-Forecasting")
GOOGLE_EARTH_DIR = PROJECT_ROOT / "data" / "raw" / "satellite" / "google_earth"
ANNOTATIONS_DIR = PROJECT_ROOT / "results" / "annotations"
MODELS_DIR = PROJECT_ROOT / "data" / "models" / "satellite"

# Models
PORTS_MODEL = MODELS_DIR / "ports_dota_yolo11_20251127_013205" / "weights" / "best.pt"
RETAIL_MODEL = MODELS_DIR / "retail_yolo11_20251126_150811" / "weights" / "best.pt"

# Detection parameters
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
TILE_SIZE = 1024
OVERLAP = 128

# Dataset configurations
DATASETS = {
    'ports': {
        'locations': ['Port_of_LA', 'Port_of_hongkong', 'Port_of_Tanjung_priok', 'Port_of_Salalah'],
        'model': PORTS_MODEL,
        'dataset_name': 'google_earth_ports',
        'description': 'Port activity detection for maritime trade analysis'
    },
    'retail': {
        'locations': ['Mall_of_america'],
        'model': RETAIL_MODEL,
        'dataset_name': 'google_earth_retail',
        'description': 'Retail activity detection for consumer spending analysis'
    }
}


def process_location(
    location_name: str,
    detector: TiledDetector,
    manager: StructuredAnnotationManager,
    dataset_name: str,
    model_name: str
):
    """
    Process all images for a location using tiled detection.
    
    Args:
        location_name: Name of location (e.g., 'Port_of_LA')
        detector: TiledDetector instance
        manager: StructuredAnnotationManager instance
        dataset_name: Name of dataset for saving
        model_name: Model identifier
    """
    location_dir = GOOGLE_EARTH_DIR / location_name
    
    if not location_dir.exists():
        print(f"   ⚠️  Location not found: {location_name}")
        return
    
    print(f"\n{'='*60}")
    print(f"📍 Processing: {location_name.replace('_', ' ').title()}")
    print(f"{'='*60}")
    
    # Find all year directories
    year_dirs = sorted([d for d in location_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    
    if not year_dirs:
        print(f"   ⚠️  No year directories found. Run reorganization first!")
        return
    
    total_images = 0
    total_detections = 0
    
    for year_dir in year_dirs:
        year = int(year_dir.name)
        
        # Find all images
        image_files = sorted(year_dir.glob("*.jpg")) + sorted(year_dir.glob("*.png"))
        
        if not image_files:
            continue
        
        print(f"\n📅 Year {year}: {len(image_files)} images")
        
        for img_path in tqdm(image_files, desc=f"  Processing {year}"):
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"   ⚠️  Cannot read: {img_path.name}")
                    continue
                
                h, w = img.shape[:2]
                
                # Process with tiled detection
                annotated, stats = detector.process_image(img, CONF_THRESHOLD, IOU_THRESHOLD)
                
                # Prepare detection data
                detections = {
                    'total_detections': stats['total_detections'],
                    'class_counts': stats['class_counts'],
                    'num_tiles': stats['num_tiles'],
                    'detections_per_tile': stats['detections_per_tile'],
                    'confidence_scores': stats['confidence_scores']
                }
                
                # Save annotation
                manager.save_annotation(
                    dataset_name=dataset_name,
                    location=location_name,
                    year=year,
                    image_name=img_path.name,
                    annotated_image=annotated,
                    detections=detections,
                    metadata={
                        'source': 'Google Earth',
                        'original_path': str(img_path.relative_to(PROJECT_ROOT)),
                        'resolution': f'{w}x{h}',
                        'model': model_name,
                        'method': 'tiled',
                        'tile_size': TILE_SIZE,
                        'overlap': OVERLAP,
                        'conf_threshold': CONF_THRESHOLD,
                        'iou_threshold': IOU_THRESHOLD
                    }
                )
                
                total_images += 1
                total_detections += stats['total_detections']
                
            except Exception as e:
                print(f"   ❌ Error processing {img_path.name}: {e}")
    
    # Create location summary
    if total_images > 0:
        print(f"\n📊 Creating location summary...")
        manager.create_location_summary(dataset_name, location_name)
        
        print(f"\n✅ {location_name} Complete:")
        print(f"   Images processed: {total_images}")
        print(f"   Total detections: {total_detections}")
        print(f"   Average detections/image: {total_detections/total_images:.2f}")


def process_dataset(dataset_type: str, specific_location: str = None):
    """
    Process a dataset (ports or retail).
    
    Args:
        dataset_type: 'ports', 'retail', or 'all'
        specific_location: Optional specific location to process
    """
    if dataset_type == 'all':
        datasets_to_process = ['ports', 'retail']
    else:
        datasets_to_process = [dataset_type]
    
    manager = StructuredAnnotationManager(ANNOTATIONS_DIR)
    
    for ds_type in datasets_to_process:
        config = DATASETS[ds_type]
        
        print(f"\n{'='*60}")
        print(f"🌍 PROCESSING: {ds_type.upper()}")
        print(f"{'='*60}")
        print(f"📦 Model: {config['model'].parent.parent.name}")
        print(f"📋 Tile size: {TILE_SIZE}x{TILE_SIZE}, Overlap: {OVERLAP}px")
        
        # Load model
        print(f"\n📦 Loading model...")
        model = YOLO(str(config['model']))
        detector = TiledDetector(model, TILE_SIZE, OVERLAP)
        print(f"   ✅ Model loaded")
        
        # Get locations to process
        if specific_location:
            if specific_location in config['locations']:
                locations = [specific_location]
            else:
                print(f"   ⚠️  {specific_location} not in {ds_type} dataset")
                continue
        else:
            locations = config['locations']
        
        print(f"\n📍 Locations: {len(locations)}")
        for loc in locations:
            print(f"   - {loc.replace('_', ' ')}")
        
        # Process each location
        for location in locations:
            process_location(
                location,
                detector,
                manager,
                config['dataset_name'],
                config['model'].parent.parent.name
            )
        
        # Create dataset README
        print(f"\n📝 Creating dataset documentation...")
        manager.create_dataset_readme(
            config['dataset_name'],
            config['description']
        )
    
    # Create master index
    print(f"\n📝 Creating master index...")
    manager.create_master_index()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Process satellite imagery with tiled detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all ports
  python process_satellite_data.py --dataset ports
  
  # Process retail
  python process_satellite_data.py --dataset retail
  
  # Process specific location
  python process_satellite_data.py --dataset ports --location Port_of_LA
  
  # Process everything
  python process_satellite_data.py --dataset all
        """
    )
    
    parser.add_argument(
        '--dataset',
        choices=['ports', 'retail', 'all'],
        required=True,
        help='Dataset to process'
    )
    
    parser.add_argument(
        '--location',
        type=str,
        help='Specific location to process (optional)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("🛰️  SATELLITE DATA PROCESSING - TILED METHOD")
    print("="*60)
    
    # Process
    process_dataset(args.dataset, args.location)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"✅ PROCESSING COMPLETE!")
    print(f"{'='*60}")
    
    print(f"\n📁 Results Location:")
    print(f"   {ANNOTATIONS_DIR}")
    
    print(f"\n📄 Documentation:")
    print(f"   Master Index: {ANNOTATIONS_DIR / 'INDEX.md'}")
    
    print(f"\n🎯 Next Steps:")
    print(f"   1. Review annotations in results/annotations/")
    print(f"   2. Analyze trends using all_years_summary.csv files")
    print(f"   3. Build forecasting models with detection data")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
