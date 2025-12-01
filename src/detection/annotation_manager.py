#!/usr/bin/env python3
"""
Structured Annotation Results Manager

Creates a clear, organized directory structure for all detection results:
- Annotated images with bounding boxes
- Detection metadata (JSON + CSV)
- Summary statistics
- Organized by dataset, location, and year

Directory Structure:
    results/annotations/
      ├── google_earth/
      │   ├── Port_of_LA/
      │   │   ├── 2017/
      │   │   │   ├── annotated/
      │   │   │   │   ├── 2017-1_annotated.jpg
      │   │   │   │   └── 2017-2_annotated.jpg
      │   │   │   ├── detections.csv
      │   │   │   ├── detections.json
      │   │   │   └── summary.txt
      │   │   ├── 2018/
      │   │   └── all_years_summary.csv
      │   └── README.md
      │
      ├── sentinel2/
      │   ├── Port_of_Los_Angeles/
      │   │   ├── 2017/
      │   │   │   ├── tiles/
      │   │   │   │   ├── tile_r00_c00.jpg
      │   │   │   │   └── tile_r00_c01.jpg
      │   │   │   ├── detections.csv
      │   │   │   └── summary.txt
      │   │   └── all_years_summary.csv
      │   └── README.md
      │
      └── INDEX.md (master index of all annotations)
"""

from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import shutil


class StructuredAnnotationManager:
    """Manages structured storage of detection annotations"""
    
    def __init__(self, base_output_dir: Path):
        """
        Initialize annotation manager.
        
        Args:
            base_output_dir: Base directory for all annotations (e.g., results/annotations)
        """
        self.base_dir = Path(base_output_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create master index
        self.index_file = self.base_dir / "INDEX.md"
        self.stats = {
            'total_images': 0,
            'total_detections': 0,
            'datasets': {},
            'last_updated': None
        }
        
    def get_dataset_dir(self, dataset_name: str) -> Path:
        """Get or create dataset directory (e.g., 'google_earth', 'sentinel2')"""
        dataset_dir = self.base_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        return dataset_dir
    
    def get_location_dir(self, dataset_name: str, location: str) -> Path:
        """Get or create location directory"""
        location_dir = self.get_dataset_dir(dataset_name) / location
        location_dir.mkdir(exist_ok=True)
        return location_dir
    
    def get_year_dir(self, dataset_name: str, location: str, year: int) -> Path:
        """Get or create year directory with subdirectories"""
        year_dir = self.get_location_dir(dataset_name, location) / str(year)
        year_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (year_dir / "annotated").mkdir(exist_ok=True)
        
        return year_dir
    
    def save_annotation(
        self,
        dataset_name: str,
        location: str,
        year: int,
        image_name: str,
        annotated_image: np.ndarray,
        detections: Dict,
        metadata: Optional[Dict] = None
    ) -> Path:
        """
        Save a single annotated image with its detection data.
        
        Args:
            dataset_name: Dataset identifier (e.g., 'google_earth', 'sentinel2')
            location: Location name (e.g., 'Port_of_LA', 'Port_of_Singapore')
            year: Year of the image
            image_name: Original image filename
            annotated_image: Image with bounding boxes drawn
            detections: Detection results dict with 'total_detections', 'class_counts', etc.
            metadata: Optional additional metadata
            
        Returns:
            Path to saved annotated image
        """
        year_dir = self.get_year_dir(dataset_name, location, year)
        
        # Save annotated image
        base_name = Path(image_name).stem
        annotated_path = year_dir / "annotated" / f"{base_name}_annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated_image)
        
        # Prepare detection record
        record = {
            'image_name': image_name,
            'annotated_path': str(annotated_path.relative_to(self.base_dir)),
            'timestamp': datetime.now().isoformat(),
            'total_detections': detections.get('total_detections', 0),
            'class_counts': detections.get('class_counts', {}),
            'metadata': metadata or {}
        }
        
        # Append to detections JSON
        detections_json = year_dir / "detections.json"
        if detections_json.exists():
            with open(detections_json, 'r') as f:
                all_detections = json.load(f)
        else:
            all_detections = []
        
        all_detections.append(record)
        
        with open(detections_json, 'w') as f:
            json.dump(all_detections, f, indent=2)
        
        # Update CSV
        self._update_csv(year_dir, all_detections)
        
        # Update summary
        self._update_summary(year_dir, all_detections)
        
        # Update stats
        self.stats['total_images'] += 1
        self.stats['total_detections'] += detections.get('total_detections', 0)
        
        return annotated_path
    
    def _update_csv(self, year_dir: Path, detections: List[Dict]):
        """Update CSV file with all detections"""
        csv_path = year_dir / "detections.csv"
        
        # Flatten data for CSV
        rows = []
        for det in detections:
            row = {
                'image_name': det['image_name'],
                'timestamp': det['timestamp'],
                'total_detections': det['total_detections'],
                'annotated_path': det['annotated_path']
            }
            # Add class counts as separate columns
            for cls, count in det['class_counts'].items():
                row[f'count_{cls}'] = count
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
    
    def _update_summary(self, year_dir: Path, detections: List[Dict]):
        """Update summary text file"""
        summary_path = year_dir / "summary.txt"
        
        total_images = len(detections)
        total_detections = sum(d['total_detections'] for d in detections)
        
        # Aggregate class counts
        all_classes = {}
        for det in detections:
            for cls, count in det['class_counts'].items():
                all_classes[cls] = all_classes.get(cls, 0) + count
        
        # Write summary
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Detection Summary - {year_dir.parent.name}/{year_dir.name}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total Images: {total_images}\n")
            f.write(f"Total Detections: {total_detections}\n")
            f.write(f"Average Detections per Image: {total_detections/total_images:.2f}\n\n")
            
            f.write("Detections by Class:\n")
            f.write("-" * 40 + "\n")
            for cls, count in sorted(all_classes.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {cls:20s}: {count:5d}\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def create_location_summary(self, dataset_name: str, location: str):
        """Create summary across all years for a location"""
        location_dir = self.get_location_dir(dataset_name, location)
        
        # Collect data from all years
        all_years_data = []
        year_dirs = sorted([d for d in location_dir.iterdir() if d.is_dir() and d.name.isdigit()])
        
        for year_dir in year_dirs:
            json_file = year_dir / "detections.json"
            if json_file.exists():
                with open(json_file, 'r') as f:
                    year_data = json.load(f)
                    for record in year_data:
                        record['year'] = int(year_dir.name)
                    all_years_data.extend(year_data)
        
        if not all_years_data:
            return
        
        # Create summary CSV
        summary_path = location_dir / "all_years_summary.csv"
        
        # Aggregate by year
        year_summary = {}
        for record in all_years_data:
            year = record['year']
            if year not in year_summary:
                year_summary[year] = {
                    'year': year,
                    'total_images': 0,
                    'total_detections': 0,
                    'class_counts': {}
                }
            
            year_summary[year]['total_images'] += 1
            year_summary[year]['total_detections'] += record['total_detections']
            
            for cls, count in record['class_counts'].items():
                year_summary[year]['class_counts'][cls] = \
                    year_summary[year]['class_counts'].get(cls, 0) + count
        
        # Convert to DataFrame
        rows = []
        for year, data in sorted(year_summary.items()):
            row = {
                'year': year,
                'total_images': data['total_images'],
                'total_detections': data['total_detections'],
                'avg_detections_per_image': data['total_detections'] / data['total_images']
            }
            # Add class counts
            for cls, count in data['class_counts'].items():
                row[f'total_{cls}'] = count
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(summary_path, index=False)
        
        print(f"   ✅ Created summary: {summary_path.relative_to(self.base_dir)}")
    
    def create_dataset_readme(self, dataset_name: str, description: str):
        """Create README for a dataset"""
        dataset_dir = self.get_dataset_dir(dataset_name)
        readme_path = dataset_dir / "README.md"
        
        # Count locations and years
        locations = [d for d in dataset_dir.iterdir() if d.is_dir()]
        total_images = 0
        total_detections = 0
        
        for location_dir in locations:
            summary_file = location_dir / "all_years_summary.csv"
            if summary_file.exists():
                df = pd.read_csv(summary_file)
                total_images += df['total_images'].sum()
                total_detections += df['total_detections'].sum()
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(f"# {dataset_name.replace('_', ' ').title()} Annotations\n\n")
            f.write(f"{description}\n\n")
            f.write(f"## Statistics\n\n")
            f.write(f"- **Total Locations**: {len(locations)}\n")
            f.write(f"- **Total Images**: {total_images}\n")
            f.write(f"- **Total Detections**: {total_detections}\n")
            f.write(f"- **Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Locations\n\n")
            for location_dir in sorted(locations):
                if location_dir.is_dir():
                    years = [d.name for d in location_dir.iterdir() if d.is_dir() and d.name.isdigit()]
                    f.write(f"- **{location_dir.name}**: {len(years)} years ({', '.join(sorted(years))})\n")
            
            f.write(f"\n## Directory Structure\n\n")
            f.write(f"```\n")
            f.write(f"{dataset_name}/\n")
            f.write(f"  ├── <location_name>/\n")
            f.write(f"  │   ├── <year>/\n")
            f.write(f"  │   │   ├── annotated/          # Annotated images with bounding boxes\n")
            f.write(f"  │   │   ├── detections.csv      # Detection data in CSV format\n")
            f.write(f"  │   │   ├── detections.json     # Detailed detection data\n")
            f.write(f"  │   │   └── summary.txt         # Human-readable summary\n")
            f.write(f"  │   └── all_years_summary.csv   # Aggregated data across all years\n")
            f.write(f"  └── README.md                    # This file\n")
            f.write(f"```\n")
    
    def create_master_index(self):
        """Create master index of all annotations"""
        datasets = [d for d in self.base_dir.iterdir() if d.is_dir()]
        
        with open(self.index_file, 'w', encoding='utf-8') as f:
            f.write("# Annotation Results Index\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for dataset_dir in sorted(datasets):
                if dataset_dir.name.startswith('.'):
                    continue
                    
                f.write(f"## {dataset_dir.name.replace('_', ' ').title()}\n\n")
                
                locations = [d for d in dataset_dir.iterdir() if d.is_dir()]
                
                for location_dir in sorted(locations):
                    years = sorted([d.name for d in location_dir.iterdir() 
                                  if d.is_dir() and d.name.isdigit()])
                    
                    # Count images and detections
                    total_images = 0
                    total_detections = 0
                    
                    for year in years:
                        json_file = location_dir / year / "detections.json"
                        if json_file.exists():
                            with open(json_file, 'r') as jf:
                                data = json.load(jf)
                                total_images += len(data)
                                total_detections += sum(d['total_detections'] for d in data)
                    
                    f.write(f"### {location_dir.name}\n")
                    f.write(f"- **Years**: {', '.join(years)}\n")
                    f.write(f"- **Images**: {total_images}\n")
                    f.write(f"- **Detections**: {total_detections}\n")
                    f.write(f"- **Path**: `{location_dir.relative_to(self.base_dir)}/`\n\n")
        
        print(f"\n✅ Created master index: {self.index_file}")


def example_usage():
    """Example of how to use the StructuredAnnotationManager"""
    
    # Initialize manager
    manager = StructuredAnnotationManager(
        base_output_dir=Path("results/annotations")
    )
    
    # Example: Save Google Earth annotation
    # (This would be called from your detection script)
    """
    annotated_img = cv2.imread("path/to/annotated.jpg")
    detections = {
        'total_detections': 15,
        'class_counts': {'ship': 12, 'harbor': 3}
    }
    
    manager.save_annotation(
        dataset_name='google_earth',
        location='Port_of_LA',
        year=2023,
        image_name='2023-1.jpg',
        annotated_image=annotated_img,
        detections=detections,
        metadata={'source': 'Google Earth Engine', 'resolution': '0.5m'}
    )
    """
    
    # Create location summary
    # manager.create_location_summary('google_earth', 'Port_of_LA')
    
    # Create dataset README
    # manager.create_dataset_readme(
    #     'google_earth',
    #     'High-resolution Google Earth imagery with object detection annotations.'
    # )
    
    # Create master index
    # manager.create_master_index()
    
    print("✅ Structured annotation system ready!")


if __name__ == "__main__":
    example_usage()
