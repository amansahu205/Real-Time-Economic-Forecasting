# 📸 Structured Annotation Results System

A clear, organized system for storing and managing object detection annotations across all satellite imagery datasets.

## 🎯 Purpose

This system provides:
- **Clear organization** - Easy to find any annotated image by dataset, location, and year
- **Complete metadata** - CSV, JSON, and text summaries for every detection
- **Reproducibility** - Full tracking of detection parameters and sources
- **Scalability** - Works for Google Earth, Sentinel-2, and any future datasets

## 📁 Directory Structure

```
results/annotations/
├── INDEX.md                          # Master index of all annotations
│
├── google_earth/                     # Google Earth dataset
│   ├── README.md                     # Dataset overview
│   ├── Port_of_LA/
│   │   ├── 2017/
│   │   │   ├── annotated/            # Annotated images with bounding boxes
│   │   │   │   ├── 2017-1_annotated.jpg
│   │   │   │   ├── 2017-2_annotated.jpg
│   │   │   │   └── 2017-3_annotated.jpg
│   │   │   ├── detections.csv        # Detection data (CSV format)
│   │   │   ├── detections.json       # Detailed detection data (JSON)
│   │   │   └── summary.txt           # Human-readable summary
│   │   ├── 2018/
│   │   ├── 2020/
│   │   ├── 2021/
│   │   ├── 2022/
│   │   ├── 2023/
│   │   ├── all_years_summary.csv     # Aggregated data across all years
│   │   └── visualizations/           # Charts and comparison grids
│   └── Port_of_Singapore/
│       └── ...
│
├── sentinel2/                        # Sentinel-2 dataset
│   ├── README.md                     # Dataset overview
│   ├── Port_of_Los_Angeles/
│   │   ├── 2017/
│   │   │   ├── annotated/            # Full image or tiles
│   │   │   ├── detections.csv
│   │   │   ├── detections.json
│   │   │   └── summary.txt
│   │   ├── 2018/
│   │   ├── ...
│   │   └── all_years_summary.csv
│   └── Port_of_Singapore/
│       └── ...
│
└── xview/                            # xView dataset (if needed)
    └── ...
```

## 📊 File Formats

### 1. Annotated Images (`annotated/`)
- **Format**: JPEG images with bounding boxes drawn
- **Naming**: `<original_name>_annotated.jpg`
- **Content**: Visual representation of all detections

### 2. Detections CSV (`detections.csv`)
```csv
image_name,timestamp,total_detections,annotated_path,count_ship,count_harbor,count_large-vehicle
2017-1.jpg,2025-11-30T20:00:00,7,google_earth/Port_of_LA/2017/annotated/2017-1_annotated.jpg,7,0,0
2017-2.jpg,2025-11-30T20:00:01,8,google_earth/Port_of_LA/2017/annotated/2017-2_annotated.jpg,7,1,0
```

### 3. Detections JSON (`detections.json`)
```json
[
  {
    "image_name": "2017-1.jpg",
    "annotated_path": "google_earth/Port_of_LA/2017/annotated/2017-1_annotated.jpg",
    "timestamp": "2025-11-30T20:00:00",
    "total_detections": 7,
    "class_counts": {
      "ship": 7
    },
    "metadata": {
      "source": "Google Earth",
      "resolution": "0.5m",
      "model": "ports_dota_yolo11",
      "confidence_threshold": 0.25
    }
  }
]
```

### 4. Summary Text (`summary.txt`)
```
Detection Summary - Port_of_LA/2017
============================================================

Total Images: 3
Total Detections: 21
Average Detections per Image: 7.00

Detections by Class:
----------------------------------------
  ship                :    20
  large-vehicle       :     1

============================================================
Last Updated: 2025-11-30 20:00:00
```

### 5. All Years Summary (`all_years_summary.csv`)
```csv
year,total_images,total_detections,avg_detections_per_image,total_ship,total_harbor
2017,3,21,7.00,20,1
2018,2,12,6.00,11,1
2020,3,35,11.67,35,0
2021,3,48,16.00,47,1
2022,3,57,19.00,57,0
2023,2,30,15.00,30,0
```

## 🚀 Usage

### 1. Migrate Existing Annotations

```bash
cd notebooks
python migrate_existing_annotations.py
```

This will move your existing Google Earth Port of LA annotations into the new structure.

### 2. Process New Images

#### Google Earth Images
```bash
python detect_and_save_structured.py \
    --dataset google_earth \
    --location Port_of_LA \
    --model-type ports \
    --conf 0.25
```

#### Sentinel-2 Images
```bash
python detect_and_save_structured.py \
    --dataset sentinel2 \
    --location Port_of_Singapore \
    --year 2023 \
    --model-type ports \
    --conf 0.25
```

### 3. Programmatic Access

```python
from save_structured_annotations import StructuredAnnotationManager

# Initialize manager
manager = StructuredAnnotationManager(
    base_output_dir=Path("results/annotations")
)

# Save an annotation
manager.save_annotation(
    dataset_name='google_earth',
    location='Port_of_LA',
    year=2023,
    image_name='2023-1.jpg',
    annotated_image=annotated_img,
    detections={
        'total_detections': 15,
        'class_counts': {'ship': 12, 'harbor': 3}
    },
    metadata={'source': 'Google Earth', 'resolution': '0.5m'}
)

# Create summaries
manager.create_location_summary('google_earth', 'Port_of_LA')
manager.create_dataset_readme('google_earth', 'Description...')
manager.create_master_index()
```

## 📈 Analysis Examples

### Load Detection Data
```python
import pandas as pd

# Load single year
df_2023 = pd.read_csv('annotations/google_earth/Port_of_LA/2023/detections.csv')

# Load all years
df_all = pd.read_csv('annotations/google_earth/Port_of_LA/all_years_summary.csv')

# Analyze trends
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(df_all['year'], df_all['total_detections'], marker='o')
plt.xlabel('Year')
plt.ylabel('Total Detections')
plt.title('Port Activity Over Time')
plt.grid(True)
plt.show()
```

### Compare Locations
```python
import glob

# Load all location summaries
summaries = []
for csv_file in glob.glob('annotations/google_earth/*/all_years_summary.csv'):
    location = Path(csv_file).parent.name
    df = pd.read_csv(csv_file)
    df['location'] = location
    summaries.append(df)

df_combined = pd.concat(summaries)

# Compare 2023 activity across locations
df_2023 = df_combined[df_combined['year'] == 2023]
df_2023.plot(x='location', y='total_detections', kind='bar')
```

## 🔍 Finding Annotations

### By Dataset and Location
```
annotations/google_earth/Port_of_LA/
```

### By Year
```
annotations/google_earth/Port_of_LA/2023/
```

### Specific Annotated Image
```
annotations/google_earth/Port_of_LA/2023/annotated/2023-1_annotated.jpg
```

### All Visualizations
```
annotations/google_earth/Port_of_LA/visualizations/
```

## 📋 Master Index

The `INDEX.md` file provides a complete overview:
- All datasets
- All locations
- Year coverage
- Image and detection counts
- Direct paths to results

## 🎨 Visualizations

Each location can have a `visualizations/` folder containing:
- Comparison grids
- Trend charts
- Detection heatmaps
- Time-lapse frames

## ✅ Benefits

1. **Easy Navigation**: Find any result by dataset → location → year
2. **Multiple Formats**: CSV for analysis, JSON for programs, TXT for humans
3. **Complete Tracking**: Every detection includes metadata and timestamps
4. **Aggregated Summaries**: Quick overview without loading individual files
5. **Scalable**: Add new datasets, locations, or years without reorganization
6. **Reproducible**: Full parameter tracking for scientific rigor

## 🔄 Migration Status

- ✅ Google Earth Port of LA (2017-2023): Ready to migrate
- ⏳ Sentinel-2 Ports: Can be added as needed
- ⏳ Sentinel-2 Retail: Can be added as needed
- ⏳ xView Training Data: Optional

## 📞 Support

For questions or issues with the annotation system:
1. Check the master `INDEX.md` for overview
2. Review dataset-specific `README.md` files
3. Examine `summary.txt` files for quick stats
4. Load CSV files for detailed analysis

---

**Last Updated**: 2025-11-30  
**System Version**: 1.0  
**Maintained By**: Economic Forecasting Project
