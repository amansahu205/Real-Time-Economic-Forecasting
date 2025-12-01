# Satellite Imagery Data

## xView Dataset

### Overview
- **Dataset**: hassanmojab/xview-dataset
- **Size**: 20.8 GB
- **Images**: High-resolution satellite imagery
- **Objects**: 60 classes (vehicles, buildings, infrastructure)
- **Resolution**: 0.3m per pixel
- **Purpose**: Detect commercial activity at retail locations

### Download Instructions

#### Option 1: Direct Kaggle Command (Simplest)
```bash
cd data/raw/satellite/xview
kaggle datasets download hassanmojab/xview-dataset
```

#### Option 2: Use Download Script
```bash
cd data/raw/satellite/xview
.\download.ps1  # PowerShell
# or
bash download.sh  # Bash/Git Bash
```

#### Option 3: Python Script
```bash
python xview_download.py
```

### Expected Files After Download

```
xview/
├── xview-dataset.zip          # Downloaded file (20.8 GB)
├── train_images/              # Training images (after extraction)
├── val_images/                # Validation images
└── labels/                    # Annotations (JSON/GeoJSON)
```

### Extract the Dataset
```powershell
# PowerShell
Expand-Archive xview-dataset.zip -DestinationPath .

# Or use 7-Zip, WinRAR, etc.
```

### Object Classes for Economic Analysis

**Vehicles (Retail Activity):**
- Small Car
- Bus
- Pickup Truck
- Cargo Truck
- Container Truck

**Buildings (Context):**
- Building
- Warehouse
- Shopping Mall
- Retail Store

**Infrastructure:**
- Parking Lot
- Loading Dock
- Storage Tank

### Why xView is Best

1. ✅ **Context-aware**: Includes buildings, not just cars
2. ✅ **High resolution**: Can accurately count vehicles
3. ✅ **Multiple object types**: 60 classes for rich features
4. ✅ **Well-annotated**: Ready for ML training
5. ✅ **Manageable size**: 20.8 GB vs 150 GB full dataset

### Data Processing Pipeline

```
1. Download xView dataset (20.8 GB)
   ↓
2. Extract images and labels
   ↓
3. Filter for retail areas
   ↓
4. Run object detection (YOLOv8)
   ↓
5. Count vehicles near retail buildings
   ↓
6. Calculate parking occupancy rates
   ↓
7. Generate daily activity metrics
```

### Feature Engineering

From satellite images, extract:
- `cars_at_retail`: Vehicle count near stores
- `parking_occupancy`: Percentage of parking spaces filled
- `retail_building_count`: Number of stores in area
- `loading_dock_activity`: Trucks at loading zones
- `construction_sites`: New development (growth signal)

### Alternative Datasets

If xView doesn't work:
- **COWC**: Cars Overhead With Context (15 GB)
- **DOTA**: Aerial object detection (2.5 GB)
- **SpaceNet**: Building footprints

### Kaggle Dataset Info

- **URL**: https://www.kaggle.com/datasets/hassanmojab/xview-dataset
- **Downloads**: 2,844
- **Rating**: 0.69/1.0
- **Last Updated**: Aug 2021

### Team Responsibility

- **Aman Kumar Sahu**: Download and setup
- **Ankur Sheth**: Image processing and feature extraction

### Next Steps

1. Download dataset (1-2 hours)
2. Extract files
3. Run `scripts/process_satellite.py`
4. Verify extracted features
