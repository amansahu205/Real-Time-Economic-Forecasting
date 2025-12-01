# 🛰️ Real-Time Economic Forecasting

**Cloud-based economic forecasting using satellite imagery, AIS maritime data, and ML.**

## 🎯 Overview

This project predicts economic indicators (trade volume, retail activity) using:
- **Satellite Imagery** - Object detection on ports and retail centers
- **AIS Maritime Data** - Ship tracking for trade flow analysis
- **Machine Learning** - Time series forecasting

## 📁 Project Structure

```
Real-Time-Economic-Forecasting/
├── src/                          # Core source code
│   ├── config.py                 # Configuration settings
│   ├── detection/                # Object detection modules
│   │   ├── tiled_detector.py     # Tiled YOLO detection
│   │   └── annotation_manager.py # Annotation management
│   ├── features/                 # Feature extraction
│   │   ├── satellite_features.py # Satellite feature extraction
│   │   ├── ais_features.py       # AIS feature extraction
│   │   └── feature_fusion.py     # Data fusion
│   └── forecasting/              # Forecasting models
│       └── model.py              # Economic forecaster
│
├── scripts/                      # Pipeline scripts
│   ├── run_pipeline.py           # End-to-end pipeline
│   ├── aws_upload.py             # AWS S3 upload
│   ├── process_ais_data.py       # AIS processing
│   └── process_satellite_data.py # Satellite processing
│
├── notebooks/                    # Analysis notebooks
│   ├── Economic_Forecasting_Model.ipynb
│   └── Port_LA_Analysis.ipynb
│
├── data/                         # Data directory
│   ├── raw/                      # Raw data
│   │   ├── satellite/            # Satellite imagery
│   │   └── ais/                  # AIS maritime data
│   ├── processed/                # Processed data
│   ├── features/                 # Extracted features
│   └── models/                   # Trained models
│
├── results/                      # Results
│   └── annotations/              # Detection results
│
└── docs/                         # Documentation
    ├── AWS_ARCHITECTURE.md       # AWS infrastructure
    └── PROCESSING_GUIDE.md       # Processing guide
```

## 🚀 Quick Start

### 1. Run End-to-End Pipeline

```bash
python scripts/run_pipeline.py --all
```

### 2. Upload to AWS

```bash
# Configure AWS CLI first
aws configure

# Upload data
python scripts/aws_upload.py --all
```

### 3. Individual Steps

```bash
# Process satellite data
python scripts/process_satellite_data.py --dataset ports

# Process AIS data
python scripts/process_ais_data.py --year 2017

# Extract features only
python scripts/run_pipeline.py --features-only

# Train model only
python scripts/run_pipeline.py --train-only
```

## 📊 Data Sources

| Source | Coverage | Records |
|--------|----------|---------|
| **Satellite (Google Earth)** | 5 locations, 2017-2024 | 129 images |
| **AIS Maritime** | Port of LA, 2017 | 365 days |
| **YOLO Models** | Ports, Retail, City | 3 models |

## 🏗️ AWS Architecture

```
S3 (Data Lake) → Glue (Catalog) → Batch/SageMaker (Processing)
     ↓                                      ↓
  Lambda (Ingestion)              EMR (Data Fusion)
     ↓                                      ↓
  EventBridge (Schedule)         SageMaker (Forecasting)
     ↓                                      ↓
  Step Functions (Orchestration) → QuickSight (Dashboard)
```

See `docs/AWS_ARCHITECTURE.md` for details.

## 📈 Pipeline Flow

```
1. Data Ingestion
   └── Satellite images + AIS data → S3

2. Object Detection
   └── YOLO models → Ship/vehicle counts

3. Feature Extraction
   └── Daily metrics, trends, ratios

4. Data Fusion
   └── Merge satellite + AIS + sentiment

5. Forecasting
   └── ML model → Economic predictions

6. Visualization
   └── QuickSight dashboard
```

## 🎯 Models

| Model | Training Data | Classes |
|-------|---------------|---------|
| **Ports** | DOTA + xView | ship, harbor, storage-tank |
| **Retail** | xView | vehicle, building |
| **City** | xView | urban activity |

## 📖 Documentation

- `docs/AWS_ARCHITECTURE.md` - AWS infrastructure design
- `docs/AWS_ARCHITECTURE_DETAILED.md` - Detailed specifications
- `docs/PROCESSING_GUIDE.md` - Data processing guide
- `docs/AIS_DATA_GUIDE.md` - AIS data documentation

## 👥 Team

- **Aman Sahu** - Satellite data, ML pipeline
- **Akul** - AWS architecture
- **Ankur** - SageMaker deployment
- **Sahil** - Data fusion, AIS
- **Supriya** - Forecasting, visualization

---

**Course:** DATA-650 (Fall 2025)  
**Status:** Production Ready  
**Last Updated:** 2025-12-01
