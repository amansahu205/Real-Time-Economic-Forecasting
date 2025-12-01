# 🛰️ Real-Time Economic Forecasting

**Cloud-based economic forecasting using satellite imagery, AIS maritime data, and ML.**

## 🎯 Overview

This project predicts economic indicators (trade volume, retail activity) using:
- **Satellite Imagery** - YOLO object detection on ports and retail centers
- **AIS Maritime Data** - Ship tracking for trade flow analysis
- **Machine Learning** - Forecasting with Random Forest, Gradient Boosting

## 📁 Project Structure

```
Real-Time-Economic-Forecasting/
├── src/                          # Core source code
│   ├── config.py                 # Configuration settings
│   ├── aws_utils.py              # AWS/S3 utilities
│   ├── detection/                # Object detection
│   │   ├── tiled_detector.py     # Tiled YOLO for large images
│   │   └── annotation_manager.py # Detection result management
│   ├── features/                 # Feature extraction
│   │   ├── satellite_features.py # Ship/vehicle counts
│   │   ├── ais_features.py       # Maritime traffic metrics
│   │   └── feature_fusion.py     # Multi-source fusion
│   └── forecasting/              # ML models
│       └── model.py              # Economic forecaster
│
├── scripts/                      # Pipeline scripts
│   ├── run_pipeline.py           # End-to-end pipeline
│   ├── aws_upload.py             # S3 data upload
│   ├── process_ais_data.py       # AIS data processing
│   ├── process_satellite_data.py # Satellite processing
│   ├── download_ais_data.py      # AIS data download
│   ├── download_ais_daily.py     # Daily AIS download
│   ├── validate_ais_satellite.py # Data validation
│   └── preprocessing/            # Dataset preparation
│       ├── preprocess_dota_ports.py
│       ├── create_retail_2class.py
│       └── create_optimized_datasets.py
│
├── notebooks/
│   ├── demo/                     # 🎯 DEMO NOTEBOOKS (run these)
│   │   ├── Demo_1_YOLO_Training.ipynb
│   │   ├── Demo_2_Object_Detection.ipynb
│   │   ├── Demo_3_AIS_Data.ipynb
│   │   ├── Demo_4_Data_Fusion.ipynb
│   │   ├── Demo_5_Forecasting.ipynb
│   │   └── PRESENTATION_SCRIPTS.md
│   └── full/                     # Full analysis notebooks
│       ├── Port_LA_Analysis.ipynb
│       ├── Economic_Forecasting_Model.ipynb
│       └── News_Sentiment_Analysis.ipynb
│
├── data/                         # Data (gitignored)
│   ├── raw/satellite/            # Satellite imagery
│   ├── raw/ais/                  # AIS maritime data
│   ├── processed/                # Processed features
│   └── models/                   # Trained YOLO models
│
└── results/                      # Detection results
    └── annotations/
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

## 🎬 Demo Notebooks

Run these in order for presentation:

| Demo | Description | Time |
|------|-------------|------|
| **Demo 1** | YOLO Model Training | 3-4 min |
| **Demo 2** | Ship/Car Detection | 5-6 min |
| **Demo 3** | AIS Maritime Data | 3-4 min |
| **Demo 4** | Data Fusion | 3-4 min |
| **Demo 5** | Economic Forecasting | 5-6 min |

See `notebooks/demo/PRESENTATION_SCRIPTS.md` for talking points.

## 📊 Key Results

| Metric | Finding |
|--------|---------|
| Port ships (2020 vs 2019) | +27% (supply chain backup) |
| Mall cars (2020 vs 2019) | -63% (COVID lockdown) |
| Trade forecast error | ~3% MAE |
| Retail forecast error | ~2% MAE |

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
