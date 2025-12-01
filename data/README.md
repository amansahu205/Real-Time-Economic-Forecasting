# Data Directory Structure

This directory contains all data for the Real-Time Economic Forecasting project.

## Directory Organization

```
data/
├── raw/                    # Raw, unprocessed data from sources
│   ├── satellite/         # Satellite imagery data
│   │   ├── xview/        # xView dataset (150 GB)
│   │   └── samples/      # Sample images for testing
│   ├── news/             # Financial news data
│   │   ├── historical/   # Historical news datasets
│   │   └── live/         # Live API data cache
│   ├── shipping/         # Maritime/shipping data
│   │   ├── ais/         # AIS vessel tracking data
│   │   └── ports/       # Port-specific data
│   └── economic/         # Economic indicators
│       ├── fred/        # FRED data (retail sales, GDP, etc.)
│       └── worldbank/   # World Bank data
│
├── processed/             # Cleaned and preprocessed data
│   ├── satellite/        # Processed satellite images
│   ├── news/            # Cleaned news articles
│   ├── shipping/        # Processed AIS data
│   └── economic/        # Processed economic indicators
│
├── features/             # Engineered features for ML
│   ├── satellite/       # Parking occupancy, building counts, etc.
│   ├── news/           # Sentiment scores, topic features
│   ├── shipping/       # Port activity metrics
│   └── combined/       # Merged multi-modal features
│
└── models/              # Trained models and checkpoints
    ├── satellite/       # Object detection models
    ├── sentiment/       # NLP sentiment models
    ├── forecasting/     # Economic forecasting models
    └── checkpoints/     # Training checkpoints

```

## Data Sources

### 1. Satellite Imagery (150 GB)
- **Source**: xView Dataset
- **Location**: `raw/satellite/xview/`
- **Format**: TIFF images + JSON annotations
- **Download**: http://xviewdataset.org/
- **Purpose**: Detect commercial activity (cars, buildings, infrastructure)

### 2. Financial News (3 GB + Live)
- **Historical**: All the News 2.0 dataset
- **Live**: NewsAPI, Finnhub, Alpha Vantage
- **Location**: `raw/news/`
- **Format**: CSV, JSON
- **Purpose**: Sentiment analysis for market optimism

### 3. Shipping Data (50 GB)
- **Source**: Danish Maritime Authority AIS
- **Location**: `raw/shipping/ais/`
- **Format**: CSV
- **Purpose**: Track global trade flow via vessel movements

### 4. Economic Indicators (< 1 GB)
- **Source**: FRED API, World Bank API
- **Location**: `raw/economic/`
- **Format**: CSV, JSON
- **Purpose**: Target variables for forecasting

## Storage Requirements

| Directory | Estimated Size | Description |
|-----------|---------------|-------------|
| `raw/satellite/` | 150 GB | xView full dataset |
| `raw/news/` | 3 GB | Historical news |
| `raw/shipping/` | 50 GB | AIS data |
| `raw/economic/` | < 1 GB | Economic indicators |
| `processed/` | 20 GB | Cleaned data |
| `features/` | 10 GB | Engineered features |
| `models/` | 20 GB | Trained models |
| **Total** | **~253 GB** | |

## Data Pipeline Flow

```
RAW DATA → PROCESSED DATA → FEATURES → MODELS → PREDICTIONS
```

1. **Raw**: Download from sources
2. **Processed**: Clean, normalize, filter
3. **Features**: Extract meaningful signals
4. **Models**: Train ML models
5. **Predictions**: Generate forecasts

## Important Notes

- **Do NOT commit raw data to Git** (see .gitignore)
- Raw data should be downloaded to D: drive (291 GB available)
- Processed data and features can be shared via S3
- Models should be versioned and stored in S3/model registry

## Data Download Instructions

See individual README files in each subdirectory for specific download instructions:
- `raw/satellite/README.md` - xView download guide
- `raw/news/README.md` - News data sources
- `raw/shipping/README.md` - AIS data download
- `raw/economic/README.md` - FRED/World Bank API setup


