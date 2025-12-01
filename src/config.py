"""
Configuration settings for the Economic Forecasting Pipeline
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"

# Satellite data paths
SATELLITE_RAW_DIR = RAW_DATA_DIR / "satellite"
GOOGLE_EARTH_DIR = SATELLITE_RAW_DIR / "google_earth"
SENTINEL_DIR = SATELLITE_RAW_DIR / "sentinel-2-l2a"

# AIS data paths
AIS_RAW_DIR = RAW_DATA_DIR / "ais" / "noaa"
AIS_PROCESSED_DIR = PROCESSED_DATA_DIR / "ais"

# Model paths
SATELLITE_MODELS_DIR = MODELS_DIR / "satellite"
PORTS_MODEL = SATELLITE_MODELS_DIR / "ports_dota_yolo11_20251127_013205" / "weights" / "best.pt"
RETAIL_MODEL = SATELLITE_MODELS_DIR / "retail_yolo11_20251126_150811" / "weights" / "best.pt"
CITY_MODEL = SATELLITE_MODELS_DIR / "city_yolo11_20251127_184743" / "weights" / "best.pt"

# Results paths
ANNOTATIONS_DIR = RESULTS_DIR / "annotations"
DETECTIONS_DIR = RESULTS_DIR / "detections"

# Detection settings
TILE_SIZE = 1024
TILE_OVERLAP = 128
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Port of LA bounding box (for AIS filtering)
PORT_LA_BOUNDS = {
    'min_lat': 33.65,
    'max_lat': 33.85,
    'min_lon': -118.35,
    'max_lon': -118.15
}

# Locations for processing
LOCATIONS = {
    'ports': [
        'Port_of_LA',
        'Port_of_hongkong',
        'Port_of_Salalah',
        'Port_of_Tanjung_priok'
    ],
    'retail': [
        'Mall_of_america'
    ]
}

# AWS S3 configuration
AWS_CONFIG = {
    'raw_bucket': 'economic-forecast-raw',
    'models_bucket': 'economic-forecast-models',
    'processed_bucket': 'economic-forecast-processed',
    'region': 'us-east-1'
}

# Feature engineering settings
FEATURE_CONFIG = {
    'moving_avg_windows': [7, 14, 30],
    'growth_periods': [7, 30],
    'lag_periods': [1, 7, 14, 30]
}
