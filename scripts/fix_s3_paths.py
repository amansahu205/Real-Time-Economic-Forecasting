#!/usr/bin/env python3
"""
Fix demo notebooks with correct S3 paths based on actual bucket structure.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEMO_DIR = PROJECT_ROOT / "notebooks" / "demo"

# Corrected setup code matching actual S3 structure
NEW_SETUP_CODE = [
    "# Setup - Works both locally and in SageMaker\n",
    "import sys\n",
    "import os\n",
    "from pathlib import Path\n",
    "\n",
    "# Install dependencies in SageMaker\n",
    "IS_SAGEMAKER = os.path.exists('/home/ec2-user/SageMaker') or os.environ.get('SM_MODEL_DIR') is not None\n",
    "\n",
    "if IS_SAGEMAKER:\n",
    "    print('📦 Installing dependencies...')\n",
    "    import subprocess\n",
    "    subprocess.run(['pip', 'install', 'ultralytics', 'opencv-python-headless', '-q'], check=True)\n",
    "    print('✅ Dependencies installed')\n",
    "\n",
    "# Core imports\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# YOLO import\n",
    "try:\n",
    "    from ultralytics import YOLO\n",
    "    YOLO_AVAILABLE = True\n",
    "except ImportError:\n",
    "    YOLO_AVAILABLE = False\n",
    "    print('⚠️ YOLO not available - run: pip install ultralytics')\n",
    "\n",
    "# Environment detection\n",
    "if IS_SAGEMAKER:\n",
    "    PROJECT_ROOT = Path('/home/ec2-user/SageMaker/Real-Time-Economic-Forecasting')\n",
    "    USE_S3 = True\n",
    "    print('🌩️  Running in AWS SageMaker')\n",
    "else:\n",
    "    PROJECT_ROOT = Path.cwd().parent.parent\n",
    "    USE_S3 = False\n",
    "    print('💻 Running locally')\n",
    "\n",
    "# ===========================================\n",
    "# S3 BUCKET CONFIGURATION (ACTUAL STRUCTURE)\n",
    "# ===========================================\n",
    "S3_RAW = 'economic-forecast-raw'\n",
    "S3_MODELS = 'economic-forecast-models'\n",
    "S3_PROCESSED = 'economic-forecast-processed'\n",
    "\n",
    "# S3 Paths (matching actual bucket structure)\n",
    "S3_PATHS = {\n",
    "    'satellite': f's3://{S3_RAW}/satellite/google_earth',\n",
    "    'port_la_images': f's3://{S3_RAW}/satellite/google_earth/Port_of_LA',\n",
    "    'mall_images': f's3://{S3_RAW}/satellite/google_earth/Mall_of_america',\n",
    "    'models': f's3://{S3_MODELS}/yolo',\n",
    "    'port_model': f's3://{S3_MODELS}/yolo/ports/best.pt',\n",
    "    'retail_model': f's3://{S3_MODELS}/yolo/retail/best.pt',\n",
    "    'city_model': f's3://{S3_MODELS}/yolo/city/best.pt',\n",
    "    'ais': f's3://{S3_PROCESSED}/ais',\n",
    "    'ais_la': f's3://{S3_PROCESSED}/ais/Port_of_LA_ais_features.csv',\n",
    "    'detections': f's3://{S3_PROCESSED}/detections',\n",
    "    'news': f's3://{S3_RAW}/news/sentiment/data',\n",
    "}\n",
    "\n",
    "# Local paths\n",
    "LOCAL_PATHS = {\n",
    "    'satellite': PROJECT_ROOT / 'data' / 'raw' / 'satellite' / 'google_earth',\n",
    "    'port_la_images': PROJECT_ROOT / 'data' / 'raw' / 'satellite' / 'google_earth' / 'Port_of_LA',\n",
    "    'mall_images': PROJECT_ROOT / 'data' / 'raw' / 'satellite' / 'google_earth' / 'Mall_of_america',\n",
    "    'models': PROJECT_ROOT / 'data' / 'models' / 'satellite',\n",
    "    'port_model': PROJECT_ROOT / 'data' / 'models' / 'satellite' / 'ports_dota_yolo11_20251127_013205' / 'weights' / 'best.pt',\n",
    "    'retail_model': PROJECT_ROOT / 'data' / 'models' / 'satellite' / 'retail_yolo11_20251126_150811' / 'weights' / 'best.pt',\n",
    "    'ais': PROJECT_ROOT / 'data' / 'processed' / 'ais',\n",
    "    'ais_la': PROJECT_ROOT / 'data' / 'processed' / 'ais' / 'Port_of_LA_ais_features.csv',\n",
    "    'detections': PROJECT_ROOT / 'results' / 'annotations',\n",
    "}\n",
    "\n",
    "def get_path(key):\n",
    "    '''Get path - S3 or local based on environment.'''\n",
    "    if USE_S3:\n",
    "        return S3_PATHS.get(key, S3_PATHS.get('satellite'))\n",
    "    else:\n",
    "        return LOCAL_PATHS.get(key, LOCAL_PATHS.get('satellite'))\n",
    "\n",
    "def download_model(model_type='port'):\n",
    "    '''Download model from S3 to local temp for inference.'''\n",
    "    if not USE_S3:\n",
    "        # Return local path\n",
    "        if model_type == 'port':\n",
    "            return LOCAL_PATHS['port_model']\n",
    "        elif model_type == 'retail':\n",
    "            return LOCAL_PATHS['retail_model']\n",
    "        return None\n",
    "    \n",
    "    import boto3\n",
    "    import tempfile\n",
    "    \n",
    "    s3 = boto3.client('s3')\n",
    "    \n",
    "    model_keys = {\n",
    "        'port': 'yolo/ports/best.pt',\n",
    "        'retail': 'yolo/retail/best.pt',\n",
    "        'city': 'yolo/city/best.pt',\n",
    "    }\n",
    "    \n",
    "    key = model_keys.get(model_type)\n",
    "    if not key:\n",
    "        print(f'❌ Unknown model type: {model_type}')\n",
    "        return None\n",
    "    \n",
    "    local_path = Path(tempfile.gettempdir()) / f'{model_type}_best.pt'\n",
    "    \n",
    "    if not local_path.exists():\n",
    "        print(f'📥 Downloading {model_type} model from S3...')\n",
    "        s3.download_file(S3_MODELS, key, str(local_path))\n",
    "        print(f'✅ Model saved to {local_path}')\n",
    "    else:\n",
    "        print(f'✅ Using cached model: {local_path}')\n",
    "    \n",
    "    return local_path\n",
    "\n",
    "def list_s3_images(prefix):\n",
    "    '''List images in S3 bucket.'''\n",
    "    import boto3\n",
    "    s3 = boto3.client('s3')\n",
    "    \n",
    "    # Parse bucket and prefix from s3:// path\n",
    "    if prefix.startswith('s3://'):\n",
    "        parts = prefix.replace('s3://', '').split('/', 1)\n",
    "        bucket = parts[0]\n",
    "        prefix = parts[1] if len(parts) > 1 else ''\n",
    "    else:\n",
    "        bucket = S3_RAW\n",
    "    \n",
    "    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)\n",
    "    \n",
    "    images = []\n",
    "    for obj in response.get('Contents', []):\n",
    "        key = obj['Key']\n",
    "        if key.endswith(('.jpg', '.jpeg', '.png', '.tif')):\n",
    "            images.append(f's3://{bucket}/{key}')\n",
    "    \n",
    "    return images\n",
    "\n",
    "def download_image(s3_path, local_dir='/tmp'):\n",
    "    '''Download single image from S3.'''\n",
    "    import boto3\n",
    "    s3 = boto3.client('s3')\n",
    "    \n",
    "    parts = s3_path.replace('s3://', '').split('/', 1)\n",
    "    bucket = parts[0]\n",
    "    key = parts[1]\n",
    "    \n",
    "    filename = key.split('/')[-1]\n",
    "    local_path = Path(local_dir) / filename\n",
    "    \n",
    "    s3.download_file(bucket, key, str(local_path))\n",
    "    return local_path\n",
    "\n",
    "print(f'✅ Setup complete | S3: {USE_S3} | YOLO: {YOLO_AVAILABLE}')\n",
    "print(f'📁 Project: {PROJECT_ROOT}')\n"
]


def update_notebook(nb_path):
    """Update a single notebook."""
    print(f"Updating: {nb_path.name}")
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    updated = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = cell['source']
            source_str = ''.join(source) if isinstance(source, list) else source
            
            if 'Setup' in source_str and 'PROJECT_ROOT' in source_str:
                cell['source'] = NEW_SETUP_CODE
                updated = True
                print(f"  Updated cell {i}")
                break
    
    if updated:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"  ✅ Saved")
    
    return updated


def main():
    print("="*60)
    print("Fixing S3 Paths in Demo Notebooks")
    print("="*60)
    
    notebooks = list(DEMO_DIR.glob("*.ipynb"))
    print(f"\nFound {len(notebooks)} notebooks\n")
    
    for nb_path in sorted(notebooks):
        update_notebook(nb_path)
    
    print(f"\n{'='*60}")
    print("✅ All notebooks updated with correct S3 paths")
    print("="*60)


if __name__ == "__main__":
    main()
