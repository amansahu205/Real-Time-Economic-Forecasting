#!/usr/bin/env python3
"""
Fix demo notebooks to include proper imports for SageMaker.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEMO_DIR = PROJECT_ROOT / "notebooks" / "demo"

# Updated setup code with all necessary imports and pip installs
NEW_SETUP_CODE = [
    "# Setup - Works both locally and in SageMaker\n",
    "import sys\n",
    "import os\n",
    "from pathlib import Path\n",
    "\n",
    "# Install dependencies in SageMaker\n",
    "if os.path.exists('/home/ec2-user/SageMaker'):\n",
    "    print('📦 Installing dependencies...')\n",
    "    !pip install ultralytics opencv-python-headless -q\n",
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
    "# Detect environment\n",
    "IS_SAGEMAKER = os.path.exists('/home/ec2-user/SageMaker') or os.environ.get('SM_MODEL_DIR') is not None\n",
    "\n",
    "if IS_SAGEMAKER:\n",
    "    PROJECT_ROOT = Path('/home/ec2-user/SageMaker/Real-Time-Economic-Forecasting')\n",
    "    USE_S3 = True\n",
    "    print('🌩️  Running in AWS SageMaker')\n",
    "else:\n",
    "    PROJECT_ROOT = Path.cwd().parent.parent  # notebooks/demo/ -> project root\n",
    "    USE_S3 = False\n",
    "    print('💻 Running locally')\n",
    "\n",
    "# S3 Configuration\n",
    "S3_RAW = 'economic-forecast-raw'\n",
    "S3_MODELS = 'economic-forecast-models'\n",
    "S3_PROCESSED = 'economic-forecast-processed'\n",
    "\n",
    "# Path helper\n",
    "def get_path(path_type):\n",
    "    '''Get path for data - S3 or local based on environment.'''\n",
    "    paths_s3 = {\n",
    "        'satellite': f's3://{S3_RAW}/satellite/google_earth',\n",
    "        'models': f's3://{S3_MODELS}/yolo',\n",
    "        'ais': f's3://{S3_PROCESSED}/ais',\n",
    "        'results': f's3://{S3_PROCESSED}/annotations',\n",
    "    }\n",
    "    paths_local = {\n",
    "        'satellite': PROJECT_ROOT / 'data' / 'raw' / 'satellite' / 'google_earth',\n",
    "        'models': PROJECT_ROOT / 'data' / 'models' / 'satellite',\n",
    "        'ais': PROJECT_ROOT / 'data' / 'processed' / 'ais',\n",
    "        'results': PROJECT_ROOT / 'results' / 'annotations',\n",
    "    }\n",
    "    return paths_s3.get(path_type) if USE_S3 else paths_local.get(path_type)\n",
    "\n",
    "# S3 helper for downloading models\n",
    "def download_model_from_s3(model_name):\n",
    "    '''Download model from S3 to local temp directory.'''\n",
    "    import boto3\n",
    "    import tempfile\n",
    "    \n",
    "    s3 = boto3.client('s3')\n",
    "    \n",
    "    # Model paths in S3\n",
    "    model_keys = {\n",
    "        'port': 'yolo/ports_dota_yolo11_20251127_013205/weights/best.pt',\n",
    "        'retail': 'yolo/retail_yolo11_20251126_150811/weights/best.pt',\n",
    "    }\n",
    "    \n",
    "    key = model_keys.get(model_name)\n",
    "    if not key:\n",
    "        return None\n",
    "    \n",
    "    local_path = Path(tempfile.gettempdir()) / f'{model_name}_best.pt'\n",
    "    \n",
    "    if not local_path.exists():\n",
    "        print(f'📥 Downloading {model_name} model from S3...')\n",
    "        s3.download_file(S3_MODELS, key, str(local_path))\n",
    "        print(f'✅ Downloaded to {local_path}')\n",
    "    \n",
    "    return local_path\n",
    "\n",
    "print(f'✅ Setup complete | S3: {USE_S3} | YOLO: {YOLO_AVAILABLE}')\n"
]


def update_notebook(nb_path):
    """Update a single notebook with fixed imports."""
    print(f"Updating: {nb_path.name}")
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find and update the setup cell
    updated = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = cell['source']
            source_str = ''.join(source) if isinstance(source, list) else source
            
            # Find setup cell
            if 'Setup' in source_str and 'PROJECT_ROOT' in source_str:
                cell['source'] = NEW_SETUP_CODE
                updated = True
                print(f"  Updated cell {i}")
                break
    
    if updated:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        print(f"  ✅ Saved")
    else:
        print(f"  ⚠️  No setup cell found")
    
    return updated


def main():
    print("="*60)
    print("Fixing Demo Notebooks - Adding YOLO imports")
    print("="*60)
    
    notebooks = list(DEMO_DIR.glob("*.ipynb"))
    print(f"\nFound {len(notebooks)} notebooks\n")
    
    updated_count = 0
    for nb_path in sorted(notebooks):
        if update_notebook(nb_path):
            updated_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Updated {updated_count} notebooks")
    print("="*60)


if __name__ == "__main__":
    main()
