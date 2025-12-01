#!/usr/bin/env python3
"""
Update demo notebooks to support both local and S3 paths.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEMO_DIR = PROJECT_ROOT / "notebooks" / "demo"

# New setup code that works both locally and in SageMaker
NEW_SETUP_CODE = [
    "# Setup - Works both locally and in SageMaker\n",
    "import sys\n",
    "import os\n",
    "from pathlib import Path\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# Detect environment\n",
    "IS_SAGEMAKER = os.path.exists('/home/ec2-user/SageMaker') or os.environ.get('SM_MODEL_DIR') is not None\n",
    "\n",
    "if IS_SAGEMAKER:\n",
    "    PROJECT_ROOT = Path('/home/ec2-user/SageMaker/Real-Time-Economic-Forecasting')\n",
    "    USE_S3 = True\n",
    "    print('\\U0001F329\\uFE0F  Running in AWS SageMaker')\n",
    "else:\n",
    "    PROJECT_ROOT = Path.cwd().parent.parent  # notebooks/demo/ -> project root\n",
    "    USE_S3 = False\n",
    "    print('\\U0001F4BB Running locally')\n",
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
    "print(f'\\u2705 Setup complete | S3: {USE_S3}')\n"
]


def update_notebook(nb_path):
    """Update a single notebook with S3-compatible setup."""
    print(f"Updating: {nb_path.name}")
    
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find and update the setup cell
    updated = False
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = cell['source']
            source_str = ''.join(source) if isinstance(source, list) else source
            
            # Find setup cell with PROJECT_ROOT
            if 'PROJECT_ROOT' in source_str and ('Path.cwd()' in source_str or 'setup' in source_str.lower()):
                # Replace with new setup code
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
    print("Updating Demo Notebooks for S3 Compatibility")
    print("="*60)
    
    notebooks = list(DEMO_DIR.glob("Demo_*.ipynb"))
    print(f"\nFound {len(notebooks)} demo notebooks\n")
    
    updated_count = 0
    for nb_path in sorted(notebooks):
        if update_notebook(nb_path):
            updated_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Updated {updated_count}/{len(notebooks)} notebooks")
    print("="*60)


if __name__ == "__main__":
    main()
