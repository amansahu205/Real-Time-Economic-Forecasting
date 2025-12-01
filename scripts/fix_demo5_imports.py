#!/usr/bin/env python3
"""
Fix Demo_5_Forecasting.ipynb to add missing sklearn imports.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "demo" / "Demo_5_Forecasting.ipynb"

# New setup cell with sklearn imports
NEW_SETUP_CELL = [
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
    "    subprocess.run(['pip', 'install', 'scikit-learn', '-q'], capture_output=True, check=True)\n",
    "    print('✅ Dependencies installed')\n",
    "\n",
    "# Core imports\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# ML imports\n",
    "from sklearn.linear_model import LinearRegression, Ridge\n",
    "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
    "from sklearn.metrics import mean_absolute_error, r2_score\n",
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
    "# S3 Configuration\n",
    "S3_RAW = 'economic-forecast-raw'\n",
    "S3_MODELS = 'economic-forecast-models'\n",
    "S3_PROCESSED = 'economic-forecast-processed'\n",
    "\n",
    "S3_PATHS = {\n",
    "    'ais_la': f's3://{S3_PROCESSED}/ais/Port_of_LA_ais_features.csv',\n",
    "    'detections': f's3://{S3_PROCESSED}/detections',\n",
    "}\n",
    "\n",
    "LOCAL_PATHS = {\n",
    "    'ais_la': PROJECT_ROOT / 'data' / 'processed' / 'ais' / 'Port_of_LA_ais_features.csv',\n",
    "    'detections': PROJECT_ROOT / 'results' / 'annotations',\n",
    "}\n",
    "\n",
    "print(f'✅ Setup complete | S3: {USE_S3}')\n",
    "print(f'📁 Project: {PROJECT_ROOT}')\n",
    "print(f'🔧 Models: LinearRegression, Ridge, RandomForest, GradientBoosting')\n"
]


def fix_notebook():
    print(f"Fixing: {NOTEBOOK_PATH.name}")
    
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find and replace setup cell
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            # Setup cell
            if 'Setup' in source and 'IS_SAGEMAKER' in source:
                print(f"  Updating setup cell (index {i})")
                cell['source'] = NEW_SETUP_CELL
                break
    
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("✅ Notebook fixed with sklearn imports!")


if __name__ == "__main__":
    fix_notebook()
