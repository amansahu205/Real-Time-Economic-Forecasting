#!/usr/bin/env python3
"""
Update Demo_2_Object_Detection.ipynb to use tiled detection.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "demo" / "Demo_2_Object_Detection.ipynb"

# New setup cell with TiledDetector
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
    "    subprocess.run(['pip', 'install', 'ultralytics', 'opencv-python-headless', '-q'], \n",
    "                   capture_output=True, check=True)\n",
    "    print('✅ Dependencies installed')\n",
    "\n",
    "# Core imports\n",
    "import cv2\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from PIL import Image\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# YOLO import\n",
    "try:\n",
    "    from ultralytics import YOLO\n",
    "    YOLO_AVAILABLE = True\n",
    "except ImportError:\n",
    "    YOLO_AVAILABLE = False\n",
    "    print('⚠️ YOLO not available')\n",
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
    "# Add src to path for TiledDetector\n",
    "sys.path.insert(0, str(PROJECT_ROOT / 'src'))\n",
    "\n",
    "# S3 Configuration\n",
    "S3_RAW = 'economic-forecast-raw'\n",
    "S3_MODELS = 'economic-forecast-models'\n",
    "S3_PROCESSED = 'economic-forecast-processed'\n",
    "\n",
    "S3_PATHS = {\n",
    "    'satellite': f's3://{S3_RAW}/satellite/google_earth',\n",
    "    'port_la_images': f's3://{S3_RAW}/satellite/google_earth/Port_of_LA',\n",
    "    'mall_images': f's3://{S3_RAW}/satellite/google_earth/Mall_of_america',\n",
    "    'models': f's3://{S3_MODELS}/yolo',\n",
    "    'port_model': f's3://{S3_MODELS}/yolo/ports/best.pt',\n",
    "    'retail_model': f's3://{S3_MODELS}/yolo/retail/best.pt',\n",
    "}\n",
    "\n",
    "LOCAL_PATHS = {\n",
    "    'satellite': PROJECT_ROOT / 'data' / 'raw' / 'satellite' / 'google_earth',\n",
    "    'port_la_images': PROJECT_ROOT / 'data' / 'raw' / 'satellite' / 'google_earth' / 'Port_of_LA',\n",
    "    'mall_images': PROJECT_ROOT / 'data' / 'raw' / 'satellite' / 'google_earth' / 'Mall_of_america',\n",
    "    'port_model': PROJECT_ROOT / 'data' / 'models' / 'satellite' / 'ports_dota_yolo11_20251127_013205' / 'weights' / 'best.pt',\n",
    "    'retail_model': PROJECT_ROOT / 'data' / 'models' / 'satellite' / 'retail_yolo11_20251126_150811' / 'weights' / 'best.pt',\n",
    "}\n",
    "\n",
    "def download_model(model_type='port'):\n",
    "    '''Download model from S3.'''\n",
    "    if not USE_S3:\n",
    "        return LOCAL_PATHS.get(f'{model_type}_model')\n",
    "    \n",
    "    import boto3\n",
    "    import tempfile\n",
    "    s3 = boto3.client('s3')\n",
    "    \n",
    "    model_keys = {'port': 'yolo/ports/best.pt', 'retail': 'yolo/retail/best.pt'}\n",
    "    key = model_keys.get(model_type)\n",
    "    if not key:\n",
    "        return None\n",
    "    \n",
    "    local_path = Path(tempfile.gettempdir()) / f'{model_type}_best.pt'\n",
    "    if not local_path.exists():\n",
    "        print(f'📥 Downloading {model_type} model from S3...')\n",
    "        s3.download_file(S3_MODELS, key, str(local_path))\n",
    "        print(f'✅ Downloaded to {local_path}')\n",
    "    else:\n",
    "        print(f'✅ Using cached: {local_path}')\n",
    "    return local_path\n",
    "\n",
    "def list_s3_images(prefix):\n",
    "    '''List images in S3.'''\n",
    "    import boto3\n",
    "    s3 = boto3.client('s3')\n",
    "    if prefix.startswith('s3://'):\n",
    "        parts = prefix.replace('s3://', '').split('/', 1)\n",
    "        bucket, prefix = parts[0], parts[1] if len(parts) > 1 else ''\n",
    "    else:\n",
    "        bucket = S3_RAW\n",
    "    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)\n",
    "    return [f's3://{bucket}/{obj[\"Key\"]}' for obj in response.get('Contents', []) \n",
    "            if obj['Key'].endswith(('.jpg', '.jpeg', '.png'))]\n",
    "\n",
    "def download_image(s3_path, local_dir='/tmp'):\n",
    "    '''Download image from S3.'''\n",
    "    import boto3\n",
    "    s3 = boto3.client('s3')\n",
    "    parts = s3_path.replace('s3://', '').split('/', 1)\n",
    "    bucket, key = parts[0], parts[1]\n",
    "    local_path = Path(local_dir) / key.split('/')[-1]\n",
    "    s3.download_file(bucket, key, str(local_path))\n",
    "    return local_path\n",
    "\n",
    "print(f'✅ Setup complete | S3: {USE_S3} | YOLO: {YOLO_AVAILABLE}')\n"
]

# Tiled detector cell
TILED_DETECTOR_CELL = [
    "# Tiled Detection for High-Resolution Images\n",
    "# ============================================\n",
    "# Large satellite images need to be split into tiles for accurate detection\n",
    "# of small objects like ships and cars.\n",
    "\n",
    "class TiledDetector:\n",
    "    '''\n",
    "    Tiled object detection for high-resolution satellite imagery.\n",
    "    \n",
    "    Why Tiling?\n",
    "    - YOLO processes images at 640x640\n",
    "    - Our satellite images are 4000x3000+\n",
    "    - Without tiling: small objects get lost when image is resized\n",
    "    - With tiling: each tile preserves detail for small object detection\n",
    "    '''\n",
    "    \n",
    "    def __init__(self, model, tile_size=1024, overlap=128):\n",
    "        self.model = model\n",
    "        self.tile_size = tile_size\n",
    "        self.overlap = overlap\n",
    "        self.stride = tile_size - overlap\n",
    "    \n",
    "    def process_image(self, image, conf=0.25):\n",
    "        '''Process image with tiling and return annotated image + stats.'''\n",
    "        h, w = image.shape[:2]\n",
    "        all_detections = []\n",
    "        \n",
    "        # Calculate tiles\n",
    "        rows = (h - self.overlap) // self.stride + 1\n",
    "        cols = (w - self.overlap) // self.stride + 1\n",
    "        total_tiles = rows * cols\n",
    "        \n",
    "        print(f'   📐 Image size: {w}x{h}')\n",
    "        print(f'   🔲 Tiles: {cols}x{rows} = {total_tiles} tiles')\n",
    "        print(f'   🔍 Processing...')\n",
    "        \n",
    "        # Process each tile\n",
    "        for row in range(rows):\n",
    "            for col in range(cols):\n",
    "                y1 = row * self.stride\n",
    "                x1 = col * self.stride\n",
    "                y2 = min(y1 + self.tile_size, h)\n",
    "                x2 = min(x1 + self.tile_size, w)\n",
    "                \n",
    "                # Handle edge tiles\n",
    "                if y2 == h and y2 - y1 < self.tile_size:\n",
    "                    y1 = max(0, h - self.tile_size)\n",
    "                if x2 == w and x2 - x1 < self.tile_size:\n",
    "                    x1 = max(0, w - self.tile_size)\n",
    "                \n",
    "                tile = image[y1:y2, x1:x2]\n",
    "                \n",
    "                # Run detection\n",
    "                results = self.model(tile, conf=conf, verbose=False)[0]\n",
    "                \n",
    "                # Convert to global coordinates\n",
    "                for box in results.boxes:\n",
    "                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()\n",
    "                    all_detections.append({\n",
    "                        'bbox': [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1],\n",
    "                        'confidence': float(box.conf[0]),\n",
    "                        'class_id': int(box.cls[0]),\n",
    "                        'class_name': self.model.names[int(box.cls[0])]\n",
    "                    })\n",
    "        \n",
    "        # Apply NMS to remove duplicates from overlapping tiles\n",
    "        detections = self._nms(all_detections)\n",
    "        \n",
    "        # Draw detections\n",
    "        annotated = self._draw(image.copy(), detections)\n",
    "        \n",
    "        # Stats\n",
    "        class_counts = {}\n",
    "        for d in detections:\n",
    "            cls = d['class_name']\n",
    "            class_counts[cls] = class_counts.get(cls, 0) + 1\n",
    "        \n",
    "        return annotated, {'total': len(detections), 'by_class': class_counts, 'tiles': total_tiles}\n",
    "    \n",
    "    def _nms(self, detections, iou_thresh=0.5):\n",
    "        '''Non-Maximum Suppression to remove duplicate detections.'''\n",
    "        if not detections:\n",
    "            return []\n",
    "        \n",
    "        # Group by class\n",
    "        by_class = {}\n",
    "        for d in detections:\n",
    "            cls = d['class_name']\n",
    "            by_class.setdefault(cls, []).append(d)\n",
    "        \n",
    "        final = []\n",
    "        for cls, dets in by_class.items():\n",
    "            boxes = np.array([d['bbox'] for d in dets])\n",
    "            scores = np.array([d['confidence'] for d in dets])\n",
    "            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.0, iou_thresh)\n",
    "            if len(indices) > 0:\n",
    "                for idx in indices.flatten():\n",
    "                    final.append(dets[idx])\n",
    "        return final\n",
    "    \n",
    "    def _draw(self, image, detections):\n",
    "        '''Draw bounding boxes.'''\n",
    "        colors = {'ship': (0,255,0), 'large-vehicle': (0,165,255), 'small-vehicle': (0,255,255),\n",
    "                  'car': (0,255,0), 'harbor': (255,0,0), 'storage-tank': (255,255,0)}\n",
    "        for d in detections:\n",
    "            x1, y1, x2, y2 = map(int, d['bbox'])\n",
    "            color = colors.get(d['class_name'], (255,255,255))\n",
    "            cv2.rectangle(image, (x1,y1), (x2,y2), color, 2)\n",
    "            label = f\"{d['class_name']} {d['confidence']:.2f}\"\n",
    "            cv2.putText(image, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)\n",
    "        return image\n",
    "\n",
    "print('✅ TiledDetector ready')\n",
    "print('   • Tile size: 1024x1024')\n",
    "print('   • Overlap: 128px (prevents edge artifacts)')\n",
    "print('   • NMS: Removes duplicate detections')\n"
]

# Model loading cell
MODEL_LOADING_CELL = [
    "# Load Trained Models from S3\n",
    "print('📥 Loading trained models...')\n",
    "\n",
    "# Download from S3\n",
    "port_model_path = download_model('port')\n",
    "retail_model_path = download_model('retail')\n",
    "\n",
    "# Load models\n",
    "port_model = YOLO(str(port_model_path)) if port_model_path else None\n",
    "retail_model = YOLO(str(retail_model_path)) if retail_model_path else None\n",
    "\n",
    "# Create tiled detectors\n",
    "if port_model:\n",
    "    port_detector = TiledDetector(port_model, tile_size=1024, overlap=128)\n",
    "    print('   ✅ Port detector ready (DOTA-trained for ships)')\n",
    "\n",
    "if retail_model:\n",
    "    retail_detector = TiledDetector(retail_model, tile_size=1024, overlap=128)\n",
    "    print('   ✅ Retail detector ready (trained for cars)')\n",
    "\n",
    "print('\\n✅ Models loaded with tiled detection!')\n"
]

# Port detection cell
PORT_DETECTION_CELL = [
    "# Port of LA: Ship Detection with Tiling\n",
    "print('🚢 PORT OF LA - SHIP DETECTION')\n",
    "print('='*50)\n",
    "\n",
    "# Get image from S3\n",
    "if USE_S3:\n",
    "    port_images = list_s3_images(f'{S3_PATHS[\"port_la_images\"]}/2024')\n",
    "    if not port_images:\n",
    "        port_images = list_s3_images(f'{S3_PATHS[\"port_la_images\"]}/2023')\n",
    "    sample_port_image = download_image(port_images[0]) if port_images else None\n",
    "else:\n",
    "    port_dir = LOCAL_PATHS['port_la_images'] / '2024'\n",
    "    port_images = list(port_dir.glob('*.jpg')) if port_dir.exists() else []\n",
    "    sample_port_image = port_images[0] if port_images else None\n",
    "\n",
    "if sample_port_image and port_model:\n",
    "    # Load image\n",
    "    img = cv2.imread(str(sample_port_image))\n",
    "    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "    \n",
    "    print(f'\\n📸 Image: {sample_port_image}')\n",
    "    \n",
    "    # Run tiled detection\n",
    "    annotated, stats = port_detector.process_image(img, conf=0.25)\n",
    "    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)\n",
    "    \n",
    "    # Display results\n",
    "    fig, axes = plt.subplots(1, 2, figsize=(16, 8))\n",
    "    \n",
    "    axes[0].imshow(img_rgb)\n",
    "    axes[0].set_title('Original Image', fontsize=14)\n",
    "    axes[0].axis('off')\n",
    "    \n",
    "    axes[1].imshow(annotated_rgb)\n",
    "    axes[1].set_title(f'Detected: {stats[\"total\"]} objects ({stats[\"tiles\"]} tiles)', fontsize=14)\n",
    "    axes[1].axis('off')\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "    \n",
    "    # Print stats\n",
    "    print(f'\\n📊 DETECTION RESULTS')\n",
    "    print('='*50)\n",
    "    print(f'Total objects: {stats[\"total\"]}')\n",
    "    print(f'Tiles processed: {stats[\"tiles\"]}')\n",
    "    print('\\nBy class:')\n",
    "    for cls, count in sorted(stats['by_class'].items(), key=lambda x: -x[1]):\n",
    "        print(f'   • {cls}: {count}')\n",
    "else:\n",
    "    print('❌ No image or model available')\n"
]

# Mall detection cell
MALL_DETECTION_CELL = [
    "# Mall of America: Vehicle Detection with Tiling\n",
    "print('🛒 MALL OF AMERICA - VEHICLE DETECTION')\n",
    "print('='*50)\n",
    "\n",
    "# Get image from S3\n",
    "if USE_S3:\n",
    "    mall_images = list_s3_images(f'{S3_PATHS[\"mall_images\"]}/2017')\n",
    "    sample_mall_image = download_image(mall_images[0]) if mall_images else None\n",
    "else:\n",
    "    mall_dir = LOCAL_PATHS['mall_images'] / '2017'\n",
    "    mall_images = list(mall_dir.glob('*.jpg')) if mall_dir.exists() else []\n",
    "    sample_mall_image = mall_images[0] if mall_images else None\n",
    "\n",
    "if sample_mall_image and retail_model:\n",
    "    # Load image\n",
    "    img = cv2.imread(str(sample_mall_image))\n",
    "    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "    \n",
    "    print(f'\\n📸 Image: {sample_mall_image}')\n",
    "    \n",
    "    # Run tiled detection\n",
    "    annotated, stats = retail_detector.process_image(img, conf=0.25)\n",
    "    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)\n",
    "    \n",
    "    # Display results\n",
    "    fig, axes = plt.subplots(1, 2, figsize=(16, 8))\n",
    "    \n",
    "    axes[0].imshow(img_rgb)\n",
    "    axes[0].set_title('Original Image', fontsize=14)\n",
    "    axes[0].axis('off')\n",
    "    \n",
    "    axes[1].imshow(annotated_rgb)\n",
    "    axes[1].set_title(f'Detected: {stats[\"total\"]} vehicles ({stats[\"tiles\"]} tiles)', fontsize=14)\n",
    "    axes[1].axis('off')\n",
    "    \n",
    "    plt.tight_layout()\n",
    "    plt.show()\n",
    "    \n",
    "    # Print stats\n",
    "    print(f'\\n📊 DETECTION RESULTS')\n",
    "    print('='*50)\n",
    "    print(f'Total vehicles: {stats[\"total\"]}')\n",
    "    print(f'Tiles processed: {stats[\"tiles\"]}')\n",
    "    print('\\nBy class:')\n",
    "    for cls, count in sorted(stats['by_class'].items(), key=lambda x: -x[1]):\n",
    "        print(f'   • {cls}: {count}')\n",
    "else:\n",
    "    print('❌ No image or model available')\n"
]


def update_notebook():
    print(f"Updating: {NOTEBOOK_PATH.name}")
    
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find and replace cells
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            # Setup cell
            if 'Setup' in source and 'IS_SAGEMAKER' in source:
                print(f"  Updating setup cell (index {i})")
                cell['source'] = NEW_SETUP_CELL
            
            # Model loading cell
            elif 'Load' in source and 'trained models' in source.lower():
                print(f"  Updating model loading cell (index {i})")
                cell['source'] = MODEL_LOADING_CELL
            
            # Port detection cell
            elif 'Port of LA' in source and ('detection' in source.lower() or 'ship' in source.lower()):
                if 'Run' in source or 'results' in source:
                    print(f"  Updating port detection cell (index {i})")
                    cell['source'] = PORT_DETECTION_CELL
            
            # Mall detection cell  
            elif 'Mall' in source and ('detection' in source.lower() or 'vehicle' in source.lower()):
                if 'Run' in source or 'results' in source:
                    print(f"  Updating mall detection cell (index {i})")
                    cell['source'] = MALL_DETECTION_CELL
    
    # Insert TiledDetector cell after setup
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown' and 'Load Trained Models' in ''.join(cell.get('source', [])):
            # Insert before this markdown
            tiled_cell = {
                'cell_type': 'code',
                'execution_count': None,
                'metadata': {},
                'outputs': [],
                'source': TILED_DETECTOR_CELL
            }
            nb['cells'].insert(i, tiled_cell)
            print(f"  Inserted TiledDetector cell at index {i}")
            break
    
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("✅ Notebook updated with tiled detection!")


if __name__ == "__main__":
    update_notebook()
