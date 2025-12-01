#!/usr/bin/env python3
"""
Fix Demo_2_Object_Detection.ipynb to use trained models from S3.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "demo" / "Demo_2_Object_Detection.ipynb"

# New model loading cell that uses S3
MODEL_LOADING_CELL = [
    "# Load our trained models from S3\n",
    "print(\"📥 Loading trained models...\")\n",
    "\n",
    "# Download models from S3 (or use local if available)\n",
    "port_model_path = download_model('port')\n",
    "retail_model_path = download_model('retail')\n",
    "\n",
    "# Load YOLO models\n",
    "if port_model_path and Path(port_model_path).exists():\n",
    "    port_model = YOLO(str(port_model_path))\n",
    "    print(f\"   ✅ Port model loaded (trained on DOTA dataset)\")\n",
    "else:\n",
    "    print(\"   ❌ Port model not found!\")\n",
    "    port_model = None\n",
    "\n",
    "if retail_model_path and Path(retail_model_path).exists():\n",
    "    retail_model = YOLO(str(retail_model_path))\n",
    "    print(f\"   ✅ Retail model loaded (trained for car detection)\")\n",
    "else:\n",
    "    print(\"   ❌ Retail model not found!\")\n",
    "    retail_model = None\n",
    "\n",
    "print(\"\\n✅ Models ready for inference!\")\n"
]

# New port image loading cell that uses S3
PORT_IMAGE_CELL = [
    "# Load Port of LA image from S3\n",
    "import cv2\n",
    "from PIL import Image\n",
    "\n",
    "print(\"📸 Loading Port of LA satellite image...\")\n",
    "\n",
    "if USE_S3:\n",
    "    # List images from S3\n",
    "    port_images = list_s3_images(f'{S3_PATHS[\"port_la_images\"]}/2024')\n",
    "    if port_images:\n",
    "        # Download first image\n",
    "        sample_port_image = download_image(port_images[0])\n",
    "        print(f\"   ✅ Downloaded: {port_images[0].split('/')[-1]}\")\n",
    "    else:\n",
    "        print(\"   ⚠️ No 2024 images, trying 2023...\")\n",
    "        port_images = list_s3_images(f'{S3_PATHS[\"port_la_images\"]}/2023')\n",
    "        if port_images:\n",
    "            sample_port_image = download_image(port_images[0])\n",
    "        else:\n",
    "            sample_port_image = None\n",
    "else:\n",
    "    # Local path\n",
    "    port_images_dir = LOCAL_PATHS['port_la_images'] / '2024'\n",
    "    port_images = list(port_images_dir.glob('*.jpg')) if port_images_dir.exists() else []\n",
    "    sample_port_image = port_images[0] if port_images else None\n",
    "\n",
    "if sample_port_image:\n",
    "    # Display original image\n",
    "    img = Image.open(sample_port_image)\n",
    "    plt.figure(figsize=(12, 8))\n",
    "    plt.imshow(img)\n",
    "    plt.title(f'🚢 Port of LA - Original Satellite Image', fontsize=14, fontweight='bold')\n",
    "    plt.axis('off')\n",
    "    plt.show()\n",
    "    print(f\"\\n📐 Image size: {img.size}\")\n",
    "else:\n",
    "    print(\"❌ No port images found!\")\n"
]

# New mall image loading cell that uses S3
MALL_IMAGE_CELL = [
    "# Load Mall of America image from S3\n",
    "print(\"📸 Loading Mall of America satellite image...\")\n",
    "\n",
    "if USE_S3:\n",
    "    # List images from S3\n",
    "    mall_images = list_s3_images(f'{S3_PATHS[\"mall_images\"]}/2017')\n",
    "    if mall_images:\n",
    "        sample_mall_image = download_image(mall_images[0])\n",
    "        print(f\"   ✅ Downloaded: {mall_images[0].split('/')[-1]}\")\n",
    "    else:\n",
    "        sample_mall_image = None\n",
    "else:\n",
    "    mall_images_dir = LOCAL_PATHS['mall_images'] / '2017'\n",
    "    mall_images = list(mall_images_dir.glob('*.jpg')) if mall_images_dir.exists() else []\n",
    "    sample_mall_image = mall_images[0] if mall_images else None\n",
    "\n",
    "if sample_mall_image:\n",
    "    img = Image.open(sample_mall_image)\n",
    "    plt.figure(figsize=(12, 8))\n",
    "    plt.imshow(img)\n",
    "    plt.title(f'🛒 Mall of America - Original Satellite Image', fontsize=14, fontweight='bold')\n",
    "    plt.axis('off')\n",
    "    plt.show()\n",
    "else:\n",
    "    print(\"❌ No mall images found!\")\n"
]


def fix_notebook():
    print(f"Fixing: {NOTEBOOK_PATH.name}")
    
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find and replace cells
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            # Fix model loading cell
            if 'Load our trained models' in source and 'port_model_path' in source:
                print(f"  Fixing model loading cell (index {i})")
                cell['source'] = MODEL_LOADING_CELL
            
            # Fix port image cell
            elif 'Find a sample Port of LA image' in source or 'port_images_dir' in source:
                print(f"  Fixing port image cell (index {i})")
                cell['source'] = PORT_IMAGE_CELL
            
            # Fix mall image cell
            elif 'Find a sample Mall of America image' in source or 'mall_images_dir' in source:
                print(f"  Fixing mall image cell (index {i})")
                cell['source'] = MALL_IMAGE_CELL
    
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("✅ Notebook fixed!")


if __name__ == "__main__":
    fix_notebook()
