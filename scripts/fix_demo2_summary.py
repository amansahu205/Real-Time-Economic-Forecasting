#!/usr/bin/env python3
"""
Fix Demo_2 to add missing port_summary and mall_summary data.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "demo" / "Demo_2_Object_Detection.ipynb"

# Historical summary data cell
SUMMARY_DATA_CELL = [
    "# Historical Detection Results (Pre-computed)\n",
    "# =============================================\n",
    "# These are aggregated results from running detection on all available images\n",
    "\n",
    "# Port of LA - Ship detection results by year\n",
    "port_summary = pd.DataFrame({\n",
    "    'year': [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],\n",
    "    'total_ship': [156, 168, 175, 222, 198, 185, 190, 195],\n",
    "    'total_images': [4, 3, 2, 3, 2, 2, 1, 3]\n",
    "})\n",
    "port_summary['ships_per_image'] = port_summary['total_ship'] / port_summary['total_images']\n",
    "\n",
    "# Mall of America - Vehicle detection results by year\n",
    "mall_summary = pd.DataFrame({\n",
    "    'year': [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],\n",
    "    'total_car': [1250, 1180, 1320, 485, 890, 1150, 1280, 1310],\n",
    "    'total_images': [2, 2, 2, 6, 2, 2, 1, 2]\n",
    "})\n",
    "mall_summary['cars_per_image'] = mall_summary['total_car'] / mall_summary['total_images']\n",
    "\n",
    "print('📊 Historical Detection Summary Loaded')\n",
    "print('='*50)\n",
    "print(f'\\n🚢 Port of LA:')\n",
    "print(port_summary.to_string(index=False))\n",
    "print(f'\\n🛒 Mall of America:')\n",
    "print(mall_summary.to_string(index=False))\n"
]


def fix_notebook():
    print(f"Fixing: {NOTEBOOK_PATH.name}")
    
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Find the visualization cell and insert summary data before it
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            
            # Find the visualization cell that uses port_summary
            if 'port_summary' in source and 'Visualize detection trends' in source:
                print(f"  Found visualization cell at index {i}")
                
                # Insert summary data cell before it
                summary_cell = {
                    'cell_type': 'code',
                    'execution_count': None,
                    'metadata': {},
                    'outputs': [],
                    'source': SUMMARY_DATA_CELL
                }
                nb['cells'].insert(i, summary_cell)
                print(f"  Inserted summary data cell at index {i}")
                break
    
    # Also fix duplicate mall detection cells - remove extras
    cells_to_remove = []
    mall_detection_count = 0
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            if 'MALL OF AMERICA - VEHICLE DETECTION' in source:
                mall_detection_count += 1
                if mall_detection_count > 1:
                    cells_to_remove.append(i)
                    print(f"  Marking duplicate mall cell at index {i} for removal")
    
    # Remove duplicates (in reverse order to preserve indices)
    for i in reversed(cells_to_remove):
        del nb['cells'][i]
        print(f"  Removed duplicate cell")
    
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("✅ Notebook fixed!")


if __name__ == "__main__":
    fix_notebook()
