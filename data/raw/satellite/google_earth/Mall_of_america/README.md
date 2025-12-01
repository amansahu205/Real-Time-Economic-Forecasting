# Mall of America

Google Earth high-resolution satellite imagery of Mall of America (Minnesota, USA).

## Overview

- **Location**: Bloomington, Minnesota, USA
- **Type**: Retail (Shopping Mall)
- **Total Images**: 19
- **Years Covered**: 2017 - 2024
- **Number of Years**: 8

## About Mall of America

- Largest shopping mall in the United States
- Over 500 stores
- Major economic indicator for retail activity
- Parking lot occupancy correlates with consumer spending

## Images by Year

| Year | Images |
|------|--------|
| 2017 | 2 |
| 2018 | 2 |
| 2019 | 2 |
| 2020 | 6 |
| 2021 | 2 |
| 2022 | 2 |
| 2023 | 1 |
| 2024 | 2 |

## Directory Structure

```
Mall_of_america/
  ├── 2017/
  │   └── 2 images (2017-1.jpg to 2017-2.jpg)
  ├── 2018/
  │   └── 2 images (2018-1.jpg to 2018-2.jpg)
  ├── 2019/
  │   └── 2 images (2019-1.jpg to 2019-2.jpg)
  ├── 2020/
  │   └── 6 images (2020-1.jpg to 2020-6.jpg)
  ├── 2021/
  │   └── 2 images (2021-1.jpg to 2021-2.jpg)
  ├── 2022/
  │   └── 2 images (2022-1.jpg to 2022-2.jpg)
  ├── 2023/
  │   └── 1 images (2023-1.jpg to 2023-1.jpg)
  ├── 2024/
  │   └── 2 images (2024-1.jpg to 2024-2.jpg)
  └── README.md
```

## Detection Objects

Using the retail detection model, we can identify:
- **Vehicles**: Cars in parking lots (retail activity indicator)
- **Buildings**: Mall structures
- **Parking Areas**: Occupancy levels

## Processing

Process with tiled comparison:

```bash
python process_retail_tiled.py
```

This will:
1. Run both full image and tiled detection
2. Compare which method detects more vehicles
3. Save annotated images and statistics
4. Generate comparison visualizations
