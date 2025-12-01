# Google Earth Satellite Imagery

High-resolution satellite imagery from Google Earth for port activity monitoring.

## Dataset Overview

- **Total Locations**: 4
- **Total Images**: 110
- **Years Covered**: 2017 - 2024
- **Resolution**: ~0.5m (varies by location and date)

## Locations

| Location | Images | Years | Year Range |
|----------|--------|-------|------------|
| Port of LA | 35 | 8 | 2017-2024 |
| Port of Salalah | 18 | 7 | 2017-2023 |
| Port of Tanjung priok | 28 | 8 | 2017-2024 |
| Port of hongkong | 29 | 8 | 2017-2024 |

## Directory Structure

```
google_earth/
  ├── Port_of_LA/
  │   ├── 2017/  (4 images)
  │   ├── 2018/  (3 images)
  │   ├── 2019/  (3 images)
  │   ├── 2020/  (6 images)
  │   ├── 2021/  (4 images)
  │   ├── 2022/  (5 images)
  │   ├── 2023/  (3 images)
  │   ├── 2024/  (5 images)
  │   └── README.md
  ├── Port_of_Salalah/
  │   ├── 2017/  (2 images)
  │   ├── 2018/  (5 images)
  │   ├── 2019/  (2 images)
  │   ├── 2020/  (4 images)
  │   ├── 2021/  (2 images)
  │   ├── 2022/  (2 images)
  │   ├── 2023/  (1 images)
  │   └── README.md
  ├── Port_of_Tanjung_priok/
  │   ├── 2017/  (4 images)
  │   ├── 2018/  (2 images)
  │   ├── 2019/  (7 images)
  │   ├── 2020/  (6 images)
  │   ├── 2021/  (3 images)
  │   ├── 2022/  (2 images)
  │   ├── 2023/  (2 images)
  │   ├── 2024/  (2 images)
  │   └── README.md
  ├── Port_of_hongkong/
  │   ├── 2017/  (5 images)
  │   ├── 2018/  (4 images)
  │   ├── 2019/  (5 images)
  │   ├── 2020/  (6 images)
  │   ├── 2021/  (3 images)
  │   ├── 2022/  (3 images)
  │   ├── 2023/  (1 images)
  │   ├── 2024/  (2 images)
  │   └── README.md
  └── README.md (this file)
```

## Processing Pipeline

### 1. Run Object Detection

Process all locations:

```bash
cd notebooks

# Port of LA
python detect_and_save_structured.py --dataset google_earth --location Port_of_LA

# Port of Salalah
python detect_and_save_structured.py --dataset google_earth --location Port_of_Salalah

# Port of Tanjung priok
python detect_and_save_structured.py --dataset google_earth --location Port_of_Tanjung_priok

# Port of hongkong
python detect_and_save_structured.py --dataset google_earth --location Port_of_hongkong

```

### 2. View Results

Results are saved in structured format:
```
results/annotations/google_earth/<location>/<year>/
  ├── annotated/       # Images with bounding boxes
  ├── detections.csv   # Detection data
  └── summary.txt      # Statistics
```

## Data Quality

- **Source**: Google Earth Engine
- **Format**: JPEG (RGB)
- **Coverage**: Port areas with ships, containers, cranes
- **Temporal**: Annual snapshots (2017-2024)

## Use Cases

1. **Ship Detection**: Count vessels in port
2. **Activity Monitoring**: Track port utilization over time
3. **Economic Forecasting**: Correlate port activity with trade data
4. **Comparative Analysis**: Compare activity across global ports
