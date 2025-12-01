# Port Of La

Google Earth high-resolution satellite imagery.

## Overview

- **Total Images**: 35
- **Years Covered**: 2017 - 2024
- **Number of Years**: 8

## Images by Year

| Year | Images |
|------|--------|
| 2017 | 4 |
| 2018 | 3 |
| 2019 | 3 |
| 2020 | 6 |
| 2021 | 4 |
| 2022 | 5 |
| 2023 | 3 |
| 2024 | 5 |

## Directory Structure

```
Port_of_LA/
  ├── 2017/
  │   └── 4 images (2017-1.jpg to 2017-4.jpg)
  ├── 2018/
  │   └── 3 images (2018-1.jpg to 2018-3.jpg)
  ├── 2019/
  │   └── 3 images (2019-1.jpg to 2019-3.jpg)
  ├── 2020/
  │   └── 6 images (2020-1.jpg to 2020-6.jpg)
  ├── 2021/
  │   └── 4 images (2021-1.jpg to 2021-4.jpg)
  ├── 2022/
  │   └── 5 images (2022-1.jpg to 2022-5.jpg)
  ├── 2023/
  │   └── 3 images (2023-1.jpg to 2023-3.jpg)
  ├── 2024/
  │   └── 5 images (2024-1.jpg to 2024-5.jpg)
  └── README.md
```

## Usage

Process these images with object detection:

```bash
python detect_and_save_structured.py \
    --dataset google_earth \
    --location Port_of_LA \
    --model-type ports
```
