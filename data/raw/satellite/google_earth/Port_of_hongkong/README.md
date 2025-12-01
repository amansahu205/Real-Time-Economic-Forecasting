# Port Of Hongkong

Google Earth high-resolution satellite imagery.

## Overview

- **Total Images**: 29
- **Years Covered**: 2017 - 2024
- **Number of Years**: 8

## Images by Year

| Year | Images |
|------|--------|
| 2017 | 5 |
| 2018 | 4 |
| 2019 | 5 |
| 2020 | 6 |
| 2021 | 3 |
| 2022 | 3 |
| 2023 | 1 |
| 2024 | 2 |

## Directory Structure

```
Port_of_hongkong/
  ├── 2017/
  │   └── 5 images (2017-1.jpg to 2017-5.jpg)
  ├── 2018/
  │   └── 4 images (2018-1.jpg to 2018-4.jpg)
  ├── 2019/
  │   └── 5 images (2019-1.jpg to 2019-5.jpg)
  ├── 2020/
  │   └── 6 images (2020-1.jpg to 2020-6.jpg)
  ├── 2021/
  │   └── 3 images (2021-1.jpg to 2021-3.jpg)
  ├── 2022/
  │   └── 3 images (2022-1.jpg to 2022-3.jpg)
  ├── 2023/
  │   └── 1 images (2023-1.jpg to 2023-1.jpg)
  ├── 2024/
  │   └── 2 images (2024-1.jpg to 2024-2.jpg)
  └── README.md
```

## Usage

Process these images with object detection:

```bash
python detect_and_save_structured.py \
    --dataset google_earth \
    --location Port_of_hongkong \
    --model-type ports
```
