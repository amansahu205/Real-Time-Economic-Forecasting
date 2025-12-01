# Port Of Salalah

Google Earth high-resolution satellite imagery.

## Overview

- **Total Images**: 18
- **Years Covered**: 2017 - 2023
- **Number of Years**: 7

## Images by Year

| Year | Images |
|------|--------|
| 2017 | 2 |
| 2018 | 5 |
| 2019 | 2 |
| 2020 | 4 |
| 2021 | 2 |
| 2022 | 2 |
| 2023 | 1 |

## Directory Structure

```
Port_of_Salalah/
  ├── 2017/
  │   └── 2 images (2017-1.jpg to 2017-2.jpg)
  ├── 2018/
  │   └── 5 images (2018-1.jpg to 2018-5.jpg)
  ├── 2019/
  │   └── 2 images (2019-1.jpg to 2019-2.jpg)
  ├── 2020/
  │   └── 4 images (2020-1.jpg to 2020-4.jpg)
  ├── 2021/
  │   └── 2 images (2021-1.jpg to 2021-2.jpg)
  ├── 2022/
  │   └── 2 images (2022-1.jpg to 2022-2.jpg)
  ├── 2023/
  │   └── 1 images (2023-1.jpg to 2023-1.jpg)
  └── README.md
```

## Usage

Process these images with object detection:

```bash
python detect_and_save_structured.py \
    --dataset google_earth \
    --location Port_of_Salalah \
    --model-type ports
```
