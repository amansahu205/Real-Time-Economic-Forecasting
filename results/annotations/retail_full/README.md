# Retail Full Annotations

Retail detection using full image method (Mall of America)

## Statistics

- **Total Locations**: 1
- **Total Images**: 19
- **Total Detections**: 122
- **Last Updated**: 2025-11-30 22:00:36

## Locations

- **Mall_of_america**: 8 years (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024)

## Directory Structure

```
retail_full/
  ├── <location_name>/
  │   ├── <year>/
  │   │   ├── annotated/          # Annotated images with bounding boxes
  │   │   ├── detections.csv      # Detection data in CSV format
  │   │   ├── detections.json     # Detailed detection data
  │   │   └── summary.txt         # Human-readable summary
  │   └── all_years_summary.csv   # Aggregated data across all years
  └── README.md                    # This file
```
