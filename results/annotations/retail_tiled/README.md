# Retail Tiled Annotations

Retail detection using tiled method (Mall of America)

## Statistics

- **Total Locations**: 1
- **Total Images**: 19
- **Total Detections**: 1542
- **Last Updated**: 2025-11-30 22:00:36

## Locations

- **Mall_of_america**: 8 years (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024)

## Directory Structure

```
retail_tiled/
  ├── <location_name>/
  │   ├── <year>/
  │   │   ├── annotated/          # Annotated images with bounding boxes
  │   │   ├── detections.csv      # Detection data in CSV format
  │   │   ├── detections.json     # Detailed detection data
  │   │   └── summary.txt         # Human-readable summary
  │   └── all_years_summary.csv   # Aggregated data across all years
  └── README.md                    # This file
```
