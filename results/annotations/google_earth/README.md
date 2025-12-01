# Google Earth Annotations

High-resolution Google Earth imagery with object detection annotations for port activity monitoring and economic forecasting.

## Statistics

- **Total Locations**: 4
- **Total Images**: 140
- **Total Detections**: 1562
- **Last Updated**: 2025-11-30 21:18:30

## Locations

- **Port_of_hongkong**: 8 years (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024)
- **Port_of_LA**: 8 years (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024)
- **Port_of_Salalah**: 7 years (2017, 2018, 2019, 2020, 2021, 2022, 2023)
- **Port_of_Tanjung_priok**: 8 years (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024)

## Directory Structure

```
google_earth/
  ├── <location_name>/
  │   ├── <year>/
  │   │   ├── annotated/          # Annotated images with bounding boxes
  │   │   ├── detections.csv      # Detection data in CSV format
  │   │   ├── detections.json     # Detailed detection data
  │   │   └── summary.txt         # Human-readable summary
  │   └── all_years_summary.csv   # Aggregated data across all years
  └── README.md                    # This file
```
