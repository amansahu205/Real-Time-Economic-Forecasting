# Google Earth Tiled Annotations

Tiled detection with overlap (improved method)

## Statistics

- **Total Locations**: 4
- **Total Images**: 108
- **Total Detections**: 6078
- **Last Updated**: 2025-11-30 21:38:34

## Locations

- **Port_of_hongkong**: 8 years (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024)
- **Port_of_LA**: 8 years (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024)
- **Port_of_Salalah**: 7 years (2017, 2018, 2019, 2020, 2021, 2022, 2023)
- **Port_of_Tanjung_priok**: 8 years (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024)

## Directory Structure

```
google_earth_tiled/
  ├── <location_name>/
  │   ├── <year>/
  │   │   ├── annotated/          # Annotated images with bounding boxes
  │   │   ├── detections.csv      # Detection data in CSV format
  │   │   ├── detections.json     # Detailed detection data
  │   │   └── summary.txt         # Human-readable summary
  │   └── all_years_summary.csv   # Aggregated data across all years
  └── README.md                    # This file
```
