#!/usr/bin/env python3
"""
Download AIS Daily Data (2018-2024 format)

NOAA changed format in 2018:
- 2017: Monthly files by Zone
- 2018+: Daily files, ALL US data combined

Each daily file is ~500MB-1GB (all US), so download selectively!

Usage:
    python download_ais_daily.py --year 2024 --month 1 --days 1-7
    python download_ais_daily.py --year 2020 --month 3 --days 15  # COVID period
"""

import requests
import argparse
from pathlib import Path
from tqdm import tqdm
import time

PROJECT_ROOT = Path(__file__).parent.parent
AIS_DIR = PROJECT_ROOT / "data" / "raw" / "ais" / "noaa_daily"


def download_daily_ais(year, month, day, retry=3):
    """
    Download NOAA AIS daily data (2018+ format)
    
    Args:
        year: 2018-2024
        month: 1-12
        day: 1-31
        retry: Number of retry attempts
    
    Returns:
        bool: True if successful
    """
    # Create directory
    output_dir = AIS_DIR / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct URL (new format: daily, all US)
    url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.zip"
    
    # Output file
    output_file = output_dir / f"AIS_{year}_{month:02d}_{day:02d}.zip"
    
    # Check if already downloaded
    if output_file.exists():
        file_size = output_file.stat().st_size / (1024 * 1024)
        if file_size > 10:  # At least 10MB (daily files are large)
            print(f"   ✅ Already downloaded: {output_file.name} ({file_size:.1f} MB)")
            return True
    
    # Download with retry
    for attempt in range(retry):
        try:
            print(f"   📥 Downloading: {output_file.name} (attempt {attempt + 1}/{retry})")
            
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"      ") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            file_size = output_file.stat().st_size / (1024 * 1024)
            print(f"   ✅ Downloaded: {output_file.name} ({file_size:.1f} MB)")
            return True
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"   ⚠️  File not available: {output_file.name} (404)")
                return False
            else:
                print(f"   ❌ HTTP Error: {e}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        if attempt < retry - 1:
            print(f"   ⏳ Retrying in 5 seconds...")
            time.sleep(5)
    
    return False


def download_date_range(year, month, start_day, end_day):
    """Download a range of days."""
    print(f"\n{'='*60}")
    print(f"📅 DOWNLOADING AIS DATA: {year}-{month:02d}")
    print(f"{'='*60}")
    print(f"Days: {start_day} to {end_day}")
    print(f"Output: {AIS_DIR / str(year)}")
    print(f"\n⚠️  Note: Each file is ~500MB-1GB (all US data)")
    
    success = 0
    failed = 0
    
    for day in range(start_day, end_day + 1):
        if download_daily_ais(year, month, day):
            success += 1
        else:
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Downloaded: {success} files")
    if failed > 0:
        print(f"❌ Failed: {failed} files")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Download NOAA AIS daily data (2018+)')
    parser.add_argument('--year', type=int, required=True, help='Year (2018-2024)')
    parser.add_argument('--month', type=int, required=True, help='Month (1-12)')
    parser.add_argument('--days', type=str, required=True, help='Day or range (e.g., 1 or 1-7)')
    
    args = parser.parse_args()
    
    # Parse days
    if '-' in args.days:
        start, end = map(int, args.days.split('-'))
    else:
        start = end = int(args.days)
    
    download_date_range(args.year, args.month, start, end)


if __name__ == "__main__":
    main()
