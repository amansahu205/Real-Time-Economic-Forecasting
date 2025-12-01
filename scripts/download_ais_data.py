#!/usr/bin/env python3
"""
Download AIS (Automatic Identification System) Maritime Data

Downloads NOAA AIS data for Port of LA region (Zone 18) from 2017-2024.
This provides maritime GPS tracking data for economic forecasting.

Usage:
    python download_ais_data.py --years 2017-2024
    python download_ais_data.py --year 2023 --months 1-6
"""

import requests
import argparse
from pathlib import Path
from tqdm import tqdm
import time

PROJECT_ROOT = Path(__file__).parent.parent
AIS_DIR = PROJECT_ROOT / "data" / "raw" / "ais" / "noaa"


def download_noaa_ais(year, month, zone=18, retry=3):
    """
    Download NOAA AIS data for specific month/year
    
    Args:
        year: 2017-2024
        month: 1-12
        zone: 18 for Port of LA (West Coast)
        retry: Number of retry attempts
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Create directory
    output_dir = AIS_DIR / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct URL
    url = f"https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_Zone{zone}.zip"
    
    # Output file
    output_file = output_dir / f"AIS_{year}_{month:02d}_Zone{zone}.zip"
    
    # Check if already downloaded
    if output_file.exists():
        file_size = output_file.stat().st_size / (1024 * 1024)  # MB
        if file_size > 1:  # At least 1MB
            print(f"   ✅ Already downloaded: {output_file.name} ({file_size:.1f} MB)")
            return True
        else:
            print(f"   ⚠️  Incomplete file, re-downloading: {output_file.name}")
            output_file.unlink()
    
    # Download with retry
    for attempt in range(retry):
        try:
            print(f"   📥 Downloading: {output_file.name} (attempt {attempt + 1}/{retry})")
            
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(output_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"      ") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Verify download
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
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
    
    print(f"   ❌ Failed after {retry} attempts: {output_file.name}")
    return False


def download_year(year, months=None, zone=18):
    """
    Download AIS data for entire year or specific months
    
    Args:
        year: Year to download
        months: List of months (1-12), or None for all months
        zone: AIS zone (18 for Port of LA)
    """
    if months is None:
        months = range(1, 13)
    
    print(f"\n{'='*60}")
    print(f"📅 DOWNLOADING AIS DATA FOR {year}")
    print(f"{'='*60}")
    print(f"Zone: {zone} (Port of LA region)")
    print(f"Months: {list(months)}")
    
    success_count = 0
    fail_count = 0
    
    for month in months:
        success = download_noaa_ais(year, month, zone)
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Downloaded: {success_count} files")
    if fail_count > 0:
        print(f"❌ Failed: {fail_count} files")
    print(f"{'='*60}")


def download_all_years(start_year=2017, end_year=2024, zone=18):
    """
    Download AIS data for multiple years
    
    Args:
        start_year: First year to download
        end_year: Last year to download (inclusive)
        zone: AIS zone
    """
    print("="*60)
    print("🚢 NOAA AIS DATA DOWNLOADER")
    print("="*60)
    print(f"\nTarget: Port of LA (Zone {zone})")
    print(f"Years: {start_year}-{end_year}")
    print(f"Output: {AIS_DIR}")
    
    years = range(start_year, end_year + 1)
    total_files = len(years) * 12
    
    print(f"\nTotal files to download: {total_files}")
    print(f"Estimated size: ~{total_files * 0.3:.1f} GB")
    print(f"Estimated time: ~{total_files * 2:.0f} minutes")
    
    input("\nPress Enter to start download...")
    
    start_time = time.time()
    
    for year in years:
        download_year(year, zone=zone)
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("✅ DOWNLOAD COMPLETE!")
    print(f"{'='*60}")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Data location: {AIS_DIR}")
    
    # Count downloaded files
    downloaded = len(list(AIS_DIR.glob("**/*.zip")))
    print(f"Total files downloaded: {downloaded}/{total_files}")


def main():
    parser = argparse.ArgumentParser(description='Download NOAA AIS maritime data')
    parser.add_argument('--year', type=int, help='Single year to download')
    parser.add_argument('--years', type=str, help='Year range (e.g., 2017-2024)')
    parser.add_argument('--months', type=str, help='Month range (e.g., 1-6)')
    parser.add_argument('--zone', type=int, default=18, help='AIS zone (default: 18 for Port of LA)')
    
    args = parser.parse_args()
    
    # Parse arguments
    if args.year:
        # Single year
        months = None
        if args.months:
            start, end = map(int, args.months.split('-'))
            months = range(start, end + 1)
        download_year(args.year, months, args.zone)
        
    elif args.years:
        # Year range
        start, end = map(int, args.years.split('-'))
        download_all_years(start, end, args.zone)
        
    else:
        # Default: all years
        download_all_years(2017, 2024, args.zone)


if __name__ == "__main__":
    main()
