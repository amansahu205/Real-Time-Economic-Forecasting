#!/usr/bin/env python3
"""
Process AIS (Automatic Identification System) Maritime Data

Extracts ship positions from NOAA AIS data, filters to Port of LA region,
and aggregates to daily metrics for economic forecasting.

Usage:
    python process_ais_data.py
    python process_ais_data.py --year 2023
"""

import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
AIS_DIR = PROJECT_ROOT / "data" / "raw" / "ais" / "noaa"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed" / "ais"

# Port of LA bounding box (expanded to capture approach areas)
PORT_LA_BOUNDS = {
    'min_lat': 33.65,
    'max_lat': 33.85,
    'min_lon': -118.35,
    'max_lon': -118.15
}

# AIS Vessel Type Codes
VESSEL_TYPES = {
    'cargo': list(range(70, 80)),      # Cargo vessels
    'tanker': list(range(80, 90)),     # Tankers
    'passenger': list(range(60, 70)),  # Passenger vessels
    'fishing': [30],                   # Fishing vessels
    'tug': [31, 32, 52],              # Tugs and pilot vessels
}


def extract_zip(zip_path):
    """Extract ZIP file and return CSV path"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get CSV filename
            csv_name = [f for f in zip_ref.namelist() if f.endswith('.csv')][0]
            
            # Extract to same directory
            zip_ref.extract(csv_name, zip_path.parent)
            
            csv_path = zip_path.parent / csv_name
            return csv_path
    except Exception as e:
        print(f"   ❌ Error extracting {zip_path.name}: {e}")
        return None


def filter_port_area(df, bounds):
    """Filter AIS data to port area"""
    mask = (
        (df['LAT'] >= bounds['min_lat']) &
        (df['LAT'] <= bounds['max_lat']) &
        (df['LON'] >= bounds['min_lon']) &
        (df['LON'] <= bounds['max_lon'])
    )
    return df[mask]


def classify_vessel(vessel_type_code):
    """Classify vessel type for economic analysis"""
    try:
        code = int(vessel_type_code)
        for category, codes in VESSEL_TYPES.items():
            if code in codes:
                return category
        return 'other'
    except:
        return 'unknown'


def process_ais_file(zip_path, bounds=PORT_LA_BOUNDS):
    """
    Process single AIS ZIP file
    
    Returns:
        DataFrame with filtered and processed AIS data
    """
    # Extract ZIP
    csv_path = extract_zip(zip_path)
    if csv_path is None:
        return None
    
    try:
        # Read CSV in chunks (files can be large)
        chunks = []
        chunk_size = 100000
        
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, low_memory=False):
            # Filter to port area
            filtered = filter_port_area(chunk, bounds)
            
            if len(filtered) > 0:
                chunks.append(filtered)
        
        if not chunks:
            # Clean up
            csv_path.unlink()
            return None
        
        # Combine chunks
        df = pd.concat(chunks, ignore_index=True)
        
        # Parse datetime
        df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')
        
        # Remove invalid timestamps
        df = df.dropna(subset=['BaseDateTime'])
        
        # Classify vessels
        df['vessel_category'] = df['VesselType'].apply(classify_vessel)
        
        # Remove duplicates (same ship, same minute)
        df['minute'] = df['BaseDateTime'].dt.floor('T')
        df = df.drop_duplicates(subset=['MMSI', 'minute'])
        df = df.drop(columns=['minute'])
        
        # Clean up extracted CSV
        csv_path.unlink()
        
        return df
        
    except Exception as e:
        print(f"   ❌ Error processing {zip_path.name}: {e}")
        if csv_path and csv_path.exists():
            csv_path.unlink()
        return None


def aggregate_daily_metrics(df):
    """
    Aggregate AIS data to daily metrics
    
    Returns:
        DataFrame with daily port activity metrics
    """
    df['date'] = df['BaseDateTime'].dt.date
    
    # Daily aggregations
    daily = df.groupby('date').agg({
        'MMSI': 'nunique',  # Unique ships per day
    }).rename(columns={'MMSI': 'unique_ships'})
    
    # Count by vessel category
    for category in ['cargo', 'tanker', 'passenger', 'fishing', 'tug']:
        daily[f'{category}_ships'] = df[df['vessel_category'] == category].groupby(
            df['BaseDateTime'].dt.date
        )['MMSI'].nunique()
    
    # Fill NaN with 0
    daily = daily.fillna(0).astype(int)
    
    return daily.reset_index()


def calculate_economic_features(daily_df):
    """
    Calculate economic indicators from daily AIS data
    
    Returns:
        DataFrame with economic features
    """
    df = daily_df.copy()
    
    # Ensure date is datetime
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 1. Port Activity Index (weighted by economic importance)
    df['port_activity_index'] = (
        df['cargo_ships'] * 0.4 +
        df['tanker_ships'] * 0.3 +
        df['unique_ships'] * 0.2 +
        df['tug_ships'] * 0.1
    )
    
    # 2. Moving Averages (7-day and 30-day)
    df['ships_ma7'] = df['unique_ships'].rolling(7, min_periods=1).mean()
    df['ships_ma30'] = df['unique_ships'].rolling(30, min_periods=1).mean()
    df['cargo_ma7'] = df['cargo_ships'].rolling(7, min_periods=1).mean()
    df['cargo_ma30'] = df['cargo_ships'].rolling(30, min_periods=1).mean()
    
    # 3. Growth Rates (week-over-week, month-over-month)
    df['ships_wow_growth'] = df['unique_ships'].pct_change(7) * 100
    df['ships_mom_growth'] = df['unique_ships'].pct_change(30) * 100
    df['cargo_wow_growth'] = df['cargo_ships'].pct_change(7) * 100
    
    # 4. Volatility (7-day standard deviation)
    df['ships_volatility'] = df['unique_ships'].rolling(7, min_periods=1).std()
    
    # 5. Cargo Ratio (cargo ships / total ships)
    df['cargo_ratio'] = df['cargo_ships'] / df['unique_ships'].replace(0, np.nan)
    df['cargo_ratio'] = df['cargo_ratio'].fillna(0)
    
    # 6. Activity Level (compared to 30-day average)
    df['activity_level'] = df['unique_ships'] / df['ships_ma30'].replace(0, np.nan)
    df['activity_level'] = df['activity_level'].fillna(1)
    
    return df


def process_all_ais_data(year=None):
    """
    Process all downloaded AIS data
    
    Args:
        year: Process specific year only, or None for all years
    """
    print("="*60)
    print("🚢 PROCESSING AIS DATA FOR PORT OF LA")
    print("="*60)
    
    # Find all ZIP files
    if year:
        zip_files = sorted(AIS_DIR.glob(f"{year}/*.zip"))
        print(f"\nProcessing year: {year}")
    else:
        zip_files = sorted(AIS_DIR.glob("**/*.zip"))
        print(f"\nProcessing all years")
    
    print(f"Found {len(zip_files)} files to process")
    
    if len(zip_files) == 0:
        print("\n❌ No AIS data found!")
        print(f"   Please run: python scripts/download_ais_data.py")
        return None, None
    
    print(f"\nPort Area: {PORT_LA_BOUNDS}")
    print()
    
    all_data = []
    
    # Process each file
    for zip_file in tqdm(zip_files, desc="Processing files"):
        df = process_ais_file(zip_file)
        if df is not None and len(df) > 0:
            all_data.append(df)
            tqdm.write(f"   ✅ {zip_file.name}: {len(df):,} positions, {df['MMSI'].nunique()} ships")
    
    if not all_data:
        print("\n❌ No data found in port area!")
        return None, None
    
    # Combine all data
    print("\n📊 Combining all data...")
    full_data = pd.concat(all_data, ignore_index=True)
    
    print(f"\n{'='*60}")
    print("📊 RAW DATA SUMMARY")
    print(f"{'='*60}")
    print(f"Total AIS records: {len(full_data):,}")
    print(f"Date range: {full_data['BaseDateTime'].min()} to {full_data['BaseDateTime'].max()}")
    print(f"Unique ships: {full_data['MMSI'].nunique():,}")
    print(f"\nVessel breakdown:")
    print(full_data['vessel_category'].value_counts())
    
    # Save raw data
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DIR / "Port_of_LA_ais_raw.parquet"
    full_data.to_parquet(output_file)
    print(f"\n✅ Saved raw data: {output_file}")
    
    # Aggregate to daily metrics
    print("\n📊 Aggregating to daily metrics...")
    daily_metrics = aggregate_daily_metrics(full_data)
    
    print(f"\n{'='*60}")
    print("📊 DAILY METRICS SUMMARY")
    print(f"{'='*60}")
    print(f"Total days: {len(daily_metrics)}")
    print(f"Date range: {daily_metrics['date'].min()} to {daily_metrics['date'].max()}")
    print(f"\nDaily statistics:")
    print(daily_metrics[['unique_ships', 'cargo_ships', 'tanker_ships']].describe())
    
    # Save daily metrics
    output_file = PROCESSED_DIR / "Port_of_LA_ais_daily.csv"
    daily_metrics.to_csv(output_file, index=False)
    print(f"\n✅ Saved daily metrics: {output_file}")
    
    # Calculate economic features
    print("\n📊 Calculating economic features...")
    features = calculate_economic_features(daily_metrics)
    
    # Save features
    output_file = PROCESSED_DIR / "Port_of_LA_ais_features.csv"
    features.to_csv(output_file, index=False)
    print(f"✅ Saved economic features: {output_file}")
    
    print(f"\n{'='*60}")
    print("✅ PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nOutput files:")
    print(f"  - Raw data: data/processed/ais/Port_of_LA_ais_raw.parquet")
    print(f"  - Daily metrics: data/processed/ais/Port_of_LA_ais_daily.csv")
    print(f"  - Economic features: data/processed/ais/Port_of_LA_ais_features.csv")
    
    return full_data, features


def main():
    parser = argparse.ArgumentParser(description='Process NOAA AIS maritime data')
    parser.add_argument('--year', type=int, help='Process specific year only')
    
    args = parser.parse_args()
    
    full_data, features = process_all_ais_data(year=args.year)
    
    if features is not None:
        print("\n📊 Sample economic features:")
        print(features[['date', 'unique_ships', 'cargo_ships', 'port_activity_index', 
                       'ships_ma7', 'cargo_ratio']].tail(10))


if __name__ == "__main__":
    main()
