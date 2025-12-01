"""
AWS Lambda: AIS Data Processor
Processes AIS maritime data and extracts features.
"""

import json
import boto3
import pandas as pd
import io
import os
from datetime import datetime

s3 = boto3.client('s3')

# Configuration
RAW_BUCKET = os.environ.get('RAW_BUCKET', 'economic-forecast-raw')
PROCESSED_BUCKET = os.environ.get('PROCESSED_BUCKET', 'economic-forecast-processed')

# Port of LA bounding box
PORT_LA_BBOX = {
    'lat_min': 33.70,
    'lat_max': 33.78,
    'lon_min': -118.30,
    'lon_max': -118.15
}

# Vessel type mapping
VESSEL_TYPES = {
    'cargo': range(70, 80),
    'tanker': range(80, 90),
    'passenger': range(60, 70),
    'fishing': range(30, 38),
    'tug': range(50, 60)
}


def lambda_handler(event, context):
    """
    Process AIS data file.
    
    Input event:
    {
        "bucket": "economic-forecast-raw",
        "key": "ais/2024/01/ais_data.csv"
    }
    """
    
    print(f"🚢 AIS Processor started")
    print(f"Event: {json.dumps(event)}")
    
    # Get input parameters
    bucket = event.get('bucket', RAW_BUCKET)
    key = event.get('key', '')
    
    if not key:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'No key provided'})
        }
    
    try:
        # Download AIS data from S3
        print(f"📥 Downloading: s3://{bucket}/{key}")
        response = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(io.BytesIO(response['Body'].read()))
        
        print(f"   Loaded {len(df)} records")
        
        # Process AIS data
        features = process_ais_data(df)
        
        # Save processed features to S3
        output_key = f"features/ais/{key.split('/')[-1].replace('.csv', '_features.csv')}"
        save_to_s3(features, PROCESSED_BUCKET, output_key)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'AIS data processed successfully',
                'input_records': len(df),
                'output_key': f"s3://{PROCESSED_BUCKET}/{output_key}",
                'features': features.to_dict() if isinstance(features, pd.Series) else features
            })
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def process_ais_data(df):
    """Process AIS data and extract features."""
    
    print("🔄 Processing AIS data...")
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()
    
    # Identify lat/lon columns
    lat_col = next((c for c in df.columns if 'lat' in c), None)
    lon_col = next((c for c in df.columns if 'lon' in c), None)
    
    if not lat_col or not lon_col:
        print("⚠️ No lat/lon columns found")
        return pd.Series({'error': 'No coordinates found'})
    
    # Filter to Port of LA region
    df_port = df[
        (df[lat_col] >= PORT_LA_BBOX['lat_min']) &
        (df[lat_col] <= PORT_LA_BBOX['lat_max']) &
        (df[lon_col] >= PORT_LA_BBOX['lon_min']) &
        (df[lon_col] <= PORT_LA_BBOX['lon_max'])
    ].copy()
    
    print(f"   Filtered to Port of LA: {len(df_port)} records")
    
    # Extract features
    features = {}
    
    # Total unique vessels
    mmsi_col = next((c for c in df_port.columns if 'mmsi' in c), None)
    if mmsi_col:
        features['total_vessels'] = df_port[mmsi_col].nunique()
    
    # Vessel type breakdown
    vessel_type_col = next((c for c in df_port.columns if 'vesseltype' in c or 'vessel_type' in c), None)
    if vessel_type_col:
        for vtype, vrange in VESSEL_TYPES.items():
            count = df_port[df_port[vessel_type_col].isin(vrange)][mmsi_col].nunique() if mmsi_col else 0
            features[f'{vtype}_vessels'] = count
    
    # Speed statistics
    speed_col = next((c for c in df_port.columns if 'sog' in c or 'speed' in c), None)
    if speed_col:
        features['avg_speed'] = round(df_port[speed_col].mean(), 2)
        features['max_speed'] = round(df_port[speed_col].max(), 2)
        # Stationary vessels (speed < 0.5 knots)
        features['stationary_pct'] = round((df_port[speed_col] < 0.5).mean() * 100, 2)
    
    # Time-based features
    time_col = next((c for c in df_port.columns if 'time' in c or 'date' in c), None)
    if time_col:
        try:
            df_port['timestamp'] = pd.to_datetime(df_port[time_col])
            features['date_range_start'] = df_port['timestamp'].min().isoformat()
            features['date_range_end'] = df_port['timestamp'].max().isoformat()
        except:
            pass
    
    features['total_records'] = len(df_port)
    features['processed_at'] = datetime.utcnow().isoformat()
    
    print(f"   Extracted {len(features)} features")
    
    return pd.Series(features)


def save_to_s3(data, bucket, key):
    """Save data to S3."""
    
    if isinstance(data, pd.Series):
        data = data.to_frame().T
    
    csv_buffer = io.StringIO()
    data.to_csv(csv_buffer, index=False)
    
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=csv_buffer.getvalue(),
        ContentType='text/csv'
    )
    
    print(f"✅ Saved to s3://{bucket}/{key}")
