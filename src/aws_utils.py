"""
AWS Utilities for SageMaker Notebooks

Provides seamless switching between local and S3 paths.
Automatically detects if running in SageMaker or locally.

Usage:
    from src.aws_utils import get_data_paths, is_sagemaker
    
    paths = get_data_paths()
    satellite_dir = paths['satellite']
    models_dir = paths['models']
"""

import os
from pathlib import Path

# S3 Bucket Configuration
S3_BUCKETS = {
    'raw': 'economic-forecast-raw',
    'models': 'economic-forecast-models', 
    'processed': 'economic-forecast-processed'
}

# S3 Paths
S3_PATHS = {
    'satellite': f"s3://{S3_BUCKETS['raw']}/satellite/google_earth",
    'sentinel': f"s3://{S3_BUCKETS['raw']}/satellite/sentinel-2-l2a",
    'ais_raw': f"s3://{S3_BUCKETS['raw']}/ais",
    'models': f"s3://{S3_BUCKETS['models']}/yolo",
    'processed_ais': f"s3://{S3_BUCKETS['processed']}/ais",
    'features': f"s3://{S3_BUCKETS['processed']}/features",
    'annotations': f"s3://{S3_BUCKETS['processed']}/annotations",
}


def is_sagemaker():
    """Check if running in SageMaker environment."""
    # SageMaker sets these environment variables
    sagemaker_indicators = [
        'SM_CHANNEL_TRAINING',
        'SM_MODEL_DIR', 
        'SM_OUTPUT_DATA_DIR',
        'SAGEMAKER_INTERNAL_IMAGE_URI'
    ]
    
    # Also check for typical SageMaker paths
    sagemaker_paths = [
        '/home/ec2-user/SageMaker',
        '/opt/ml'
    ]
    
    # Check environment variables
    for var in sagemaker_indicators:
        if os.environ.get(var):
            return True
    
    # Check paths
    for path in sagemaker_paths:
        if os.path.exists(path):
            return True
    
    return False


def get_project_root():
    """Get project root directory."""
    if is_sagemaker():
        # In SageMaker, repo is cloned to /home/ec2-user/SageMaker/
        sagemaker_root = Path('/home/ec2-user/SageMaker/Real-Time-Economic-Forecasting')
        if sagemaker_root.exists():
            return sagemaker_root
        # Fallback to current directory
        return Path.cwd()
    else:
        # Local development
        return Path(__file__).parent.parent


def get_data_paths(use_s3=None):
    """
    Get data paths - automatically uses S3 in SageMaker, local otherwise.
    
    Args:
        use_s3: Force S3 (True) or local (False). None = auto-detect.
    
    Returns:
        dict with paths for satellite, models, ais, etc.
    """
    if use_s3 is None:
        use_s3 = is_sagemaker()
    
    if use_s3:
        return {
            'satellite': S3_PATHS['satellite'],
            'sentinel': S3_PATHS['sentinel'],
            'ais_raw': S3_PATHS['ais_raw'],
            'models': S3_PATHS['models'],
            'processed_ais': S3_PATHS['processed_ais'],
            'features': S3_PATHS['features'],
            'annotations': S3_PATHS['annotations'],
            'is_s3': True
        }
    else:
        project_root = get_project_root()
        return {
            'satellite': project_root / 'data' / 'raw' / 'satellite' / 'google_earth',
            'sentinel': project_root / 'data' / 'raw' / 'satellite' / 'sentinel-2-l2a',
            'ais_raw': project_root / 'data' / 'raw' / 'ais',
            'models': project_root / 'data' / 'models' / 'satellite',
            'processed_ais': project_root / 'data' / 'processed' / 'ais',
            'features': project_root / 'data' / 'features',
            'annotations': project_root / 'results' / 'annotations',
            'is_s3': False
        }


def download_from_s3(s3_path, local_path):
    """Download file from S3 to local path."""
    import boto3
    
    # Parse S3 path
    s3_path = s3_path.replace('s3://', '')
    bucket = s3_path.split('/')[0]
    key = '/'.join(s3_path.split('/')[1:])
    
    # Download
    s3 = boto3.client('s3')
    local_path = Path(local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    
    s3.download_file(bucket, key, str(local_path))
    return local_path


def upload_to_s3(local_path, s3_path):
    """Upload file from local to S3."""
    import boto3
    
    # Parse S3 path
    s3_path = s3_path.replace('s3://', '')
    bucket = s3_path.split('/')[0]
    key = '/'.join(s3_path.split('/')[1:])
    
    # Upload
    s3 = boto3.client('s3')
    s3.upload_file(str(local_path), bucket, key)
    return f"s3://{bucket}/{key}"


def list_s3_files(s3_path, pattern=None):
    """List files in S3 path."""
    import boto3
    
    # Parse S3 path
    s3_path = s3_path.replace('s3://', '')
    bucket = s3_path.split('/')[0]
    prefix = '/'.join(s3_path.split('/')[1:])
    if not prefix.endswith('/'):
        prefix += '/'
    
    # List objects
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    
    files = []
    for obj in response.get('Contents', []):
        key = obj['Key']
        if pattern is None or pattern in key:
            files.append(f"s3://{bucket}/{key}")
    
    return files


def read_csv_from_s3(s3_path):
    """Read CSV directly from S3."""
    import pandas as pd
    return pd.read_csv(s3_path)


def read_parquet_from_s3(s3_path):
    """Read Parquet directly from S3."""
    import pandas as pd
    return pd.read_parquet(s3_path)


# Print environment info when imported
if __name__ == "__main__":
    print(f"Running in SageMaker: {is_sagemaker()}")
    print(f"Project root: {get_project_root()}")
    print(f"\nData paths:")
    for key, value in get_data_paths().items():
        print(f"  {key}: {value}")
