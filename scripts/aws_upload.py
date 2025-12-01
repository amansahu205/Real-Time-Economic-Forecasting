#!/usr/bin/env python3
"""
AWS S3 Upload Script

Uploads project data to AWS S3 buckets.

Prerequisites:
    1. AWS CLI configured: aws configure
    2. Proper IAM permissions for S3

Usage:
    python scripts/aws_upload.py --check      # Check AWS connection
    python scripts/aws_upload.py --create     # Create S3 buckets
    python scripts/aws_upload.py --upload     # Upload all data
    python scripts/aws_upload.py --all        # Create buckets and upload
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    DATA_DIR, MODELS_DIR, RESULTS_DIR, FEATURES_DIR,
    AWS_CONFIG
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def run_aws_command(cmd: list, check: bool = True) -> tuple:
    """Run AWS CLI command."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check
        )
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, "", str(e)
    except FileNotFoundError:
        return False, "", "AWS CLI not found. Install with: pip install awscli"


def check_aws_connection():
    """Check if AWS CLI is configured and working."""
    logger.info("🔍 Checking AWS connection...")
    
    success, stdout, stderr = run_aws_command(['aws', 'sts', 'get-caller-identity'], check=False)
    
    if success:
        logger.info("✅ AWS connection successful")
        logger.info(f"   {stdout.strip()}")
        return True
    else:
        logger.error("❌ AWS connection failed")
        logger.error(f"   {stderr}")
        logger.info("\n📋 To configure AWS CLI:")
        logger.info("   1. Run: aws configure")
        logger.info("   2. Enter your Access Key ID")
        logger.info("   3. Enter your Secret Access Key")
        logger.info("   4. Enter region: us-east-1")
        return False


def create_s3_buckets():
    """Create S3 buckets for the project."""
    logger.info("\n📦 Creating S3 buckets...")
    
    buckets = [
        AWS_CONFIG['raw_bucket'],
        AWS_CONFIG['models_bucket'],
        AWS_CONFIG['processed_bucket']
    ]
    
    region = AWS_CONFIG['region']
    
    for bucket in buckets:
        logger.info(f"   Creating {bucket}...")
        
        # Check if bucket exists
        success, _, _ = run_aws_command(
            ['aws', 's3api', 'head-bucket', '--bucket', bucket],
            check=False
        )
        
        if success:
            logger.info(f"   ✅ {bucket} already exists")
            continue
        
        # Create bucket
        if region == 'us-east-1':
            cmd = ['aws', 's3', 'mb', f's3://{bucket}']
        else:
            cmd = ['aws', 's3', 'mb', f's3://{bucket}', '--region', region]
        
        success, stdout, stderr = run_aws_command(cmd, check=False)
        
        if success:
            logger.info(f"   ✅ Created {bucket}")
        else:
            logger.error(f"   ❌ Failed to create {bucket}: {stderr}")


def upload_satellite_data():
    """Upload satellite images to S3."""
    logger.info("\n🛰️ Uploading satellite data...")
    
    source = DATA_DIR / "raw" / "satellite" / "google_earth"
    dest = f"s3://{AWS_CONFIG['raw_bucket']}/satellite/google_earth/"
    
    if not source.exists():
        logger.warning(f"   ⚠️ Source not found: {source}")
        return
    
    cmd = [
        'aws', 's3', 'sync',
        str(source), dest,
        '--exclude', '*.git/*',
        '--exclude', '*.DS_Store'
    ]
    
    success, stdout, stderr = run_aws_command(cmd, check=False)
    
    if success:
        logger.info(f"   ✅ Uploaded to {dest}")
    else:
        logger.error(f"   ❌ Upload failed: {stderr}")


def upload_models():
    """Upload trained models to S3."""
    logger.info("\n🤖 Uploading models...")
    
    source = MODELS_DIR / "satellite"
    dest = f"s3://{AWS_CONFIG['models_bucket']}/yolo/"
    
    if not source.exists():
        logger.warning(f"   ⚠️ Source not found: {source}")
        return
    
    # Only upload best.pt weights (not all checkpoints)
    for model_dir in source.iterdir():
        if model_dir.is_dir():
            weights_file = model_dir / "weights" / "best.pt"
            if weights_file.exists():
                model_dest = f"{dest}{model_dir.name}/weights/best.pt"
                cmd = ['aws', 's3', 'cp', str(weights_file), model_dest]
                
                success, _, stderr = run_aws_command(cmd, check=False)
                if success:
                    logger.info(f"   ✅ Uploaded {model_dir.name}/weights/best.pt")
                else:
                    logger.error(f"   ❌ Failed: {stderr}")


def upload_ais_data():
    """Upload AIS data to S3."""
    logger.info("\n🚢 Uploading AIS data...")
    
    # Raw AIS data
    raw_source = DATA_DIR / "raw" / "ais" / "noaa"
    raw_dest = f"s3://{AWS_CONFIG['raw_bucket']}/ais/noaa/"
    
    if raw_source.exists():
        cmd = ['aws', 's3', 'sync', str(raw_source), raw_dest, '--exclude', '*.csv']
        success, _, stderr = run_aws_command(cmd, check=False)
        if success:
            logger.info(f"   ✅ Uploaded raw AIS to {raw_dest}")
    
    # Processed AIS data
    processed_source = DATA_DIR / "processed" / "ais"
    processed_dest = f"s3://{AWS_CONFIG['processed_bucket']}/ais/"
    
    if processed_source.exists():
        cmd = ['aws', 's3', 'sync', str(processed_source), processed_dest]
        success, _, stderr = run_aws_command(cmd, check=False)
        if success:
            logger.info(f"   ✅ Uploaded processed AIS to {processed_dest}")


def upload_features():
    """Upload extracted features to S3."""
    logger.info("\n📊 Uploading features...")
    
    source = FEATURES_DIR
    dest = f"s3://{AWS_CONFIG['processed_bucket']}/features/"
    
    if not source.exists():
        logger.warning(f"   ⚠️ Source not found: {source}")
        return
    
    cmd = ['aws', 's3', 'sync', str(source), dest]
    success, _, stderr = run_aws_command(cmd, check=False)
    
    if success:
        logger.info(f"   ✅ Uploaded to {dest}")
    else:
        logger.error(f"   ❌ Upload failed: {stderr}")


def upload_results():
    """Upload detection results to S3."""
    logger.info("\n📈 Uploading results...")
    
    source = RESULTS_DIR / "annotations"
    dest = f"s3://{AWS_CONFIG['processed_bucket']}/annotations/"
    
    if not source.exists():
        logger.warning(f"   ⚠️ Source not found: {source}")
        return
    
    cmd = [
        'aws', 's3', 'sync',
        str(source), dest,
        '--exclude', '*.png',
        '--exclude', '*.jpg'
    ]
    
    success, _, stderr = run_aws_command(cmd, check=False)
    
    if success:
        logger.info(f"   ✅ Uploaded to {dest}")
    else:
        logger.error(f"   ❌ Upload failed: {stderr}")


def upload_all():
    """Upload all data to S3."""
    logger.info("="*60)
    logger.info("🚀 UPLOADING ALL DATA TO AWS S3")
    logger.info("="*60)
    
    upload_satellite_data()
    upload_models()
    upload_ais_data()
    upload_features()
    upload_results()
    
    logger.info("\n" + "="*60)
    logger.info("✅ UPLOAD COMPLETE")
    logger.info("="*60)
    
    # Show bucket contents
    logger.info("\n📋 Bucket Contents:")
    for bucket in [AWS_CONFIG['raw_bucket'], AWS_CONFIG['models_bucket'], AWS_CONFIG['processed_bucket']]:
        logger.info(f"\n{bucket}:")
        run_aws_command(['aws', 's3', 'ls', f's3://{bucket}/', '--summarize'], check=False)


def main():
    parser = argparse.ArgumentParser(description='AWS S3 Upload Script')
    parser.add_argument('--check', action='store_true', help='Check AWS connection')
    parser.add_argument('--create', action='store_true', help='Create S3 buckets')
    parser.add_argument('--upload', action='store_true', help='Upload all data')
    parser.add_argument('--all', action='store_true', help='Create buckets and upload')
    
    args = parser.parse_args()
    
    if args.check:
        check_aws_connection()
    elif args.create:
        if check_aws_connection():
            create_s3_buckets()
    elif args.upload:
        if check_aws_connection():
            upload_all()
    elif args.all:
        if check_aws_connection():
            create_s3_buckets()
            upload_all()
    else:
        # Default: show help
        parser.print_help()


if __name__ == "__main__":
    main()
