"""
AWS Lambda: S3 Upload Trigger
Triggered when new files are uploaded to S3 buckets.
Sends notifications and can trigger Step Functions pipeline.
"""

import json
import boto3
import urllib.parse
import os
from datetime import datetime

# Initialize AWS clients
s3 = boto3.client('s3')
sns = boto3.client('sns')
stepfunctions = boto3.client('stepfunctions')

# Environment variables (set in Lambda configuration)
SNS_TOPIC_ARN = os.environ.get('SNS_TOPIC_ARN', '')
STEP_FUNCTION_ARN = os.environ.get('STEP_FUNCTION_ARN', '')


def lambda_handler(event, context):
    """
    Main handler for S3 trigger events.
    
    Triggered when:
    - New satellite image uploaded
    - New AIS data uploaded
    - New model uploaded
    """
    
    print(f"Event received: {json.dumps(event)}")
    
    # Extract S3 event details
    record = event['Records'][0]
    bucket = record['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(record['s3']['object']['key'])
    size = record['s3']['object'].get('size', 0)
    event_time = record['eventTime']
    
    print(f"📁 New file: s3://{bucket}/{key}")
    print(f"   Size: {size / 1024:.2f} KB")
    print(f"   Time: {event_time}")
    
    # Determine file type and action
    file_info = classify_file(bucket, key)
    
    # Log to CloudWatch
    log_event(bucket, key, file_info)
    
    # Send SNS notification
    if SNS_TOPIC_ARN:
        send_notification(bucket, key, file_info)
    
    # Trigger Step Functions if needed
    if file_info['trigger_pipeline'] and STEP_FUNCTION_ARN:
        trigger_pipeline(bucket, key, file_info)
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'S3 event processed successfully',
            'bucket': bucket,
            'key': key,
            'file_type': file_info['type'],
            'action': file_info['action']
        })
    }


def classify_file(bucket, key):
    """Classify uploaded file and determine action."""
    
    file_info = {
        'type': 'unknown',
        'action': 'logged',
        'trigger_pipeline': False,
        'message': ''
    }
    
    key_lower = key.lower()
    
    # Satellite imagery
    if 'satellite' in key_lower and key_lower.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
        file_info['type'] = 'satellite_image'
        file_info['action'] = 'queue_detection'
        file_info['trigger_pipeline'] = True
        file_info['message'] = f"🛰️ New satellite image uploaded: {key.split('/')[-1]}"
        
        # Extract location from path
        if 'port_of_la' in key_lower:
            file_info['location'] = 'Port of LA'
        elif 'mall' in key_lower:
            file_info['location'] = 'Mall of America'
        else:
            file_info['location'] = 'Unknown'
    
    # AIS maritime data
    elif 'ais' in key_lower and key_lower.endswith('.csv'):
        file_info['type'] = 'ais_data'
        file_info['action'] = 'process_ais'
        file_info['trigger_pipeline'] = True
        file_info['message'] = f"🚢 New AIS data uploaded: {key.split('/')[-1]}"
    
    # YOLO model weights
    elif key_lower.endswith('.pt'):
        file_info['type'] = 'model_weights'
        file_info['action'] = 'register_model'
        file_info['trigger_pipeline'] = False
        file_info['message'] = f"🤖 New model uploaded: {key.split('/')[-1]}"
    
    # Processed features
    elif 'features' in key_lower and key_lower.endswith('.csv'):
        file_info['type'] = 'features'
        file_info['action'] = 'update_catalog'
        file_info['trigger_pipeline'] = False
        file_info['message'] = f"📊 New features uploaded: {key.split('/')[-1]}"
    
    # Detection results
    elif 'detection' in key_lower or 'annotation' in key_lower:
        file_info['type'] = 'detection_results'
        file_info['action'] = 'update_catalog'
        file_info['trigger_pipeline'] = False
        file_info['message'] = f"🔍 Detection results uploaded: {key.split('/')[-1]}"
    
    else:
        file_info['message'] = f"📁 File uploaded: {key.split('/')[-1]}"
    
    return file_info


def log_event(bucket, key, file_info):
    """Log event details to CloudWatch."""
    
    log_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'bucket': bucket,
        'key': key,
        'file_type': file_info['type'],
        'action': file_info['action'],
        'trigger_pipeline': file_info['trigger_pipeline']
    }
    
    print(f"📝 Log: {json.dumps(log_data)}")


def send_notification(bucket, key, file_info):
    """Send SNS notification."""
    
    try:
        subject = f"Economic Forecast: {file_info['type'].replace('_', ' ').title()}"
        
        message = f"""
{file_info['message']}

Details:
- Bucket: {bucket}
- Key: {key}
- Type: {file_info['type']}
- Action: {file_info['action']}
- Pipeline Triggered: {file_info['trigger_pipeline']}
- Time: {datetime.utcnow().isoformat()}

---
Economic Forecasting System
        """
        
        response = sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Message=message.strip(),
            Subject=subject
        )
        
        print(f"✅ SNS notification sent: {response['MessageId']}")
        
    except Exception as e:
        print(f"❌ SNS error: {str(e)}")


def trigger_pipeline(bucket, key, file_info):
    """Trigger Step Functions pipeline."""
    
    try:
        # Prepare input for Step Functions
        pipeline_input = {
            'bucket': bucket,
            'key': key,
            'file_type': file_info['type'],
            'action': file_info['action'],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Generate unique execution name
        execution_name = f"{file_info['type']}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        
        response = stepfunctions.start_execution(
            stateMachineArn=STEP_FUNCTION_ARN,
            name=execution_name,
            input=json.dumps(pipeline_input)
        )
        
        print(f"✅ Pipeline triggered: {response['executionArn']}")
        
    except Exception as e:
        print(f"❌ Step Functions error: {str(e)}")
