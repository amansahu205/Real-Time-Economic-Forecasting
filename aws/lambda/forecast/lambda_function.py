"""
AWS Lambda: Economic Forecasting
Runs ML predictions using processed features.
"""

import json
import boto3
import pandas as pd
import numpy as np
import pickle
import io
import os
from datetime import datetime

s3 = boto3.client('s3')

# Configuration
MODELS_BUCKET = os.environ.get('MODELS_BUCKET', 'economic-forecast-models')
PROCESSED_BUCKET = os.environ.get('PROCESSED_BUCKET', 'economic-forecast-processed')


def lambda_handler(event, context):
    """
    Run economic forecasting.
    
    Input event:
    {
        "features_key": "features/fused/2024_features.csv",
        "model_type": "trade"  # or "retail"
    }
    """
    
    print(f"🔮 Forecasting Lambda started")
    print(f"Event: {json.dumps(event)}")
    
    features_key = event.get('features_key', '')
    model_type = event.get('model_type', 'trade')
    
    try:
        # Load features
        features = load_features(features_key)
        
        # Load model
        model = load_model(model_type)
        
        # Run prediction
        predictions = run_prediction(model, features, model_type)
        
        # Save results
        output_key = f"predictions/{model_type}/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        save_predictions(predictions, output_key)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Forecast completed successfully',
                'model_type': model_type,
                'predictions': predictions,
                'output_key': f"s3://{PROCESSED_BUCKET}/{output_key}"
            })
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def load_features(key):
    """Load features from S3."""
    
    if not key:
        # Use default latest features
        key = 'features/fused/latest_features.csv'
    
    print(f"📥 Loading features: s3://{PROCESSED_BUCKET}/{key}")
    
    response = s3.get_object(Bucket=PROCESSED_BUCKET, Key=key)
    df = pd.read_csv(io.BytesIO(response['Body'].read()))
    
    print(f"   Loaded {len(df)} records with {len(df.columns)} features")
    
    return df


def load_model(model_type):
    """Load trained model from S3."""
    
    model_key = f"ml_models/{model_type}_model.pkl"
    
    print(f"📥 Loading model: s3://{MODELS_BUCKET}/{model_key}")
    
    try:
        response = s3.get_object(Bucket=MODELS_BUCKET, Key=model_key)
        model = pickle.loads(response['Body'].read())
        print(f"   Model loaded successfully")
        return model
    except s3.exceptions.NoSuchKey:
        print(f"⚠️ Model not found, using simple prediction")
        return None


def run_prediction(model, features, model_type):
    """Run prediction using model or simple heuristics."""
    
    print(f"🔮 Running {model_type} prediction...")
    
    predictions = {
        'model_type': model_type,
        'timestamp': datetime.utcnow().isoformat(),
        'predictions': []
    }
    
    if model is not None:
        # Use trained model
        feature_cols = [c for c in features.columns if c not in ['year', 'date', 'target']]
        X = features[feature_cols].fillna(0)
        
        y_pred = model.predict(X)
        
        for i, pred in enumerate(y_pred):
            predictions['predictions'].append({
                'index': i,
                'predicted_value': round(float(pred), 4),
                'confidence': 0.85  # Placeholder
            })
    else:
        # Simple heuristic prediction
        if model_type == 'trade':
            # Use ship counts to estimate trade volume
            if 'total_ships' in features.columns:
                base_value = features['total_ships'].mean() * 0.05  # Million TEUs
            else:
                base_value = 9.5  # Historical average for Port of LA
            
            predictions['predictions'].append({
                'metric': 'trade_volume_million_teus',
                'predicted_value': round(base_value, 2),
                'confidence': 0.75,
                'method': 'heuristic'
            })
            
        elif model_type == 'retail':
            # Use car counts to estimate foot traffic
            if 'total_cars' in features.columns:
                base_value = features['total_cars'].mean() * 0.1  # Index
            else:
                base_value = 100  # Baseline index
            
            predictions['predictions'].append({
                'metric': 'retail_foot_traffic_index',
                'predicted_value': round(base_value, 2),
                'confidence': 0.75,
                'method': 'heuristic'
            })
    
    print(f"   Generated {len(predictions['predictions'])} predictions")
    
    return predictions


def save_predictions(predictions, key):
    """Save predictions to S3."""
    
    s3.put_object(
        Bucket=PROCESSED_BUCKET,
        Key=key,
        Body=json.dumps(predictions, indent=2),
        ContentType='application/json'
    )
    
    print(f"✅ Saved predictions to s3://{PROCESSED_BUCKET}/{key}")
