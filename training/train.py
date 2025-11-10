#!/usr/bin/env python3
"""
Model training for fraud detection
Uses XGBoost classifier
"""

import pandas as pd
import numpy as np
import boto3
import os
import json
import joblib
import argparse
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from datetime import datetime

def download_from_s3(bucket, key, local_path):
    """Download file from S3"""
    s3 = boto3.client('s3')
    print(f"Downloading s3://{bucket}/{key}")
    s3.download_file(bucket, key, local_path)

def upload_to_s3(local_path, bucket, key):
    """Upload file to S3"""
    s3 = boto3.client('s3')
    print(f"Uploading to s3://{bucket}/{key}")
    s3.upload_file(local_path, bucket, key)

def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost model"""
    print("\nTraining XGBoost model...")
    
    # Model parameters
    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        'random_state': 42
    }
    
    # Train model
    model = xgb.XGBClassifier(**params)
    
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=True
    )
    
    # Training metrics
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'train_accuracy': float(accuracy_score(y_train, train_pred)),
        'train_precision': float(precision_score(y_train, train_pred)),
        'train_recall': float(recall_score(y_train, train_pred)),
        'train_f1': float(f1_score(y_train, train_pred)),
        'val_accuracy': float(accuracy_score(y_val, val_pred)),
        'val_precision': float(precision_score(y_val, val_pred)),
        'val_recall': float(recall_score(y_val, val_pred)),
        'val_f1': float(f1_score(y_val, val_pred))
    }
    
    print("\nTraining Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return model, metrics

def main():
    print("=" * 80)
    print("FRAUD DETECTION - MODEL TRAINING")
    print("=" * 80)
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    args = parser.parse_args()
    
    # For local testing
    if not os.path.exists(args.train):
        print("Running in local mode - downloading from S3")
        s3_bucket = os.environ.get('S3_DATA_BUCKET', 'mlops-project-dev-ml-data-abc123')
        
        os.makedirs('/tmp/data', exist_ok=True)
        download_from_s3(s3_bucket, 'processed-data/train.csv', '/tmp/data/train.csv')
        download_from_s3(s3_bucket, 'processed-data/validation.csv', '/tmp/data/validation.csv')
        
        args.train = '/tmp/data'
        args.validation = '/tmp/data'
        args.model_dir = '/tmp/model'
        os.makedirs(args.model_dir, exist_ok=True)
    
    # Load training data
    print("\nLoading training data...")
    train_df = pd.read_csv(os.path.join(args.train, 'train.csv'), header=None)
    val_df = pd.read_csv(os.path.join(args.validation, 'validation.csv'), header=None)
    
    # Split features and target (last column is target)
    X_train = train_df.iloc[:, :-1]
    y_train = train_df.iloc[:, -1]
    X_val = val_df.iloc[:, :-1]
    y_val = val_df.iloc[:, -1]
    
    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Fraud rate (train): {y_train.mean():.2%}")
    print(f"Fraud rate (val): {y_val.mean():.2%}")
    
    # Train model
    model, metrics = train_model(X_train, y_train, X_val, y_val)
    
    # Save model
    model_path = os.path.join(args.model_dir, 'model.pkl')
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save metrics
    metrics_path = os.path.join(args.model_dir, 'training_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    # Send metrics to CloudWatch
    try:
        cloudwatch = boto3.client('cloudwatch')
        cloudwatch.put_metric_data(
            Namespace='MLOps/Training',
            MetricData=[
                {
                    'MetricName': 'ValidationAccuracy',
                    'Value': metrics['val_accuracy'],
                    'Unit': 'None'
                },
                {
                    'MetricName': 'ValidationF1Score',
                    'Value': metrics['val_f1'],
                    'Unit': 'None'
                }
            ]
        )
        print("Metrics sent to CloudWatch")
    except Exception as e:
        print(f"Warning: Could not send metrics to CloudWatch: {e}")
    
    print("\nTraining complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
