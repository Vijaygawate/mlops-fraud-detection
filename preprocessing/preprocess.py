#!/usr/bin/env python3
"""
Data preprocessing for fraud detection model
Reads raw data from S3, processes it, and saves train/validation/test splits
"""

import pandas as pd
import numpy as np
import boto3
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Configuration
S3_BUCKET = os.environ.get('S3_DATA_BUCKET', 'mlops-project-dev-ml-data-abc123')
INPUT_KEY = 'raw-data/transactions.csv'
OUTPUT_PREFIX = 'processed-data/'

def download_from_s3(bucket, key, local_path):
    """Download file from S3"""
    s3 = boto3.client('s3')
    print(f"Downloading s3://{bucket}/{key} to {local_path}")
    s3.download_file(bucket, key, local_path)
    print(f"Download complete")

def upload_to_s3(local_path, bucket, key):
    """Upload file to S3"""
    s3 = boto3.client('s3')
    print(f"Uploading {local_path} to s3://{bucket}/{key}")
    s3.upload_file(local_path, bucket, key)
    print(f"Upload complete")

def preprocess_data(df):
    """Preprocess the data"""
    print("Starting preprocessing...")
    
    # Remove any duplicates
    df = df.drop_duplicates()
    
    # Handle missing values
    df = df.fillna(df.median())
    
    # Feature engineering
    df['amount_per_day_ratio'] = df['transaction_amount'] / (df['days_since_last_transaction'] + 1)
    df['transaction_frequency'] = df['num_transactions_last_30days'] / 30
    df['is_high_value'] = (df['transaction_amount'] > df['transaction_amount'].quantile(0.95)).astype(int)
    
    # Log transform for skewed features
    df['log_amount'] = np.log1p(df['transaction_amount'])
    df['log_account_age'] = np.log1p(df['account_age_days'])
    
    print(f"Preprocessing complete. Shape: {df.shape}")
    return df

def main():
    print("=" * 80)
    print("FRAUD DETECTION - DATA PREPROCESSING")
    print("=" * 80)
    
    # Download data from S3
    local_input = '/tmp/transactions.csv'
    download_from_s3(S3_BUCKET, INPUT_KEY, local_input)
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(local_input)
    print(f"Loaded {len(df)} records")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    
    # Preprocess
    df = preprocess_data(df)
    
    # Separate features and target
    feature_columns = [col for col in df.columns if col != 'is_fraud']
    X = df[feature_columns]
    y = df['is_fraud']
    
    # Split data: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train)} samples ({y_train.sum()} fraud)")
    print(f"  Validation: {len(X_val)} samples ({y_val.sum()} fraud)")
    print(f"  Test: {len(X_test)} samples ({y_test.sum()} fraud)")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    scaler_path = '/tmp/scaler.pkl'
    joblib.dump(scaler, scaler_path)
    upload_to_s3(scaler_path, S3_BUCKET, f'{OUTPUT_PREFIX}scaler.pkl')
    
    # Create DataFrames with scaled data
    train_df = pd.DataFrame(X_train_scaled, columns=feature_columns)
    train_df['is_fraud'] = y_train.values
    
    val_df = pd.DataFrame(X_val_scaled, columns=feature_columns)
    val_df['is_fraud'] = y_val.values
    
    test_df = pd.DataFrame(X_test_scaled, columns=feature_columns)
    test_df['is_fraud'] = y_test.values
    
    # Save processed data
    print("\nSaving processed data...")
    train_path = '/tmp/train.csv'
    val_path = '/tmp/validation.csv'
    test_path = '/tmp/test.csv'
    
    train_df.to_csv(train_path, index=False, header=False)
    val_df.to_csv(val_path, index=False, header=False)
    test_df.to_csv(test_path, index=False, header=False)
    
    # Upload to S3
    upload_to_s3(train_path, S3_BUCKET, f'{OUTPUT_PREFIX}train.csv')
    upload_to_s3(val_path, S3_BUCKET, f'{OUTPUT_PREFIX}validation.csv')
    upload_to_s3(test_path, S3_BUCKET, f'{OUTPUT_PREFIX}test.csv')
    
    # Save feature names
    feature_info = {
        'feature_columns': feature_columns,
        'num_features': len(feature_columns)
    }
    feature_info_path = '/tmp/feature_info.json'
    with open(feature_info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    upload_to_s3(feature_info_path, S3_BUCKET, f'{OUTPUT_PREFIX}feature_info.json')
    
    # Log metrics
    print("\n" + "=" * 80)
    print("PREPROCESSING METRICS")
    print("=" * 80)
    metrics = {
        'total_records': len(df),
        'train_records': len(train_df),
        'val_records': len(val_df),
        'test_records': len(test_df),
        'num_features': len(feature_columns),
        'fraud_rate_train': float(y_train.mean()),
        'fraud_rate_val': float(y_val.mean()),
        'fraud_rate_test': float(y_test.mean())
    }
    
    for key, value in metrics.items():
        print(f"{key}: {value}")
    
    # Send metrics to CloudWatch
    try:
        cloudwatch = boto3.client('cloudwatch')
        cloudwatch.put_metric_data(
            Namespace='MLOps/Preprocessing',
            MetricData=[
                {
                    'MetricName': 'RecordsProcessed',
                    'Value': metrics['total_records'],
                    'Unit': 'Count'
                },
                {
                    'MetricName': 'TrainRecords',
                    'Value': metrics['train_records'],
                    'Unit': 'Count'
                }
            ]
        )
        print("\nMetrics sent to CloudWatch")
    except Exception as e:
        print(f"\nWarning: Could not send metrics to CloudWatch: {e}")
    
    print("\nPreprocessing complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
