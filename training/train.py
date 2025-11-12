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
import tarfile

def download_from_s3(bucket, key, local_path):
    """Download file from S3"""
    s3 = boto3.client('s3')
    print(f"Downloading s3://{bucket}/{key}")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
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
    
    print(f"Class balance:")
    print(f"  Non-fraud: {len(y_train[y_train == 0])}")
    print(f"  Fraud: {len(y_train[y_train == 1])}")
    print(f"  Scale pos weight: {params['scale_pos_weight']:.2f}")
    
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
    
    # Get environment variables
    s3_data_bucket = os.environ.get('S3_DATA_BUCKET', 'mlops-project-dev-ml-data-b618d389')
    model_artifacts_bucket = os.environ.get('MODEL_ARTIFACTS_BUCKET', 'mlops-project-dev-model-artifacts-b618d389')
    
    print(f"S3 Data Bucket: {s3_data_bucket}")
    print(f"Model Artifacts Bucket: {model_artifacts_bucket}")
    
    if not s3_data_bucket or not model_artifacts_bucket:
        raise ValueError("S3_DATA_BUCKET and MODEL_ARTIFACTS_BUCKET environment variables must be set!")
    
    # Setup paths
    print("\nSetting up paths...")
    os.makedirs('/tmp/data', exist_ok=True)
    
    # Download training data
    print("\nDownloading training data from S3...")
    download_from_s3(s3_data_bucket, 'processed-data/train.csv', '/tmp/data/train.csv')
    download_from_s3(s3_data_bucket, 'processed-data/validation.csv', '/tmp/data/validation.csv')
    
    # Load training data
    print("\nLoading training data...")
    train_df = pd.read_csv('/tmp/data/train.csv', header=None)
    val_df = pd.read_csv('/tmp/data/validation.csv', header=None)
    
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
    
    # Save model directly (not in subdirectory)
    model_path = '/tmp/model.pkl'
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved to {model_path}")
    
    # Save metrics
    metrics_path = '/tmp/training_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to {metrics_path}")
    
    # CRITICAL FIX: Create inference.py for SageMaker endpoint
    print("\n📝 Creating inference script...")
    inference_script = '''import joblib
import json
import numpy as np

def model_fn(model_dir):
    """Load the model from model directory"""
    import os
    model_path = os.path.join(model_dir, 'model.pkl')
    model = joblib.load(model_path)
    return model

def input_fn(request_body, content_type='application/json'):
    """Deserialize and prepare the prediction input"""
    if content_type == 'application/json':
        data = json.loads(request_body)
        # Handle different input formats
        if 'instances' in data:
            return np.array(data['instances'])
        elif isinstance(data, list):
            return np.array(data)
        else:
            raise ValueError(f"Unsupported JSON format: {data}")
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Apply model to the incoming request"""
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)
    return {
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist()
    }

def output_fn(prediction, accept='application/json'):
    """Serialize and prepare the prediction output"""
    if accept == 'application/json':
        return json.dumps(prediction), accept
    raise ValueError(f"Unsupported accept type: {accept}")
'''
    
    inference_path = '/tmp/inference.py'
    with open(inference_path, 'w') as f:
        f.write(inference_script)
    print(f"✅ Inference script created at {inference_path}")
    
    # CRITICAL FIX: Create tarball with FLAT structure (files at root, not in subdirectory)
    print("\n📦 Creating model tarball with proper structure...")
    model_tar_path = '/tmp/model.tar.gz'
    with tarfile.open(model_tar_path, 'w:gz') as tar:
        # Add files with arcname to ensure they're at root of tarball
        tar.add(model_path, arcname='model.pkl')
        tar.add(inference_path, arcname='inference.py')
    
    # Verify tarball structure
    print("\n✅ Tarball contents:")
    with tarfile.open(model_tar_path, 'r:gz') as tar:
        for member in tar.getmembers():
            print(f"   ✓ {member.name}")
    
    # Get file size
    tar_size = os.path.getsize(model_tar_path)
    print(f"\n   Tarball size: {tar_size / 1024:.2f} KB")
    
    # Upload to S3 with timestamp
    timestamp = datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')
    training_job_name = f'job-{timestamp}'
    model_s3_key = f'training-jobs/{training_job_name}/output/model.tar.gz'
    metrics_s3_key = f'training-jobs/{training_job_name}/output/training_metrics.json'
    
    print(f"\n📤 Uploading model to S3...")
    print(f"   Bucket: {model_artifacts_bucket}")
    print(f"   Key: {model_s3_key}")
    
    s3 = boto3.client('s3')
    s3.upload_file(model_tar_path, model_artifacts_bucket, model_s3_key)
    print(f"✅ Model uploaded to s3://{model_artifacts_bucket}/{model_s3_key}")
    
    # Upload metrics
    s3.upload_file(metrics_path, model_artifacts_bucket, metrics_s3_key)
    print(f"✅ Metrics uploaded to s3://{model_artifacts_bucket}/{metrics_s3_key}")
    
    # Save training job name for next stages
    with open('/tmp/training_job_name.txt', 'w') as f:
        f.write(training_job_name)
    print(f"✅ Training job name: {training_job_name}")
    
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
        print("✅ Metrics sent to CloudWatch")
    except Exception as e:
        print(f"Warning: Could not send metrics to CloudWatch: {e}")
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Model location: s3://{model_artifacts_bucket}/{model_s3_key}")
    print(f"Training job: {training_job_name}")
    print("=" * 80)

if __name__ == "__main__":
    main()
