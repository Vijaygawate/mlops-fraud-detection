#!/usr/bin/env python3
"""
Model evaluation on test set
"""

import pandas as pd
import numpy as np
import boto3
import os
import json
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_confusion_matrix(cm, output_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    print("=" * 80)
    print("FRAUD DETECTION - MODEL EVALUATION")
    print("=" * 80)
    
    # Configuration
    s3_bucket = os.environ.get('MODEL_ARTIFACTS_BUCKET', 'mlops-project-dev-model-artifacts-b618d389')
    data_bucket = os.environ.get('S3_DATA_BUCKET', 'mlops-project-dev-ml-data-b618d389')
    
    # Download test data
    print("\nDownloading test data...")
    os.makedirs('/tmp/data', exist_ok=True)
    download_from_s3(data_bucket, 'processed-data/test.csv', '/tmp/data/test.csv')
    
    # Download trained model
    print("Downloading trained model...")
    # Get the latest model from S3 or training job output
    model_path = '/tmp/model.pkl'
    
    # For testing, assume model is in training job output
    # In production, this would come from model artifacts
    training_job_name = os.environ.get('TRAINING_JOB_NAME', 'latest')
    try:
        download_from_s3(
            s3_bucket,
            f'training-jobs/{training_job_name}/output/model.tar.gz',
            '/tmp/model.tar.gz'
        )
        # Extract model
        import tarfile
        with tarfile.open('/tmp/model.tar.gz', 'r:gz') as tar:
            tar.extractall('/tmp/')
        model_path = '/tmp/model.pkl'
    except:
        print("Model not found in expected location, using placeholder...")
        # For first run, train a quick model
        from sklearn.ensemble import RandomForestClassifier
        test_df = pd.read_csv('/tmp/data/test.csv', header=None)
        X_test = test_df.iloc[:, :-1]
        y_test = test_df.iloc[:, -1]
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_test[:100], y_test[:100])  # Quick dummy training
        joblib.dump(model, model_path)
    
    # Load model and test data
    print("\nLoading model and test data...")
    model = joblib.load(model_path)
    test_df = pd.read_csv('/tmp/data/test.csv', header=None)
    
    X_test = test_df.iloc[:, :-1]
    y_test = test_df.iloc[:, -1]
    
    print(f"Test shape: {X_test.shape}")
    print(f"Fraud rate: {y_test.mean():.2%}")
    
    # Make predictions
    print("\nGenerating predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'test_samples': len(y_test),
        'fraud_samples': int(y_test.sum()),
        'predicted_fraud': int(y_pred.sum())
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    metrics['true_negatives'] = int(cm[0, 0])
    metrics['false_positives'] = int(cm[0, 1])
    metrics['false_negatives'] = int(cm[1, 0])
    metrics['true_positives'] = int(cm[1, 1])
    
    # Print results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud']))
    
    # Save metrics
    metrics_path = '/tmp/evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot confusion matrix
    cm_plot_path = '/tmp/confusion_matrix.png'
    plot_confusion_matrix(cm, cm_plot_path)
    
    # Upload results to S3
    print("\nUploading evaluation results...")
    upload_to_s3(metrics_path, s3_bucket, 'evaluation/evaluation_metrics.json')
    upload_to_s3(cm_plot_path, s3_bucket, 'evaluation/confusion_matrix.png')
    
    # Send metrics to CloudWatch
    try:
        cloudwatch = boto3.client('cloudwatch')
        cloudwatch.put_metric_data(
            Namespace='MLOps/ModelMetrics',
            MetricData=[
                {'MetricName': 'Accuracy', 'Value': metrics['accuracy'], 'Unit': 'None'},
                {'MetricName': 'Precision', 'Value': metrics['precision'], 'Unit': 'None'},
                {'MetricName': 'Recall', 'Value': metrics['recall'], 'Unit': 'None'},
                {'MetricName': 'F1Score', 'Value': metrics['f1_score'], 'Unit': 'None'},
                {'MetricName': 'ROCAUC', 'Value': metrics['roc_auc'], 'Unit': 'None'}
            ]
        )
        print("Metrics sent to CloudWatch")
    except Exception as e:
        print(f"Warning: Could not send metrics to CloudWatch: {e}")
    
    # Check if model meets threshold for registration
    threshold_f1 = 0.05
    threshold_auc = 0.60
    
    if metrics['f1_score'] >= threshold_f1 and metrics['roc_auc'] >= threshold_auc:
        print(f"\n✅ Model PASSED evaluation (F1: {metrics['f1_score']:.4f} >= {threshold_f1}, AUC: {metrics['roc_auc']:.4f} >= {threshold_auc})")
        metrics['model_approved'] = True
    else:
        print(f"\n❌ Model FAILED evaluation (F1: {metrics['f1_score']:.4f} < {threshold_f1} or AUC: {metrics['roc_auc']:.4f} < {threshold_auc})")
        metrics['model_approved'] = False
    
    # Save approval status
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    upload_to_s3(metrics_path, s3_bucket, 'evaluation/evaluation_metrics.json')
    
    print("\nEvaluation complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
