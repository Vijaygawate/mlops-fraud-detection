#!/usr/bin/env python3
"""
Register approved model in SageMaker Model Registry
"""

import boto3
import json
import os
from datetime import datetime

def download_from_s3(bucket, key, local_path):
    """Download file from S3"""
    s3 = boto3.client('s3')
    print(f"Downloading s3://{bucket}/{key}")
    s3.download_file(bucket, key, local_path)

def main():
    print("=" * 80)
    print("FRAUD DETECTION - MODEL REGISTRATION")
    print("=" * 80)
    
    # Configuration
    model_package_group_name = os.environ.get('MODEL_PACKAGE_GROUP_NAME', 'fraud-detection-models')
    s3_bucket = os.environ.get('MODEL_ARTIFACTS_BUCKET', 'mlops-project-dev-model-artifacts-abc123')
    sagemaker_role = os.environ.get('SAGEMAKER_ROLE_ARN', 'arn:aws:iam::123456789012:role/SageMakerRole')
    
    # Download evaluation metrics
    print("\nDownloading evaluation metrics...")
    metrics_path = '/tmp/evaluation_metrics.json'
    download_from_s3(s3_bucket, 'evaluation/evaluation_metrics.json', metrics_path)
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    print(f"\nModel Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Approved:  {metrics.get('model_approved', False)}")
    
    # Check if model is approved
    if not metrics.get('model_approved', False):
        print("\n❌ Model did not meet approval criteria. Skipping registration.")
        return
    
    # Get model artifacts location
    training_job_name = os.environ.get('TRAINING_JOB_NAME', 'latest')
    model_data_url = f"s3://{s3_bucket}/training-jobs/{training_job_name}/output/model.tar.gz"
    
    # Create SageMaker client
    sagemaker = boto3.client('sagemaker')
    
    # Register model
    print("\nRegistering model in Model Registry...")
    
    model_package_description = f"Fraud detection model - F1: {metrics['f1_score']:.4f}, AUC: {metrics['roc_auc']:.4f}"
    
    try:
        response = sagemaker.create_model_package(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageDescription=model_package_description,
            InferenceSpecification={
                'Containers': [
                    {
                        # 'Image': '492215442770.dkr.ecr.eu-central-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3', (central)
                        'Image': '662702820516.dkr.ecr.eu-north-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
                        'ModelDataUrl': model_data_url,
                    }
                ],
                'SupportedContentTypes': ['application/json', 'text/csv'],
                'SupportedResponseMIMETypes': ['application/json'],
                'SupportedRealtimeInferenceInstanceTypes': [
                    'ml.t2.medium',
                    'ml.m5.large',
                    'ml.m5.xlarge'
                ],
                'SupportedTransformInstanceTypes': [
                    'ml.m5.large',
                    'ml.m5.xlarge'
                ]
            },
            ModelApprovalStatus='PendingManualApproval',
            MetadataProperties={
                'GeneratedBy': 'MLOps Pipeline',
                'TrainingJobName': training_job_name
            },
            CustomerMetadataProperties={
                'Accuracy': str(metrics['accuracy']),
                'F1Score': str(metrics['f1_score']),
                'ROCAUC': str(metrics['roc_auc']),
                'Timestamp': datetime.utcnow().isoformat()
            }
        )
        
        model_package_arn = response['ModelPackageArn']
        print(f"\n✅ Model registered successfully!")
        print(f"Model Package ARN: {model_package_arn}")
        
        # Save model package ARN for deployment stage
        with open('/tmp/model_package_arn.txt', 'w') as f:
            f.write(model_package_arn)
        
        # Upload to S3
        s3 = boto3.client('s3')
        s3.upload_file('/tmp/model_package_arn.txt', s3_bucket, 'deployment/model_package_arn.txt')
        
        # Send metric to CloudWatch
        cloudwatch = boto3.client('cloudwatch')
        cloudwatch.put_metric_data(
            Namespace='MLOps',
            MetricData=[
                {
                    'MetricName': 'ModelRegistered',
                    'Value': 1,
                    'Unit': 'Count'
                }
            ]
        )
        
    except Exception as e:
        print(f"\n❌ Error registering model: {e}")
        raise
    
    print("\nRegistration complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
