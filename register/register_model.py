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
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, key, local_path)

def main():
    print("=" * 80)
    print("FRAUD DETECTION - MODEL REGISTRATION")
    print("=" * 80)
    
    # Configuration
    model_package_group_name = os.environ.get('MODEL_PACKAGE_GROUP_NAME', 'fraud-detection-models')
    s3_bucket = os.environ.get('MODEL_ARTIFACTS_BUCKET', 'mlops-project-dev-model-artifacts-8c2241a2')
    sagemaker_role = os.environ.get('SAGEMAKER_ROLE_ARN', 'arn:aws:iam::446468849132:role/mlops-project-dev-sagemaker-execution')
    
    print(f"Model Package Group: {model_package_group_name}")
    print(f"Artifacts Bucket: {s3_bucket}")
    
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
    
    print("\n✅ Model approved! Proceeding with registration...")
    
    # Find the latest trained model (FIXED: Don't use hardcoded training_job_name)
    s3 = boto3.client('s3')
    print("\nFinding latest trained model...")
    
    response = s3.list_objects_v2(
        Bucket=s3_bucket,
        Prefix='training-jobs/'
    )
    
    objects = sorted(response.get('Contents', []), key=lambda x: x['LastModified'], reverse=True)
    
    model_data_url = None
    training_job_name = None
    
    for obj in objects:
        if obj['Key'].endswith('model.tar.gz'):
            model_data_url = f"s3://{s3_bucket}/{obj['Key']}"
            # Extract training job name from path: training-jobs/job-YYYY-MM-DD-HH-MM-SS/output/model.tar.gz
            training_job_name = obj['Key'].split('/')[1]
            print(f"Found model: {model_data_url}")
            print(f"Training job: {training_job_name}")
            break
    
    if not model_data_url:
        raise Exception("No trained model found in S3!")
    
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
            # FIXED: Removed MetadataProperties - TrainingJobName not supported there
            CustomerMetadataProperties={
                'Accuracy': str(metrics['accuracy']),
                'F1Score': str(metrics['f1_score']),
                'ROCAUC': str(metrics['roc_auc']),
                'TrainingJobName': training_job_name,  # Moved here
                'Timestamp': datetime.utcnow().isoformat(),
                'GeneratedBy': 'MLOps-Pipeline'  # Moved here
            }
        )
        
        model_package_arn = response['ModelPackageArn']
        print(f"\n✅ Model registered successfully!")
        print(f"Model Package ARN: {model_package_arn}")
        
        # Save model package ARN for deployment stage
        # Save to both locations
        with open('/tmp/model_package_arn.txt', 'w') as f:
            f.write(model_package_arn)
        
        with open('model_package_arn.txt', 'w') as f:
            f.write(model_package_arn)
        
        print("Model package ARN saved locally")
        
        # Upload to S3
        s3.upload_file('/tmp/model_package_arn.txt', s3_bucket, 'deployment/model_package_arn.txt')
        print("Model package ARN uploaded to S3")
        
        # Send metric to CloudWatch
        try:
            cloudwatch = boto3.client('cloudwatch')
            cloudwatch.put_metric_data(
                Namespace='MLOps/Registration',
                MetricData=[
                    {
                        'MetricName': 'ModelRegistered',
                        'Value': 1,
                        'Unit': 'Count'
                    }
                ]
            )
            print("✅ Metrics sent to CloudWatch")
        except Exception as e:
            print(f"Warning: Could not send metrics to CloudWatch: {e}")
        
    except Exception as e:
        print(f"\n❌ Error registering model: {e}")
        raise
    
    print("\n✅ Registration complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
