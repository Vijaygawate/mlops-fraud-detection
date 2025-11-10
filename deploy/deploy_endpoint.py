#!/usr/bin/env python3
"""
Deploy approved model to SageMaker endpoint
"""

import boto3
import json
import os
import time
from datetime import datetime

def download_from_s3(bucket, key, local_path):
    """Download file from S3"""
    s3 = boto3.client('s3')
    print(f"Downloading s3://{bucket}/{key}")
    s3.download_file(bucket, key, local_path)

def wait_for_endpoint(endpoint_name, sagemaker_client, timeout=600):
    """Wait for endpoint to be in service"""
    print(f"\nWaiting for endpoint '{endpoint_name}' to be in service...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        status = response['EndpointStatus']
        print(f"Status: {status}")
        
        if status == 'InService':
            print("✅ Endpoint is in service!")
            return True
        elif status == 'Failed':
            print(f"❌ Endpoint creation failed: {response.get('FailureReason', 'Unknown')}")
            return False
        
        time.sleep(30)
    
    print(f"❌ Timeout waiting for endpoint")
    return False

def test_endpoint(endpoint_name, runtime_client):
    """Test the deployed endpoint"""
    print(f"\nTesting endpoint '{endpoint_name}'...")
    
    # Sample test data (should match training data format)
    test_payload = {
        "instances": [[100.0, 5, 14, 2.5, 365, 15, 50.0, 0, 2, 20.0, 0.5, 1, 4.61, 5.90]]
    }
    
    try:
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(test_payload)
        )
        
        result = json.loads(response['Body'].read().decode())
        print(f"✅ Test successful! Prediction: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("=" * 80)
    print("FRAUD DETECTION - MODEL DEPLOYMENT")
    print("=" * 80)
    
    # Configuration
    endpoint_name = os.environ.get('ENDPOINT_NAME', 'fraud-detection-endpoint')
    s3_bucket = os.environ.get('MODEL_ARTIFACTS_BUCKET', 'mlops-project-dev-model-artifacts-abc123')
    sagemaker_role = os.environ.get('SAGEMAKER_ROLE_ARN', 'arn:aws:iam::123456789012:role/SageMakerRole')
    instance_type = os.environ.get('ENDPOINT_INSTANCE_TYPE', 'ml.m5.large')
    instance_count = int(os.environ.get('ENDPOINT_INSTANCE_COUNT', '2'))
    
    # Download model package ARN
    print("\nDownloading model package information...")
    arn_path = '/tmp/model_package_arn.txt'
    download_from_s3(s3_bucket, 'deployment/model_package_arn.txt', arn_path)
    
    with open(arn_path, 'r') as f:
        model_package_arn = f.read().strip()
    
    print(f"Model Package ARN: {model_package_arn}")
    
    # Create SageMaker clients
    sagemaker = boto3.client('sagemaker')
    runtime = boto3.client('sagemaker-runtime')
    
    # Update model package to Approved status
    print("\nApproving model package...")
    sagemaker.update_model_package(
        ModelPackageArn=model_package_arn,
        ModelApprovalStatus='Approved'
    )
    
    # Create model name
    model_name = f"fraud-detection-model-{datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')}"
    
    # Create model from model package
    print(f"\nCreating model '{model_name}'...")
    sagemaker.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'ModelPackageName': model_package_arn
        },
        ExecutionRoleArn=sagemaker_role
    )
    
    # Create endpoint configuration
    endpoint_config_name = f"{model_name}-config"
    print(f"\nCreating endpoint configuration '{endpoint_config_name}'...")
    
    sagemaker.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InitialInstanceCount': instance_count,
                'InstanceType': instance_type,
                'InitialVariantWeight': 1.0
            }
        ]
    )
    
    # Check if endpoint exists
    try:
        sagemaker.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
        print(f"\nEndpoint '{endpoint_name}' exists. Updating...")
    except:
        endpoint_exists = False
        print(f"\nCreating new endpoint '{endpoint_name}'...")
    
    # Create or update endpoint
    if endpoint_exists:
        sagemaker.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    else:
        sagemaker.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    
    # Wait for endpoint
    if not wait_for_endpoint(endpoint_name, sagemaker, timeout=900):
        print("❌ Endpoint deployment failed!")
        return
    
    # Test endpoint
    if test_endpoint(endpoint_name, runtime):
        print("\n✅ Deployment successful!")
    else:
        print("\n⚠️  Endpoint deployed but test failed")
    
    # Save endpoint name
    with open('/tmp/endpoint_name.txt', 'w') as f:
        f.write(endpoint_name)
    
    # Upload to S3
    s3 = boto3.client('s3')
    s3.upload_file('/tmp/endpoint_name.txt', s3_bucket, 'deployment/endpoint_name.txt')
    
    # Send metrics to CloudWatch
    try:
        cloudwatch = boto3.client('cloudwatch')
        cloudwatch.put_metric_data(
            Namespace='MLOps',
            MetricData=[
                {
                    'MetricName': 'EndpointDeployed',
                    'Value': 1,
                    'Unit': 'Count'
                }
            ]
        )
    except Exception as e:
        print(f"Warning: Could not send metrics: {e}")
    
    print(f"\nEndpoint Name: {endpoint_name}")
    print(f"Instance Type: {instance_type}")
    print(f"Instance Count: {instance_count}")
    print("\nDeployment complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
