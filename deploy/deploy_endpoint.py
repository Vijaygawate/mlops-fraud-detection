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
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    s3.download_file(bucket, key, local_path)

def wait_for_endpoint(endpoint_name, sagemaker_client, timeout=900):
    """Wait for endpoint to be in service"""
    print(f"\nWaiting for endpoint '{endpoint_name}' to be in service...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            print(f"Status: {status}")
            
            if status == 'InService':
                print("✅ Endpoint is in service!")
                return True
            elif status == 'Failed':
                failure_reason = response.get('FailureReason', 'Unknown')
                print(f"❌ Endpoint creation failed: {failure_reason}")
                return False
            
            time.sleep(30)
        except Exception as e:
            print(f"Error checking endpoint status: {e}")
            time.sleep(30)
    
    print(f"❌ Timeout waiting for endpoint after {timeout} seconds")
    return False

def test_endpoint(endpoint_name, runtime_client):
    """Test the deployed endpoint"""
    print(f"\nTesting endpoint '{endpoint_name}'...")
    
    # Sample test data (should match training data format - 11 features)
    # Based on preprocessed data: transaction_amount, merchant_category, transaction_hour, etc.
    test_payload = {
        "instances": [[100.0, 5, 14, 2.5, 365, 15, 50.0, 0, 2, 20.0, 4.61]]
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
        print(f"   Make sure test payload has {len(test_payload['instances'][0])} features")
        return False

def main():
    print("=" * 80)
    print("FRAUD DETECTION - MODEL DEPLOYMENT")
    print("=" * 80)
    
    # Configuration
    endpoint_name = os.environ.get('ENDPOINT_NAME', 'fraud-detection-endpoint')
    s3_bucket = os.environ.get('MODEL_ARTIFACTS_BUCKET', 'mlops-project-dev-model-artifacts-b618d389')
    sagemaker_role = os.environ.get('SAGEMAKER_ROLE_ARN', 'arn:aws:iam::446468849132:role/mlops-project-dev-sagemaker-execution')
    instance_type = os.environ.get('ENDPOINT_INSTANCE_TYPE', 'ml.m5.large')
    instance_count = int(os.environ.get('ENDPOINT_INSTANCE_COUNT', '1'))
    
    print(f"Endpoint: {endpoint_name}")
    print(f"Bucket: {s3_bucket}")
    print(f"Role: {sagemaker_role}")
    print(f"Instance: {instance_type} x {instance_count}")
    
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
    try:
        sagemaker.update_model_package(
            ModelPackageArn=model_package_arn,
            ModelApprovalStatus='Approved'
        )
        print("✅ Model package approved")
    except Exception as e:
        print(f"⚠️ Could not approve model package: {e}")
        print("   (May already be approved)")
    
    # Create model name with timestamp
    model_name = f"fraud-detection-model-{datetime.utcnow().strftime('%Y-%m-%d-%H-%M-%S')}"
    
    # Create model from model package
    print(f"\nCreating model '{model_name}'...")
    try:
        sagemaker.create_model(
            ModelName=model_name,
            PrimaryContainer={
                'ModelPackageName': model_package_arn
            },
            ExecutionRoleArn=sagemaker_role
        )
        print(f"✅ Model created: {model_name}")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        raise
    
    # Create endpoint configuration
    endpoint_config_name = f"{model_name}-config"
    print(f"\nCreating endpoint configuration '{endpoint_config_name}'...")
    
    try:
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
        print(f"✅ Endpoint config created: {endpoint_config_name}")
    except Exception as e:
        print(f"❌ Error creating endpoint config: {e}")
        raise
    
    # Check if endpoint exists
    endpoint_exists = False
    try:
        sagemaker.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
        print(f"\n⚠️ Endpoint '{endpoint_name}' already exists. Updating...")
    except sagemaker.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            endpoint_exists = False
            print(f"\n✅ Creating new endpoint '{endpoint_name}'...")
        else:
            raise
    
    # Create or update endpoint
    try:
        if endpoint_exists:
            sagemaker.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            print("✅ Endpoint update initiated")
        else:
            sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            print("✅ Endpoint creation initiated")
    except Exception as e:
        print(f"❌ Error creating/updating endpoint: {e}")
        raise
    
    # Wait for endpoint (increased timeout to 15 minutes)
    print("\n" + "=" * 80)
    print("WAITING FOR ENDPOINT (this takes 5-10 minutes)")
    print("=" * 80)
    
    if not wait_for_endpoint(endpoint_name, sagemaker, timeout=900):
        print("\n❌ Endpoint deployment failed!")
        print("Check CloudWatch logs:")
        print(f"  aws logs tail /aws/sagemaker/Endpoints/{endpoint_name} --follow --region eu-north-1")
        raise Exception("Endpoint deployment failed")
    
    # Test endpoint
    print("\n" + "=" * 80)
    print("TESTING ENDPOINT")
    print("=" * 80)
    
    if test_endpoint(endpoint_name, runtime):
        print("\n✅ Deployment and testing successful!")
    else:
        print("\n⚠️ Endpoint deployed but test failed")
        print("   Check that test payload matches model input format")
    
    # Save endpoint name to both locations
    print("\nSaving endpoint information...")
    with open('/tmp/endpoint_name.txt', 'w') as f:
        f.write(endpoint_name)
    
    with open('endpoint_name.txt', 'w') as f:
        f.write(endpoint_name)
    
    # Upload to S3
    s3 = boto3.client('s3')
    s3.upload_file('/tmp/endpoint_name.txt', s3_bucket, 'deployment/endpoint_name.txt')
    print("✅ Endpoint name saved")
    
    # Send metrics to CloudWatch
    try:
        cloudwatch = boto3.client('cloudwatch')
        cloudwatch.put_metric_data(
            Namespace='MLOps/Deployment',
            MetricData=[
                {
                    'MetricName': 'EndpointDeployed',
                    'Value': 1,
                    'Unit': 'Count',
                    'Dimensions': [
                        {'Name': 'Endpoint', 'Value': endpoint_name},
                        {'Name': 'InstanceType', 'Value': instance_type}
                    ]
                }
            ]
        )
        print("✅ Metrics sent to CloudWatch")
    except Exception as e:
        print(f"⚠️ Could not send metrics: {e}")
    
    print("\n" + "=" * 80)
    print("✅ DEPLOYMENT COMPLETE!")
    print("=" * 80)
    print(f"Endpoint Name: {endpoint_name}")
    print(f"Instance Type: {instance_type}")
    print(f"Instance Count: {instance_count}")
    print(f"Model Name: {model_name}")
    print(f"\nTest the endpoint:")
    print(f"  aws sagemaker-runtime invoke-endpoint \\")
    print(f"    --endpoint-name {endpoint_name} \\")
    print(f"    --body '{{\"instances\": [[100.0, 5, 14, 2.5, 365, 15, 50.0, 0, 2, 20.0, 4.61]]}}' \\")
    print(f"    --content-type application/json \\")
    print(f"    output.json")
    print("=" * 80)

if __name__ == "__main__":
    main()
