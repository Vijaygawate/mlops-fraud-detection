import os
import boto3
import datetime

endpoint_name = os.environ.get("ENDPOINT_NAME", "fraud-detection-endpoint")
role_arn = os.environ.get("SAGEMAKER_ROLE_ARN")
model_package_arn = os.environ.get("ROLLBACK_MODEL_PACKAGE_ARN")
instance_type = os.environ.get("ENDPOINT_INSTANCE_TYPE", "ml.m5.large")

if not model_package_arn:
    raise ValueError("ROLLBACK_MODEL_PACKAGE_ARN must be set for rollback")

sm_client = boto3.client("sagemaker")

# Get an endpoint config name that includes timestamp
timestamp = datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
endpoint_config_name = f"{endpoint_name}-rollback-{timestamp}"

print(f"Creating model from package ARN: {model_package_arn}")
model_name = f"{endpoint_name}-rollback-model-{timestamp}"

try:
    sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={
            "ModelPackageName": model_package_arn
        }
    )
    print(f"Model created: {model_name}")
except sm_client.exceptions.ClientError as e:
    if "already exists" in str(e):
        print(f"Model {model_name} already exists, continuing...")
    else:
        raise

print(f"Creating endpoint config: {endpoint_config_name}")
sm_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            "ModelName": model_name,
            "VariantName": "AllTraffic",
            "InstanceType": instance_type,
            "InitialInstanceCount": 1
        }
    ]
)

print(f"Updating endpoint {endpoint_name} to use config {endpoint_config_name}")
sm_client.update_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

print(f"✅ Rollback complete. Endpoint '{endpoint_name}' now serves model package {model_package_arn}")
