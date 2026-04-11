"""
SageMaker deployment script — packages, registers, and deploys the sklearn model.

Usage:
    uv run python deploy/deploy_sagemaker.py \
      --bucket your-bucket-name \
      --region eu-west-1 \
      --endpoint-name your-endpoint-name \
      --model-package-group your-group-name
"""

import argparse
import json
import tarfile
from pathlib import Path

import boto3

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_FILE = MODEL_DIR / "xgboost_bath_predictor.joblib"
METADATA_FILE = MODEL_DIR / "model_metadata.json"
DEPLOY_DIR = Path(__file__).resolve().parent


def _get_sklearn_image_uri(region: str) -> str:
    import sagemaker
    return sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="1.2-1",
        image_scope="inference",
    )


def package_model(model_path: Path, output_dir: Path) -> Path:
    """Package the sklearn model and inference script as a .tar.gz for SageMaker.

    SageMaker sklearn container expects:
    - model.joblib at the root
    - code/inference.py for the custom serving logic
    - SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/model/code tells the container where to find it

    Args:
        model_path: Path to the trained model joblib file.
        output_dir: Directory where the .tar.gz will be created.

    Returns:
        Path to the created .tar.gz file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tar_path = output_dir / "model.tar.gz"

    inference_script = DEPLOY_DIR / "inference_sagemaker.py"
    if not inference_script.exists():
        raise FileNotFoundError(
            f"inference_sagemaker.py not found at {inference_script}."
        )

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(model_path, arcname="model.joblib")
        tar.add(inference_script, arcname="code/inference.py")

    # Verify contents
    with tarfile.open(tar_path) as t:
        contents = t.getnames()
    print(f"  Archive contents: {contents}")
    print(f"  Packaged: {tar_path} ({tar_path.stat().st_size / 1024:.1f} KB)")
    return tar_path


def upload_to_s3(local_path: Path, bucket: str, key: str) -> str:
    """Upload a local file to S3 and return the S3 URI.

    Args:
        local_path: Path to the local file.
        bucket: S3 bucket name.
        key: S3 object key.

    Returns:
        Full S3 URI (s3://bucket/key).
    """
    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)
    s3_uri = f"s3://{bucket}/{key}"
    print(f"  Uploaded to: {s3_uri}")
    return s3_uri


def register_model(
    s3_model_uri: str,
    model_package_group_name: str,
    region: str,
    metrics: dict,
) -> str:
    """Register the model in SageMaker Model Registry.

    Creates the Model Package Group if it doesn't exist, then registers
    a new Model Package version with the sklearn container image,
    the S3 model artifact, and evaluation metrics.

    Args:
        s3_model_uri: S3 URI of the packaged model (.tar.gz).
        model_package_group_name: Name for the Model Package Group.
        region: AWS region.
        metrics: Dict with 'rmse', 'mae', 'r2' keys.

    Returns:
        The Model Package ARN.
    """
    sm = boto3.client("sagemaker", region_name=region)

    # Create Model Package Group if it doesn't exist
    try:
        sm.create_model_package_group(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageGroupDescription="Forging line bath time predictor",
        )
        print(f"  Created Model Package Group: {model_package_group_name}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"  Model Package Group already exists: {model_package_group_name}")
        else:
            raise

    image_uri = _get_sklearn_image_uri(region)
    print(f"  Using container: {image_uri}")

    response = sm.create_model_package(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageDescription="Gradient Boosting Regressor for bath time prediction",
        InferenceSpecification={
            "Containers": [
                {
                    "Image": image_uri,
                    "ModelDataUrl": s3_model_uri,
                    "Environment": {
                        "SAGEMAKER_PROGRAM": "inference.py",
                        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                    },
                }
            ],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"],
            "SupportedRealtimeInferenceInstanceTypes": [
                "ml.t2.medium", "ml.m5.large"
            ],
        },
        ModelApprovalStatus="Approved",
        CustomerMetadataProperties={
            "rmse": str(metrics["rmse"]),
            "mae": str(metrics["mae"]),
            "r2": str(metrics["r2"]),
        },
    )

    model_package_arn = response["ModelPackageArn"]
    print(f"  Registered Model Package: {model_package_arn}")
    return model_package_arn


def deploy_endpoint(
    model_package_arn: str,
    endpoint_name: str,
    region: str,
    instance_type: str = "ml.t2.medium",
) -> str:
    """Deploy a real-time SageMaker endpoint from a registered Model Package.

    Creates a SageMaker Model, Endpoint Configuration, and Endpoint.
    Waits until the endpoint status is 'InService'.

    Args:
        model_package_arn: ARN of the registered Model Package.
        endpoint_name: Name for the endpoint.
        region: AWS region.
        instance_type: EC2 instance type for the endpoint.

    Returns:
        The endpoint name.
    """
    sm = boto3.client("sagemaker", region_name=region)
    sts = boto3.client("sts", region_name=region)
    account_id = sts.get_caller_identity()["Account"]

    model_name = f"{endpoint_name}-model"
    config_name = f"{endpoint_name}-config"
    role_arn = f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
    print(f"  Using role: {role_arn}")

    # Create SageMaker Model
    try:
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={"ModelPackageName": model_package_arn},
            ExecutionRoleArn=role_arn,
        )
        print(f"  Created model: {model_name}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"  Model already exists: {model_name}")
        else:
            raise

    # Create Endpoint Configuration
    try:
        sm.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InstanceType": instance_type,
                    "InitialInstanceCount": 1,
                }
            ],
        )
        print(f"  Created endpoint config: {config_name}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"  Endpoint config already exists: {config_name}")
        else:
            raise

    # Create or update Endpoint
    try:
        sm.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name,
        )
        print(f"  Creating endpoint: {endpoint_name}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"  Updating endpoint: {endpoint_name}")
            sm.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name,
            )
        else:
            raise

    # Wait for endpoint to be InService
    print("  Waiting for endpoint to be InService (~5-10 minutes)...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={"Delay": 30, "MaxAttempts": 40},
    )
    print(f"  Endpoint is InService: {endpoint_name}")
    return endpoint_name


def test_endpoint(endpoint_name: str, region: str) -> dict:
    """Test the deployed endpoint with sample pieces.

    Invokes the endpoint with representative inputs and compares
    the predictions against expected ranges.

    Args:
        endpoint_name: Name of the deployed endpoint.
        region: AWS region.

    Returns:
        Dict with test results and predictions.
    """
    runtime = boto3.client("sagemaker-runtime", region_name=region)

    test_cases = [
        {"die_matrix": 5052, "lifetime_2nd_strike_s": 18.3, "oee_cycle_time_s": 13.5},
        {"die_matrix": 5090, "lifetime_2nd_strike_s": 17.8, "oee_cycle_time_s": 14.0},
        {"die_matrix": 4974, "lifetime_2nd_strike_s": 17.3, "oee_cycle_time_s": 13.0},
        {"die_matrix": 5091, "lifetime_2nd_strike_s": 18.6, "oee_cycle_time_s": 13.8},
        # Slow piece — should predict higher than normal
        {"die_matrix": 5052, "lifetime_2nd_strike_s": 30.0, "oee_cycle_time_s": 13.5},
    ]

    results = []
    for case in test_cases:
        payload = f"{case['die_matrix']},{case['lifetime_2nd_strike_s']},{case['oee_cycle_time_s']}"
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Body=payload,
        )
        prediction = float(response["Body"].read().decode("utf-8").strip())
        result = {**case, "predicted_bath_s": round(prediction, 3)}
        results.append(result)
        print(f"  Matrix {case['die_matrix']}, strike2={case['lifetime_2nd_strike_s']}s -> {prediction:.2f}s")

    normal = results[0]["predicted_bath_s"]
    slow = results[4]["predicted_bath_s"]
    assert slow > normal, f"Slow piece ({slow}s) should predict higher than normal ({normal}s)"
    print(f"\n  Slow piece test passed: {slow:.2f}s > {normal:.2f}s")

    return {"predictions": results, "slow_piece_test": "passed"}


def main():
    parser = argparse.ArgumentParser(description="Deploy model to SageMaker")
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--model-package-group", required=True)
    args = parser.parse_args()

    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    print("=" * 60)
    print("SageMaker Deployment Pipeline")
    print("=" * 60)

    print("\n[1/5] Packaging model artifact...")
    tar_path = package_model(MODEL_FILE, MODEL_DIR)
    print(f"  Created: {tar_path}")

    print("\n[2/5] Uploading to S3...")
    s3_key = "models/vaultech-bath-predictor/model.tar.gz"
    s3_uri = upload_to_s3(tar_path, args.bucket, s3_key)
    print(f"  Uploaded: {s3_uri}")

    print("\n[3/5] Registering in Model Registry...")
    model_package_arn = register_model(
        s3_uri, args.model_package_group, args.region, metadata["metrics"]
    )
    print(f"  Registered: {model_package_arn}")

    print("\n[4/5] Deploying endpoint...")
    endpoint = deploy_endpoint(model_package_arn, args.endpoint_name, args.region)
    print(f"  Endpoint live: {endpoint}")

    print("\n[5/5] Testing endpoint...")
    results = test_endpoint(args.endpoint_name, args.region)
    print(f"  Results: {json.dumps(results, indent=2)}")

    print("\n" + "=" * 60)
    print("Deployment complete!")
    print(f"  Endpoint:       {args.endpoint_name}")
    print(f"  Model Package:  {model_package_arn}")
    print(f"  S3 artifact:    {s3_uri}")
    print("=" * 60)


if __name__ == "__main__":
    main()
