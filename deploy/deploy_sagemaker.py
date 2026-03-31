"""
SageMaker deployment script — packages, registers, and deploys the XGBoost model.

Usage:
    uv run python deploy/deploy_sagemaker.py \
      --bucket your-bucket-name \
      --region eu-west-1 \
      --endpoint-name your-endpoint-name \
      --model-package-group your-group-name
"""

import argparse
import json
from pathlib import Path


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_FILE = MODEL_DIR / "xgboost_bath_predictor.json"
METADATA_FILE = MODEL_DIR / "model_metadata.json"


def package_model(model_path: Path, output_dir: Path) -> Path:
    """Package the XGBoost model as a .tar.gz archive for SageMaker.

    SageMaker's built-in XGBoost container expects a file named
    'xgboost-model' at the root of the archive.

    Args:
        model_path: Path to the trained model JSON file.
        output_dir: Directory where the .tar.gz will be created.

    Returns:
        Path to the created .tar.gz file.
    """
    # TODO: implement
    raise NotImplementedError


def upload_to_s3(local_path: Path, bucket: str, key: str) -> str:
    """Upload a local file to S3.

    Args:
        local_path: Path to the local file.
        bucket: S3 bucket name.
        key: S3 object key.

    Returns:
        Full S3 URI (s3://bucket/key).
    """
    # TODO: implement
    raise NotImplementedError


def register_model(
    s3_model_uri: str,
    model_package_group_name: str,
    region: str,
    metrics: dict,
) -> str:
    """Register the model in SageMaker Model Registry.

    Creates the Model Package Group if it doesn't exist, then registers
    a new Model Package version with the XGBoost container image,
    the S3 model artifact, and evaluation metrics.

    Args:
        s3_model_uri: S3 URI of the packaged model (.tar.gz).
        model_package_group_name: Name for the Model Package Group.
        region: AWS region.
        metrics: Dict with 'rmse', 'mae', 'r2' keys.

    Returns:
        The Model Package ARN.
    """
    # TODO: implement
    raise NotImplementedError


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
    # TODO: implement
    raise NotImplementedError


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
    # TODO: implement
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description="Deploy XGBoost model to SageMaker")
    parser.add_argument("--bucket", required=True, help="S3 bucket for model artifact")
    parser.add_argument("--region", default="eu-west-1", help="AWS region")
    parser.add_argument("--endpoint-name", required=True, help="SageMaker endpoint name")
    parser.add_argument("--model-package-group", required=True, help="Model Package Group name")
    args = parser.parse_args()

    # Load model metadata for metrics
    with open(METADATA_FILE) as f:
        metadata = json.load(f)

    print("=" * 60)
    print("SageMaker Deployment Pipeline")
    print("=" * 60)

    # Step 1: Package
    print("\n[1/5] Packaging model artifact...")
    tar_path = package_model(MODEL_FILE, MODEL_DIR)
    print(f"  Created: {tar_path}")

    # Step 2: Upload to S3
    print("\n[2/5] Uploading to S3...")
    s3_key = "models/xgboost-bath-predictor/model.tar.gz"
    s3_uri = upload_to_s3(tar_path, args.bucket, s3_key)
    print(f"  Uploaded: {s3_uri}")

    # Step 3: Register in Model Registry
    print("\n[3/5] Registering in Model Registry...")
    model_package_arn = register_model(
        s3_uri, args.model_package_group, args.region, metadata["metrics"]
    )
    print(f"  Registered: {model_package_arn}")

    # Step 4: Deploy endpoint
    print("\n[4/5] Deploying endpoint...")
    endpoint = deploy_endpoint(model_package_arn, args.endpoint_name, args.region)
    print(f"  Endpoint live: {endpoint}")

    # Step 5: Test
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
