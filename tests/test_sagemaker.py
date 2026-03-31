"""
Validation tests for SageMaker deployment (Task 10).

These tests verify that:
1. The model is registered in SageMaker Model Registry
2. The real-time endpoint is InService
3. The endpoint returns valid predictions

Usage:
    export SAGEMAKER_MODEL_PACKAGE_GROUP="your-group-name"
    export SAGEMAKER_ENDPOINT_NAME="your-endpoint-name"
    export AWS_DEFAULT_REGION="your-region"       # e.g. eu-west-1
    uv run pytest tests/test_sagemaker.py -v
"""

import json
import os

import boto3
import pytest


MODEL_PACKAGE_GROUP = os.environ.get("SAGEMAKER_MODEL_PACKAGE_GROUP")
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-west-1")


def _skip_if_not_configured():
    if not MODEL_PACKAGE_GROUP or not ENDPOINT_NAME:
        pytest.skip(
            "SAGEMAKER_MODEL_PACKAGE_GROUP and SAGEMAKER_ENDPOINT_NAME "
            "environment variables must be set"
        )


@pytest.fixture(scope="module")
def sm_client():
    _skip_if_not_configured()
    return boto3.client("sagemaker", region_name=REGION)


@pytest.fixture(scope="module")
def runtime_client():
    _skip_if_not_configured()
    return boto3.client("sagemaker-runtime", region_name=REGION)


# --- Model Registry ---


def test_model_package_group_exists(sm_client):
    """The Model Package Group must exist in the registry."""
    response = sm_client.describe_model_package_group(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP
    )
    assert response["ModelPackageGroupName"] == MODEL_PACKAGE_GROUP


def test_model_package_group_has_versions(sm_client):
    """At least one model version must be registered in the group."""
    response = sm_client.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        MaxResults=5,
    )
    packages = response["ModelPackageSummaryList"]
    assert len(packages) >= 1, "No model versions registered in the group"


def test_latest_model_has_metrics(sm_client):
    """The latest registered model must include evaluation metrics."""
    packages = sm_client.list_model_packages(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    latest_arn = packages["ModelPackageSummaryList"][0]["ModelPackageArn"]
    detail = sm_client.describe_model_package(ModelPackageName=latest_arn)

    # Metrics can be in CustomerMetadataProperties or ModelMetrics
    has_customer_metadata = bool(detail.get("CustomerMetadataProperties"))
    has_model_metrics = bool(detail.get("ModelMetrics"))
    assert has_customer_metadata or has_model_metrics, (
        "Model package has no evaluation metrics attached"
    )


# --- Endpoint ---


def test_endpoint_exists_and_in_service(sm_client):
    """The endpoint must exist and be InService."""
    response = sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    assert response["EndpointStatus"] == "InService", (
        f"Endpoint status is {response['EndpointStatus']}, expected InService"
    )


def test_endpoint_returns_prediction(runtime_client):
    """Invoke the endpoint with a sample piece and verify the response."""
    # CSV format: die_matrix, lifetime_2nd_strike_s, oee_cycle_time_s
    payload = "5052,18.3,13.5"

    response = runtime_client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="text/csv",
        Body=payload,
    )

    body = response["Body"].read().decode("utf-8").strip()
    prediction = float(body)
    assert 40 < prediction < 80, (
        f"Prediction {prediction}s is outside plausible range (40-80s)"
    )


def test_endpoint_prediction_per_matrix(runtime_client):
    """Verify the endpoint returns different predictions for different matrices."""
    predictions = {}
    for matrix in [4974, 5052, 5090, 5091]:
        payload = f"{matrix},18.0,13.5"
        response = runtime_client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="text/csv",
            Body=payload,
        )
        body = response["Body"].read().decode("utf-8").strip()
        predictions[matrix] = float(body)

    # All predictions must be valid
    for matrix, pred in predictions.items():
        assert 40 < pred < 80, f"Matrix {matrix}: prediction {pred}s out of range"

    # Predictions should differ across matrices (different tooling = different times)
    unique_rounded = set(round(p, 1) for p in predictions.values())
    assert len(unique_rounded) > 1, "All matrices returned the same prediction"


def test_endpoint_slow_piece_higher_prediction(runtime_client):
    """A piece that is slow at the 2nd strike should have a higher predicted bath time."""
    normal_payload = "5052,18.0,13.5"
    slow_payload = "5052,30.0,13.5"

    resp_normal = runtime_client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME, ContentType="text/csv", Body=normal_payload,
    )
    resp_slow = runtime_client.invoke_endpoint(
        EndpointName=ENDPOINT_NAME, ContentType="text/csv", Body=slow_payload,
    )

    normal_pred = float(resp_normal["Body"].read().decode("utf-8").strip())
    slow_pred = float(resp_slow["Body"].read().decode("utf-8").strip())

    assert slow_pred > normal_pred, (
        f"Slow piece ({slow_pred}s) should predict higher than normal ({normal_pred}s)"
    )
