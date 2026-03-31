# Deployment

## SageMaker deployment

Run the deployment script:

```bash
uv run python deploy/deploy_sagemaker.py \
  --bucket <your-bucket> \
  --region <your-region> \
  --endpoint-name <your-endpoint-name> \
  --model-package-group <your-group-name>
```

## Resource names

| Resource | Name |
|---|---|
| S3 bucket | TODO |
| Model Package Group | TODO |
| Endpoint name | TODO |
| AWS region | TODO |

## Validate

```bash
export SAGEMAKER_MODEL_PACKAGE_GROUP="<your-group-name>"
export SAGEMAKER_ENDPOINT_NAME="<your-endpoint-name>"
export AWS_DEFAULT_REGION="<your-region>"
uv run pytest tests/test_sagemaker.py -v
```
