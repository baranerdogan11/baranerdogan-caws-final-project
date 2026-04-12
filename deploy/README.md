# Deployment

## SageMaker deployment

### Resource names

| Resource | Name |
|---|---|
| S3 bucket | `caws-final-project` |
| Model Package Group | `vaultech-bath-predictor` |
| Endpoint name | `vaultech-bath-endpoint` |
| AWS region | `eu-west-1` |

### Prerequisites

1. AWS credentials configured (`aws configure`)
2. A SageMaker execution role named `SageMakerExecutionRole` with `AmazonSageMakerFullAccess` and `AmazonS3FullAccess` policies
3. S3 bucket `caws-final-project` in `eu-west-1`
4. Model retrained with numpy 1.24.4 / sklearn 1.2.2 (matching the SageMaker container):

```bash
docker run --rm \
  -v "$(pwd)/models:/models" \
  -v "$(pwd)/data:/data" \
  python:3.9-slim \
  bash -c "pip install 'numpy==1.24.4' scikit-learn==1.2.2 pandas joblib pyarrow -q && python -c \"
import pandas as pd, joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
df = pd.read_parquet('/data/gold/pieces.parquet')
FEATURES = ['die_matrix', 'lifetime_2nd_strike_s', 'oee_cycle_time_s']
TARGET = 'lifetime_bath_s'
df_model = df[FEATURES + [TARGET]].dropna().copy()
df_model['die_matrix'] = df_model['die_matrix'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(df_model[FEATURES], df_model[TARGET], test_size=0.2, random_state=42, stratify=df_model['die_matrix'])
model = GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, min_samples_leaf=10, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, '/models/xgboost_bath_predictor.joblib')
print('Model saved')
\""
```

### Run the deployment

```bash
uv run python deploy/deploy_sagemaker.py \
  --bucket caws-final-project \
  --region eu-west-1 \
  --endpoint-name vaultech-bath-endpoint \
  --model-package-group vaultech-bath-predictor
```

### Validate

```bash
export SAGEMAKER_MODEL_PACKAGE_GROUP="vaultech-bath-predictor"
export SAGEMAKER_ENDPOINT_NAME="vaultech-bath-endpoint"
export AWS_DEFAULT_REGION="eu-west-1"
uv run pytest tests/test_sagemaker.py -v
```

### What the script does

1. **Package** — zips `model.joblib` + `code/inference.py` into `model.tar.gz`
2. **Upload** — pushes the archive to `s3://caws-final-project/models/vaultech-bath-predictor/model.tar.gz`
3. **Register** — creates a Model Package Group and registers the model with RMSE, MAE, R² metrics
4. **Deploy** — creates a SageMaker endpoint (`ml.t2.medium`) and waits for `InService`
5. **Test** — invokes the endpoint with 5 sample pieces and validates predictions

### Important notes

- The SageMaker sklearn 1.2-1 container uses Python 3.9 + numpy 1.24.4. The model **must** be trained with the same numpy version or it will fail to load.
- The `inference_sagemaker.py` script is packaged inside `code/` in the tar archive and located via `SAGEMAKER_SUBMIT_DIRECTORY=/opt/ml/model/code`.
- AWS session tokens expire — refresh credentials before running if you get auth errors.

### Cleanup (to avoid AWS charges)

```bash
aws sagemaker delete-endpoint --endpoint-name vaultech-bath-endpoint --region eu-west-1
aws sagemaker delete-endpoint-config --endpoint-config-name vaultech-bath-endpoint-config --region eu-west-1
aws sagemaker delete-model --model-name vaultech-bath-endpoint-model --region eu-west-1
```
