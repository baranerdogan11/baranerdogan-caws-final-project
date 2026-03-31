# Final Project — Expected Tasks

## Timeline

- **Project shared**: 25 March
- **Dedicated class session**: 8 April
- **Delivery deadline**: 14 April

## Starting point

Base repository with:

- A local development environment configured with Python/uv and Docker
- Data in a PostgreSQL database with flyway migrations to set up the schema
- A fixed directory structure with placeholders for notebooks, scripts, and the Streamlit app
- Guide about the final project and the expected tasks (FINAL_PROJECT.md)

## Data lifecycle: Bronze–Silver–Gold

This project follows the **Bronze–Silver–Gold (BSG)** data lifecycle model — a separation of responsibilities across the stages of data maturity, not just a folder structure. Reference: [Bronze–Silver–Gold as a Data Lifecycle Model](https://github.com/oriolrius/data-platform-ops/blob/main/bronze-silver-gold.md)

In this project:

- **Bronze** (PostgreSQL `bronze` schema) preserves the raw PLC signals exactly as captured — no cleaning, no interpretation. It answers: *what did the sensors say?*
- **Silver** (PostgreSQL `silver` schema) applies all cleaning rules to produce validated, consistent facts — one row per piece with trustworthy cumulative times. It answers: *what is correct?*
- **Gold** (Parquet file `data/gold/pieces.parquet`) enriches the clean data with partial times and production context for analytics and ML. It answers: *what does this mean for operations?*

Each task below operates within a specific layer. The architecture details are in `docs/04_DataArchitecture.md`.

---

## Tasks

Below is the list of activities you need to complete. Each task describes what to build, why, and what to deliver. Deliverables are the concrete files or results that will be reviewed. The [evaluation rubric](#evaluation-rubric) at the end of this document maps each task to a grading block and weight.

**Important**: when you finish each task, create a **git tag** (`task-01`, `task-02`, ..., `task-12`) so the instructor can check out the exact state of the code at that point. For example:

```bash
git tag task-03
git push origin task-03
```

### Task 1: Set up the environment and connect to the database

First, initialize your git repository — this is required before you start working:

```bash
git init
git add -A
git commit -m "initial: student starter template"
```

Then follow `README.md` to set up the local development environment: install dependencies with `uv sync`, start the PostgreSQL database with Docker Compose, run Flyway migrations to create the schema, and seed the data. Launch JupyterLab with `uv run lab` and verify you can query the bronze tables.

Deliverable: nothing to submit — but tasks 2+ require a working environment. Git tag: `task-01`

### Task 2: Understand the case and review the data

Read the project documentation to understand the forging line process (`docs/01_TheProduct.md`, `docs/02_ManufacturingProcess.md`) and explore the raw data in `bronze.raw_lifetime` and `bronze.raw_piece_info`. Identify the signals, their meaning, value ranges, and data quality issues (zeros, outliers, the broken upsetting signal). Understand why all analysis must be segmented by die matrix.

Deliverable: `notebooks/00_explore_data.ipynb`. Git tag: `task-02`

### Task 3: Clean the raw data (bronze → silver)

The raw PLC data contains noise that must be removed before any analysis. The cleaning rules and their industrial justification are documented in `docs/03_CleaningUpData.md`. The data architecture (why bronze/silver/gold) is explained in `docs/04_DataArchitecture.md`.

**What you need to build**: a notebook that reads the raw signal tables (`bronze.raw_lifetime`, `bronze.raw_piece_info`), applies all cleaning rules, and writes validated pieces to `silver.clean_pieces`.

The cleaning pipeline should:

1. **Filter bad signals**: exclude the upsetting (1st strike) signal — it is broken at the PLC level — and remove zero values (tracking failures where the PLC lost track of a piece)
2. **Deduplicate**: the PLC occasionally registers the same piece twice at the same timestamp — keep only the last reading
3. **Pivot and join**: transform the tall signal/value format into one row per piece with all cumulative times as columns, joining lifetime readings with piece identification (piece_id, die_matrix)
4. **Drop unidentified pieces**: pieces without a piece_id or die_matrix cannot be analyzed per matrix
5. **Remove outliers**: extreme values (Q3 + 3×IQR **per signal per die matrix**) are stuck pieces, unclosed cycles, or machine stops — not slow pieces
6. **Validate physical order**: cumulative times must increase monotonically along the process (2nd strike < 3rd strike < 4th strike < auxiliary press < bath) — violations indicate sensor errors
7. **Compute OEE cycle time**: rolling average of the last 5 inter-piece timestamp intervals, with values outside 11–16s set to NULL
8. **Print a cleaning report**: how many records were removed at each step and why

After cleaning, you should have approximately **169,000 valid pieces** (from ~180,000 raw).

Deliverable: `notebooks/01_bronze_to_silver.ipynb`. Git tag: `task-03`

### Task 4: Enrich and export to gold (silver → gold)

Silver contains only valid, clean pieces — but it lacks the **inter-stage partial times** that are essential for diagnosing delays. This task enriches the clean data and exports it as a portable parquet file for analytics and ML, then verifies the output quality.

**What you need to build**: a notebook that reads `silver.clean_pieces`, computes derived features, and exports to `data/gold/pieces.parquet`, plus a quality gate notebook that verifies the gold dataset before downstream analysis.

The enrichment should:

1. **Compute partial times between stages**: since the lifetime columns are cumulative from furnace exit, subtract consecutive stages to get the time spent in each segment:

   - Furnace → 2nd strike (robot pick, transfer to main press)
   - 2nd strike → 3rd strike (press retraction, repositioning)
   - 3rd strike → 4th strike (transfer to drill station)
   - 4th strike → Auxiliary press (exit main press, deburring and coining)
   - Auxiliary press → Bath (transport to quench bath)

   These partial times are the key diagnostic values: when a piece is slow, the partial that deviates from the reference tells you which segment caused the delay.
2. **Mark production gaps**: flag pieces that follow a gap > 5 minutes and assign a `production_run_id` to group consecutive pieces within the same uninterrupted run — this prevents time-series analysis from interpolating across weekends, maintenance stops, or changeovers.
3. **Export to parquet**: save the enriched dataset with columns in physical process order. This file is what the analysis notebooks, the ML model, and the Streamlit app will consume.

The quality gate should:

1. **Verify data quality**: confirm no zeros remain, no outliers, monotonic order holds, and OEE values are within the valid range
2. **Inspect per-matrix statistics**: compute partial time statistics per die matrix to confirm the gold dataset matches expected reference values
3. **Validate the exported parquet**: reload the file and verify row counts, column structure, and round-trip integrity

Deliverables: `notebooks/02_silver_to_gold.ipynb`, `notebooks/03_build_clean_dataset.ipynb` (quality gate). Git tag: `task-04`

### Task 5: Analyze per die matrix

Each die matrix has different tooling geometry and expected travel times — a piece that is "slow" for matrix 4974 (median 56s) may be perfectly normal for matrix 5091 (median 59s). This task finds the **reference behavior** for each matrix and uses it to detect and diagnose delays.

**What you need to build**: a notebook that reads the gold parquet and performs per-matrix analysis.

The analysis should:

1. **Establish reference profiles**: compute median cumulative and partial times per die matrix — these are the "normal" values that represent a piece traveling the line without issues
2. **Measure variability per segment**: compute the coefficient of variation (CV = std / median) for each partial time per matrix — high CV indicates an unstable segment, which is a candidate for process improvement
3. **Compute deviations from reference**: for each piece, calculate how much each partial time deviates from its matrix reference — positive deviation = slower than expected
4. **Identify slow pieces**: flag pieces where the total bath time exceeds the 90th percentile for their matrix, and for each slow piece find which segment contributed the most delay (the "penalized segment")
5. **Detect drift over time**: compare daily median bath times at the start and end of each matrix's active period — progressive increases may indicate tooling wear or process degradation

Deliverable: `notebooks/04_analyze_per_matrix.ipynb`. Git tag: `task-05`

### Task 6: Feature selection and predictive model

The goal is to predict the total travel time to the quench bath (`lifetime_bath_s`) using only data available **early in the process** — after the 2nd strike, approximately 18 seconds into the ~58-second journey. If we can predict the total time this early, we can raise real-time alerts for pieces likely to be slow while they are still on the line. The model design is documented in `docs/05_PredictiveModel.md`.

**What you need to build**: a notebook that selects features, trains an XGBoost model, evaluates it, and saves the artifacts for the inference service.

The notebook should:

1. **Select features under a constraint**: only information available after the 2nd strike can be used — later stages (3rd strike, 4th strike, auxiliary press) would make the prediction useless because the piece is almost done. Justify why each feature is included or excluded.
2. **Analyze feature correlations**: compute Pearson correlation between each feature and the target to validate that the selected features have predictive value.
3. **Split train/test**: 80/20 split stratified by die matrix, with a fixed random seed for reproducibility.
4. **Train an XGBoost Regressor**: configure hyperparameters (trees, depth, learning rate, subsampling) and fit on the training set.
5. **Evaluate on the test set**: report RMSE, MAE, and R² — interpret what the error means in the context of a ~58-second bath time.
6. **Check per-matrix performance**: verify the model works equally well across all 4 die matrices, not just on average.
7. **Inspect feature importance**: confirm which features the model relies on most, validating the feature selection rationale.
8. **Save the model and metadata**: export the trained model (`models/xgboost_bath_predictor.json`) and a metadata file (`models/model_metadata.json`) with the feature list, metrics, and training info — these are loaded by the inference service.

Deliverable: `notebooks/05_feature_selection_and_model.ipynb`. Git tag: `task-06`

### Task 7: Build an inference service

The trained model needs to be usable outside the notebook — by the Streamlit app, from the command line, or by any Python consumer. This task wraps the model into a `Predictor` class that provides **bath time prediction** from early-stage features.

**What you need to build**:

1. **A `Predictor` class** (`src/vaultech_analysis/inference.py`) that:

   - Loads the saved XGBoost model and metadata from `models/`
   - `predict(die_matrix, lifetime_2nd_strike_s, oee_cycle_time_s)` → returns predicted bath time and model metrics
   - `predict_batch(df)` → returns a Series of predictions for a DataFrame of pieces
   - Handles missing OEE gracefully (defaults to the median ~13.8s)
   - Validates die_matrix and returns an error for unknown values
2. **A CLI entry point** so the service can be tested from the terminal:

   ```bash
   uv run python -m vaultech_analysis.inference --die-matrix 5052 --strike2 18.3 --oee 13.5
   ```
3. **Tests** (`tests/test_inference.py`): verify that predictions return valid results.

Deliverables:

- `src/vaultech_analysis/inference.py`
- `tests/test_inference.py`
- Git tag: `task-07`

### Task 8: Integrate into the Streamlit app

The prediction must be visible to the user through a dashboard. This task builds a Streamlit app that loads the gold dataset, runs the predictor on all pieces, and lets the user explore individual piece timing. The app is launched with `uv run app`.

**What you need to build**:

1. **Data loading**: read the gold parquet, run `predict_batch()` on all pieces to add predicted bath times and prediction errors. Cache the result so it's only computed once per session.
2. **Sidebar filters**: let the user filter by die matrix, date range, and optionally show only slow pieces (bath time > 90th percentile for their matrix).
3. **Summary metrics**: display total pieces, median bath time, median predicted time, and MAE for the filtered selection.
4. **Pieces table**: a scrollable table showing timestamp, piece ID, die matrix, actual bath time, predicted bath time, prediction error, and OEE. The table should support row selection.
5. **Piece detail panel**: when the user selects a piece from the table, show:
   - Cumulative travel times at each stage vs the die matrix reference (with deviation)
   - Partial times between stages vs reference (with deviation and OK/slow status)
   - A bar chart comparing actual vs reference partial times (process synoptic)

Deliverable: `app/streamlit_app.py`. Git tag: `task-08`

### Task 9: Package the application locally

Before deploying to AWS, the Streamlit app must run correctly as a Docker container.

**What you need to build**:

1. **Write a Dockerfile** that packages only what's needed to run the app: the Streamlit application (`app/`), the inference module (`src/`), the trained model (`models/`), and the gold parquet (`data/gold/`). Use `uv` for dependency management and layer caching. Exclude notebooks, tests, docs, migrations, and raw data.
2. **Build and test locally**: build the image and run it — verify the app loads and predictions work.

Deliverable: `Dockerfile` (in the root folder of the project). Git tag: `task-09`

### Task 10: Deploy the model as a SageMaker inference endpoint

Before wiring the app to AWS, the model itself must be deployed and validated as a standalone SageMaker real-time endpoint.

**What you need to build**: a deployment script (`deploy/deploy_sagemaker.py`) that automates the full SageMaker deployment pipeline. The script has a provided skeleton with function signatures — implement each function.

The script must implement these functions:

1. **`package_model(model_path, output_dir)`** → `Path`: Package the model artifact as a `.tar.gz` archive. SageMaker expects the model file at the root of the archive. The built-in XGBoost container (1.7+) accepts `xgboost-model` as filename. Rename your model JSON, create the archive, and return the path to the `.tar.gz` file.
2. **`upload_to_s3(local_path, bucket, key)`** → `str`: Upload the packaged model to an S3 bucket you control. Return the full S3 URI (`s3://bucket/key`).
3. **`register_model(s3_model_uri, model_package_group_name, region, metrics)`** → `str`: Create a Model Package Group (if it doesn't exist), then register the model as a Model Package. Include the XGBoost container image URI (use `sagemaker.image_uris.retrieve()`), the S3 model artifact path, and your evaluation metrics (RMSE, MAE, R²). Return the Model Package ARN.
4. **`deploy_endpoint(model_package_arn, endpoint_name, region, instance_type)`** → `str`: From the registered Model Package, create a SageMaker Model, an Endpoint Configuration (use `ml.t2.medium`), and an Endpoint. Wait for the endpoint to reach `InService` status. Return the endpoint name.
5. **`test_endpoint(endpoint_name, region)`** → `dict`: Invoke the endpoint with sample pieces (using `boto3` `sagemaker-runtime` `invoke_endpoint`) and return the predictions. Compare against the local model's output for the same inputs — this is your proof that the endpoint works before integrating it into the app.

**Important**: all AWS resources must be created in the **`eu-west-1` (Ireland)** region.

The script should be runnable end-to-end from the command line:

```bash
uv run python deploy/deploy_sagemaker.py \
  --bucket your-bucket-name \
  --region eu-west-1 \
  --endpoint-name your-endpoint-name \
  --model-package-group your-group-name
```

Once your endpoint is live, the provided test script `tests/test_sagemaker.py` must pass. See `tests/README.md` for setup and execution instructions. The tests validate:

| Test                                           | What it checks                                           |
| ---------------------------------------------- | -------------------------------------------------------- |
| `test_model_package_group_exists`            | The Model Package Group exists in the registry           |
| `test_model_package_group_has_versions`      | At least one model version is registered                 |
| `test_latest_model_has_metrics`              | The latest version has RMSE, MAE, R² attached           |
| `test_endpoint_exists_and_in_service`        | The endpoint is live (`InService`)                     |
| `test_endpoint_returns_prediction`           | A sample piece returns a prediction in the 40–80s range |
| `test_endpoint_prediction_per_matrix`        | Different die matrices produce different predictions     |
| `test_endpoint_slow_piece_higher_prediction` | A slow 2nd strike predicts a higher bath time            |

Deliverables:

- `deploy/deploy_sagemaker.py` (deployment script with all 5 functions implemented)
- `deploy/README.md` (documents the names you chose for bucket, endpoint, model package group, and how to re-run)
- All 7 tests in `tests/test_sagemaker.py` passing
- Git tag: `task-10`

### Task 11: Wire the Streamlit app to the SageMaker endpoint and deploy

Now that the endpoint is live and tested, adapt the Streamlit app to use it instead of loading the model locally, and deploy the whole app to AWS.

**What you need to build**:

1. **Adapt the inference**: modify the `Predictor` class (or create an alternative) so that `predict()` and `predict_batch()` call the SageMaker endpoint via `boto3` `invoke_endpoint` instead of loading the XGBoost model from disk. The app should no longer need the `models/` directory — the model lives in SageMaker.
2. **Add an inference debug panel**: when the user selects a piece, show the details of the inference call — the input payload sent to the endpoint, the raw response received, and the round-trip latency. This helps debug and demonstrates that the prediction is coming from SageMaker, not a local model.
3. **Push to Amazon ECR**: build the updated Docker image (which no longer needs `models/` since inference is remote), create a repository in your AWS account, tag and push.
4. **Deploy on ECS/Fargate**: create a cluster, register a task definition (Fargate launch type, `awsvpc` network mode, port 8501), and create a service with a public IP. The task's IAM role must have permission to invoke the SageMaker endpoint.
5. **Verify end-to-end**: access the public URL, select a piece, and confirm the prediction comes from the SageMaker endpoint (visible in the debug panel).

You will need a public subnet, a security group allowing inbound TCP on port 8501, and the `ecsTaskExecutionRole` IAM role with `sagemaker:InvokeEndpoint` permission. 

Deliverables:

- Updated `src/vaultech_analysis/inference.py` (SageMaker endpoint integration)
- Updated `app/streamlit_app.py` (inference debug panel)
- Updated `Dockerfile` (no longer needs `models/`)
- Git tag: `task-11`

### Task 12: Architecture diagram and demo video

**What you need to deliver**:

1. **Architecture diagram**: draw the complete AWS infrastructure showing how the components connect end-to-end. It must include: the user's browser, the Streamlit app on ECS/Fargate, the SageMaker real-time endpoint, the model artifact on S3, the Model Registry, and ECR. Show the data flow with arrows — what happens when a user selects a piece and a prediction is requested.
2. **Demo video (max 5 minutes)**: record a screencast walking through the running application and its architecture. The video must cover:

   - Show the architecture diagram and explain each component and how they connect
   - Open the app in a browser via the public Fargate URL
   - Select a piece and show the prediction coming from SageMaker (visible in the inference debug panel — input payload, endpoint response, latency)
   - Explain the data flow step by step: browser → Fargate (Streamlit) → SageMaker endpoint → XGBoost model → prediction response back to the app
   - Be concise — focus on the architecture and the data flow, not on code walkthrough

Deliverables:

- Architecture diagram (in `solutions/architecture_diagram.png`)
- Demo video (max 5 minutes, in `solutions/demo_video.mp4`)
- Git tag: `task-12`

---

## Delivery requirements

**⚠️⚠️⚠️ If any of these requirements are not met, the project will not be reviewed. ⚠️⚠️⚠️**

1. **Repository URL**: create a file `solutions/repos.md` containing only the URL of your private GitHub repository with the completed code. Not forget to invite the github user: joseporiol.rius (joseporiol.rius@esade.edu); **NO OTHER ACCOUNTS!**
2. **Git tags**: each task must have its corresponding tag (`task-01` through `task-12`) pushed to the repository.
3. **ZIP file**: package the **complete project folder** as a `.zip` file, including all hidden folders (`.git`, `.github`, etc.) ⚠️⚠️⚠️**except `.venv`** (excluded for space reasons)⚠️⚠️⚠️. Example:

   ```bash
   zip -r project.zip . -x ".venv/*"
   ```
4. **Upload to eCampus**: submit the `.zip` file to the designated eCampus assignment before the delivery deadline.
5. **Directory structure**: must follow the structure provided in the base repository. **No invent new folders!**

**➡️ Recommendation**: invest maximum effort in this project — the final exam will build directly on top of it. ℹ️

## Evaluation rubric

| Block                        | Weight  | Tasks | What is evaluated                                                     |
| ---------------------------- | ------- | ----- | --------------------------------------------------------------------- |
| Case understanding           | 10%     | 1–2  | Why analysis is per matrix, what each variable represents             |
| Data cleaning + architecture | 5% + 5% | 3–4  | Coherent cleaning rules, justification, bronze/silver/gold pipeline   |
| Per-matrix analysis          | 10%     | 5     | Reference patterns, variability, deviation, slow piece diagnosis      |
| Predictive model + inference | 5% + 5% | 6–7  | Feature selection, XGBoost metrics, Predictor class, tests            |
| Streamlit app                | 5% + 5% | 8–9  | Dashboard with predictions, detail panel, local Docker                |
| SageMaker deployment         | 20%     | 10    | Model packaging, Model Registry, real-time endpoint, validation tests |
| App on AWS                   | 20%     | 11    | Endpoint integration, debug panel, ECR/Fargate, public URL            |
| Architecture + demo          | 10%     | 12    | Architecture diagram, 5-min video walkthrough, documentation          |

## Minimum passing criteria

To pass, **all** of the following must be true:

1. The delivery requirements are met (ZIP, eCampus, repos.md, git tags)
2. The project runs locally without critical errors following the README instructions
3. Data cleaning is implemented and justified — silver table contains valid pieces
4. The gold parquet is generated with partial times and production runs
5. The XGBoost model is trained, evaluated, and saved to `models/`
6. The Streamlit app runs locally and shows predictions for each piece
7. The Dockerfile builds and the app runs in a container

Tasks 10–12 (SageMaker, Fargate, architecture video) are **not required to pass** but account for **50% of the grade**. A project that only completes tasks 1–9 can reach a maximum of 50%.
