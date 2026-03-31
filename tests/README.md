# Tests

## Running tests

Run all local tests (no AWS credentials needed):

```bash
uv run pytest -v
```

Run only the notebook validation tests:

```bash
uv run pytest tests/test_nb*.py -v
```

Run only the inference and causes tests:

```bash
uv run pytest tests/test_inference.py tests/test_causes.py -v
```

Run the end-to-end browser tests (requires `playwright install chromium`):

```bash
# Against local app (starts `uv run app` automatically)
uv run pytest tests/test_app_e2e.py -v

# Against Docker container (builds image, runs container on :8501)
VAULTECH_E2E_DOCKER=1 uv run pytest tests/test_app_e2e.py -v
```

Run the AWS SageMaker tests (requires credentials + deployed endpoint):

```bash
export SAGEMAKER_MODEL_PACKAGE_GROUP="your-group-name"
export SAGEMAKER_ENDPOINT_NAME="your-endpoint-name"
export AWS_DEFAULT_REGION="your-region"   # e.g. eu-west-1

uv run pytest tests/test_sagemaker.py -v
```

## Skip behavior

Tests skip automatically when their prerequisites are not available:

| Prerequisite | Tests affected | How to satisfy |
|---|---|---|
| PostgreSQL running | `test_nb00`, `test_nb01` | `cd infra && docker compose up -d` |
| `silver.clean_pieces` populated | `test_nb01` | Run notebook 01 |
| `data/gold/pieces.parquet` exists | `test_nb02`, `test_nb03`, `test_nb04` | Run notebook 02 |
| `models/` files exist | `test_nb05`, `test_inference` | Run notebook 05 |
| `data/gold/pieces.parquet` + `models/` | `test_app_e2e` | Run notebooks 02 + 05, then `uv run playwright install chromium` |
| AWS env vars set | `test_sagemaker` | Export env vars (see above) |

---

## Test files

### `test_nb00_explore.py` — Bronze data (Task 2)

Validates that the bronze tables exist and contain expected data.

| Test | What it validates |
|---|---|
| `test_bronze_raw_lifetime_exists` | `bronze.raw_lifetime` has >1M rows |
| `test_bronze_raw_piece_info_exists` | `bronze.raw_piece_info` has >300k rows |
| `test_bronze_v_pieces_view_works` | `bronze.v_pieces` view returns >150k rows |
| `test_bronze_has_expected_signals` | 2nd strike, 3rd strike, and bath signals are present |
| `test_bronze_has_four_die_matrices` | Die matrices 4974, 5052, 5090, 5091 all present |

### `test_nb01_bronze_to_silver.py` — Cleaning pipeline (Task 3)

Validates that `silver.clean_pieces` is populated correctly after cleaning.

| Test | What it validates |
|---|---|
| `test_row_count_approximately_169k` | ~169k clean pieces (from ~180k raw) |
| `test_four_die_matrices` | All 4 die matrices present |
| `test_no_first_strike_column` | 1st strike (bad data) excluded from silver |
| `test_no_zero_values` | No zero values in any lifetime column |
| `test_no_null_piece_id` | Every piece has identification |
| `test_no_null_die_matrix` | Every piece has a die matrix |
| `test_monotonic_order` | 2nd strike < 3rd strike < 4th strike < aux press < bath |
| `test_oee_within_valid_range` | All non-null OEE values between 11–16s |
| `test_oee_has_nulls` | Some OEE values are NULL (production gaps) |
| `test_unique_timestamps` | No duplicate timestamps |

### `test_nb02_silver_to_gold.py` — Gold dataset structure (Task 4)

Validates that `data/gold/pieces.parquet` has the expected structure and content.

| Test | What it validates |
|---|---|
| `test_gold_file_exists` | Parquet file exists on disk |
| `test_gold_has_expected_columns` | All 17 expected columns present |
| `test_row_count` | ~169k rows |
| `test_four_die_matrices` | All 4 die matrices present |
| `test_partial_times_are_positive` | No negative partial times |
| `test_partial_times_sum_to_bath` | Sum of partials equals bath time |
| `test_production_run_ids_exist` | >100 production runs identified |
| `test_after_gap_is_boolean` | `after_gap` column is boolean type |
| `test_after_gap_matches_run_boundaries` | Gap flags align with run ID boundaries |

### `test_nb03_quality_gate.py` — Gold data quality (Task 4)

Validates the gold dataset passes all quality checks.

| Test | What it validates |
|---|---|
| `test_no_zero_lifetime_values` | No zeros in any lifetime column |
| `test_monotonic_order_holds` | Cumulative time order is valid |
| `test_oee_within_range` | OEE values between 11–16s |
| `test_no_extreme_outliers_per_matrix` | No values >100s; outlier rate <1% per matrix |
| `test_median_bath_times_per_matrix` | Medians within expected range per matrix |
| `test_partial_times_plausible_per_matrix` | Median partials match physical process timing |
| `test_parquet_round_trip` | File can be re-read with identical shape and columns |

### `test_nb04_analyze_per_matrix.py` — Per-matrix analysis (Task 5)

Validates that per-matrix analysis produces expected reference patterns.

| Test | What it validates |
|---|---|
| `test_each_matrix_has_enough_pieces` | >10k pieces per matrix |
| `test_reference_profiles_differ_across_matrices` | Matrices have distinct median bath times (>1s spread) |
| `test_coefficient_of_variation_computed` | CV is positive and <100% for all segments |
| `test_furnace_to_2nd_strike_most_variable` | Furnace→2nd strike has highest CV in every matrix |
| `test_slow_pieces_approximately_10_percent` | ~10% of pieces above p90 threshold |
| `test_deviation_from_reference_centered_near_zero` | Mean deviation from median is close to zero |

### `test_nb05_model.py` — Trained model (Task 6)

Validates that the trained model and metadata are saved correctly.

| Test | What it validates |
|---|---|
| `test_model_file_loadable` | XGBoost model loads from JSON |
| `test_metadata_has_required_keys` | Metadata has model_type, target, features, metrics, etc. |
| `test_target_is_bath_time` | Target is `lifetime_bath_s` |
| `test_features_are_early_stage_only` | Only die_matrix, 2nd strike, OEE as features |
| `test_no_late_stage_features` | 3rd/4th strike, aux press, bath not in features |
| `test_metrics_present_and_plausible` | RMSE <3s, MAE <2s, R² >0.5 |
| `test_four_die_matrices_in_metadata` | All 4 matrices listed |
| `test_training_rows_plausible` | >80k training rows, >20k test rows |
| `test_model_produces_predictions` | Sample prediction in 40–80s range |
| `test_model_predicts_different_per_matrix` | Different matrices produce different predictions |

### `test_app_e2e.py` — Streamlit dashboard end-to-end (Tasks 8–9)

Playwright browser tests that validate the running Streamlit app. Supports two modes:
- **Local**: starts `uv run app` automatically, runs tests against `localhost:8501`
- **Docker**: builds the Docker image, runs the container, tests against `localhost:8501`

| Test | What it validates |
|---|---|
| `test_app_health_endpoint` | Streamlit health endpoint returns HTTP 200 |
| `test_page_title` | Dashboard title "Piece Travel Time Dashboard" is visible |
| `test_four_summary_metrics` | At least 4 metric cards rendered (Total Pieces, Median Bath, etc.) |
| `test_total_pieces_metric` | "Total Pieces" metric present |
| `test_median_bath_time_metric` | "Median Bath Time" metric present |
| `test_mae_metric` | "MAE" metric present |
| `test_dataframe_visible` | Pieces data table is visible |
| `test_processed_pieces_header` | "Processed Pieces" subheader visible |
| `test_sidebar_die_matrix_filter` | Die Matrix selectbox in sidebar |
| `test_sidebar_date_range_filter` | Date range input in sidebar |
| `test_sidebar_slow_pieces_checkbox` | "Show slow pieces only" checkbox in sidebar |
| `test_sidebar_pieces_shown_metric` | "Pieces shown" count in sidebar |
| `test_info_message_before_selection` | Info message prompting row selection |

### `test_inference.py` — Predictor class (Task 7)

Local tests for the inference service:

| Test | What it validates |
|---|---|
| `test_predict_returns_prediction` | `predict()` returns a bath time in the plausible range |
| `test_predict_without_oee` | Missing OEE defaults to median without error |
| `test_predict_all_matrices` | All 4 die matrices return valid predictions |
| `test_predict_invalid_matrix` | Unknown matrix returns an error dict |
| `test_predict_includes_metrics` | Response includes model metrics (RMSE, MAE, R²) |
| `test_predict_slow_piece_higher_prediction` | A slow 2nd strike predicts a higher bath time |
| `test_predict_batch` | `predict_batch()` handles a DataFrame with mixed values |

### `test_causes.py` — Cause reference table

Local tests for the delay cause reference:

| Test | What it validates |
|---|---|
| `test_all_segments_have_causes` | Every segment has a label and at least one cause |
| `test_segment_order` | Segments are in physical process order |
| `test_get_causes_known_segment` | Known segment returns label + causes |
| `test_get_causes_unknown_segment` | Unknown segment returns empty causes |
| `test_partial_to_segment_mapping` | All 5 partial columns map to valid segment keys |

### `test_sagemaker.py` — SageMaker deployment (Task 10)

AWS tests that validate the model is registered in SageMaker Model Registry and the inference endpoint is working:

| Test | What it validates |
|---|---|
| `test_model_package_group_exists` | The Model Package Group exists in the registry |
| `test_model_package_group_has_versions` | At least one model version is registered |
| `test_latest_model_has_metrics` | The latest version has evaluation metrics (RMSE, MAE, R²) attached |
| `test_endpoint_exists_and_in_service` | The endpoint exists and its status is `InService` |
| `test_endpoint_returns_prediction` | A sample piece (matrix 5052, 2nd strike 18.3s, OEE 13.5s) returns a prediction in the 40–80s range |
| `test_endpoint_prediction_per_matrix` | Different die matrices produce different predictions |
| `test_endpoint_slow_piece_higher_prediction` | A piece slow at the 2nd strike (30s) predicts a higher bath time than a normal piece (18s) |
