# Cleaning Up Data

The raw data captured by the forging line sensors contains noise, tracking failures, and known bad signals that must be addressed before any meaningful analysis. This document defines the cleaning rules, explains why each one is necessary, and proposes an execution order.

All rules are derived from the industrial process specification (see [02_ManufacturingProcess.md](02_ManufacturingProcess.md)) and confirmed by exploratory data analysis.

## 1. Discard the 1st strike signal (upsetting)

**Rule**: Drop all records from the `forging_line.main_press.maintenance.forging_line_upsetting_lifetime_piecedata` signal. Do not use it in any analysis, model, or visualization.

**Why**: This signal is supposed to measure the cumulative time from furnace exit to the upsetting operation (1st strike) on the main press. However, it is incorrectly calculated at the PLC level and produces meaningless values.

**Evidence**:
- Values range from 0.00 to 6.70 seconds, while the next stage (2nd strike) has a median of ~18 seconds
- 22.8% of records are zero (vs ~1.2% for other signals)
- The distribution is concentrated around 0.1 seconds with no meaningful variance

This signal cannot be repaired — it must be excluded entirely.

**Database**: When querying `v_pieces`, ignore the `lifetime_1st_strike_s` column.

## 2. Remove zero values

**Rule**: Remove all records where the signal value is `0.00`.

**Why**: A zero value means the PLC did not successfully capture the piece at that stage. This can happen when:
- The piece was not detected by the sensor (missed read)
- The tracking cycle was not properly opened or closed in the PLC program
- A communication error occurred between the sensor and the PLC

A value of 0.00 seconds does not represent a real piece that arrived at a stage in zero time — it is always a tracking failure.

**Impact**: ~1.2% of records across most signals. Small enough to not affect overall statistics, but large enough to skew min values and distort distributions if not removed.

**Database**: Filter with `WHERE lifetime_bath_s > 0` (or the relevant column) in all queries against `v_pieces`.

## 3. Validate cumulative time order

**Rule**: For each piece, verify that cumulative times increase monotonically along the process:

```
2nd strike < 3rd strike < 4th strike < auxiliary press < bath
```

Flag or remove any piece where this order is violated.

**Why**: The signals represent cumulative elapsed time from furnace exit. Since each stage is physically downstream from the previous one, a piece must always arrive at each stage after the previous one. The 1st strike (upsetting) is excluded because its signal is bad data.

A violation means:
- A sensor recorded a value from a different piece (misattribution)
- The PLC assigned timestamps or values incorrectly
- A tracking cycle overlap occurred (new cycle started before the previous one closed)

**Example of a valid piece**:
| Stage | Cumulative time |
|---|---|
| 2nd strike (1st operation) | 17.9s |
| 3rd strike (2nd operation) | 24.6s |
| 4th strike (drill) | 38.0s |
| Auxiliary press | 54.5s |
| Bath | 56.2s |

**Example of an invalid piece** (3rd strike > 4th strike — physically impossible):
| Stage | Cumulative time |
|---|---|
| 2nd strike (1st operation) | 18.1s |
| 3rd strike (2nd operation) | 42.3s |
| 4th strike (drill) | 37.5s |
| Auxiliary press | 54.0s |
| Bath | 58.0s |

## 4. Remove extreme outliers

**Rule**: Remove records with values that fall outside a statistical threshold. Recommended: values beyond `Q3 + 3×IQR` per signal, computed **per die matrix**.

**Why**: The dataset contains extreme values — maximum readings reach 500–730 seconds while the 95th percentile is around 25–71 seconds depending on the stage. These are 10–27× the typical range and represent:
- **Stuck pieces**: a piece that got jammed in the press and was not cleared before the next piece arrived
- **Unclosed tracking cycles**: the PLC started timing a piece but never registered its arrival at the next stage, so the timer kept running
- **Machine stops during tracking**: production was halted (e.g. for maintenance or an emergency) while a piece was mid-process, inflating its travel time

**Reference values per signal**:

| Signal | Process stage | Median | 95th pct | Max | Outlier ratio |
|---|---|---|---|---|---|
| `lifetime_2nd_strike_s` | 2nd strike (1st operation) | 18.1s | 25.1s | 683.3s | 27× |
| `lifetime_3rd_strike_s` | 3rd strike (2nd operation) | 25.1s | 33.3s | 690.4s | 21× |
| `lifetime_4th_strike_s` | 4th strike (drill) | 38.5s | 50.6s | 716.8s | 14× |
| `lifetime_auxiliary_press_s` | Auxiliary press | 54.5s | 65.0s | 700+s | 11× |
| `lifetime_bath_s` | Quench bath | 58.4s | 70.8s | 736.6s | 10× |

**Per-matrix outlier rates (3×IQR method)**:

| Die matrix | Outlier % (bath) | Outlier % (2nd strike) |
|---|---|---|
| 4974 | 6.2% | 3.8% |
| 5052 | 6.8% | 6.3% |
| 5090 | 5.1% | 4.4% |
| 5091 | 4.9% | 4.8% |

## 5. Remove duplicate timestamps

**Rule**: When two consecutive records for the same signal have identical or near-zero (< 0.1s) time intervals, keep only the last one.

**Why**: The PLC occasionally registers the same piece reading twice at virtually the same moment. This is a data capture artifact, not a real second piece. Keeping duplicates would inflate piece counts and skew frequency statistics.

## 6. Handle production gaps

**Rule**: Do not interpolate or fill gaps between records. When gaps exceed a defined threshold (e.g. > 5 minutes), treat them as production breaks.

**Why**: Gaps up to ~353 hours (nearly 15 days) exist between consecutive records. These correspond to:
- **Weekends and holidays** — the line does not operate
- **Planned maintenance** — scheduled stops for tooling changes, inspections, or repairs
- **Unplanned stops** — equipment failures, material shortages, or quality holds

These gaps are not missing data — they are real periods where the line was not producing. Interpolating across them would create artificial readings that distort any time-series analysis.

**Practical impact**: When computing rolling averages, trend lines, or production rates, always group records within continuous production runs separated by the gap threshold.

## 7. Segment all analysis by die matrix

**Rule**: Never compute statistics, thresholds, or model features by mixing pieces from different die matrices. All analysis must be performed **per matrix**.

**Why**: Each die matrix has different tooling geometry and process parameters. The expected travel times vary by matrix — for example, the median bath time ranges from 56.0s (matrix 4974) to 59.2s (matrix 5091). A piece that is "slow" for one matrix may be perfectly normal for another.

This applies to:
- Outlier thresholds (Q3 + 3×IQR must be computed per matrix)
- Reference patterns (expected cumulative and partial times)
- Deviation detection (compare each piece to its own matrix baseline)
- Model training (separate training sets per matrix, or matrix as a feature)

**Die matrix identifier**: stored in the `die_matrix` column of `v_pieces` (values: 4974, 5052, 5090, 5091).

## 8. Handle missing 4th strike data

**Rule**: ~16% of records have `NULL` for the 4th strike signal (`lifetime_4th_strike_s`). These correspond to a period where the drill sensor was not recording.

**How to handle**:
- For analysis that requires the full cumulative profile (2nd strike → 3rd strike → 4th strike → auxiliary press → bath), exclude these records
- For analysis that only needs the total travel time (bath) or the initial stages (2nd/3rd strike), these records can be kept
- Do not impute the missing values — the sensor was offline, not misreading

## 9. Compute and filter OEE cycle time

**Rule**: Compute the OEE cycle time as the **rolling average of the last 5 inter-piece timestamp intervals**. Valid values must be between **11 and 16 seconds**. Set out-of-range values to NULL — the piece itself remains valid, but the OEE metric is not meaningful.

**Why**: The OEE cycle time measures the **time between consecutive pieces** exiting the line — it is a production throughput metric, not a per-piece travel time. The original PLC computes it as the rolling average of the last 5 pieces at the auxiliary press. We approximate it from the timestamp intervals between consecutive pieces.

**How it's computed**:
1. Sort pieces by timestamp
2. Compute the time difference (seconds) between each piece and the previous one
3. Apply a rolling mean with a window of 5
4. Values outside 11–16s are set to NULL

Values outside 11–16s indicate:
- Production stops that inflated the inter-piece time
- Die matrix changeover periods
- Gaps between production runs (weekends, maintenance)

**Result**: stored in `silver.clean_pieces.oee_cycle_time_s`. Approximately 77% of pieces have a valid OEE value (~131k of ~169k).

**Important**: Do not confuse this with the lifetime signals. The OEE cycle time (~13s typical) and the piece travel time (~58s typical) measure fundamentally different things.

## Cleaning pipeline summary

The recommended execution order ensures each step builds on a cleaner dataset:

```
Step 1.  Drop 1st strike (upsetting) signal entirely
         → removes known bad signal before any analysis

Step 2.  Remove records where value = 0
         → eliminates tracking failures

Step 3.  Remove duplicate timestamps (same signal, < 0.1s apart)
         → eliminates PLC double-reads

Step 4.  Segment data by die matrix
         → all subsequent steps operate per-matrix

Step 5.  Remove statistical outliers (Q3 + 3×IQR per signal per matrix)
         → removes stuck pieces, unclosed cycles, machine stops

Step 6.  Validate monotonic cumulative order per piece:
         2nd strike < 3rd strike < 4th strike < auxiliary press < bath
         → removes physically impossible readings

Step 7.  Compute OEE cycle time (rolling avg of 5 inter-piece intervals)
         → set values outside 11–16s to NULL

Step 8.  Mark production gaps (intervals > 5 min between records)
         → prevents cross-gap interpolation in time-series analysis
```

After cleaning, the dataset should contain approximately **160,000–170,000 valid pieces** (from ~180,000 raw) with consistent, physically plausible travel times segmented by die matrix.
