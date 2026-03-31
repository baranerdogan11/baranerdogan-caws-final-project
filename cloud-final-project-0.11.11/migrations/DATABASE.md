# Database Schema

PostgreSQL database for storing and analyzing forging line process data. Organized in a medallion architecture with bronze (raw) and silver (cleaned) layers.

## Schema Diagram

![Database Schema](assets/schema.png)

## Schemas

| Schema | Layer | Description |
|---|---|---|
| `bronze` | Bronze | Raw signal data as captured by the PLC |
| `silver` | Silver | Cleaned and pivoted piece data |

---

## Bronze — `bronze` schema

Tables in this schema are populated directly by PLC injection and should be treated as immutable.

### `raw_lifetime`

Raw cumulative elapsed times (in **seconds**) per piece from furnace exit to each process stage. One row per signal reading.

| Column | Type | Description |
|---|---|---|
| `timestamp` | `TIMESTAMPTZ` | When the reading was recorded |
| `signal` | `TEXT` | Signal identifier |
| `value` | `DOUBLE PRECISION` | Cumulative time in seconds since furnace exit |

**Indexes**: `timestamp`, `signal`

#### Signal reference

| Signal (contains) | Process stage | Typical value |
|---|---|---|
| `upsetting_lifetime` | 1st strike (upsetting) | ~0.1s (bad data) |
| `lifetime_first` | 2nd strike (1st operation) | ~17–19s |
| `lifetime_second` | 3rd strike (2nd operation) | ~24–26s |
| `lifetime_drill` | 4th strike (drill) | ~37–40s |
| `lifetime_bath` | Quench bath | ~56–59s |
| `general.maintenance` | General lifetime | ~56–59s |


#### Cumulative time order

Times are cumulative from furnace exit, monotonically increasing along the process:

```
Furnace → 2nd strike (~18s) → 3rd strike (~25s) → 4th strike (~38s) → Aux. press (~55s) → Bath (~58s)
```

Partial times between stages can be computed by subtraction:

| Segment | Calculation | Typical |
|---|---|---|
| Furnace → 2nd strike | `lifetime_2nd_strike_s` | ~17–19s |
| 2nd strike → 3rd strike | `lifetime_3rd_strike_s - lifetime_2nd_strike_s` | ~6–7s |
| 3rd strike → 4th strike | `lifetime_4th_strike_s - lifetime_3rd_strike_s` | ~13–14s |
| 4th strike → Aux. press | ~16–17s of `lifetime_bath_s - lifetime_4th_strike_s` | ~16–17s |
| Aux. press → Bath | ~1.5–2s of `lifetime_bath_s - lifetime_4th_strike_s` | ~1.5–2s |

### `raw_piece_info`

Piece identification and die matrix assignment. One row per signal reading.

| Column | Type | Description |
|---|---|---|
| `timestamp` | `TIMESTAMPTZ` | When the reading was recorded (shared with `raw_lifetime`) |
| `signal` | `TEXT` | Signal identifier |
| `value` | `TEXT` | Piece ID or die matrix number |

**Indexes**: `timestamp`, `signal`

#### Signals

| Signal (contains) | What it stores | Example |
|---|---|---|
| `piece_id` | Unique piece identifier | `251106001720` |
| `die_matrix` | Die tooling number | `5052.0` |

#### Die matrices

| Die Matrix | Pieces | Active period |
|---|---|---|
| 4974 | ~16,400 | Nov 2025 |
| 5052 | ~22,400 | Nov 2025 – Feb 2026 |
| 5090 | ~85,500 | Dec 2025 – Feb 2026 |
| 5091 | ~52,300 | Nov 2025 – Mar 2026 |

Most production days show a single active matrix, but changeovers can happen mid-day, resulting in two matrices on the same date.

### `v_pieces` (view)

Convenience view. Pivots and joins `raw_lifetime` and `raw_piece_info` into one row per piece by timestamp.

| Column | Type | Unit | Description |
|---|---|---|---|
| `timestamp` | `TIMESTAMPTZ` | — | When the piece was recorded |
| `piece_id` | `TEXT` | — | Unique piece identifier |
| `die_matrix` | `INTEGER` | — | Die tooling number |
| `lifetime_1st_strike_s` | `DOUBLE PRECISION` | seconds | 1st strike — upsetting (bad data) |
| `lifetime_2nd_strike_s` | `DOUBLE PRECISION` | seconds | Furnace exit → 2nd strike (1st operation) |
| `lifetime_3rd_strike_s` | `DOUBLE PRECISION` | seconds | Furnace exit → 3rd strike (2nd operation) |
| `lifetime_4th_strike_s` | `DOUBLE PRECISION` | seconds | Furnace exit → 4th strike (drill) |
| `lifetime_auxiliary_press_s` | `DOUBLE PRECISION` | seconds | Furnace exit → auxiliary press |
| `lifetime_bath_s` | `DOUBLE PRECISION` | seconds | Furnace exit → quench bath |
| `lifetime_general_s` | `DOUBLE PRECISION` | seconds | General lifetime (≈ bath) |

**Join key**: `timestamp` (shared between both raw tables)

### `flyway_schema_history`

Managed by Flyway. Tracks applied migrations, checksums, and timestamps.

---

## Silver — `silver` schema

### `silver.clean_pieces`

Fully cleaned and validated piece data. One row per piece. Contains only valid pieces — all signal-level noise, tracking failures, outliers, and monotonic order violations have been removed.

| Column | Type | Unit | Description |
|---|---|---|---|
| `timestamp` | `TIMESTAMPTZ` | — | When the piece was recorded |
| `piece_id` | `TEXT` | — | Unique piece identifier |
| `die_matrix` | `INTEGER` | — | Die tooling number |
| `lifetime_2nd_strike_s` | `DOUBLE PRECISION` | seconds | Furnace exit → 2nd strike (1st operation) |
| `lifetime_3rd_strike_s` | `DOUBLE PRECISION` | seconds | Furnace exit → 3rd strike (2nd operation) |
| `lifetime_4th_strike_s` | `DOUBLE PRECISION` | seconds | Furnace exit → 4th strike (drill) |
| `lifetime_auxiliary_press_s` | `DOUBLE PRECISION` | seconds | Furnace exit → auxiliary press |
| `lifetime_bath_s` | `DOUBLE PRECISION` | seconds | Furnace exit → quench bath |
| `lifetime_general_s` | `DOUBLE PRECISION` | seconds | General lifetime (≈ bath) |
| `oee_cycle_time_s` | `DOUBLE PRECISION` | seconds | OEE cycle time: rolling avg of 5 inter-piece intervals (NULL if outside 11–16s) |
| `processed_at` | `TIMESTAMPTZ` | — | When this row was written by the notebook |

**Indexes**: `timestamp` (unique), `die_matrix`, `processed_at`

Note: the 1st strike (upsetting) signal is excluded from silver — it is incorrectly calculated at the PLC and has no analytical value.

---

## Gold — Parquet file (outside database)

### `data/gold/pieces.parquet`

Analysis-ready dataset exported from silver with computed features for analytics and ML. No additional cleaning — silver already contains only valid pieces.

| Column | Type | Unit | Description |
|---|---|---|---|
| `timestamp` | `datetime` | — | When the piece was recorded |
| `piece_id` | `string` | — | Unique piece identifier |
| `die_matrix` | `int64` | — | Die tooling number |
| `lifetime_2nd_strike_s` | `float64` | seconds | Furnace exit → 2nd strike (1st operation) |
| `lifetime_3rd_strike_s` | `float64` | seconds | Furnace exit → 3rd strike (2nd operation) |
| `lifetime_4th_strike_s` | `float64` | seconds | Furnace exit → 4th strike (drill) |
| `lifetime_auxiliary_press_s` | `float64` | seconds | Furnace exit → auxiliary press |
| `lifetime_bath_s` | `float64` | seconds | Furnace exit → quench bath |
| `lifetime_general_s` | `float64` | seconds | General lifetime (≈ bath) |
| `oee_cycle_time_s` | `float64` | seconds | OEE cycle time (NULL if outside 11–16s) |
| `partial_furnace_to_2nd_strike_s` | `float64` | seconds | Time spent: furnace → 2nd strike |
| `partial_2nd_to_3rd_strike_s` | `float64` | seconds | Time spent: 2nd strike → 3rd strike |
| `partial_3rd_to_4th_strike_s` | `float64` | seconds | Time spent: 3rd strike → 4th strike |
| `partial_4th_strike_to_auxiliary_press_s` | `float64` | seconds | Time spent: 4th strike → auxiliary press |
| `partial_auxiliary_press_to_bath_s` | `float64` | seconds | Time spent: auxiliary press → bath |
| `after_gap` | `bool` | — | True if this piece follows a production gap (> 5 min) |
| `production_run_id` | `int64` | — | Groups consecutive pieces within the same production run |

---

## Relationships

- `raw_lifetime` and `raw_piece_info` share the same `timestamp` values — this is the join key
- `v_pieces` is a read-only pivot+join of both bronze tables
- `silver.clean_pieces` is populated from the bronze tables by the `01_bronze_to_silver` notebook
- `data/gold/pieces.parquet` is exported from `silver.clean_pieces` by the `02_silver_to_gold` notebook

## Data Quality Summary

| Issue | Where it exists | Where it's handled |
|---|---|---|
| 1st strike signal (bad data) | Bronze | Dropped in silver |
| Zero values (~1.2%) | Bronze | Removed in silver |
| Duplicate timestamps | Bronze | Deduplicated in silver |
| Extreme outliers (500–730s) | Bronze | Removed in silver (3×IQR per matrix) |
| Monotonic order violations | Bronze | Removed in silver |
| Missing 4th strike data (~16%) | All layers | Kept as NULL |

## Example Queries

```sql
-- Bronze: raw exploration
SELECT * FROM bronze.v_pieces WHERE die_matrix = 5052 LIMIT 10;

-- Silver: clean data
SELECT die_matrix, COUNT(*) AS pieces,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lifetime_bath_s)::NUMERIC, 1) AS median_bath_s
FROM silver.clean_pieces
GROUP BY die_matrix ORDER BY die_matrix;
```

```python
# Gold: analysis in Python
import pandas as pd
df = pd.read_parquet("data/gold/pieces.parquet")
df.groupby("die_matrix")[["partial_furnace_to_2nd_strike_s", "partial_4th_strike_to_auxiliary_press_s"]].median()
```
