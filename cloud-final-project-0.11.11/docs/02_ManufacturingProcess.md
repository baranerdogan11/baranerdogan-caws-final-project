# Manufacturing Process

Detailed description of the forging line, its process stages, timing characteristics, data capture points, and how the collected data maps to the database schema used for analysis.

## Process overview

The forging line transforms heated steel billets into track chain links (see [01_TheProduct.md](01_TheProduct.md)) through four sequential zones. Each piece is handled by robots and automated transfer systems that move it from one station to the next.

![Process overview — four zones of the forging line with cumulative times at each stage](assets/process_overview.png)

A typical piece takes **56–59 seconds** to travel the entire line from furnace exit to quench bath. A new piece exits the line approximately every **11–16 seconds** (the OEE cycle time).

## Zone 1: Furnace — billet loading and heating

<img src="assets/zone_1_furnace.png" alt="Zone 1: Furnace — billet loading and heating" width="800">

Steel billets (called "tacs" in the plant) are fed into an **induction furnace** where they are heated to the working temperature required for forging.

**What happens:**

- Billets enter through an automated feeding system
- The induction furnace heats each billet to forging temperature
- A quality check determines if the billet meets temperature conditions
- Billets that fail the check are diverted for recycling or discarded

**Data capture:** No cumulative time signal is recorded at this stage. The furnace exit is the **time zero reference** — all subsequent lifetime signals measure elapsed time from this point.

## Zone 2: Main press — stamping operations

<img src="assets/zone_2_main_press.png" alt="Zone 2: Main press — stamping operations" width="800">

This is the core of the forging process. The heated billet arrives at the main press where it undergoes four consecutive operations, each performed by the press with robots handling piece transfer and positioning between stations.

### 1st strike — upsetting (recalcat)

The first operation on the main press. The billet is compressed to redistribute material before the stamping strikes.

**Data capture:**

- Signal: `forging_line.main_press.maintenance.forging_line_upsetting_lifetime_piecedata`
- Database column: `lifetime_1st_strike_s`
- Measures: cumulative time from furnace exit to upsetting completion
- Typical value: ~0.1s (this is suspiciously low)
- **Status: known bad data** — this signal is incorrectly calculated at the source and must be excluded from all analysis

Main press operations:

<img src="assets/zone_2_operations.png" alt="Zone 2: Main press operations" width="800">

### 2nd strike — 1st operation (primera)

The first stamping operation on the main press. The billet receives its initial shape through the die.

**What happens:**

- Robot picks the piece from the upsetting station
- Transfers and positions it for the first stamping die
- Main press performs the 2nd strike

**Data capture:**

- Signal: `forging_line.main_press.maintenance.forging_line_lifetime_first_piecedata`
- Database column: `lifetime_2nd_strike_s`
- Measures: cumulative time from furnace exit to 2nd strike completion
- **Typical value: ~17–19 seconds**

**Segment timing (furnace → 2nd strike):** ~17–19s. This includes robot pick at furnace exit, gripper close/confirmation, transfer trajectory to the main press, zone interlocks/safety clearances, and positioning for the first die.

### 3rd strike — 2nd operation (segona)

The second stamping operation. The piece receives additional forming to complete the link geometry.

**What happens:**

- Press retracts from the 2nd strike
- Robot repositions the piece for the second stamping die
- Main press performs the 3rd strike

**Data capture:**

- Signal: `forging_line.main_press.maintenance.forging_line_lifetime_second_piecedata`
- Database column: `lifetime_3rd_strike_s`
- Measures: cumulative time from furnace exit to 3rd strike completion
- **Typical value: ~24–26 seconds**

**Segment timing (2nd strike → 3rd strike):** ~6–7s. This includes press retraction, robot repositioning, synchronization with the press, and placement confirmation.

### 4th strike — drill (trepant)

The drilling operation on the main press. Pin holes and mounting geometry are drilled into the forged link.

**What happens:**

- Robot picks the piece from the 3rd strike position
- Transfers it to the drill station on the main press
- Drilling operation completes the pin bores

**Data capture:**

- Signal: `forging_line.main_press.maintenance.forging_line_lifetime_drill_piecedata`
- Database column: `lifetime_4th_strike_s`
- Measures: cumulative time from furnace exit to 4th strike (drill) completion
- **Typical value: ~37–40 seconds**
- Note: ~16% of records have NULL for this signal, corresponding to a period where the drill sensor was not recording

**Segment timing (3rd strike → 4th strike):** ~13–14s. This includes robot pick with micro-corrections, wait for drill station availability, transfer trajectory, and placement (possibly with retries if orientation is wrong).

## Zone 3: Auxiliary press — deburring and coining

<img src="assets/zone_3_auxiliary_press.jpg" alt="Zone 3: Auxiliary press — deburring and coining" width="800">

After exiting the main press, the formed piece arrives at a secondary press where deburring (removal of flash/burrs from the stamping) and coining (dimensional refinement) are performed. Robots and transfer systems also operate in this zone.

**What happens:**

- Piece exits the drill station and clears the main press zone
- Robot transfers the piece to the auxiliary press
- Deburring removes excess material from the stamping edges
- Coining refines critical dimensions

**Data capture:**

- Signal: `forging_line.auxiliary_press.maintenance.forging_line_lifetime_auxiliary_press_piecedata`
- Database column: `lifetime_auxiliary_press_s`
- Measures: cumulative time from furnace exit to auxiliary press completion
- **Typical value: ~54–57 seconds**

**Segment timing (4th strike → auxiliary press):** ~16–17s. This includes: exit from main press, robot transfer to auxiliary press, deburring/coining operations, positioning and queue waits.

**Segment timing (auxiliary press → bath):** ~1.5–2s. This includes: exit from auxiliary press, short transport to quench bath, and deposit.

## Zone 4: Quench bath — thermal treatment

<img src="assets/zone_4_quench_bath.png" alt="Zone 4: Quench bath — thermal treatment" width="800">

The final stage. The fully formed piece enters a liquid quench bath for thermal treatment (hardening) to achieve the required metallurgical properties for track chain service.

**What happens:**

- Piece exits the auxiliary press
- Robot or conveyor transports it to the quench bath
- Piece is deposited into the bath
- Thermal treatment completes according to process parameters

**Data capture:**

- Signal: `forging_line.bath.maintenance.forging_line_lifetime_bath_piecedata`
- Database column: `lifetime_bath_s`
- Measures: cumulative time from furnace exit to quench bath entry
- **Typical value: ~56–59 seconds**
- This represents the **total piece travel time** through the entire forging line

There is also a general lifetime signal (`forging_line.general.maintenance.forging_line_lifetime_piecedata`, column `lifetime_general_s`) which is equivalent to the bath time.

## Timing summary

### Cumulative times (from furnace exit)

| Stage | Cumulative time | Database column |
|---|---|---|
| Furnace exit | 0s (reference) | — |
| Main press — 1st strike (upsetting) | ~0.1s (bad data) | `lifetime_1st_strike_s` |
| Main press — 2nd strike (1st operation) | ~17–19s | `lifetime_2nd_strike_s` |
| Main press — 3rd strike (2nd operation) | ~24–26s | `lifetime_3rd_strike_s` |
| Main press — 4th strike (drill) | ~37–40s | `lifetime_4th_strike_s` |
| Quench bath | ~56–59s | `lifetime_bath_s` |

### Partial times (between stages)

| Segment | Typical time | What happens |
|---|---|---|
| Furnace → 2nd strike | ~17–19s | Robot pick at furnace, transfer to main press, gripper operations, positioning |
| 2nd strike → 3rd strike | ~6–7s | Press retraction, robot repositioning, placement |
| 3rd strike → 4th strike | ~13–14s | Transfer within main press to drill station, wait for availability |
| 4th strike → Auxiliary press | ~16–17s | Exit main press, transfer to auxiliary press, deburring and coining |
| Auxiliary press → Bath | ~1.5–2s | Transport to quench bath, deposit |
| **Total** | **~56–59s** | **Full piece journey** |

### Production rate

The **OEE cycle time** measures the time between consecutive pieces exiting the line — a throughput metric. The original PLC computes it as the rolling average of the last 5 pieces at the auxiliary press. Since the auxiliary press signal is not directly captured, the silver layer approximates it from the rolling average of the last 5 inter-piece timestamp intervals.

- **Expected range: 11–16 seconds per piece**
- Computed in silver as: rolling mean of 5 consecutive `timestamp` differences
- Stored in: `silver.clean_pieces.oee_cycle_time_s`
- Values outside the 11–16s range are set to NULL (the piece is valid, but the OEE metric is not)

This means the line can process **225–327 pieces per hour** under normal conditions.

## Die matrices

Each piece is forged using a specific **die matrix** — the physical tooling installed on the main press that determines the shape of the link. Different matrices produce different link models and have different expected travel times.

| Die Matrix | Observed pieces | Active period | Notes |
|---|---|---|---|
| 4974 | ~16,400 | Nov 2025 | Shortest active period |
| 5052 | ~22,400 | Nov 2025 – Feb 2026 | First matrix in the dataset |
| 5090 | ~85,500 | Dec 2025 – Feb 2026 | Most pieces produced |
| 5091 | ~52,300 | Nov 2025 – Mar 2026 | Longest active period |

Changing the die matrix requires a physical tooling changeover on the press. In most cases, only one matrix is active per day, but changeovers can happen mid-shift, resulting in two matrices appearing on the same production date.

**All performance analysis must be segmented by die matrix** — comparing pieces across different matrices is not meaningful because each matrix has its own expected timing profile.

## Piece identification and traceability

Each piece receives a **unique identifier** as it enters the process:

- Signal: `forging_line.general.general.forging_line_piece_id_piecedata`
- Database column: `piece_id`
- Example: `251106001720` (encodes date and sequence number)

This identifier, combined with the timestamp, allows full traceability: every lifetime reading at every stage can be linked back to a specific piece and its die matrix.

## How data is captured

### Signal architecture

The forging line PLC (Programmable Logic Controller) records timestamped signal readings as each piece passes through each stage. The data arrives as **signal/value pairs**:

- **Timestamp**: when the reading was captured (millisecond precision, with timezone)
- **Signal**: a hierarchical identifier following the pattern `{line}.{zone}.{category}.{measurement}`
- **Value**: the measurement (cumulative seconds for lifetime signals, or text for identifiers)

All signals for the same piece share the **same timestamp**, which acts as the join key between lifetime readings and piece identification.

### From sensors to database

The PLC injects signal readings directly into a PostgreSQL database. The data is split across two tables based on data type:

| Database table | What it stores | Rows |
|---|---|---|
| `bronze.raw_lifetime` | 6 lifetime signals — cumulative travel times in seconds | ~1,048,000 |
| `bronze.raw_piece_info` | 2 identification signals — piece_id and die_matrix | ~360,000 |

The `v_pieces` view pivots and joins these tables into a flat structure with **one row per piece** containing all cumulative times and identification. See [DATABASE.md](../migrations/DATABASE.md) for the full schema.

### Data flow diagram

![Data flow — from PLC sensors to PostgreSQL database and the v_pieces analysis view](assets/data_flow.png)

## Possible causes of delay by segment

When a piece takes longer than expected, the partial times between stages identify which segment is responsible. Each segment has its own set of probable mechanical or programming causes:

### Furnace → 2nd strike (1st operation)

- Robot **pick time** at furnace exit (approach, alignment, search)
- **Gripper close** time and "piece gripped" confirmation
- **Retry** attempts (piece slip, misalignment)
- **Transfer trajectory** to main press (limited speed, acceleration, wait points)
- **Zone interlocks** (safety barriers, "zone occupied" signals, doors/curtains)
- **Queue/buffer** waits if the next station is not available

### 2nd strike → 3rd strike (within main press)

- **Retraction** from 2nd strike (clear zone before movement)
- Robot **pick** after strike (variable position → corrections)
- **Gripper open/close** cycles + validations (pressure, sensors)
- **PLC handshake** wait for press ready signal
- **Safety wait points** added to avoid collisions
- **Regrip** if the piece is not properly oriented

### 3rd strike → 4th strike (within main press)

- **Retraction** from 3rd strike position
- **Transfer trajectory** (speed limitations, conservative path)
- **Synchronization** waits (drill station occupied, no entry permission)
- **Positioning** at drill (slow approach, precision points)
- **Confirmation** of gripper/piece presence before release

### 4th strike → Auxiliary press

- Robot **pick** at drill with position variability (micro-corrections)
- **Exit** from main press and clear zone
- **Transfer trajectory** to auxiliary press (limited speed)
- **Queue** at auxiliary press entry (occupied, full buffer)
- **Wait** for auxiliary press permission (station occupied, safety)
- **Positioning** and validation before placing the piece
- **Interlocks** (doors, shared zones, permissions)

### Auxiliary press → Bath

- **Retraction** from auxiliary press (clear zone)
- **Transport** to quench bath (short path)
- **Queue/buffer** at bath (bath occupied, zone not available)
- **Bath permissions** (conditions OK, safety checks)
- **Deposit** into bath (slow approach, drop confirmation)
