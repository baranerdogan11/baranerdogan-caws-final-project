-- Add auxiliary press signal to v_pieces view and silver table.

DROP VIEW IF EXISTS bronze.v_pieces;

CREATE VIEW bronze.v_pieces AS
WITH lifetime_pivot AS (
    SELECT
        timestamp,
        MAX(CASE WHEN signal LIKE '%upsetting_lifetime%' THEN value END)       AS lifetime_1st_strike_s,
        MAX(CASE WHEN signal LIKE '%lifetime_first%' THEN value END)           AS lifetime_2nd_strike_s,
        MAX(CASE WHEN signal LIKE '%lifetime_second%' THEN value END)          AS lifetime_3rd_strike_s,
        MAX(CASE WHEN signal LIKE '%lifetime_drill%' THEN value END)           AS lifetime_4th_strike_s,
        MAX(CASE WHEN signal LIKE '%auxiliary_press%' THEN value END)           AS lifetime_auxiliary_press_s,
        MAX(CASE WHEN signal LIKE '%lifetime_bath%' THEN value END)            AS lifetime_bath_s,
        MAX(CASE WHEN signal LIKE '%general.maintenance%' THEN value END)      AS lifetime_general_s
    FROM bronze.raw_lifetime
    GROUP BY timestamp
),
piece_info_pivot AS (
    SELECT
        timestamp,
        MAX(CASE WHEN signal LIKE '%piece_id%' THEN value END)     AS piece_id,
        MAX(CASE WHEN signal LIKE '%die_matrix%' THEN value END)   AS die_matrix
    FROM bronze.raw_piece_info
    GROUP BY timestamp
)
SELECT
    p.timestamp,
    p.piece_id,
    p.die_matrix::NUMERIC::INTEGER AS die_matrix,
    l.lifetime_1st_strike_s,
    l.lifetime_2nd_strike_s,
    l.lifetime_3rd_strike_s,
    l.lifetime_4th_strike_s,
    l.lifetime_auxiliary_press_s,
    l.lifetime_bath_s,
    l.lifetime_general_s
FROM piece_info_pivot p
JOIN lifetime_pivot l ON p.timestamp = l.timestamp;

-- Add column to silver
ALTER TABLE silver.clean_pieces
    ADD COLUMN IF NOT EXISTS lifetime_auxiliary_press_s DOUBLE PRECISION;
