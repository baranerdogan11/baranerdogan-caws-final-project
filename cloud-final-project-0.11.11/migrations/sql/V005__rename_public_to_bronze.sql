-- Move bronze tables from public schema to dedicated bronze schema.

CREATE SCHEMA IF NOT EXISTS bronze;

ALTER TABLE public.raw_lifetime SET SCHEMA bronze;
ALTER TABLE public.raw_piece_info SET SCHEMA bronze;

-- Drop and recreate the view in the bronze schema (views can't be moved with SET SCHEMA
-- when they reference tables that changed schema).
DROP VIEW IF EXISTS public.v_pieces;

CREATE VIEW bronze.v_pieces AS
WITH lifetime_pivot AS (
    SELECT
        timestamp,
        MAX(CASE WHEN signal LIKE '%lifetime_first%' THEN value END)      AS lifetime_first_s,
        MAX(CASE WHEN signal LIKE '%lifetime_second%' THEN value END)     AS lifetime_second_s,
        MAX(CASE WHEN signal LIKE '%lifetime_drill%' THEN value END)      AS lifetime_drill_s,
        MAX(CASE WHEN signal LIKE '%lifetime_bath%' THEN value END)       AS lifetime_bath_s,
        MAX(CASE WHEN signal LIKE '%upsetting_lifetime%' THEN value END)  AS upsetting_lifetime_s,
        MAX(CASE WHEN signal LIKE '%general.maintenance%' THEN value END) AS lifetime_general_s
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
    l.lifetime_first_s,
    l.lifetime_second_s,
    l.lifetime_drill_s,
    l.lifetime_bath_s,
    l.lifetime_general_s,
    l.upsetting_lifetime_s
FROM piece_info_pivot p
JOIN lifetime_pivot l ON p.timestamp = l.timestamp;
